"""VecGathererSim: vectorized (batched-over-envs) gatherer fast-sim.

Steps B parallel gatherer episodes at once with pure numpy, producing per-env
results BYTE-IDENTICAL to the scalar AiUtopiaSimEnv (sim_env.py) on its VANILLA
gatherer path (no decision_core, no scout, no shaping flags) -- the M1 gate path.
Per-env parity vs the scalar env is the hard acceptance gate
(tests/unit/test_vec_sim_parity.py).

DESIGN (batched-array state, parity-protected): world state lives as CONTIGUOUS
numpy arrays over B -- agent_pos (B,3), logs (B,n,3), log_alive (B,n), oak (B,),
tick (B,) -- NOT a list of B scalar SimWorld objects. reset() builds the byte-faithful
_JavaRandom arena per env via SimWorld(seed).reset ONCE, then copies it into the
arrays; step() / _build_obs() then read+write the arrays directly, with ZERO per-step
np.stack over python objects and ZERO per-env list-comprehensions in the hot path.
Skill dynamics run via vec_skills.vec_apply_skills, which mutates those same arrays
in place with the float-precision-sensitive walk written in CLOSED FORM (fuzzed equal
to the scalar walk loop to ~1e-13, well under the parity atol), so the documented
walk-stall gotcha cannot drift. Obs + reward + termination are likewise vectorized: a
single batched gatherer_nearest_columns_batched pass (the ~65%-of-step-time topmost-
column scan, top-8 via argpartition), a closed-form numpy reward proven equal to
compute_reward_stage_1 (gatherer), and boolean term/trunc masks.

AUTO-RESET: step() auto-resets envs that terminated/truncated and returns the FRESH
reset obs for those rows; last_episode_return / last_episode_length carry the
just-finished episode stats (NaN / -1 otherwise). Obs is built ONCE per step over all
B; _autoreset then re-seeds the done envs' state arrays and rebuilds obs for the DONE
SUBSET ONLY (no second full-B build), scattering those fresh-reset rows back in. The
parity test compares only NON-terminating steps so the substitution never contaminates
the dynamics check; a partial-done test guards the subset-rebuild path.

IMPORT-LIGHT: numpy + aiutopia.sim.* + the pure reward/spaces/_embeds leaves only --
NEVER chromadb / py4j / torch / sentence_transformers.
"""

from __future__ import annotations

import numpy as np

from aiutopia.env._embeds import _agent_uuid_embed, gatherer_goal_embedding_stub
from aiutopia.env.reward import GAMMA, GAMMA_CLIP, TIME_PENALTY
from aiutopia.env.spaces import N_GATHERER_SKILLS, N_TARGET_CLASSES_PER_ROLE
from aiutopia.sim.obs_adapter import (
    _BIOME_ID,
    _LIGHT_LEVEL,
    _TIME_OF_DAY,
    _WEATHER,
    GRID_RADIUS,
    INV_SLOTS,
    REACH_RADIUS_BLOCKS,
    STONE_AXE_ID,
)
from aiutopia.sim.vec_obs import gatherer_nearest_columns_batched
from aiutopia.sim.vec_skills import vec_apply_skills
from aiutopia.sim.world import SimWorld

__all__ = ["VecGathererSim"]

_OAK_LOG_ID = 132  # _ITEM_ID_TO_NAME[132] == "oak_log"
_OAK_LOG_VALUE = 1.0  # LOG_VALUE["oak_log"]
_OAK_CAP = 256  # ROLE_INVENTORY_CAPS["gatherer"]["oak_log"]
_GOAL_OAK = 64  # gatherer_stub_subgoal: collect 64 oak_log

_ARENA_CENTER_X = 64.0
_ARENA_CENTER_Z = -48.0
_ARENA_HALF = 24.0
_ARENA_FLOOR_Y = 60.0

_W = 2 * GRID_RADIUS  # 32
_NCELL = _W * _W  # 1024
_GRID_CHANNELS = 6
_GRID_FLAT = _NCELL * _GRID_CHANNELS  # 6144

_AGENT_ID = "gatherer_0"
_UUID_EMBED = _agent_uuid_embed(_AGENT_ID).astype(np.float32)
_GOAL_EMBED = gatherer_goal_embedding_stub().astype(np.float32)
_ROLE_ONE_HOT = np.array([1, 0, 0, 0], dtype=np.int32)

_HARVEST = 1
_DEPOSIT_CHEST = 2
_WAIT = 4


class VecGathererSim:
    """Vectorized gatherer sim. B parallel envs, pure-numpy obs/reward/term.

    Args:
        num_envs: number of parallel envs B.
        max_episode_ticks: per-episode tick (== env-step) budget for truncation.
        spawn_jitter: TRAINING-only ± block displacement of the agent spawn.
        approach_shaping: TRAINING-only PBRS toward nearest log while masked.
        force_masked_spawn: TRAINING-only push agent out until HARVEST masked.
        randomize_layout: gate for the three curriculum knobs above (training).
    """

    def __init__(
        self,
        num_envs: int,
        max_episode_ticks: int = 300,
        *,
        spawn_jitter: float = 0.0,
        approach_shaping: bool = False,
        force_masked_spawn: bool = False,
        randomize_layout: bool = False,
    ) -> None:
        """Build B parallel envs; see the class docstring for the args."""
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.max_episode_ticks = int(max_episode_ticks)
        # TRAINING-only curriculum knobs (parity-faithful to scalar sim_env.py).
        # All default OFF so the vanilla M1 gate path stays byte-identical (the
        # test_vec_sim_parity.py contract). spawn_jitter + force_masked_spawn run
        # the SCALAR arithmetic on the per-env SimWorld at reset/autoreset
        # (parity-by-construction); only approach_shaping touches the per-step hot
        # path (vectorized below). Like the scalar env, every knob is gated on
        # randomize_layout (training); fixed-seed eval never jitters/masks/shapes.
        self._spawn_jitter = float(spawn_jitter)
        self._approach_shaping = bool(approach_shaping)
        self._force_masked_spawn = bool(force_masked_spawn)
        self._randomize_layout = bool(randomize_layout)
        # Separate prev-potential store for approach_shaping (mirrors scalar
        # _prev_phi_approach). Seeded at reset, telescoped per step while masked.
        self._prev_phi_approach = np.zeros(self.num_envs, dtype=np.float64)
        # BATCHED-ARRAY STATE (replaces a list of B scalar SimWorld objects).
        # All hot-path reads/writes hit these contiguous arrays directly -- no
        # per-step np.stack over python objects, no per-env list-comprehensions.
        # Allocated lazily in reset() once n (logs-per-env) is known.
        self.agent_pos = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.logs = np.zeros((self.num_envs, 0, 3), dtype=np.int64)
        self.log_alive = np.zeros((self.num_envs, 0), dtype=bool)
        self.oak = np.zeros(self.num_envs, dtype=np.int64)
        self.tick = np.zeros(self.num_envs, dtype=np.int64)
        self.seeds = np.zeros(self.num_envs, dtype=np.int64)
        self._ep_return = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_len = np.zeros(self.num_envs, dtype=np.int64)
        self.last_episode_return = np.full(self.num_envs, np.nan, dtype=np.float64)
        self.last_episode_length = np.full(self.num_envs, -1, dtype=np.int64)

    def _columns_batched(self):
        return gatherer_nearest_columns_batched(self.logs, self.log_alive, self.agent_pos)

    def _build_obs(self) -> dict:
        B = self.num_envs
        grid, nearest8, nearest_dist, richness = self._columns_batched()

        # (B,32,32) occupancy -> (B,32,32,6) zeros (channel 0=log), C-order flatten
        # to (B, 6144) -- byte-identical to the scalar grid.reshape(-1).
        grid6 = np.zeros((B, _W, _W, _GRID_CHANNELS), dtype=np.float32)
        grid6[..., 0] = grid
        g_resource_grid = grid6.reshape(B, _GRID_FLAT)

        pos = self.agent_pos.astype(np.float32)
        ticks = self.tick.astype(np.int32)

        inv_ids = np.zeros((B, INV_SLOTS), dtype=np.int32)
        inv_counts = np.zeros((B, INV_SLOTS), dtype=np.int32)
        inv_ids[:, 0] = STONE_AXE_ID
        inv_counts[:, 0] = 1
        self._fill_oak_slots(inv_ids, inv_counts, self.oak)

        mask = self._build_action_mask(inv_counts, nearest_dist)

        return {
            "agent_uuid_embed": np.broadcast_to(_UUID_EMBED, (B, _UUID_EMBED.size)).copy(),
            "role_one_hot": np.broadcast_to(_ROLE_ONE_HOT, (B, 4)).copy(),
            "goal_embedding": np.broadcast_to(_GOAL_EMBED, (B, _GOAL_EMBED.size)).copy(),
            "position": pos,
            "tick_in_episode": ticks[:, None],
            "inv_slot_item_ids": inv_ids,
            "inv_slot_counts": inv_counts,
            "g_resource_grid": g_resource_grid,
            "g_nearest_resources": nearest8,
            "g_richness_score": richness,
            "g_hostiles_nearby": np.zeros((B, 4, 4), dtype=np.float32),
            "action_mask": mask,
            "velocity": np.zeros((B, 3), dtype=np.float32),
            "yaw_pitch": np.zeros((B, 2), dtype=np.float32),
            "health": np.full((B, 1), 20.0, dtype=np.float32),
            "hunger": np.full((B, 1), 20.0, dtype=np.float32),
            "saturation": np.full((B, 1), 20.0, dtype=np.float32),
            "armor_value": np.zeros((B, 1), dtype=np.float32),
            "main_hand_item_id": np.full(B, STONE_AXE_ID, dtype=np.int32),
            "off_hand_item_id": np.zeros(B, dtype=np.int32),
            "goal_ticks_left": np.zeros((B, 1), dtype=np.int32),
            "time_of_day": np.full((B, 1), _TIME_OF_DAY, dtype=np.int32),
            "weather": np.full(B, _WEATHER, dtype=np.int32),
            "biome_id": np.full(B, _BIOME_ID, dtype=np.int32),
            "light_level": np.full((B, 1), _LIGHT_LEVEL, dtype=np.int32),
            "comm_payloads": np.zeros((B, 32, 128), dtype=np.float32),
            "comm_metadata": np.zeros((B, 32, 8), dtype=np.float32),
        }

    @staticmethod
    def _fill_oak_slots(inv_ids, inv_counts, oak) -> None:
        """Pack oak_log counts into slots 1.. (64/slot), mirroring _inv_slots.

        Vectorized over B: each env gets ``full = oak // 64`` slots of 64 plus one
        remainder slot of ``oak % 64`` (if nonzero), capped at the INV_SLOTS-1 usable
        slots. Byte-identical to the per-env fill loop (which stopped at slot
        INV_SLOTS); the oak cap (256) keeps every count well within 5 slots.
        """
        usable = INV_SLOTS - 1  # slots 1..INV_SLOTS-1
        oak64 = np.minimum(oak.astype(np.int64), usable * 64)  # cap to packable slots
        full = (oak64 // 64).astype(np.int64)
        rem = (oak64 % 64).astype(np.int64)
        slot_idx = np.arange(1, INV_SLOTS)[None, :]  # (1, usable)
        full_b = full[:, None]
        is_full = slot_idx <= full_b  # slots [1..full] hold 64
        is_rem = (slot_idx == (full_b + 1)) & (rem[:, None] > 0)  # next slot holds rem
        occupied = is_full | is_rem
        inv_ids[:, 1:][occupied] = _OAK_LOG_ID
        counts = np.where(is_full, 64, 0) + np.where(is_rem, rem[:, None], 0)
        inv_counts[:, 1:] = counts.astype(inv_counts.dtype)

    def _build_action_mask(self, inv_counts, nearest_dist) -> dict:
        """Vectorized compute_gatherer_action_mask for the M1B arena facts.

        Sizes to the INPUT batch (inv_counts.shape[0]), not self.num_envs, so it
        works for both the full-B step build and the done-subset autoreset build.
        """
        B = inv_counts.shape[0]
        skill = np.ones((B, N_GATHERER_SKILLS), dtype=np.int8)
        target = np.ones((B, N_GATHERER_SKILLS, N_TARGET_CLASSES_PER_ROLE), dtype=np.int8)

        inv_full = inv_counts.sum(axis=1) >= INV_SLOTS * 64  # 36*64
        skill[:, _DEPOSIT_CHEST] = 0  # no chest ever in range
        target[:, _DEPOSIT_CHEST, :] = 0
        resource_in_range = nearest_dist <= REACH_RADIUS_BLOCKS
        harvest_masked = inv_full | (~resource_in_range)
        skill[harvest_masked, _HARVEST] = 0
        target[~resource_in_range, _HARVEST, :] = 0
        none_legal = ~skill.astype(bool).any(axis=1)
        skill[none_legal, _WAIT] = 1

        return {
            "skill_type": skill,
            "target_per_skill": target,
            "comm_payload": np.ones((B, 1), dtype=np.int8),
            "should_broadcast": np.ones((B, 2), dtype=np.int8),
        }

    def _apply_spawn_curriculum(self, w: SimWorld, seed: int) -> None:
        """Apply TRAINING-only spawn jitter + force_masked_spawn to a SimWorld.

        Runs the EXACT scalar AiUtopiaSimEnv.reset arithmetic on ``w`` so it is
        parity-by-construction (bit-identical to the scalar path, not fuzz-equal).
        No-op unless ``randomize_layout`` AND the relevant knob is set. Mutates only
        ``w.agent_pos``.
        """
        if not self._randomize_layout:
            return
        import math  # noqa: PLC0415

        from aiutopia.sim.obs_adapter import (  # noqa: PLC0415
            REACH_RADIUS_BLOCKS,
            gatherer_nearest_columns,
        )

        half = _ARENA_HALF
        # spawn_jitter: seeded per-env horizontal displacement, clamped to arena.
        if self._spawn_jitter > 0.0:
            jr = np.random.default_rng(seed * 2654435761 & 0xFFFFFFFF)
            off = jr.uniform(-self._spawn_jitter, self._spawn_jitter, size=2)
            w.agent_pos[0] = float(
                np.clip(
                    w.agent_pos[0] + off[0],
                    _ARENA_CENTER_X - half,
                    _ARENA_CENTER_X + half,
                )
            )
            w.agent_pos[2] = float(
                np.clip(
                    w.agent_pos[2] + off[1],
                    _ARENA_CENTER_Z - half,
                    _ARENA_CENTER_Z + half,
                )
            )
        # force_masked_spawn: push the agent outward from the nearest perceived
        # trunk until its topmost log is out of reach (HARVEST masked), capped iters.
        if self._force_masked_spawn:
            for _ in range(40):
                _ct, nb = gatherer_nearest_columns(w)
                if not nb or math.sqrt(nb[0][5]) > REACH_RADIUS_BLOCKS:
                    break  # already masked (or nothing in perception)
                dx0 = nb[0][2]
                dz0 = nb[0][4]
                nrm = math.hypot(dx0, dz0) or 1.0
                w.agent_pos[0] = float(
                    np.clip(
                        w.agent_pos[0] - dx0 / nrm,
                        _ARENA_CENTER_X - half,
                        _ARENA_CENTER_X + half,
                    )
                )
                w.agent_pos[2] = float(
                    np.clip(
                        w.agent_pos[2] - dz0 / nrm,
                        _ARENA_CENTER_Z - half,
                        _ARENA_CENTER_Z + half,
                    )
                )

    def reset(self, seeds) -> dict:
        """Reset all envs to the given per-env seeds; return the batched obs dict.

        Builds the byte-faithful _JavaRandom arena per env via SimWorld(seed).reset
        ONCE, then copies the per-env layout into the contiguous batched-state arrays.
        After this, step()/_build_obs() touch only the arrays -- never SimWorld again.
        """
        seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
        if seeds.shape[0] != self.num_envs:
            raise ValueError(f"seeds length {seeds.shape[0]} != num_envs {self.num_envs}")
        self.seeds = seeds.copy()

        # Build per-env arenas once, then stack into batched arrays.
        worlds = [SimWorld() for _ in range(self.num_envs)]
        for i, w in enumerate(worlds):
            w.reset(int(seeds[i]), arena_mode="trees")
            # TRAINING-only spawn curriculum: run the SCALAR jitter / force-mask
            # arithmetic on this per-env SimWorld (parity-by-construction). No-op
            # unless randomize_layout AND the relevant knob is set.
            self._apply_spawn_curriculum(w, int(seeds[i]))
        n = worlds[0].logs.shape[0]
        # Uniform-n guard: batched logs (B,n,3) require equal n across envs. The
        # gate path is trees-only (n=64 always); a future mixed/cluster mode with a
        # different log count must fail loud here rather than stack ragged arrays.
        for i, w in enumerate(worlds):
            if w.logs.shape[0] != n:
                raise ValueError(
                    f"non-uniform logs-per-env: env {i} has {w.logs.shape[0]} != {n} "
                    "(batched-array state requires equal n; only trees mode is supported)"
                )

        self.agent_pos = np.stack([w.agent_pos for w in worlds]).astype(np.float64)
        self.logs = np.stack([w.logs for w in worlds]).astype(np.int64)
        self.log_alive = np.stack([w.log_alive for w in worlds]).astype(bool)
        self.oak = np.zeros(self.num_envs, dtype=np.int64)
        self.tick = np.zeros(self.num_envs, dtype=np.int64)

        self._ep_return[:] = 0.0
        self._ep_len[:] = 0
        self.last_episode_return[:] = np.nan
        self.last_episode_length[:] = -1
        # Seed the approach-shaping potential at the (post-curriculum) spawn so the
        # first step's telescoping delta is measured from the spawn (mirrors scalar
        # _prev_phi_approach = _log_potential(world) at the end of reset()).
        self._prev_phi_approach = self._log_potential_batched()
        return self._build_obs()

    def _log_potential_batched(self, idx: np.ndarray | None = None) -> np.ndarray:
        """Vectorized scalar ``_log_potential`` (-0.1 * nearest alive-log dist).

        Mirrors sim_env._log_potential EXACTLY (W=0.1, all alive logs, full 3D
        distance to ``agent_pos``). NOT the perception/floored ``nearest_dist`` --
        reusing that would be a parity bug. ``idx`` restricts to a row subset (for
        autoreset re-seeding); None means all B rows.
        """
        if idx is None:
            logs = self.logs
            alive = self.log_alive
            pos = self.agent_pos
        else:
            logs = self.logs[idx]
            alive = self.log_alive[idx]
            pos = self.agent_pos[idx]
        d = logs.astype(np.float64) - pos[:, None, :]
        dist = np.sqrt((d * d).sum(axis=2))
        dist = np.where(alive, dist, np.inf)
        min_dist = dist.min(axis=1)
        # No alive log -> potential 0.0 (scalar returns 0.0 when none alive).
        phi = np.where(np.isfinite(min_dist), -0.1 * min_dist, 0.0)
        return phi.astype(np.float64)

    def step(self, actions):
        """Advance all envs one macro-step.

        Returns (obs, reward(B,), terminated(B,), truncated(B,)). Envs that
        terminate/truncate are auto-reset (same seed) and the returned obs row
        holds the FRESH reset obs.
        """
        B = self.num_envs
        skill_type = np.asarray(actions["skill_type"]).reshape(-1).astype(np.int64)
        target_class = (
            np.asarray(actions.get("target_class", np.zeros(B, dtype=np.int64)))
            .reshape(-1)
            .astype(np.int64)
        )
        spatial = np.asarray(actions["spatial_param"], dtype=np.float64).reshape(B, 3)
        scalar = np.asarray(actions["scalar_param"], dtype=np.float64).reshape(B, -1)

        # Snapshot oak BEFORE the in-place skill mutation (Trap A: self.oak is
        # mutated in place by vec_apply_skills, so prev_oak MUST be a copy or the
        # reward delta is identically 0).
        prev_oak = self.oak.copy()

        # 1. Advance world state via the VECTORIZED batched skill dynamics
        # (vec_skills.vec_apply_skills) -- byte-identical per-env to looping the
        # scalar apply_skill (locked by tests/unit/test_vec_skills.py +
        # test_vec_sim_parity.py), but HARVEST walk-chains and NAVIGATE walks run
        # as numpy over B at once instead of B x 64 x ~75 python tick-steps. Now
        # mutates the batched-state arrays in place (no SimWorld stack/scatter).
        # Returns the per-env clipped popcount (== bin(bitset).count("1")) directly.
        n_clipped = vec_apply_skills(
            skill_type,
            target_class,
            spatial,
            scalar,
            self.agent_pos,
            self.log_alive,
            self.logs,
            self.oak,
        )
        # 2. env-step counter (one tick per step; walk-ticks NOT counted).
        self.tick += 1

        curr_oak = self.oak

        # 3. Reward (vectorized; proven == compute_reward_stage_1 gatherer).
        reward = self._gatherer_reward(prev_oak, curr_oak, n_clipped)

        # 4. Obs (vectorized) -- built EXACTLY ONCE per step.
        obs = self._build_obs()

        # 4b. approach_shaping (TRAINING-only): PBRS distance-reduction toward the
        # nearest alive log ONLY while HARVEST is masked. Mirrors scalar sim_env
        # EXACTLY: phi_a = _log_potential(world) on the POST-step world; the masked
        # gate is obs action_mask skill_type[1] == 0; add phi_a - prev ONLY when
        # masked; UPDATE prev for ALL envs every step (telescoping). Off by default.
        if self._approach_shaping:
            phi_a = self._log_potential_batched()  # (B,) post-step potential
            harvest_masked = obs["action_mask"]["skill_type"][:, _HARVEST] == 0
            reward = reward + np.where(harvest_masked, phi_a - self._prev_phi_approach, 0.0)
            self._prev_phi_approach[:] = phi_a

        # 5. Termination (goal) + truncation (tick budget OR OOB box).
        # Trap B: the OOB box test must use the float32 position (agent_pos cast to
        # float32 then back to float64) to match the scalar env's obs-position
        # round-trip exactly. Reading raw float64 self.agent_pos here would be an
        # untestable parity drift (the parity sequence never truncates), so we mirror
        # obs["position"] (which is self.agent_pos.astype(float32)).
        terminated = curr_oak >= _GOAL_OAK
        pos = obs["position"]
        dx = np.abs(pos[:, 0].astype(np.float64) - _ARENA_CENTER_X)
        dz = np.abs(pos[:, 2].astype(np.float64) - _ARENA_CENTER_Z)
        y = pos[:, 1].astype(np.float64)
        oob = (dx > _ARENA_HALF) | (dz > _ARENA_HALF) | (y < _ARENA_FLOOR_Y)
        truncated = (self.tick >= self.max_episode_ticks) | oob

        self._ep_return += reward
        self._ep_len += 1
        done = terminated | truncated
        self.last_episode_return[:] = np.nan
        self.last_episode_length[:] = -1
        if done.any():
            self.last_episode_return[done] = self._ep_return[done]
            self.last_episode_length[done] = self._ep_len[done]
            self._autoreset(done, obs)

        return obs, reward, terminated, truncated

    def _gatherer_reward(self, prev_oak, curr_oak, n_clipped):
        """Vectorized gatherer stage-1 reward == compute_reward_stage_1(gatherer).

            r_primary = d_oak * LOG_VALUE[oak_log]                (task allowlist)
            r_pbrs    = gamma*min(curr,256)*v - min(prev,256)*v   (Phi; oak-only)
            r         = r_primary + r_pbrs - TIME_PENALTY - GAMMA_CLIP*n_clipped
        (r_death and r_exploits are 0 in the M1B arena.)
        """
        v = _OAK_LOG_VALUE
        delta = (curr_oak - prev_oak).astype(np.float64)
        r_primary = delta * v
        phi_prev = np.minimum(prev_oak, _OAK_CAP).astype(np.float64) * v
        phi_curr = np.minimum(curr_oak, _OAK_CAP).astype(np.float64) * v
        r_pbrs = GAMMA * phi_curr - phi_prev
        r_clip = GAMMA_CLIP * n_clipped.astype(np.float64)
        return r_primary + r_pbrs - TIME_PENALTY - r_clip

    def _autoreset(self, done, obs) -> None:
        """Re-seed finished envs in place, then patch ONLY their obs rows.

        Fix 3: the obs for non-done rows was already built once in step(); rebuilding
        the full-B obs a second time (the old behavior) doubled obs-build cost on every
        step that had ANY done env -- which is most steps in the converged regime. Here
        we rebuild obs for the DONE SUBSET only (gatherer_nearest_columns_batched on the
        re-seeded done slice) and scatter those rows in, so obs is effectively built
        once for live rows + once for done rows, never twice for all B. The returned
        obs row for a done env equals its fresh-reset obs (parity contract preserved).
        """
        idx = np.nonzero(done)[0]
        m = idx.shape[0]
        if m == 0:
            return

        # Re-seed the done envs' state arrays from byte-faithful SimWorld arenas.
        # Under randomize_layout (training) the per-env seed is ADVANCED by num_envs
        # so each env walks a DISJOINT seed stream -- the batched analog of the scalar
        # env's global ``_train_ep`` increment per episode (documented divergence: the
        # exact global-counter schedule across B parallel envs is not reproduced; the
        # per-env streams are disjoint and cover the layout space). Default path
        # (randomize_layout=False) keeps the same-seed re-seed (existing tests green).
        for i in idx:
            ii = int(i)
            if self._randomize_layout:
                self.seeds[ii] = self.seeds[ii] + self.num_envs
            w = SimWorld()
            w.reset(int(self.seeds[ii]), arena_mode="trees")
            # Re-apply the spawn curriculum on the fresh world (parity: scalar runs
            # jitter / force-mask on every reset, not just the first episode).
            self._apply_spawn_curriculum(w, int(self.seeds[ii]))
            self.agent_pos[ii] = w.agent_pos
            self.logs[ii] = w.logs
            self.log_alive[ii] = w.log_alive
            self.oak[ii] = 0
            self.tick[ii] = 0
            self._ep_return[ii] = 0.0
            self._ep_len[ii] = 0

        # Subset obs: only the done rows, via the SAME batched primitives _build_obs
        # uses (so the patched rows are byte-identical to a fresh reset's obs row).
        sub_logs = self.logs[idx]
        sub_alive = self.log_alive[idx]
        sub_pos = self.agent_pos[idx]
        grid, nearest8, nearest_dist, richness = gatherer_nearest_columns_batched(
            sub_logs, sub_alive, sub_pos
        )
        grid6 = np.zeros((m, _W, _W, _GRID_CHANNELS), dtype=np.float32)
        grid6[..., 0] = grid
        g_resource_grid = grid6.reshape(m, _GRID_FLAT)

        inv_ids = np.zeros((m, INV_SLOTS), dtype=np.int32)
        inv_counts = np.zeros((m, INV_SLOTS), dtype=np.int32)
        inv_ids[:, 0] = STONE_AXE_ID
        inv_counts[:, 0] = 1
        self._fill_oak_slots(inv_ids, inv_counts, self.oak[idx])  # oak==0 -> just the axe
        mask = self._build_action_mask(inv_counts, nearest_dist)

        # Scatter the freshly-reset rows into the already-built obs dict.
        obs["position"][idx] = sub_pos.astype(np.float32)
        obs["tick_in_episode"][idx, 0] = self.tick[idx].astype(np.int32)
        obs["inv_slot_item_ids"][idx] = inv_ids
        obs["inv_slot_counts"][idx] = inv_counts
        obs["g_resource_grid"][idx] = g_resource_grid
        obs["g_nearest_resources"][idx] = nearest8
        obs["g_richness_score"][idx] = richness
        for mk, mv in mask.items():
            obs["action_mask"][mk][idx] = mv

        # Re-seed the approach-shaping potential for the just-reset rows to their
        # FRESH spawn potential (mirrors scalar reset() re-seeding _prev_phi_approach
        # = _log_potential(world) on the new episode). No-op when approach_shaping is
        # off; cheap, so unconditional for correctness.
        self._prev_phi_approach[idx] = self._log_potential_batched(idx)
