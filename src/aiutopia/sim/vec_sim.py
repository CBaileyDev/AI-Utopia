"""VecGathererSim: vectorized (batched-over-envs) gatherer fast-sim.

Steps B parallel gatherer episodes at once with pure numpy, producing per-env
results BYTE-IDENTICAL to the scalar AiUtopiaSimEnv (sim_env.py) on its VANILLA
gatherer path (no decision_core, no scout, no shaping flags) -- the M1 gate path.
Per-env parity vs the scalar env is the hard acceptance gate
(tests/unit/test_vec_sim_parity.py).

DESIGN (staged, parity-protected): skill dynamics are advanced by looping the
EXISTING scalar apply_skill over B independent SimWorld instances -- bit-identical
by construction, so the float-precision-sensitive walk loops (a documented project
gotcha) cannot drift. The speedup is in the OBS + REWARD + termination layers, which
ARE vectorized: a single batched gatherer_nearest_columns_batched pass (the
~65%%-of-step-time topmost-column scan), a closed-form numpy reward proven equal to
compute_reward_stage_1 (gatherer), and boolean term/trunc masks.

AUTO-RESET: step() auto-resets envs that terminated/truncated and returns the FRESH
reset obs for those rows; last_episode_return / last_episode_length carry the
just-finished episode stats (NaN / -1 otherwise). The parity test compares only
NON-terminating steps so the substitution never contaminates the dynamics check.

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
    """

    def __init__(self, num_envs: int, max_episode_ticks: int = 300) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.max_episode_ticks = int(max_episode_ticks)
        self.worlds: list[SimWorld] = [SimWorld() for _ in range(self.num_envs)]
        self.seeds = np.zeros(self.num_envs, dtype=np.int64)
        self._ep_return = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_len = np.zeros(self.num_envs, dtype=np.int64)
        self.last_episode_return = np.full(self.num_envs, np.nan, dtype=np.float64)
        self.last_episode_length = np.full(self.num_envs, -1, dtype=np.int64)

    def _columns_batched(self):
        logs = np.stack([w.logs for w in self.worlds]).astype(np.int64)
        alive = np.stack([w.log_alive for w in self.worlds]).astype(bool)
        pos = np.stack([w.agent_pos for w in self.worlds]).astype(np.float64)
        return gatherer_nearest_columns_batched(logs, alive, pos)

    def _build_obs(self) -> dict:
        B = self.num_envs
        grid, nearest8, nearest_dist, richness = self._columns_batched()

        # (B,32,32) occupancy -> (B,32,32,6) zeros (channel 0=log), C-order flatten
        # to (B, 6144) -- byte-identical to the scalar grid.reshape(-1).
        grid6 = np.zeros((B, _W, _W, _GRID_CHANNELS), dtype=np.float32)
        grid6[..., 0] = grid
        g_resource_grid = grid6.reshape(B, _GRID_FLAT)

        pos = np.stack([w.agent_pos for w in self.worlds]).astype(np.float32)
        ticks = np.array([w.tick for w in self.worlds], dtype=np.int32)

        inv_ids = np.zeros((B, INV_SLOTS), dtype=np.int32)
        inv_counts = np.zeros((B, INV_SLOTS), dtype=np.int32)
        inv_ids[:, 0] = STONE_AXE_ID
        inv_counts[:, 0] = 1
        oak = np.array([int(w.inventory.get("oak_log", 0)) for w in self.worlds], dtype=np.int64)
        self._fill_oak_slots(inv_ids, inv_counts, oak)

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
        """Pack oak_log counts into slots 1.. (64/slot), mirroring _inv_slots."""
        for i in range(oak.shape[0]):
            remaining = int(oak[i])
            slot = 1
            while remaining > 0 and slot < INV_SLOTS:
                take = min(remaining, 64)
                inv_ids[i, slot] = _OAK_LOG_ID
                inv_counts[i, slot] = take
                remaining -= take
                slot += 1

    def _build_action_mask(self, inv_counts, nearest_dist) -> dict:
        """Vectorized compute_gatherer_action_mask for the M1B arena facts."""
        B = self.num_envs
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

    def reset(self, seeds) -> dict:
        """Reset all envs to the given per-env seeds; return the batched obs dict."""
        seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
        if seeds.shape[0] != self.num_envs:
            raise ValueError(f"seeds length {seeds.shape[0]} != num_envs {self.num_envs}")
        self.seeds = seeds.copy()
        for i, w in enumerate(self.worlds):
            w.reset(int(seeds[i]), arena_mode="trees")
        self._ep_return[:] = 0.0
        self._ep_len[:] = 0
        self.last_episode_return[:] = np.nan
        self.last_episode_length[:] = -1
        return self._build_obs()

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

        prev_oak = np.array(
            [int(w.inventory.get("oak_log", 0)) for w in self.worlds], dtype=np.int64
        )

        # 1. Advance world state via the VECTORIZED batched skill dynamics
        # (vec_skills.vec_apply_skills) -- byte-identical per-env to looping the
        # scalar apply_skill (locked by tests/unit/test_vec_skills.py +
        # test_vec_sim_parity.py), but HARVEST walk-chains and NAVIGATE walks run
        # as numpy over B at once instead of B x 64 x ~75 python tick-steps.
        # Returns the per-env clipped popcount (== bin(bitset).count("1")) directly.
        n_clipped = vec_apply_skills(self.worlds, skill_type, target_class, spatial, scalar)
        # 2. env-step counter (one tick per step; walk-ticks NOT counted).
        for w in self.worlds:
            w.tick += 1

        curr_oak = np.array(
            [int(w.inventory.get("oak_log", 0)) for w in self.worlds], dtype=np.int64
        )

        # 3. Reward (vectorized; proven == compute_reward_stage_1 gatherer).
        reward = self._gatherer_reward(prev_oak, curr_oak, n_clipped)

        # 4. Obs (vectorized).
        obs = self._build_obs()

        # 5. Termination (goal) + truncation (tick budget OR OOB box).
        terminated = curr_oak >= _GOAL_OAK
        pos = obs["position"]
        dx = np.abs(pos[:, 0].astype(np.float64) - _ARENA_CENTER_X)
        dz = np.abs(pos[:, 2].astype(np.float64) - _ARENA_CENTER_Z)
        y = pos[:, 1].astype(np.float64)
        oob = (dx > _ARENA_HALF) | (dz > _ARENA_HALF) | (y < _ARENA_FLOOR_Y)
        ticks = np.array([w.tick for w in self.worlds], dtype=np.int64)
        truncated = (ticks >= self.max_episode_ticks) | oob

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
        """Re-seed finished worlds and overwrite their obs rows with fresh resets."""
        idx = np.nonzero(done)[0]
        for i in idx:
            self.worlds[int(i)].reset(int(self.seeds[int(i)]), arena_mode="trees")
            self._ep_return[int(i)] = 0.0
            self._ep_len[int(i)] = 0
        fresh = self._build_obs()
        for key, val in obs.items():
            if key == "action_mask":
                for mk in val:
                    val[mk][done] = fresh["action_mask"][mk][done]
            else:
                val[done] = fresh[key][done]
