"""AiUtopiaSimEnv: a fast, headless, single-process gatherer env that mirrors
the PettingZoo-Parallel surface of ``AiUtopiaPettingZooEnv`` (env/wrapper.py)
behind the *same* obs/action/reward/termination contract — a drop-in for the
scenario runner with no live Minecraft server.

Surface parity with the real wrapper:
  - ``__init__(config: dict)``
  - ``reset(seed) -> (obs, infos)``
  - ``step(action_dict) -> (obs, rew, term, trunc, info)``
  - ``observation_space(agent)`` / ``action_space(agent)`` from env/spaces.py
  - ``possible_agents`` / ``agents`` (== ["gatherer_0"] for M1B)
  - ``close()``

Internals (one ``SimWorld`` for the single gatherer agent):
  step -> ``apply_skill`` (skill dynamics) -> ``build_gatherer_obs`` (byte-faithful
  obs) -> ``step_reward`` (forwards to ``compute_reward_stage_1``) -> termination.

Termination parity (must match wrapper.AiUtopiaPettingZooEnv.step exactly):
  - SUCCESS terminal: ``_goal_success`` (env/_embeds.py — the SAME predicate the
    wrapper uses) over the gatherer stub goal ({oak_log: 64}), evaluated on the
    inventory reconstructed by ``reward._inventory_from_obs`` from ``obs_curr``.
    No hardcoded 64 here — it rides on the shared GoalSpec.
  - TRUNCATION: ``world.tick >= max_episode_ticks`` OR out-of-bounds
    (``abs(x-64) > 24 or abs(z+48) > 24 or y < 60`` — the wrapper's N19 arena box).
  - ``info[agent]["goal_success"]`` + ``info[agent]["skill_completion"]`` mirror the
    wrapper's terminal labels.
  - Finished agents (term OR trunc) are pruned from ``self.agents`` like the wrapper.

IMPORT-LIGHT by contract: this module imports only the sibling ``aiutopia.sim``
modules (world / skills / obs_adapter / reward_adapter), ``aiutopia.env._embeds``
(``_goal_success`` + ``gatherer_stub_subgoal``), ``aiutopia.env.spaces`` (the Dict
spaces), ``aiutopia.env.reward._inventory_from_obs`` (a leaf, pure module), and
``pettingzoo.ParallelEnv`` (the base class the real wrapper subclasses — itself
import-light: pulls NO torch/chroma/py4j) — NEVER chromadb / py4j / torch /
sentence_transformers. Verify with ``py -3.11 -c "import aiutopia.sim.sim_env"``
staying fast/clean.

Phase B (RLlib): ``AiUtopiaSimEnv`` subclasses ``pettingzoo.ParallelEnv`` so it
is wrappable by ``ray.rllib.env.wrappers.pettingzoo_env.ParallelPettingZooEnv``
exactly like the real env — the wrapper accesses ``par_env.unwrapped`` (provided
by ParallelEnv) in ``get_sub_environments``. Like the real wrapper, we do NOT
call ``super().__init__()`` (ParallelEnv has no required init state).

Tick semantics (parity note): ``world.tick`` is the env-step counter, incremented
once per ``step``. The simulated walk-ticks consumed inside ``apply_skill`` while
approaching a log are NOT counted (they are owned by the skill, see skills.py).
Truncation is evaluated on the POST-increment tick so that, e.g.,
``max_episode_ticks=5`` truncates on the 5th ``step`` (env-step 5 >= 5), matching
the gate's "within N env steps" budget semantics.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from gymnasium.spaces import Dict as DictSpace
from pettingzoo import ParallelEnv

from aiutopia.env._embeds import _goal_success, gatherer_stub_subgoal
from aiutopia.env.reward import _inventory_from_obs
from aiutopia.env.spaces import (
    build_role_action_space,
    build_role_observation_space,
)
from aiutopia.sim.obs_adapter import build_gatherer_obs, gatherer_nearest_columns
from aiutopia.sim.reward_adapter import step_reward
from aiutopia.sim.skills import apply_skill, mine_instance
from aiutopia.sim.world import SimWorld

__all__ = ["AiUtopiaSimEnv"]

# Arena box for out-of-bounds truncation (wrapper N19): centered on spawn
# (64.5, _, -47.5), ±24 blocks horizontally, floor at y=60.
_ARENA_CENTER_X = 64.0
_ARENA_CENTER_Z = -48.0
_ARENA_HALF = 24.0
_ARENA_FLOOR_Y = 60.0


def _role_of(agent_id: str) -> str:
    return agent_id.split("_", 1)[0]


# Distance-shaping weight (training-only PBRS). In DECISION-CORE mode the POLICY
# drives navigation, so a potential Φ = -W·dist_to_nearest_alive_log gives a
# learnable gradient toward resources the policy can't yet see (the blind-explore
# hop between clusters) WITHOUT changing the optimal policy (PBRS is invariant).
# This is the right tool here, unlike the old HARVEST-spam path where the skill
# auto-walked and shaping made no difference.
_SHAPING_W = 0.1


def _log_potential(world: SimWorld) -> float:
    alive = world.log_alive
    if not bool(alive.any()):
        return 0.0
    d = world.logs[alive].astype(np.float64) - world.agent_pos
    return -_SHAPING_W * float(np.sqrt((d * d).sum(axis=1)).min())


class AiUtopiaSimEnv(ParallelEnv):
    """PettingZoo-Parallel-shaped headless gatherer sim (single agent for M1B).

    Subclasses ``pettingzoo.ParallelEnv`` to match ``AiUtopiaPettingZooEnv`` so
    ``ParallelPettingZooEnv(AiUtopiaSimEnv(cfg))`` works byte-for-byte the same
    way as the real env in RLlib. Does not call ``super().__init__()`` (parity
    with the real wrapper; ParallelEnv has no required init state)."""

    metadata: ClassVar[dict] = {"name": "aiutopia_sim_v0", "render_modes": []}

    def __init__(self, config: dict[str, Any]):
        self.cfg = config
        self.active_roles = list(config["active_roles"])
        self.agents_init = [f"{r}_0" for r in self.active_roles]
        self.possible_agents = list(self.agents_init)
        self.agents: list[str] = []
        # stage defaults to 1 (M1B); the gatherer obs space is stage-1.
        self.stage = int(config.get("stage", 1))
        # Match the wrapper's config key. Default mirrors the wrapper's 6_000.
        self.max_episode_ticks = int(
            config.get("max_episode_ticks", config.get("max_ticks", 6_000))
        )

        # One world per agent. M1B ships the single gatherer_0; the dict keeps
        # the surface honest if active_roles ever grows.
        self.worlds: dict[str, SimWorld] = {a: SimWorld() for a in self.agents_init}
        self._prev_obs: dict[str, dict] = {}

        # The shared "collect 64 oak_log" stub goal — single source of truth
        # with the wrapper (env/_embeds.gatherer_stub_subgoal). Its
        # goal_specification drives the SUCCESS-termination predicate.
        self._stub_subgoal = gatherer_stub_subgoal()

        # Training-only: vary the arena layout each episode so the policy sees
        # the full layout distribution (incl. nav-requiring ones like seed 3)
        # and learns a general navigate-and-repeat strategy instead of overfitting
        # to one layout. Eval/transfer leave this off and pass explicit fixed
        # seeds (1/2/3) for the gate. Set via env_config["randomize_layout"].
        self._randomize_layout = bool(config.get("randomize_layout", False))
        self._train_ep = 0
        # M2 decision-core: when True, HARVEST is DEMOTED to mine ONLY the
        # instance the policy points at (target_class reinterpreted as a slot
        # index into g_nearest_resources) — no findNearest scan, no field-clearing
        # chaining. This forces the POLICY (not the skill) to choose which trunk +
        # the sequence, and to NAVIGATE/explore when nothing is in perception.
        # Off by default (preserves the proven HARVEST-spam M1B/survival path).
        self._decision_core = bool(config.get("decision_core", False))
        # M2 arena: "trees" (default, the Java-mirrored 4x4 grid) or "clusters"
        # (sim-only blind-explore experiment). arena_half widens the OOB
        # truncation box so a bigger arena (clusters: B ~27 blocks south of spawn)
        # is reachable instead of truncating the agent before it can explore there.
        self._arena_mode = str(config.get("arena_mode", "trees"))
        self._arena_half = float(config.get("arena_half", _ARENA_HALF))
        # Training-only PBRS toward the nearest alive log (see _log_potential) —
        # guides the decision-core policy's blind-explore. Off by default.
        self._distance_shaping = bool(config.get("distance_shaping", False))
        self._prev_phi: dict[str, float] = {}
        # M2 resource-bearing cue (Explorer-report sim): feed a direction-to-nearest
        # -resource into g_hostiles_nearby[0] so the policy can explore the RIGHT way
        # (closes the blind-explore-direction gap). Sim-only experiment.
        self._resource_bearing_cue = bool(config.get("resource_bearing_cue", False))

    # ───── PettingZoo API ─────
    def observation_space(self, agent: str) -> DictSpace:
        return build_role_observation_space(_role_of(agent), stage=self.stage)

    def action_space(self, agent: str) -> DictSpace:
        return build_role_action_space(_role_of(agent))

    def reset(self, seed: int | None = None, options: dict | None = None):
        if self._randomize_layout:
            # Training: ignore the (usually None/fixed) RLlib reset seed and
            # advance the layout each episode for diversity.
            self._train_ep += 1
            seed = self._train_ep
        elif seed is None:
            seed = 1
        self.agents = list(self.agents_init)
        obs: dict[str, dict] = {}
        for agent in self.agents:
            world = self.worlds[agent]
            world.reset(int(seed), arena_mode=self._arena_mode)
            obs[agent] = build_gatherer_obs(world, harvest_mask_on_perception=self._decision_core, resource_bearing_cue=self._resource_bearing_cue)
            self._prev_phi[agent] = _log_potential(world)
        self._prev_obs = obs
        infos = {a: {} for a in self.agents}
        return obs, infos

    def _dispatch_decision_core(
        self, world: SimWorld, action: dict
    ) -> tuple[SimWorld, dict]:
        """Decision-core dispatch: HARVEST -> MINE the policy-POINTED instance
        only (target_class is the slot index into g_nearest_resources); all other
        skills (NAVIGATE-explore, wait, ...) use the normal skill dynamics."""
        skill_type = int(np.asarray(action.get("skill_type", 5)).reshape(-1)[0])
        if skill_type != 1:  # not HARVEST
            return apply_skill(world, action)
        _col_top, nearby = gatherer_nearest_columns(world)
        if not nearby:  # nothing in perception — the policy must NAVIGATE to explore
            return world, {
                "resultCode": "IMMEDIATE_FAILURE",
                "failureReason": "no resource in perception (NAVIGATE to explore)",
                "clippedAxesBitset": 0,
            }
        k = int(np.asarray(action.get("target_class", 0)).reshape(-1)[0])
        k = max(0, min(k, len(nearby) - 1))  # clamp the pointer to visible instances
        x, z = int(nearby[k][0]), int(nearby[k][1])
        return world, mine_instance(world, x, z)

    def step(self, actions: dict[str, dict]):
        new_obs: dict[str, dict] = {}
        rew: dict[str, float] = {}
        term: dict[str, bool] = {}
        trunc: dict[str, bool] = {}
        info: dict[str, dict] = {}

        for agent in list(self.agents):
            world = self.worlds[agent]
            action = actions.get(agent, {})

            # 1. Advance world state. In decision-core mode HARVEST is demoted to
            # mine ONLY the policy-pointed instance; everything else (NAVIGATE
            # explore, wait, ...) uses the normal skill dynamics.
            if self._decision_core:
                world, completion = self._dispatch_decision_core(world, action)
                # CLAMP the agent inside the arena instead of OOB-TERMINATING the
                # episode. OOB-truncation was implicitly PUNISHING exploration
                # (wander out -> episode dies -> the policy learns NAVIGATE is fatal
                # -> greedy collapses to MINE-spam). Clamping lets the policy explore
                # safely (it just stops at the wall) so NAVIGATE can be learned.
                world.agent_pos[0] = float(np.clip(
                    world.agent_pos[0], _ARENA_CENTER_X - self._arena_half,
                    _ARENA_CENTER_X + self._arena_half))
                world.agent_pos[2] = float(np.clip(
                    world.agent_pos[2], _ARENA_CENTER_Z - self._arena_half,
                    _ARENA_CENTER_Z + self._arena_half))
            else:
                world, completion = apply_skill(world, action)
            # 2. env-step counter (one tick per step; walk-ticks NOT counted).
            world.tick += 1

            # 3. Build the post-step obs (byte-faithful gatherer obs adapter).
            obs_curr = build_gatherer_obs(world, harvest_mask_on_perception=self._decision_core, resource_bearing_cue=self._resource_bearing_cue)
            new_obs[agent] = obs_curr
            obs_prev = self._prev_obs.get(agent, obs_curr)

            # 4. Reward via the SAME pure stage-1 reward function the wrapper uses.
            n_clipped = bin(int(completion.get("clippedAxesBitset", 0))).count("1")
            env_meta = {
                "died_this_tick": False,  # no death source in the M1B arena
                "n_clipped_param_axes": n_clipped,
                "exploit_penalties": [],  # Phase A: no exploit detector in sim
            }
            rew[agent] = step_reward(obs_prev, obs_curr, action, env_meta)
            if self._distance_shaping:
                # BLIND-ONLY distance-reduction shaping toward the nearest alive log:
                # only rewarded when NOTHING is in perception (the explore phase), so
                # it guides the blind hop toward the hidden cluster WITHOUT penalizing
                # intra-cluster movement (clearing a cluster moves you among trunks,
                # often AWAY from the far cluster -> net-negative if applied always,
                # which made the policy MINE-spam). Distance-reduction form telescopes
                # to W·(dist at blind-start − dist at blind-end): pure "approached the
                # hidden cluster while exploring" reward, zero inaction drip.
                phi = _log_potential(world)  # = −W·dist (higher == closer)
                _, _nearby_now = gatherer_nearest_columns(world)
                if not _nearby_now:  # blind: nothing visible -> guide the explore
                    rew[agent] += phi - self._prev_phi.get(agent, phi)
                self._prev_phi[agent] = phi

            # 5. SUCCESS termination — same predicate the wrapper uses, over the
            # shared stub goal ({oak_log: 64}). _inventory_from_obs is the SAME
            # reconstruction the reward path uses, so success and reward never
            # disagree about the bag.
            curr_inv = _inventory_from_obs(obs_curr)
            goal_success = _goal_success(self._stub_subgoal.goal_specification, curr_inv)
            term[agent] = goal_success

            # 6. TRUNCATION — tick budget OR out of the seeded arena box (N19).
            pos = obs_curr.get("position", None)
            out_of_bounds = False
            if pos is not None and len(pos) >= 3:
                dx = float(pos[0]) - _ARENA_CENTER_X
                dz = float(pos[2]) - _ARENA_CENTER_Z
                y = float(pos[1])
                if (abs(dx) > self._arena_half) or (abs(dz) > self._arena_half) or (y < _ARENA_FLOOR_Y):
                    out_of_bounds = True
            trunc[agent] = (world.tick >= self.max_episode_ticks) or out_of_bounds

            info[agent] = {
                "skill_completion": completion,
                "goal_success": goal_success,
                "n_clipped": n_clipped,
            }

        self._prev_obs = new_obs
        # Prune finished agents like the wrapper.
        self.agents = [a for a in self.agents if not (term[a] or trunc[a])]
        return new_obs, rew, term, trunc, info

    def close(self) -> None:
        """Idempotent no-op close (no Java process or server to release)."""
        return None
