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
from aiutopia.env.reward import GAMMA, _inventory_from_obs
from aiutopia.env.spaces import (
    build_role_action_space,
    build_role_observation_space,
)
from aiutopia.sim.obs_adapter import build_gatherer_obs
from aiutopia.sim.reward_adapter import step_reward
from aiutopia.sim.skills import apply_skill
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


# Distance-shaping weight (training-only PBRS). Makes moving toward the nearest
# uncollected log immediately rewarding so the policy learns NAVIGATE's
# (otherwise delayed) value — closing gap #2's credit-assignment local optimum
# where the policy spammed HARVEST and never repositioned. PBRS is
# policy-invariant: it speeds learning the navigate behavior without changing the
# optimal policy. Off by default; the sim training config opts in.
_SHAPING_W = 0.1


def _log_potential(world: SimWorld) -> float:
    """PBRS potential Φ(s) = -_SHAPING_W * (distance to nearest uncollected log),
    0 when none remain. Higher (less negative) when closer to a log."""
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
        # Training-only PBRS distance shaping (see _log_potential).
        self._distance_shaping = bool(config.get("distance_shaping", False))
        self._prev_phi: dict[str, float] = {}
        # Training-only penalty for a FAILED skill dispatch (IMMEDIATE_FAILURE /
        # FAILED_TIMEOUT). Breaks the HARVEST-spam local optimum: once the agent
        # has cleared every log inside MAX_SEARCH_RADIUS, further HARVESTs fail
        # (the tail is >16 b away, unsearchable) — without a cost the policy
        # happily spams failing HARVEST forever; with one, repositioning via
        # NAVIGATE (which the distance shaping already rewards toward the far
        # log) becomes the better action. Not potential-based (it intentionally
        # shifts the optimum away from no-op spam); eval leaves it at 0.
        self._failure_penalty = float(config.get("failure_penalty", 0.0))

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
            world.reset(int(seed))
            obs[agent] = build_gatherer_obs(world)
            self._prev_phi[agent] = _log_potential(world)
        self._prev_obs = obs
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, dict]):
        new_obs: dict[str, dict] = {}
        rew: dict[str, float] = {}
        term: dict[str, bool] = {}
        trunc: dict[str, bool] = {}
        info: dict[str, dict] = {}

        for agent in list(self.agents):
            world = self.worlds[agent]
            action = actions.get(agent, {})

            # 1. Advance world state via the macro-skill dynamics.
            world, completion = apply_skill(world, action)
            # 2. env-step counter (one tick per step; walk-ticks NOT counted).
            world.tick += 1

            # 3. Build the post-step obs (byte-faithful gatherer obs adapter).
            obs_curr = build_gatherer_obs(world)
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
            # PBRS distance shaping (training-only): r += γ·Φ(s') − Φ(s). Makes
            # NAVIGATE toward the nearest log immediately rewarding so the policy
            # learns to reposition instead of HARVEST-spamming a depleted area.
            if self._distance_shaping:
                phi = _log_potential(world)
                rew[agent] += GAMMA * phi - self._prev_phi.get(agent, phi)
                self._prev_phi[agent] = phi
            # Training-only failed-dispatch penalty (breaks HARVEST-spam stall).
            if self._failure_penalty and completion.get("resultCode") in (
                "IMMEDIATE_FAILURE",
                "FAILED_TIMEOUT",
            ):
                rew[agent] -= self._failure_penalty

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
                if (abs(dx) > _ARENA_HALF) or (abs(dz) > _ARENA_HALF) or (y < _ARENA_FLOOR_Y):
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
