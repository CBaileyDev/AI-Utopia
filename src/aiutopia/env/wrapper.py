"""§7.3 — PettingZoo Parallel env wrapper.

M0 limitations:
  - gatherer role only
  - reward computation deferred to M1 (returns 0.0 stub)
  - episodic memory writes deferred to M1 (stub no-op)
  - exploit detection deferred to M1 (stub no-op)
  - per_worker_seed_offset is honored
  - mid-tick comm flush is wired
  - close() is implemented and idempotent
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
from gymnasium.spaces import Dict as DictSpace
from pettingzoo import ParallelEnv

from aiutopia.env.action_mask import compute_gatherer_action_mask
from aiutopia.env.bridge import FabricBridge
from aiutopia.env.spaces import (
    build_role_action_space,
    build_role_observation_space,
    CORE_KEYS,
    GATHERER_KEYS,
    GOAL_EMBED_DIM,
)


log = logging.getLogger(__name__)


def _role_of(agent_id: str) -> str:
    return agent_id.split("_", 1)[0]


def _decode_obs(raw: dict, role: str, stage: int,
                 action_mask: dict[str, np.ndarray]) -> dict[str, Any]:
    """Coerce a raw JSON dict from Java into a Gymnasium-Dict-conforming
    obs dict. M0 fills missing fields with zeros (Java side may not
    populate all keys yet)."""
    space = build_role_observation_space(role, stage=stage)
    out: dict[str, Any] = {}
    for key, sub in space.spaces.items():
        if key == "action_mask":
            out[key] = action_mask
            continue
        if key in raw:
            out[key] = np.asarray(raw[key], dtype=sub.dtype if hasattr(sub, "dtype") else None)
        else:
            out[key] = np.zeros(sub.shape, dtype=sub.dtype) if hasattr(sub, "shape") else 0
    # Always emit goal_embedding even if Java omits it (zeros = "no goal")
    if "goal_embedding" not in raw:
        out["goal_embedding"] = np.zeros(GOAL_EMBED_DIM, dtype=np.float32)
    return out


class AiUtopiaPettingZooEnv(ParallelEnv):
    metadata = {"name": "aiutopia_minecraft_v0", "render_modes": []}

    def __init__(self, config: dict[str, Any]):
        self.cfg            = config
        self.active_roles   = list(config["active_roles"])
        self.agents_init    = [f"{r}_0" for r in self.active_roles]
        self.possible_agents = list(self.agents_init)
        self.agents: list[str] = []
        self.stage          = int(config["stage"])
        self.tick_warp      = bool(config.get("tick_warp", False))
        self.max_ticks      = int(config.get("max_episode_ticks", 6_000))
        self._tick          = 0

        # Pick port from worker index (defaults to first port for tests).
        ports = config["py4j_ports"]
        widx  = int(getattr(config, "worker_index",
                            config.get("worker_index", 0))) % len(ports)
        self.bridge = FabricBridge(port=ports[widx])
        self.bridge.open()

        self.skill_counters: dict[str, int] = {}
        self._prev_obs: dict[str, Any] = {}

    # ───── PettingZoo API ─────
    def observation_space(self, agent: str) -> DictSpace:
        return build_role_observation_space(_role_of(agent), stage=self.stage)

    def action_space(self, agent: str) -> DictSpace:
        return build_role_action_space(_role_of(agent))

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is None:
            seed = self._next_seed_for_strategy()
        self.bridge.reset_world(seed)
        self.agents = list(self.agents_init)
        self.skill_counters = {a: 0 for a in self.agents}
        self._tick = 0
        obs = self._read_all_obs()
        self._prev_obs = obs
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, dict]):
        comm_msgs: list[dict] = []
        for agent, act in actions.items():
            self.skill_counters[agent] += 1
            invocation_id = f"{agent}-{self.skill_counters[agent]}"
            self.bridge.dispatch_skill(agent, act, invocation_id)
            if int(act.get("should_broadcast", 0)) == 1 and np.asarray(
                    act.get("comm_target_mask", [0, 0, 0, 0])).any():
                comm_msgs.append({"sender": agent, "action": act})
        self.bridge.flush_comm_batch(comm_msgs)            # mid-tick flush
        self.bridge.advance_tick_await_events(timeout_ms=30_000)

        new_obs = self._read_all_obs()
        rew  = {a: 0.0 for a in self.agents}               # M1: real reward stack
        term = {a: False for a in self.agents}
        trunc = {a: self._tick >= self.max_ticks for a in self.agents}
        info: dict[str, dict] = {a: {} for a in self.agents}

        self._prev_obs = new_obs
        self._tick += 1
        self.agents = [a for a in self.agents if not (term[a] or trunc[a])]
        return new_obs, rew, term, trunc, info

    def close(self) -> None:
        """Idempotent close. Without this Ray workers leak Java processes."""
        try:
            self.bridge.close()
        except Exception:        # nosec - close path
            log.exception("error closing FabricBridge")

    # ───── helpers ─────
    def _read_all_obs(self) -> dict[str, dict]:
        raw_all = self.bridge.observations_all()
        out: dict[str, dict] = {}
        for agent in self.agents:
            raw = raw_all.get(agent, {})
            mask = (compute_gatherer_action_mask(raw)
                    if _role_of(agent) == "gatherer"
                    else {})
            out[agent] = _decode_obs(raw, _role_of(agent), self.stage, mask)
        return out

    def _next_seed_for_strategy(self) -> int:
        strategy = self.cfg.get("seed_strategy", "fixed_easy")
        offset = (1
                  if self.cfg.get("per_worker_seed_offset")
                  else 0) * int(getattr(self.cfg, "worker_index",
                                         self.cfg.get("worker_index", 0)))
        seed_table = {
            "fixed_easy":   1 + offset,
            "fixed_medium": 2 + offset,
            "fixed_hard":   3 + offset,
        }
        return seed_table.get(strategy, 1 + offset)
