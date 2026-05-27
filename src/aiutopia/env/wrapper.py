"""§7.3 — PettingZoo Parallel env wrapper.

M0 limitations:
  - gatherer role only
  - reward computation deferred to M1 (returns 0.0 stub)
  - episodic memory writes deferred to M1 (stub no-op)
  - exploit detection deferred to M1 (stub no-op)
  - per_worker_seed_offset is honored
  - mid-tick comm flush is wired
  - close() is implemented and idempotent

M1-Pipeline T15:
  - GoalSpecAdapter wired; reset() embeds a hardcoded "collect 64 oak_log"
    Subgoal that ships in every obs as `goal_embedding` (replaces the M0
    zero stub). Plan B replaces this with planner-emitted goals.
  - `_normalize_raw()` bridges Java's stringly-typed obs (agent_uuid,
    agent_name, role_id strings + scalar-array distances) into the
    gym-shaped fields the obs space expects (agent_uuid_embed 384-d,
    role_one_hot 4-d, target_*_in_range booleans for the action_mask).
  - `agent_id_to_player_name` map carried through config (R6) so future
    dispatch_skill calls hit Carpet player_name, not env agent_id.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np
from gymnasium.spaces import Dict as DictSpace
from pettingzoo import ParallelEnv

from aiutopia.common.config import Paths
from aiutopia.env.action_mask import compute_gatherer_action_mask
from aiutopia.env.bridge import FabricBridge
from aiutopia.env.exploit import ExploitDetector
from aiutopia.env.reward import compute_reward_stage_1
from aiutopia.env.spaces import (
    build_role_action_space,
    build_role_observation_space,
    CORE_KEYS,
    GATHERER_KEYS,
    GOAL_EMBED_DIM,
)
from aiutopia.memory.client import open_chroma
from aiutopia.memory.writer import EpisodicMemoryWriter, EpisodicRecord
from aiutopia.planner.goal_spec import GoalSpecAdapter
from aiutopia.schemas.plan import (
    Constraints, GoalSpecification, Subgoal, TargetState, TerminationConditions,
)


log = logging.getLogger(__name__)


# ───── _normalize_raw constants (must match spaces.py) ─────
_AGENT_UUID_EMBED_DIM = 384
_ROLE_ONE_HOT_DIM     = 4
_ROLE_INDEX = {"gatherer": 0, "builder": 1, "farmer": 2, "defender": 3}
_REACH_RADIUS_BLOCKS  = 2.0   # match HarvestSkill.REACH_RADIUS / DepositChestSkill


def _agent_uuid_embed(uuid_str: str) -> np.ndarray:
    """Deterministic 384-d float32 embed from agent UUID string.
    SHA-256 → 32 bytes → 384 floats (tile to fill) → normalized [-1, 1].
    Stable across processes (M0 hash() carry-forward + R10)."""
    digest = hashlib.sha256(uuid_str.encode("utf-8")).digest()  # 32 bytes
    # Tile 32 bytes → 384 bytes (32 × 12 = 384)
    tiled = (digest * 12)[:_AGENT_UUID_EMBED_DIM]
    arr = np.frombuffer(tiled, dtype=np.uint8).astype(np.float32)
    return (arr / 127.5) - 1.0   # → [-1, 1]


def _role_one_hot(role_id: str) -> np.ndarray:
    out = np.zeros(_ROLE_ONE_HOT_DIM, dtype=np.int8)
    if role_id in _ROLE_INDEX:
        out[_ROLE_INDEX[role_id]] = 1
    return out


def _normalize_raw(raw: dict) -> dict:
    """Java emits a mix of raw ints + scalar arrays + auxiliary string fields.
    This converts the auxiliary strings into the gym-shaped fields the obs
    space expects and adds the two action-mask booleans (R4).
    Mutates a copy of `raw`; original is left untouched."""
    out = dict(raw)   # shallow copy is fine, we don't mutate nested arrays
    # R2: derive agent_uuid_embed + role_one_hot from the auxiliary strings.
    agent_uuid = out.pop("agent_uuid", "")
    role_id    = out.pop("role_id", "")
    out.pop("agent_name", None)
    out["agent_uuid_embed"] = _agent_uuid_embed(agent_uuid)
    out["role_one_hot"]     = _role_one_hot(role_id)
    # R4: derive in-range booleans for action_mask.
    nrd = float(out.get("nearest_resource_distance", [999.0])[0])
    ncd = float(out.get("nearest_chest_distance",    [999.0])[0])
    out["target_resource_in_range"] = nrd <= _REACH_RADIUS_BLOCKS
    out["target_chest_in_range"]    = ncd <= _REACH_RADIUS_BLOCKS
    return out


def _role_of(agent_id: str) -> str:
    return agent_id.split("_", 1)[0]


def _decode_obs(raw: dict, role: str, stage: int,
                 action_mask: dict[str, np.ndarray],
                 goal_embed: np.ndarray) -> dict[str, Any]:
    """Coerce a raw JSON dict from Java into a Gymnasium-Dict-conforming
    obs dict. M0 fills missing fields with zeros (Java side may not
    populate all keys yet). M1-Pipeline T15: `goal_embed` is injected by
    the env (hardcoded stub today, planner-emitted later)."""
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
    # Always emit goal_embedding (M1-Pipeline default: env-injected stub).
    if "goal_embedding" not in raw:
        out["goal_embedding"] = goal_embed
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

        # R6: env carries an explicit agent_id → Carpet player_name map
        # populated by the caller (CLI or test) after spawning. Without it,
        # dispatch_skill hits "agent player not found" and obs come back keyed
        # under the wrong name. Default empty → fall back to env_agent_id as
        # the player_name (only matches if the caller named the Carpet
        # player exactly "gatherer_0", which is fine for offline tests).
        self.agent_id_to_player_name: dict[str, str] = dict(
            config.get("agent_id_to_player_name", {})
        )
        for env_aid in self.agents_init:
            self.agent_id_to_player_name.setdefault(env_aid, env_aid)
        # Reverse lookup for _read_all_obs.
        self._player_name_to_agent_id: dict[str, str] = {
            v: k for k, v in self.agent_id_to_player_name.items()
        }

        # M1-Training T21 fix: under training, the wrapper must self-spawn
        # its agents so each EnvRunner is self-sufficient (the manual
        # `aiutopia agent spawn` CLI registers players with skin-pool names
        # like "Frida" — that breaks reset_episode/dispatch which use gym
        # ids like "gatherer_0"). Carpet's `/player spawn` silently no-ops
        # if the name already exists.
        if config.get("auto_spawn_agents", True):
            for env_aid, player_name in self.agent_id_to_player_name.items():
                role = env_aid.rsplit("_", 1)[0]
                try:
                    self.bridge.carpet_spawn(player_name, skin="", role=role)
                except Exception:
                    pass   # idempotent — already-spawned is OK

        # M1-Pipeline T16: per-agent ExploitDetector (5 rules from §5.3).
        self.exploit_detectors: dict[str, ExploitDetector] = {
            agent: ExploitDetector() for agent in self.agents_init
        }
        # R7: real memory writes — one writer shared across agents,
        # each agent's records routed to its own Chroma collection.
        # In tests pass `enable_memory_writes=False` to skip the heavy
        # Chroma init.
        if config.get("enable_memory_writes", True):
            paths = Paths.from_env()
            chroma_client = open_chroma(paths.chroma_dir)
            self.memory_writer = EpisodicMemoryWriter(chroma_client=chroma_client)
        else:
            self.memory_writer = EpisodicMemoryWriter(chroma_client=None)

        # Map env agent_id → agent_uuid (ULID) for memory writes. Populated
        # by the caller via config (same shape as agent_id_to_player_name).
        self.agent_id_to_uuid: dict[str, str] = dict(
            config.get("agent_id_to_uuid", {})
        )

        # GoalSpecAdapter — M1-Pipeline ships a hardcoded "collect 64 oak_log"
        # goal embedded into every obs. Plan B replaces this with planner-
        # emitted Subgoal objects routed via aiutopia.planner.event_queue.
        from aiutopia.planner.goal_spec import load_bge_small
        try:
            bge = load_bge_small()
        except Exception:
            # On dev machines without sentence-transformers fully installed,
            # fall back to a zero-vector encoder so tests don't block.
            # NOTE: the final embed is still 512-d — 384 zeros (BGE part) +
            # 128 structured features (role-one-hot, inventory bucket, etc.).
            # So the structured signal is preserved; only the NL signal is lost.
            class _ZeroBGE:
                def encode(self, _text: str) -> np.ndarray:
                    return np.zeros(384, dtype=np.float32)
            bge = _ZeroBGE()
        self.goal_adapter = GoalSpecAdapter(bge=bge)
        self._stub_subgoal = Subgoal(
            role="gatherer",
            priority=5,
            goal_specification=GoalSpecification(
                target_state=TargetState(inventory_delta={"oak_log": 64}),
                termination_conditions=TerminationConditions(
                    success_criteria=["inventory_meets_delta"],
                    timeout_ticks=6000,
                ),
            ),
            constraints=Constraints(),
            nl_summary="collect 64 oak_log",
        )
        self._stub_goal_embed = self.goal_adapter.embed(self._stub_subgoal).astype(np.float32)

    # ───── PettingZoo API ─────
    def observation_space(self, agent: str) -> DictSpace:
        return build_role_observation_space(_role_of(agent), stage=self.stage)

    def action_space(self, agent: str) -> DictSpace:
        return build_role_action_space(_role_of(agent))

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is None:
            seed = self._next_seed_for_strategy()
        self.bridge.reset_world(seed)
        # M1B: per-episode reset (T13) — teleport, clear inventory, place
        # seeded oak_log ring for each registered agent.
        for agent_id in self.agents_init:
            player_name = self.agent_id_to_player_name.get(agent_id)
            if player_name:
                self.bridge.reset_episode(player_name, int(seed))
        # Reset exploit detectors for the new episode
        from aiutopia.env.exploit import ExploitDetector
        for agent_id in list(self.exploit_detectors.keys()):
            self.exploit_detectors[agent_id] = ExploitDetector()
        self.agents = list(self.agents_init)
        self.skill_counters = {a: 0 for a in self.agents}
        self._tick = 0
        obs = self._read_all_obs()
        self._prev_obs = obs
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, dict]):
        # 1. Dispatch each agent's action via Py4J (mid-tick comm flush).
        # R6: use player_name from agent_id_to_player_name; Java only knows
        # player names, NOT env agent_ids.
        comm_msgs: list[dict] = []
        for agent, act in actions.items():
            self.skill_counters[agent] += 1
            invocation_id = f"{agent}-{self.skill_counters[agent]}"
            player_name = self.agent_id_to_player_name.get(agent, agent)
            self.bridge.dispatch_skill(player_name, act, invocation_id)
            if int(act.get("should_broadcast", 0)) == 1 and np.asarray(
                    act.get("comm_target_mask", [0, 0, 0, 0])).any():
                comm_msgs.append({"sender": player_name, "action": act})
        if comm_msgs:
            self.bridge.flush_comm_batch(comm_msgs)

        # 2. Advance world; collect SkillCompletionEvents.
        # Completion events come back keyed by player_name (Java side).
        # We translate back to env agent_id for downstream consumers.
        completion_jsons = self.bridge.advance_tick_await_events(timeout_ms=30_000)
        completions_by_agent: dict[str, dict] = {}
        for j in completion_jsons:
            try:
                evt = json.loads(j) if isinstance(j, str) else j
                env_aid = self._player_name_to_agent_id.get(
                    evt.get("agentId", ""), evt.get("agentId", ""))
                completions_by_agent[env_aid] = evt
            except Exception:
                continue

        # 3. Batched observation read (translates Java player_name keys
        # → env agent_id and applies _normalize_raw — see T15 _read_all_obs).
        new_obs = self._read_all_obs()
        rew: dict[str, float] = {}
        term: dict[str, bool] = {}
        trunc: dict[str, bool] = {}
        info: dict[str, dict] = {}

        for agent in list(self.agents):
            completion = completions_by_agent.get(agent, {})
            n_clipped = bin(int(completion.get("clippedAxesBitset", 0))).count("1")
            # health is now a numpy shape-(1,) array after _normalize_raw.
            prev_h = float(self._prev_obs.get(agent, {}).get("health", np.array([20.0]))[0])
            curr_h = float(new_obs.get(agent, {}).get("health", np.array([20.0]))[0])
            died_this_tick = curr_h <= 0 and prev_h > 0
            # Run exploit detector
            exploit_penalties = self.exploit_detectors[agent].step(
                role=_role_of(agent),
                obs_prev=self._prev_obs.get(agent, {}),
                obs_curr=new_obs.get(agent, {}),
                action=actions.get(agent, {}),
                env_meta={
                    "global_step": self._tick,
                    "skill_result_code": completion.get("resultCode", "RUNNING"),
                },
            )
            env_meta = {
                "died_this_tick": died_this_tick,
                "n_clipped_param_axes": n_clipped,
                "exploit_penalties": [(n.value, p) for n, p in exploit_penalties],
            }
            rew[agent] = compute_reward_stage_1(
                role=_role_of(agent),
                obs_prev=self._prev_obs.get(agent, {}),
                obs_curr=new_obs.get(agent, {}),
                action=actions.get(agent, {}),
                env_meta=env_meta,
            )
            term[agent]  = died_this_tick
            trunc[agent] = self._tick >= self.max_ticks
            info[agent]  = {
                "skill_completion":   completion,
                "exploit_penalties":  [(n.value, p) for n, p in exploit_penalties],
                "n_clipped":          n_clipped,
            }

            # R7: episodic memory write per tick. Importance heuristic:
            #  - |reward| normalized (clip to [0,1] by /5.0)
            #  - +0.3 if a skill completed this tick (any outcome)
            #  - +0.4 if the agent died this tick
            # Three classes per spec §4.9 (HIGH ≥0.7 immediate / MEDIUM ≥0.3 batched /
            # below skipped). EpisodicMemoryWriter does the bucketing.
            agent_uuid = self.agent_id_to_uuid.get(agent)
            if agent_uuid:  # only write if we have a real ULID (CLI-spawned)
                abs_r = min(1.0, abs(rew[agent]) / 5.0)
                completed_bonus = 0.3 if completion.get("resultCode") in {
                    "COMPLETED", "FAILED_TIMEOUT", "IMMEDIATE_FAILURE",
                } else 0.0
                death_bonus = 0.4 if died_this_tick else 0.0
                importance = min(1.0, abs_r + completed_bonus + death_bonus)
                self.memory_writer.maybe_write(EpisodicRecord(
                    agent_uuid=agent_uuid,
                    timestamp=self._tick,
                    event_type=completion.get("resultCode", "tick"),
                    participants=[],
                    importance_score=importance,
                    summary=(f"r={rew[agent]:.2f} "
                             f"skill={actions.get(agent, {}).get('skill_type', '?')} "
                             f"out={completion.get('resultCode', 'RUNNING')}"),
                    embedding=None,
                ))

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
            # R6: Java keys obs by Carpet player_name, not env agent_id.
            player_name = self.agent_id_to_player_name.get(agent, agent)
            raw = _normalize_raw(raw_all.get(player_name, {}))
            mask = (compute_gatherer_action_mask(raw)
                    if _role_of(agent) == "gatherer"
                    else {})
            out[agent] = _decode_obs(raw, _role_of(agent), self.stage, mask,
                                       self._stub_goal_embed)
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
