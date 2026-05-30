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

import json
import logging
from typing import Any

import numpy as np
from gymnasium.spaces import Dict as DictSpace
from pettingzoo import ParallelEnv

from aiutopia.common.config import Paths
from aiutopia.env._embeds import (
    _agent_uuid_embed,
    _goal_success,
    gatherer_stub_subgoal,
)
from aiutopia.env.action_mask import compute_gatherer_action_mask
from aiutopia.env.bridge import FabricBridge
from aiutopia.env.exploit import ExploitDetector
from aiutopia.env.reward import _inventory_from_obs, compute_reward_stage_1
from aiutopia.env.spaces import (
    build_role_action_space,
    build_role_observation_space,
)
from aiutopia.memory.client import open_chroma
from aiutopia.memory.writer import EpisodicMemoryWriter, EpisodicRecord
from aiutopia.planner.goal_spec import GoalSpecAdapter

log = logging.getLogger(__name__)

# `_agent_uuid_embed` and `_goal_success` are re-exported from `env._embeds`
# (lifted there so the import-light `aiutopia.sim` package can reuse them
# without pulling chromadb/py4j/sentence_transformers). They remain importable
# as `aiutopia.env.wrapper._goal_success` for existing callers/tests.
__all__ = ["AiUtopiaPettingZooEnv", "_agent_uuid_embed", "_goal_success"]


# ───── _normalize_raw constants (must match spaces.py) ─────
_ROLE_ONE_HOT_DIM = 4
_ROLE_INDEX = {"gatherer": 0, "builder": 1, "farmer": 2, "defender": 3}
_REACH_RADIUS_BLOCKS = 4.5  # N16b: bumped 3.0->4.5 to match HarvestSkill/
# DepositChestSkill REACH_RADIUS (vanilla creative
# reach). The 3.0 value re-created the float-precision
# attractor at a different radius — Java side now uses
# full WALK_PER_TICK + 4.5 to eliminate it entirely.
# Action mask uses this to gate HARVEST availability.


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
    out = dict(raw)  # shallow copy is fine, we don't mutate nested arrays
    # R2: derive agent_uuid_embed + role_one_hot from the auxiliary strings.
    agent_uuid = out.pop("agent_uuid", "")
    role_id = out.pop("role_id", "")
    out.pop("agent_name", None)
    out["agent_uuid_embed"] = _agent_uuid_embed(agent_uuid)
    out["role_one_hot"] = _role_one_hot(role_id)
    # R4: derive in-range booleans for action_mask.
    nrd = float(out.get("nearest_resource_distance", [999.0])[0])
    ncd = float(out.get("nearest_chest_distance", [999.0])[0])
    out["target_resource_in_range"] = nrd <= _REACH_RADIUS_BLOCKS
    out["target_chest_in_range"] = ncd <= _REACH_RADIUS_BLOCKS
    return out


def _role_of(agent_id: str) -> str:
    return agent_id.split("_", 1)[0]


def _decode_obs(
    raw: dict, role: str, stage: int, action_mask: dict[str, np.ndarray], goal_embed: np.ndarray
) -> dict[str, Any]:
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
        self.cfg = config
        self.active_roles = list(config.get("active_roles", ["gatherer"]))
        self.agents_init = [f"{r}_0" for r in self.active_roles]
        self.possible_agents = list(self.agents_init)
        self.agents: list[str] = []
        self.stage = int(config.get("stage", 1))
        self.tick_warp = bool(config.get("tick_warp", False))
        self.max_ticks = int(config.get("max_episode_ticks", 6_000))
        self._tick = 0

        # Arena-bounds truncation box (N19-followup). step() hard-truncates the
        # agent outside this box so wandering training episodes reset fast and
        # the on-policy buffer stays oak_log-adjacent (see step()). The box is
        # configurable so non-flat-arena operation (e.g. natural-forest recon)
        # can run without instant truncation. DEFAULTS ARE BYTE-IDENTICAL to the
        # hardcoded N19 box (center 64.5/-47.5, ±24 horizontally, y>=60):
        #   arena_bounds_check   — False disables the box entirely (no bounds trunc)
        #   arena_center_xz      — (cx, cz) horizontal box center
        #   arena_half           — half-extent in blocks (|dx|>half or |dz|>half)
        #   arena_min_y          — minimum y (below → out of bounds)
        self.arena_bounds_check = bool(config.get("arena_bounds_check", True))
        _ac = config.get("arena_center_xz", (64.5, -47.5))
        self.arena_center_xz = (float(_ac[0]), float(_ac[1]))
        self.arena_half = float(config.get("arena_half", 24.0))
        self.arena_min_y = float(config.get("arena_min_y", 60.0))

        # N10: per-skill server-side timeout_ticks injected into every action
        # before dispatch. The Java defaults (NAVIGATE/HARVEST = 6000 ticks)
        # were tuned for vanilla TPS=20 (= 5 min wall) which becomes 20s wall
        # at tick_warp 300 TPS. Under random initial-policy sampling almost
        # every NAVIGATE picks an unreachable target and runs the full 6000
        # ticks → 20s per env.step → train iter takes hours and Ray's
        # sample_timeout_s fires before any fragment fills (the v13 "silent
        # hang" symptom). 400 ticks at 300 TPS = ~1.3s — enough for the agent
        # to traverse MAX_NAV_RANGE=32 blocks (~150 ticks walking) with slack.
        # Override per-skill via env_config["skill_timeout_ticks"].
        # Keys are gym skill_type ints from spaces.py (0=NAVIGATE 1=HARVEST
        # 2=DEPOSIT_CHEST 3=SEARCH 4=WAIT 5=NOOP_BROADCAST). Only NAVIGATE,
        # HARVEST, DEPOSIT_CHEST honor `timeout_ticks` server-side; SEARCH +
        # WAIT compute duration from scalar_param × MAX_DURATION (~200 ticks).
        default_skill_timeout = {0: 400, 1: 400, 2: 400}
        self.skill_timeout_ticks: dict[int, int] = {
            int(k): int(v)
            for k, v in {
                **default_skill_timeout,
                **dict(config.get("skill_timeout_ticks", {})),
            }.items()
        }

        # Pick port from worker index (defaults to first port for tests).
        ports = config.get("py4j_ports", [25100])
        widx = int(getattr(config, "worker_index", config.get("worker_index", 0))) % len(ports)
        self.bridge = FabricBridge(port=ports[widx])
        self.bridge.open()

        # N9 / N14: populate the module-level int→name table in reward.py
        # from the Java side's ItemId registry. The mapping is the contiguous
        # one emitted by `dev.aiutopia.mod.obs.ItemIdTable` (NOT the old
        # `rawId & 0x3FF` scheme) and is exposed by
        # `Py4JEntryPoint.getItemIdNameTable()`. Without this, the reward
        # function falls back to the static eager seed in `reward.py` —
        # complete for LOG_VALUE coverage but missing the long tail of
        # items the obs builder may emit.
        #
        # N14 finding: the original `log.info(...)` here was getting
        # filtered out by Ray's default log routing (workers don't print
        # INFO to the parent train.log), which led a previous session to
        # incorrectly conclude the update was silently failing. The dict
        # WAS populating; the log line just wasn't visible. Upgraded to
        # WARNING so future audits can confirm by tailing train.log.
        try:
            jmap = self.bridge.entry_point.getItemIdNameTable()
            from aiutopia.env import reward as _rwd

            before = len(_rwd._ITEM_ID_TO_NAME)
            _rwd._ITEM_ID_TO_NAME.update({int(k): str(jmap.get(k)) for k in jmap.keySet()})
            after = len(_rwd._ITEM_ID_TO_NAME)
            log.warning(
                "N9/N14 ItemId table loaded from Java (port=%d): %d entries "
                "(eager-seed=%d, added=%d)",
                ports[widx],
                after,
                before,
                after - before,
            )
        except Exception as e:
            log.warning(
                "N9/N14 could not fetch ItemIdNameTable from Java (port=%d): %s "
                "— falling back to module-level eager seed (%d entries)",
                ports[widx],
                e,
                __import__(
                    "aiutopia.env.reward", fromlist=["_ITEM_ID_TO_NAME"]
                )._ITEM_ID_TO_NAME.__len__(),
            )

        # M1-Training likely-to-break fix #1: per-worker AIUTOPIA_ROOT so 4
        # concurrent EnvRunners don't collide on the same identity.db and
        # Chroma collections. Each worker gets its own root suffixed with its
        # worker_index. The root is set BEFORE any Paths.from_env() call below.
        if config.get("aiutopia_root_per_worker", True) and ports:
            import os as _os

            from aiutopia.common.config import per_worker_root

            base = _os.environ.get("AIUTOPIA_ROOT", "")
            if base:
                # IDEMPOTENT: re-instantiating this env in the same worker process
                # must NOT compound the suffix (the _w0_w0_w0 bug that scatters
                # Chroma/identity across phantom dirs). See per_worker_root.
                _os.environ["AIUTOPIA_ROOT"] = per_worker_root(base, widx)

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
                    pass  # idempotent — already-spawned is OK

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
        self.agent_id_to_uuid: dict[str, str] = dict(config.get("agent_id_to_uuid", {}))

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
        # The "collect 64 oak_log" stub Subgoal now lives in env._embeds as the
        # single source of truth shared with the sim. The wrapper still
        # BGE-embeds it here (real NL vec when sentence-transformers loads,
        # zeros via _ZeroBGE otherwise) — byte-identical to the old inline path.
        self._stub_subgoal = gatherer_stub_subgoal()
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
            # N10: inject server-side timeout_ticks per skill_type so the
            # Java executor doesn't burn 6000 ticks (20s wall at TPS=300) on
            # a single failed NAVIGATE during random-policy sampling.
            # Mutate a shallow copy so we don't poison the policy's dict.
            try:
                skill_type = int(np.asarray(act.get("skill_type", -1)).item())
            except Exception:
                skill_type = -1
            if "timeout_ticks" not in act and skill_type in self.skill_timeout_ticks:
                act = {**act, "timeout_ticks": int(self.skill_timeout_ticks[skill_type])}
            self.bridge.dispatch_skill(player_name, act, invocation_id)
            if (
                int(act.get("should_broadcast", 0)) == 1
                and np.asarray(act.get("comm_target_mask", [0, 0, 0, 0])).any()
            ):
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
                    evt.get("agentId", ""), evt.get("agentId", "")
                )
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
            term[agent] = died_this_tick
            # Phase-0 fix #5: SUCCESS-TERMINATION. Reward for THIS tick is
            # already banked above (computed from obs, not from `term`), so
            # terminating now does not retroactively change the final-step
            # reward — it just stops the episode so TIME_PENALTY/PBRS stop
            # accruing past the goal and the policy gets an episodic "win"
            # signal (terminated=True ⇒ no value bootstrap). Success derives
            # from the GoalSpec the env already builds (inventory_meets_delta
            # over inventory_delta); no hardcoded 64. Reuses the SAME
            # _inventory_from_obs the reward path uses so success and reward
            # never disagree about the bag. Composes with death (also
            # terminated) via OR; out_of_bounds (truncation) is handled below.
            curr_inv = _inventory_from_obs(new_obs.get(agent, {}))
            # A SUCCESS terminal is one reached ALIVE — if the agent died on
            # the same tick it crossed the goal, that is a DEATH terminal, so
            # the eval-runner label (info["goal_success"]) must not call it a
            # win. term[agent] is already True from death either way, so the
            # primary contract holds regardless; this only governs the label.
            goal_success = (not died_this_tick) and _goal_success(
                self._stub_subgoal.goal_specification, curr_inv
            )
            if goal_success:
                term[agent] = True
            # N19-followup: also truncate when the agent strays out of the
            # seeded arena. v18/v19/v20 inventory probes consistently showed
            # only 1 of 4 instances accumulating oak_log — the other 3
            # NAVIGATEd out of the 8-log ring (spawn at 64.5/66/-47.5,
            # logs within 7 blocks horizontally at y=66) and ended up
            # mining cobblestone at y=40-55 underground. With LOG_VALUE
            # cobblestone=0.091 vs oak_log=1.000, the cobblestone-mining
            # trajectories still produced positive reward, splitting the
            # policy gradient between two reward attractors and preventing
            # convergence on the M1B goal. Hard-bound the agent to the
            # arena box (±24b horizontally from spawn, y>=60) so episodes
            # that wander get terminated quickly and reset_episode pulls
            # the agent back to the ring. This narrows the on-policy
            # buffer to oak_log-adjacent trajectories.
            agent_pos = new_obs.get(agent, {}).get("position", None)
            out_of_bounds = False
            if self.arena_bounds_check and agent_pos is not None and len(agent_pos) >= 3:
                dx = float(agent_pos[0]) - self.arena_center_xz[0]
                dz = float(agent_pos[2]) - self.arena_center_xz[1]
                y = float(agent_pos[1])
                if (
                    (abs(dx) > self.arena_half)
                    or (abs(dz) > self.arena_half)
                    or (y < self.arena_min_y)
                ):
                    out_of_bounds = True
            trunc[agent] = (self._tick >= self.max_ticks) or out_of_bounds
            info[agent] = {
                "skill_completion": completion,
                "exploit_penalties": [(n.value, p) for n, p in exploit_penalties],
                "n_clipped": n_clipped,
                # Phase-0 fix #5: distinguish a SUCCESS terminal from a DEATH
                # terminal. The eval runner reads this on the terminal obs.
                "goal_success": goal_success,
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
                completed_bonus = (
                    0.3
                    if completion.get("resultCode")
                    in {
                        "COMPLETED",
                        "FAILED_TIMEOUT",
                        "IMMEDIATE_FAILURE",
                    }
                    else 0.0
                )
                death_bonus = 0.4 if died_this_tick else 0.0
                importance = min(1.0, abs_r + completed_bonus + death_bonus)
                self.memory_writer.maybe_write(
                    EpisodicRecord(
                        agent_uuid=agent_uuid,
                        timestamp=self._tick,
                        event_type=completion.get("resultCode", "tick"),
                        participants=[],
                        importance_score=importance,
                        summary=(
                            f"r={rew[agent]:.2f} "
                            f"skill={actions.get(agent, {}).get('skill_type', '?')} "
                            f"out={completion.get('resultCode', 'RUNNING')}"
                        ),
                        embedding=None,
                    )
                )

        self._prev_obs = new_obs
        self._tick += 1
        self.agents = [a for a in self.agents if not (term[a] or trunc[a])]
        return new_obs, rew, term, trunc, info

    def close(self) -> None:
        """Idempotent close. Without this Ray workers leak Java processes."""
        try:
            self.bridge.close()
        except Exception:  # nosec - close path
            log.exception("error closing FabricBridge")

    # ───── helpers ─────
    def _read_all_obs(self) -> dict[str, dict]:
        """Read observations for all active agents, dispatching by role.

        Java side: If role-specific obs methods exist (observationsGatherer, observationsExplorer,
        observationsFarmer), uses them. Otherwise falls back to observations_all() (Gatherer-only).
        """
        out: dict[str, dict] = {}
        for agent in self.agents:
            role = _role_of(agent)
            player_name = self.agent_id_to_player_name.get(agent, agent)

            # Try role-specific method first (requires Java 2b+ with multi-role obs builders)
            raw = {}
            try:
                if role == "gatherer" and hasattr(self.bridge.entry_point, "observationsGatherer"):
                    raw = self.bridge.entry_point.observationsGatherer(player_name) or {}
                elif role == "explorer" and hasattr(self.bridge.entry_point, "observationsExplorer"):
                    raw = self.bridge.entry_point.observationsExplorer(player_name) or {}
                elif role == "farmer" and hasattr(self.bridge.entry_point, "observationsFarmer"):
                    raw = self.bridge.entry_point.observationsFarmer(player_name) or {}
            except Exception:
                pass  # Fallback to batch below

            # Fallback: batch observations_all (current Java, Gatherer-only)
            if not raw:
                raw_all = self.bridge.observations_all()
                raw = raw_all.get(player_name, {})

            raw = _normalize_raw(raw)
            mask = compute_gatherer_action_mask(raw) if role == "gatherer" else {}
            out[agent] = _decode_obs(raw, role, self.stage, mask, self._stub_goal_embed)
        return out

    def _next_seed_for_strategy(self) -> int:
        strategy = self.cfg.get("seed_strategy", "fixed_easy")
        offset = (1 if self.cfg.get("per_worker_seed_offset") else 0) * int(
            getattr(self.cfg, "worker_index", self.cfg.get("worker_index", 0))
        )
        seed_table = {
            "fixed_easy": 1 + offset,
            "fixed_medium": 2 + offset,
            "fixed_hard": 3 + offset,
        }
        return seed_table.get(strategy, 1 + offset)
