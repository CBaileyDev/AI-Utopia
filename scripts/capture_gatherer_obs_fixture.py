"""Capture a golden obs trace from the REAL Minecraft env for Phase-A Task 3b.

This is the one-time, server-dependent step that produces the fixture
``tests/fixtures/gatherer_obs_trace_seed1.json``. The committed fixture then
makes ``tests/unit/test_sim_obs_parity.py::test_sim_obs_matches_real_golden_trace``
run offline forever — it is the only test that validates the sim against
*Minecraft* rather than against the sim's own assumptions (channel-id map,
``oak_log -> 132`` inventory id, the ``dy = +1`` grid/nearest off-by-one).

Modeled on ``scripts/n14_reward_probe.py``: build the real
``AiUtopiaPettingZooEnv`` on a free port, ``reset(seed=1)``, record
``obs["gatherer_0"]`` after reset and after each action in the fixed
``SCRIPTED_ACTIONS`` sequence (1x WAIT, then 3x HARVEST(oak_log, cap=1)), and
serialize the per-step obs (numpy -> list) to JSON.

The ``SCRIPTED_ACTIONS`` sequence below is the SINGLE SOURCE OF TRUTH shared
with the parity test, which imports it from this module so capture and replay
can never drift.

Usage (on a FREE Fabric instance — the 12-instance M1B run occupies
25001-25012, so use a dedicated instance/port, e.g. 25013):

    PYTHONPATH=src py -3.11 scripts/capture_gatherer_obs_fixture.py \
        --port 25013 --out tests/fixtures/gatherer_obs_trace_seed1.json

Then commit the fixture. IMPORT-NOTE: this script imports the heavy wrapper
(FabricBridge / py4j / chroma) by design — it is NOT import-light and is NEVER
imported by the sim package or its tests. Only ``SCRIPTED_ACTIONS`` (a pure
NumPy constant) is shared with the test.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _wait_action() -> dict:
    return {
        "skill_type": 4,  # WAIT
        "target_class": 0,
        "spatial_param": np.zeros(3, dtype=np.float32),
        "scalar_param": np.zeros(1, dtype=np.float32),
        "comm_payload": np.zeros(128, dtype=np.float32),
        "should_broadcast": 0,
        "comm_target_mask": np.zeros(4, dtype=np.int8),
    }


def _harvest_one() -> dict:
    return {
        "skill_type": 1,  # HARVEST
        "target_class": 0,  # oak_log
        "spatial_param": np.zeros(3, dtype=np.float32),
        "scalar_param": np.asarray([1.0 / 64.0], dtype=np.float32),  # cap=1
        "comm_payload": np.zeros(128, dtype=np.float32),
        "should_broadcast": 0,
        "comm_target_mask": np.zeros(4, dtype=np.int8),
    }


# Single source of truth for the capture + replay action sequence. The parity
# test imports this constant so the sim is driven with the EXACT actions the
# real env was. Index i of the trace is the obs AFTER applying SCRIPTED_ACTIONS[i]
# (trace[0] is the post-reset obs, before any action).
SCRIPTED_ACTIONS: list[dict] = [
    _wait_action(),
    _harvest_one(),
    _harvest_one(),
    _harvest_one(),
]


def _jsonable(obs: dict) -> dict:
    """Convert a gym-Dict obs (numpy arrays, possibly nested action_mask) into a
    JSON-serializable structure (lists/ints/floats)."""
    out: dict = {}
    for key, val in obs.items():
        if isinstance(val, dict):
            out[key] = _jsonable(val)
        elif isinstance(val, np.ndarray):
            out[key] = val.tolist()
        elif isinstance(val, np.integer):
            out[key] = int(val)
        elif isinstance(val, np.floating):
            out[key] = float(val)
        else:
            out[key] = val
    return out


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True, help="free Fabric instance port")
    ap.add_argument("--out", type=str, required=True, help="fixture output path (JSON)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    # Heavy import is intentional and confined to this capture script.
    from aiutopia.env.wrapper import AiUtopiaPettingZooEnv

    env = AiUtopiaPettingZooEnv(
        {
            "stage": 1,
            "active_roles": ["gatherer"],
            "seed_strategy": "fixed_easy",
            "py4j_ports": [args.port],
            "tick_warp": True,
            "max_episode_ticks": 1000,
            "per_worker_seed_offset": False,
            "enable_memory_writes": False,
            "aiutopia_root_per_worker": False,
            "skill_timeout_ticks": {0: 800, 1: 800, 2: 800},
        }
    )
    agent = "gatherer_0"

    obs, _ = env.reset(seed=args.seed)
    # Guard against the cold-start spawn race (first /tp before the Carpet
    # player exists) — re-reset if the agent isn't in the arena.
    for attempt in range(3):
        pos = list(obs[agent]["position"])
        if abs(pos[0] - 64.0) < 30 and pos[1] >= 60:
            break
        _p(f"[capture] reset attempt {attempt}: agent at {pos} (not arena) — re-resetting")
        obs, _ = env.reset(seed=args.seed)

    trace: list[dict] = [_jsonable(obs[agent])]
    _p(f"[capture] post-reset obs recorded; pos={list(obs[agent]['position'])}")

    for i, act in enumerate(SCRIPTED_ACTIONS):
        obs, rew, term, trunc, info = env.step({agent: act})
        trace.append(_jsonable(obs[agent]))
        comp = info.get(agent, {}).get("skill_completion", {})
        _p(
            f"[capture] step {i}: skill={act['skill_type']} rew={float(rew[agent]):+.3f} "
            f"rc={comp.get('resultCode', '?')} pos={list(obs[agent]['position'])}"
        )
        if term.get(agent) or trunc.get(agent):
            _p(f"[capture] episode ended at step {i}")
            break

    env.close()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(trace, fh)
    _p(f"[capture] wrote {len(trace)} obs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
