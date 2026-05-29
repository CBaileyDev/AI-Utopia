"""N20b — reproduce the golden-trace stall via the ENV WRAPPER (not raw bridge).

The golden-trace capture (scripts/capture_gatherer_obs_fixture.py) drives
AiUtopiaPettingZooEnv: reset(seed=1) -> WAIT -> HARVEST x3, and saw 1,0,0.
The raw-bridge probe (n20) saw 8/8. So the stall is induced by something the
ENV does between dispatches that the raw bridge does not. This script runs the
EXACT env path and logs per step: resultCode, failureReason, reward,
inventory-from-obs, position, and whether the agent/agents set changed.

Repeated 3x to test determinism. Also runs an extended HARVEST x10 (no WAIT)
through the env to compare with n14's 6/6.

INVESTIGATION ONLY — uses warm instance-1 (port 25001). No production code edits.
"""
from __future__ import annotations

import sys

import numpy as np

from aiutopia.env.wrapper import AiUtopiaPettingZooEnv
from aiutopia.env.reward import _inventory_from_obs


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _wait():
    return {
        "skill_type": 4, "target_class": 0,
        "spatial_param": np.zeros(3, dtype=np.float32),
        "scalar_param": np.zeros(1, dtype=np.float32),
        "comm_payload": np.zeros(128, dtype=np.float32),
        "should_broadcast": 0, "comm_target_mask": np.zeros(4, dtype=np.int8),
    }


def _harvest():
    return {
        "skill_type": 1, "target_class": 0,
        "spatial_param": np.zeros(3, dtype=np.float32),
        "scalar_param": np.asarray([1.0 / 64.0], dtype=np.float32),
        "comm_payload": np.zeros(128, dtype=np.float32),
        "should_broadcast": 0, "comm_target_mask": np.zeros(4, dtype=np.int8),
    }


def _build_env():
    return AiUtopiaPettingZooEnv({
        "stage": 1, "active_roles": ["gatherer"], "seed_strategy": "fixed_easy",
        "py4j_ports": [25001], "tick_warp": True, "max_episode_ticks": 1000,
        "per_worker_seed_offset": False, "enable_memory_writes": False,
        "aiutopia_root_per_worker": False,
        "skill_timeout_ticks": {0: 800, 1: 800, 2: 800},
    })


def _step_log(env, agent, act, label):
    obs, rew, term, trunc, info = env.step({agent: act})
    o = obs.get(agent, {})
    comp = info.get(agent, {}).get("skill_completion", {})
    inv = _inventory_from_obs(o)
    pos = list(o.get("position", [None, None, None])) if agent in obs else "AGENT_GONE"
    gs = info.get(agent, {}).get("goal_success", None)
    _p(
        f"    {label}: rc={comp.get('resultCode','?'):<16} rew={float(rew.get(agent,0.0)):+.3f} "
        f"term={term.get(agent)} trunc={trunc.get(agent)} goal_success={gs} "
        f"inv={inv} pos={pos}"
    )
    fr = comp.get("failureReason", "")
    if fr:
        _p(f"        fr={fr!r}")
    return obs, rew, term, trunc, info


def golden_repro(env, rep):
    agent = "gatherer_0"
    obs, _ = env.reset(seed=1)
    o = obs[agent]
    _p(f"  [golden rep={rep}] post-reset pos={list(o['position'])} inv={_inventory_from_obs(o)} agents={env.agents}")
    seq = [("WAIT", _wait()), ("HARVEST#1", _harvest()),
           ("HARVEST#2", _harvest()), ("HARVEST#3", _harvest())]
    for label, act in seq:
        if agent not in env.agents:
            _p(f"    {label}: SKIPPED — agent removed from env.agents (terminated/truncated)")
            continue
        obs, rew, term, trunc, info = _step_log(env, agent, act, label)


def extended_repro(env, rep, n=10):
    agent = "gatherer_0"
    obs, _ = env.reset(seed=1)
    o = obs[agent]
    _p(f"  [ext rep={rep}] post-reset pos={list(o['position'])} inv={_inventory_from_obs(o)} agents={env.agents}")
    for i in range(n):
        if agent not in env.agents:
            _p(f"    HARVEST[{i}]: SKIPPED — agent removed (term/trunc)")
            continue
        _step_log(env, agent, _harvest(), f"HARVEST[{i}]")


def main() -> int:
    env = _build_env()
    try:
        _p("=== GOLDEN-TRACE REPRO (reset -> WAIT -> HARVEST x3), 3 reps ===")
        for rep in range(3):
            golden_repro(env, rep)
        _p("\n=== EXTENDED (reset -> HARVEST x10, no WAIT), 3 reps ===")
        for rep in range(3):
            extended_repro(env, rep, n=10)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
