"""Phase-0 integration proof: a SCRIPTED gatherer collects 64 oak_log and the
gate is recognised as PASSED.

Exercises all four task-side P0 fixes end-to-end against a live Fabric server:
  #1 WorldOps.resetEpisode places 64 REACHABLE oak_log (all must be collectable).
  #2 reward.py rewards ONLY oak_log for the gatherer (single attractor).
  #5 wrapper.py terminates the episode the moment 64 oak_log are held.
  #3 scenario_runner._gatherer_collected_64_oak_log returns True on that obs
     (and would NOT have, on the deleted false-pass path).

This is the "first verified nonzero gate pass" — the eval pass-case has only
ever been run against a random policy (expected success=0). We drive HARVEST
deterministically (cap=1 log/dispatch) so success comes from the WORLD being
winnable, not from a trained policy.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from aiutopia.env.reward import _inventory_from_obs
from aiutopia.env.wrapper import AiUtopiaPettingZooEnv
from aiutopia.train.scenario_runner import M1_OAK_LOG_TARGET, _gatherer_collected_64_oak_log


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _harvest_one() -> dict:
    """HARVEST oak_log, cap=1 block (scalar 1/64) — one log per dispatch."""
    return {
        "skill_type": 1,  # HARVEST
        "target_class": 0,  # oak_log
        "spatial_param": np.zeros(3, dtype=np.float32),
        "scalar_param": np.asarray([1.0 / 64.0], dtype=np.float32),
        "should_broadcast": 0,
        "comm_target_mask": np.zeros(4, dtype=np.int8),
        "comm_payload": np.zeros(128, dtype=np.float32),
    }


def main() -> int:
    t0 = time.time()
    env = AiUtopiaPettingZooEnv(
        {
            "stage": 1,
            "active_roles": ["gatherer"],
            "seed_strategy": "fixed_easy",
            "py4j_ports": [25001],
            "tick_warp": True,
            "max_episode_ticks": 1000,  # gate horizon (env steps)
            "per_worker_seed_offset": False,
            "enable_memory_writes": False,
            "aiutopia_root_per_worker": False,
            "skill_timeout_ticks": {0: 800, 1: 800, 2: 800},
        }
    )
    agent = "gatherer_0"

    # Reset, guarding against the known cold-start spawn race (first /tp can fire
    # before the Carpet player exists, leaving it at world origin). Re-reset if
    # the agent isn't in the arena (|x-64| should be small).
    obs, _ = env.reset(seed=1)
    for attempt in range(3):
        pos = list(obs[agent]["position"])
        if abs(pos[0] - 64.0) < 30 and pos[1] >= 60:
            break
        _p(f"[gate] reset attempt {attempt}: agent at {pos} (not arena) — re-resetting")
        obs, _ = env.reset(seed=1)
    _p(f"[gate] {time.time()-t0:5.1f}s reset; agent at {list(obs[agent]['position'])}")

    final_obs = obs
    term = {agent: False}
    trunc = {agent: False}
    last_inv = 0
    steps = 0
    MAX_STEPS = 120  # 64 logs + slack for any misses
    for steps in range(1, MAX_STEPS + 1):
        obs, rew, term, trunc, info = env.step({agent: _harvest_one()})
        final_obs = obs
        inv = _inventory_from_obs(obs.get(agent, {}))
        n = inv.get("oak_log", 0)
        if n != last_inv:
            _p(
                f"[gate] {time.time()-t0:5.1f}s step {steps}: oak_log={n} "
                f"rew={float(rew[agent]):+.2f} term={term.get(agent)} "
                f"goal_success={info.get(agent, {}).get('goal_success')}"
            )
            last_inv = n
        if term.get(agent) or trunc.get(agent):
            break

    n_final = _inventory_from_obs(final_obs.get(agent, {})).get("oak_log", 0)
    goal_success = info.get(agent, {}).get("goal_success", False)
    gate_pass = _gatherer_collected_64_oak_log(final_obs)
    env.close()

    _p("[gate] " + "=" * 56)
    _p(f"[gate] target               : {M1_OAK_LOG_TARGET} oak_log")
    _p(f"[gate] collected            : {n_final} oak_log in {steps} env-steps")
    _p(f"[gate] #5 episode terminated : {term.get(agent)}  (goal_success={goal_success})")
    _p(f"[gate] #3 gate predicate     : {gate_pass}")
    _p(f"[gate] truncated (budget hit): {trunc.get(agent)}")
    ok = (n_final >= M1_OAK_LOG_TARGET) and term.get(agent) and goal_success and gate_pass
    _p(f"[gate] RESULT               : {'PASS ✅' if ok else 'FAIL ❌'}")
    _p("[gate] " + "=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
