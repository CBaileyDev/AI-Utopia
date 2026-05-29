"""SPIKE probe (advisor-requested): is survival-timed (multi-tick) HARVEST
deterministic across repeated dispatches on real MC? The whole survival-forest
milestone rests on this. Runs the SAME reset(seed)+HARVEST sequence N times and
checks the per-step oak_log trajectories are identical. Also reports per-episode
wall time (timed break should be much slower than the ~8 s instant clear) and
logs-per-400-tick-dispatch (for the tick-budget tuning the advisor flagged).

Run (instance-1 warm on 25001, spike jar deployed):
  PYTHONPATH=src AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data \
    py -3.11 scripts/n21_breaktiming_determinism.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

from aiutopia.env.reward import _inventory_from_obs


def _p(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _oak(obs: dict) -> int:
    inv = _inventory_from_obs(obs.get("gatherer_0", {}))
    return int(sum(c for n, c in inv.items() if n == "oak_log"))


def _harvest() -> dict:
    return {
        "gatherer_0": {
            "skill_type": 1,
            "target_class": 0,
            "spatial_param": np.zeros(3, np.float32),
            "scalar_param": np.array([1.0], np.float32),
            "comm_payload": np.zeros(128, np.float32),
            "should_broadcast": 0,
            "comm_target_mask": np.zeros(4, np.int8),
        }
    }


REAL_CONFIG = {
    "stage": 1,
    "active_roles": ["gatherer"],
    "seed_strategy": "fixed_easy",
    "py4j_ports": [25001],
    "tick_warp": True,
    "per_worker_seed_offset": False,
    "enable_memory_writes": False,
    "aiutopia_root_per_worker": False,
}


def run_episode(seed: int, max_steps: int = 14):
    from aiutopia.env.wrapper import AiUtopiaPettingZooEnv

    env = AiUtopiaPettingZooEnv({**REAL_CONFIG, "max_episode_ticks": 1000})
    t0 = time.time()
    try:
        obs, _ = env.reset(seed=seed)
        traj = [_oak(obs)]
        rcodes = []
        for _ in range(max_steps):
            obs, _rew, term, trunc, info = env.step(_harvest())
            traj.append(_oak(obs))
            comp = info.get("gatherer_0", {}).get("skill_completion")
            rcodes.append(comp.get("resultCode") if isinstance(comp, dict) else "?")
            if term.get("gatherer_0") or trunc.get("gatherer_0"):
                break
        wall = time.time() - t0
        return tuple(traj), tuple(rcodes), round(wall, 1)
    finally:
        env.close()


def main() -> int:
    _p("=" * 70)
    _p("SPIKE: survival break-timing determinism on real MC (instance-1)")
    _p("=" * 70)

    # Warm-up reset to consume the cold-start spawn race (first reset on a fresh
    # server can strand the agent at world origin); discard.
    _p("[warmup] discarding first episode (cold-start spawn race)...")
    run_episode(1, max_steps=14)

    plan = [1, 1, 1, 2, 2, 3, 3]
    results: dict[int, list] = {}
    for seed in plan:
        traj, rcodes, wall = run_episode(seed)
        results.setdefault(seed, []).append((traj, rcodes, wall))
        _p(f"  seed={seed}  oak_traj={list(traj)}  wall={wall}s  rcodes={list(rcodes)}")

    _p("")
    _p("=== DETERMINISM VERDICT ===")
    all_det = True
    for seed, runs in sorted(results.items()):
        trajs = {r[0] for r in runs}
        det = len(trajs) == 1
        all_det = all_det and det
        finals = [r[0][-1] for r in runs]
        _p(
            f"  seed={seed}: {len(runs)} runs  finals={finals}  "
            f"{'DETERMINISTIC ✓' if det else 'NON-DETERMINISTIC ✗ ' + str(trajs)}"
        )
    # logs-per-dispatch (budget tuning): first step of a fresh full grid
    s1 = results.get(1, [])
    if s1:
        first_step_logs = s1[0][0][1] - s1[0][0][0]
        _p(f"  logs cleared in the FIRST 400-tick dispatch (budget calib): ~{first_step_logs}")
    _p("")
    _p(f">>> survival break-timing is {'DETERMINISTIC' if all_det else 'NON-DETERMINISTIC'} on real MC <<<")
    return 0


if __name__ == "__main__":
    sys.exit(main())
