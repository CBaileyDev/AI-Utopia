"""THROWAWAY (investigation-only, do not commit). Seed 1/2/3 arena geometry +
greedy nearest-neighbor HARVEST simulation under a *shared per-dispatch* tick
budget, to explain why seeds 1,2 pass but seed 3 stalls at ~55/64 in real MC.

No live MC. Pure sim geometry (SimWorld) + a faithful re-implementation of the
real HarvestSkill dispatch loop:
  - one dispatch = one HARVEST env-step.
  - shared budget = 400 ticks (wrapper injects skill_timeout_ticks[1]=400).
  - per log: walk-ticks = ceil((dist_to_reach)/WALK_PER_TICK), +1 break tick.
  - greedy nearest-alive within MAX_SEARCH_RADIUS=16 of agent's *current* pos.
  - cap=64 (the policy's observed big HARVEST: scalar≈1.0 -> cap 64).
  - agent settles ~REACH_RADIUS from the log it last broke (we leave it AT the
    log center minus reach along approach dir; horizontal-only approximation).

Run:
  PYTHONPATH=src py -3.11 scripts/_seed3_geom.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aiutopia.sim.world import SimWorld  # noqa: E402

WALK_PER_TICK = 4.3 / 20.0       # 0.215 b/tick
REACH_RADIUS = 4.5
MAX_SEARCH_RADIUS = 16.0
DISPATCH_BUDGET = 400            # wrapper's injected HARVEST timeout_ticks
SPAWN = np.array([64.5, 65.0, -47.5], dtype=np.float64)


def _nearest_alive(pos, logs, alive):
    """Index of nearest alive log within MAX_SEARCH_RADIUS of pos (3D, block
    centers), or -1. Mirrors HarvestSkill.findNearest scanning integer offsets
    from floor(pos); we approximate with continuous 3D distance to centers."""
    centers = logs.astype(np.float64) + 0.5
    d = centers - pos
    dist = np.sqrt((d * d).sum(axis=1))
    in_range = alive & (dist <= MAX_SEARCH_RADIUS)
    if not in_range.any():
        return -1, None
    masked = np.where(in_range, dist, np.inf)
    idx = int(np.argmin(masked))
    return idx, float(dist[idx])


def simulate_one_harvest_dispatch(pos, logs, alive, *, budget=DISPATCH_BUDGET, cap=64):
    """Faithful real-HARVEST dispatch: shared tick budget across walk+break.

    Returns (broken, new_pos, alive, stranded_reason). agent walks at
    WALK_PER_TICK toward each nearest log until within REACH_RADIUS, breaks it
    (1 tick), repeats. Budget is shared. Leaves agent where budget ran out.
    """
    pos = pos.copy()
    alive = alive.copy()
    ticks_left = budget
    broken = 0
    reason = None
    while broken < cap and ticks_left > 0:
        idx, dist = _nearest_alive(pos, logs, alive)
        if idx < 0:
            reason = f"no oak_log within {MAX_SEARCH_RADIUS} of pos (stranded)"
            break
        center = logs[idx].astype(np.float64) + 0.5
        # walk-ticks to come within REACH_RADIUS
        gap = max(0.0, dist - REACH_RADIUS)
        walk_ticks = int(np.ceil(gap / WALK_PER_TICK)) if gap > 0 else 0
        if walk_ticks + 1 > ticks_left:
            # can't reach + break this log within remaining budget; walk as far
            # as budget allows, then stop (stranded mid-approach).
            walk_dist = ticks_left * WALK_PER_TICK
            direction = (center - pos) / dist
            pos = pos + direction * walk_dist
            reason = (f"budget exhausted mid-approach to log#{idx} "
                      f"(needed {walk_ticks+1} ticks, had {ticks_left})")
            ticks_left = 0
            break
        # move to reach point, break
        if walk_ticks > 0:
            direction = (center - pos) / dist
            pos = pos + direction * (gap)  # land exactly at reach distance
        ticks_left -= (walk_ticks + 1)
        alive[idx] = False
        broken += 1
    if broken >= cap:
        reason = f"hit cap={cap}"
    elif ticks_left <= 0 and reason is None:
        reason = "budget exhausted"
    return broken, pos, alive, reason


def analyze_seed(seed: int):
    w = SimWorld()
    w.reset(seed)
    logs = w.logs.copy()
    alive = w.log_alive.copy()
    pos = SPAWN.copy()

    # distances from spawn
    centers = logs.astype(np.float64) + 0.5
    d0 = np.sqrt(((centers - pos) ** 2).sum(axis=1))
    n_in16_spawn = int((d0 <= MAX_SEARCH_RADIUS).sum())

    print(f"\n=== SEED {seed} ===")
    print(f"  spawn={pos.tolist()}  logs={len(logs)}  "
          f"within 16 of spawn: {n_in16_spawn}/64  "
          f"dist range from spawn: [{d0.min():.1f}, {d0.max():.1f}]")

    # Simulate the FIRST big HARVEST dispatch (cap=64, budget=400).
    broken, new_pos, alive_after, reason = simulate_one_harvest_dispatch(
        pos, logs, alive, budget=DISPATCH_BUDGET, cap=64)
    print(f"  big HARVEST(cap=64, budget={DISPATCH_BUDGET}): broke {broken}  "
          f"end_pos={[round(float(x),1) for x in new_pos]}  reason='{reason}'")

    # After the big dispatch: how many logs remain, and are they within 16 of
    # the agent's resting position? (= can the next HARVEST chain them, or is a
    # NAVIGATE required?)
    remaining = int(alive_after.sum())
    if remaining > 0:
        idx, dist = _nearest_alive(new_pos, logs, alive_after)
        centers_a = logs[alive_after].astype(np.float64) + 0.5
        da = np.sqrt(((centers_a - new_pos) ** 2).sum(axis=1))
        n_in16 = int((da <= MAX_SEARCH_RADIUS).sum())
        print(f"  AFTER: remaining={remaining}  within 16 of resting pos: "
              f"{n_in16}/{remaining}  nearest remaining dist={dist if dist else float('nan'):.1f}"
              if idx >= 0 else
              f"  AFTER: remaining={remaining}  within 16 of resting pos: 0/{remaining} "
              f"(ALL STRANDED beyond 16 -> need NAVIGATE)  "
              f"nearest remaining dist={da.min():.1f}")
        if idx < 0:
            print(f"        -> next HARVEST returns IMMEDIATE_FAILURE 'no oak_log within 16' "
                  f"(case d: stranded, never moves)")
        else:
            print(f"        -> next HARVEST CAN chain (nearest within 16); "
                  f"seed should keep collecting")
    else:
        print(f"  AFTER: remaining=0  -> SUCCESS in one dispatch")

    # Greedy multi-dispatch WITHOUT navigate (what the open-loop policy does):
    # repeatedly HARVEST from wherever it rests, never repositioning.
    pos2, alive2 = pos.copy(), w.log_alive.copy()
    total = 0
    for disp in range(200):
        b, pos2, alive2, r = simulate_one_harvest_dispatch(
            pos2, logs, alive2, budget=DISPATCH_BUDGET, cap=64)
        total += b
        if b == 0:
            print(f"  open-loop (HARVEST-only) STALLS at {total}/64 after "
                  f"dispatch {disp+1} (reason='{r}')")
            break
        if alive2.sum() == 0:
            print(f"  open-loop (HARVEST-only) SUCCEEDS at {total}/64 in "
                  f"{disp+1} dispatches")
            break
    else:
        print(f"  open-loop did not converge in 200 dispatches (total={total})")

    return new_pos, alive_after, logs


for s in (1, 2, 3):
    analyze_seed(s)
