"""N14 HARVEST-only reward sanity probe.

Dispatches HARVEST repeatedly against the seeded oak_log ring placed by
reset_episode and verifies the gatherer primary reward path emits
positive signal when a log breaks.

HARVEST has its own internal navigation (scans 16 blocks, walks to
target, breaks it). We don't need NAVIGATE first — bypasses the
NAVIGATE clip-penalty confound.

Captures full skill_completion dict per step so we know whether
resultCode is COMPLETED / FAILED_TIMEOUT / IMMEDIATE_FAILURE / RUNNING.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from aiutopia.env.wrapper import AiUtopiaPettingZooEnv
from aiutopia.env.reward import _inventory_from_obs, _ITEM_ID_TO_NAME


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    _p(f"[probe] _ITEM_ID_TO_NAME BEFORE env init: {len(_ITEM_ID_TO_NAME)}")

    t0 = time.time()
    _p(f"[probe] {time.time()-t0:5.1f}s: constructing env …")
    env = AiUtopiaPettingZooEnv({
        "stage": 1,
        "active_roles": ["gatherer"],
        "seed_strategy": "fixed_easy",
        "py4j_ports": [25001],
        "tick_warp": True,
        "max_episode_ticks": 200,
        "per_worker_seed_offset": False,
        "enable_memory_writes": False,
        "aiutopia_root_per_worker": False,
        # HARVEST=1: 800 ticks @ 60 TPS = 13.3s wall. Walking 16 blocks
        # at 0.215 b/tick ≈ 75 ticks. Breaking 1 log ≈ 5 ticks. 800 is
        # ample slack.
        "skill_timeout_ticks": {0: 800, 1: 800, 2: 800},
    })
    _p(f"[probe] {time.time()-t0:5.1f}s: env built. table size: {len(_ITEM_ID_TO_NAME)}")

    obs, _ = env.reset(seed=1)
    _p(f"[probe] {time.time()-t0:5.1f}s: reset done. agents={list(obs.keys())}")
    agent = list(obs.keys())[0]
    pos0 = list(obs[agent]["position"])
    nrd0 = float(obs[agent].get("nearest_resource_distance", [999.0])[0])
    _p(f"[probe]   pos0={pos0} nrd0={nrd0:.2f}")

    rewards: list[float] = []
    for t in range(6):
        ts = time.time() - t0
        pos_now = list(obs[agent]["position"])
        # ALWAYS HARVEST oak_log (target_class=0, by Java TARGET_CLASS_TABLE)
        # scalar_param=1/64 → cap=1 block so each dispatch breaks just one
        act = {
            "skill_type":       1,                                      # HARVEST
            "target_class":     0,                                      # oak_log
            "spatial_param":    np.zeros(3, dtype=np.float32),          # unused for HARVEST
            "scalar_param":     np.asarray([1.0/64.0], dtype=np.float32),
            "should_broadcast": 0,
            "comm_target_mask": np.zeros(4, dtype=np.int8),
            "comm_payload":     np.zeros(128, dtype=np.float32),
        }
        _p(f"[probe] {ts:5.1f}s t={t}: dispatch HARVEST(oak_log, 1 block); pos={pos_now} …")
        sstart = time.time()
        obs, rew, term, trunc, info = env.step({agent: act})
        sdur = time.time() - sstart
        r = float(rew[agent])
        rewards.append(r)
        inv = _inventory_from_obs(obs[agent])
        pos_after = list(obs[agent]["position"])
        comp = info[agent].get("skill_completion", {}) if info else {}
        rc = comp.get("resultCode", "?")
        fr = comp.get("failureReason", "")
        clip = comp.get("clippedAxesBitset", 0)
        _p(f"[probe] {time.time()-t0:5.1f}s t={t} dur={sdur:.1f}s rew={r:+.3f} "
           f"rc={rc} clipBits={clip} fr={fr!r}")
        _p(f"[probe]   pos_after={pos_after} inv={inv}")
        if r > 0.1:
            _p(f"[probe]   ★ POSITIVE REWARD ★ at t={t}")
        if all(term.values()) or all(trunc.values()):
            _p(f"[probe] episode end at t={t}")
            break

    env.close()
    r = np.asarray(rewards)
    _p(f"[probe] === n={len(r)} mean={r.mean():.3f} max={r.max():.3f} min={r.min():.3f} sum={r.sum():.3f} ===")
    return 0 if r.max() > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
