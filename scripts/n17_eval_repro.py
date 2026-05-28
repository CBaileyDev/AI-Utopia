"""N17: reproduce + diagnose the eval scenario numpy.object_ bug.

Builds the same env scenario_runner.run_scenario uses, then walks each
obs key and reports which one np.asarray turns into dtype object.
"""
from __future__ import annotations

import sys

import numpy as np

from aiutopia.env.wrapper import AiUtopiaPettingZooEnv


def _p(msg): print(msg, file=sys.stderr, flush=True)


def main() -> int:
    env = AiUtopiaPettingZooEnv({
        "stage": 1,
        "active_roles": ["gatherer"],
        "seed_strategy": "fixed_easy",
        "py4j_ports": [25001],
        "tick_warp": True,
        "max_episode_ticks": 1000,
        "per_worker_seed_offset": False,
        "enable_memory_writes": False,
        "aiutopia_root_per_worker": False,
        "skill_timeout_ticks": {0: 800, 1: 800, 2: 800},
    })
    obs, _info = env.reset(seed=1)
    agent_id = list(obs.keys())[0]
    agent_obs = obs[agent_id]
    _p(f"[n17] obs has {len(agent_obs)} keys; checking dtype after np.asarray …")
    for k, v in agent_obs.items():
        try:
            arr = np.asarray(v)
            tag = "OK" if arr.dtype != np.object_ else "🔥 OBJECT_"
            _p(f"[n17] {tag:13s} {k!r:42s} shape={arr.shape} dtype={arr.dtype} "
               f"raw_type={type(v).__name__}")
            if arr.dtype == np.object_:
                # describe what's inside
                _p(f"[n17]   raw value repr: {repr(v)[:200]}")
                if hasattr(v, '__iter__'):
                    for i, x in enumerate(list(v)[:3]):
                        _p(f"[n17]     elem[{i}]: type={type(x).__name__} repr={repr(x)[:120]}")
        except Exception as e:
            _p(f"[n17] EXCEPTION on {k!r}: {e}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
