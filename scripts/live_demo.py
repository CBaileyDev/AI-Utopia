"""Live demo: drive the gatherer bot through N real-MC episodes back-to-back so
a connected spectator can watch it clear the 64-oak_log arena. Reuses the proven
transfer_eval load + instrumented loop against instance-1 (Py4J 25001).

Run (server must be warm on 25001):
  PYTHONPATH=src AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data \
    DEMO_ROUNDS=6 py -3.11 scripts/live_demo.py
"""
from __future__ import annotations

import os
import sys
import time

from transfer_eval import GATHERER_MODULE_DIR, run_instrumented


def _p(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _load():
    # Pre-import so from_checkpoint reconstructs the concrete torch class.
    from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule  # noqa: F401
    from ray.rllib.core.rl_module.rl_module import RLModule

    m = RLModule.from_checkpoint(GATHERER_MODULE_DIR)
    if hasattr(m, "eval"):
        m.eval()
    if hasattr(m, "to"):
        m.to("cpu")
    return m


def main() -> int:
    rounds = int(os.environ.get("DEMO_ROUNDS", "6"))
    pause = float(os.environ.get("DEMO_PAUSE_S", "3.0"))
    _p(f"=== LIVE DEMO: {rounds} episodes on instance-1 (Py4J 25001) ===")
    module = _load()

    from aiutopia.env.wrapper import AiUtopiaPettingZooEnv
    from aiutopia.train.scenario_runner import M1_SCENARIOS

    real_config = {
        "stage": 1,
        "active_roles": ["gatherer"],
        "seed_strategy": "fixed_easy",
        "py4j_ports": [25001],
        "tick_warp": True,
        "per_worker_seed_offset": False,
        "enable_memory_writes": False,
        "aiutopia_root_per_worker": False,
    }

    for r in range(rounds):
        scn = M1_SCENARIOS[r % len(M1_SCENARIOS)]
        res = run_instrumented(
            scn,
            env_factory=lambda cfg: AiUtopiaPettingZooEnv(cfg),
            env_config=real_config,
            rl_module=module,
            device="cpu",
            wall_budget_s=120,
        )
        _p(
            f"  round {r + 1}/{rounds}  seed={scn.seed}  "
            f"oak_log={res['oak_log']}/64  steps={res['steps_used']}  wall={res['wall_s']}s"
            + ("  ✓" if res["success"] else "")
        )
        time.sleep(pause)  # let the spectator see the fresh grid before the next clear
    _p("=== demo complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
