"""Sim-only rollout check: does the v5 (PBRS-shaped) policy NAVIGATE?

Loads the newest sim checkpoint and runs the 3 M1 gate scenarios IN THE SIM
(fixed seeds, no randomize, no shaping — shaping never affects greedy action
choice). Prints the per-seed skill histogram + oak_log + steps. The decisive
signal: a HARVEST-only histogram = still stuck in the local optimum; a mix of
NAVIGATE + HARVEST that clears the field = gap #2 closed in sim.

Run:
  PYTHONPATH=src AIUTOPIA_ROOT=C:/Users/Carte/OneDrive/Desktop/AiUtopia \
    AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data py -3.11 scripts/sim_rollout_check.py
"""
from __future__ import annotations

import sys
from collections import Counter

from transfer_eval import (  # reuse the exact instrumented-loop harness
    GATHERER_MODULE_DIR,
    run_instrumented,
)


def _p(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _load_module():
    """Load the gatherer RLModule directly (the standalone-proven path —
    importing transfer_eval.load_gatherer_module hits a load-order quirk where
    from_checkpoint returns a base RLModule)."""
    # Pre-import so from_checkpoint can reconstruct the concrete class (a cold
    # first from_checkpoint call otherwise returns a useless base RLModule).
    from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule  # noqa: F401
    from ray.rllib.core.rl_module.rl_module import RLModule

    m = RLModule.from_checkpoint(GATHERER_MODULE_DIR)
    if hasattr(m, "eval"):
        m.eval()
    if hasattr(m, "to"):
        m.to("cpu")
    _p(f"[load] {type(m).__name__} params="
       f"{sum(p.numel() for p in m.parameters())}")
    return m


def main() -> int:
    _p("=" * 72)
    _p("SIM rollout check (v5 PBRS-shaped policy) — does it NAVIGATE?")
    _p(f"checkpoint: {GATHERER_MODULE_DIR}")
    _p("=" * 72)

    module = _load_module()

    from aiutopia.train.scenario_runner import M1_SCENARIOS
    from aiutopia.train.sim_env_factory import make_aiutopia_sim_env

    sim_config = {
        "stage": 1,
        "active_roles": ["gatherer"],
        "seed_strategy": "fixed_easy",
        "per_worker_seed_offset": False,
        # explicit: fixed-seed eval, NOT randomized, NOT shaped.
        "randomize_layout": False,
        "distance_shaping": False,
    }

    any_nav = False
    for scn in M1_SCENARIOS:
        r = run_instrumented(
            scn,
            env_factory=lambda cfg: make_aiutopia_sim_env(cfg),
            env_config=sim_config,
            rl_module=module,
            device="cpu",
            wall_budget_s=120,
        )
        hist = Counter(e["skill"] for e in r["trace"])
        nav = hist.get("NAVIGATE", 0)
        any_nav = any_nav or nav > 0
        _p(
            f"  seed={scn.seed}  oak_log={r['oak_log']}/64  success={r['success']}  "
            f"steps={r['steps_used']}  NAVIGATE={nav}  HARVEST={hist.get('HARVEST', 0)}"
        )
        _p(f"    full histogram: {dict(hist)}")
    _p("")
    _p(f">>> policy uses NAVIGATE: {any_nav} <<<")
    return 0


if __name__ == "__main__":
    sys.exit(main())
