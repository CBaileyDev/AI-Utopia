"""Evaluate a DECISION-CORE policy on the 2-cluster blind-explore arena.

Loads the newest sim checkpoint and runs it GREEDILY in decision_core mode on the
clusters arena (seeds 1/2/3, fixed). The decisive signals:
  - oak_log >= 64  -> cleared BOTH clusters = learned to explore-when-blind + select
  - oak_log ~= 32  -> cleared cluster A only = learned to MINE but NOT to explore to B
  - NAVIGATE count  -> is the policy actually exploring (vs HARVEST-only)?

Run: PYTHONPATH=src AIUTOPIA_ROOT=... AIUTOPIA_DATA_DIR=... \
       py -3.11 scripts/decision_core_rollout.py
"""
from __future__ import annotations

import sys
from collections import Counter

from transfer_eval import GATHERER_MODULE_DIR, run_instrumented


def _p(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _load():
    from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule  # noqa: F401
    from ray.rllib.core.rl_module.rl_module import RLModule

    m = RLModule.from_checkpoint(GATHERER_MODULE_DIR)
    if hasattr(m, "eval"):
        m.eval()
    if hasattr(m, "to"):
        m.to("cpu")
    return m


def main() -> int:
    _p("=" * 70)
    _p("DECISION-CORE rollout — 2-cluster blind-explore arena")
    _p(f"checkpoint: {GATHERER_MODULE_DIR}")
    _p("=" * 70)
    module = _load()

    from aiutopia.train.scenario_runner import M1_SCENARIOS
    from aiutopia.train.sim_env_factory import make_aiutopia_sim_env

    for arena_mode, arena_half, note in (
        ("trees", 24.0, "all 16 trunks perceivable as the agent migrates (NO blind hop)"),
        ("clusters", 34.0, "8 trunks beyond perception -> requires a BLIND explore hop"),
    ):
        _p(f"  -- arena: {arena_mode} ({note}) --")
        env_config = {
            "stage": 1,
            "active_roles": ["gatherer"],
            "decision_core": True,
            "arena_mode": arena_mode,
            "arena_half": arena_half,
            "randomize_layout": False,
            "distance_shaping": False,  # shaping is training-only; eval is pure reward
        }
        for scn in M1_SCENARIOS:
            r = run_instrumented(
                scn,
                env_factory=lambda cfg: make_aiutopia_sim_env(cfg),
                env_config=env_config,
                rl_module=module,
                device="cpu",
                wall_budget_s=120,
            )
            hist = Counter(e["skill"] for e in r["trace"])
            tag = (
                "CLEARED ✓" if r["oak_log"] >= 64
                else ("cluster A only" if r["oak_log"] >= 28 else "stuck")
            )
            _p(
                f"    seed={scn.seed}  oak_log={r['oak_log']}/64  steps={r['steps_used']}  "
                f"NAVIGATE={hist.get('NAVIGATE', 0)}  MINE={hist.get('HARVEST', 0)}  -> {tag}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
