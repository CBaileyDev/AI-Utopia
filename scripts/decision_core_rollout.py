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

    env_config = {
        "stage": 1,
        "active_roles": ["gatherer"],
        "decision_core": True,
        "arena_mode": "clusters",
        "arena_half": 34.0,
        # eval: fixed seeds, no layout randomization
        "randomize_layout": False,
        "distance_shaping": False,  # shaping is training-only; eval is pure reward
    }
    any_explore = False
    n_full = 0
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
        nav = hist.get("NAVIGATE", 0)
        any_explore = any_explore or nav > 0
        full = r["oak_log"] >= 64
        n_full += int(full)
        clusterB = "BOTH clusters" if full else ("cluster A only" if r["oak_log"] >= 28 else "stuck early")
        _p(
            f"  seed={scn.seed}  oak_log={r['oak_log']}/64  steps={r['steps_used']}  "
            f"NAVIGATE={nav}  MINE={hist.get('HARVEST', 0)}  -> {clusterB}"
        )
    _p("")
    _p(f">>> learned to EXPLORE (NAVIGATE>0): {any_explore} ; cleared BOTH clusters on {n_full}/3 seeds <<<")
    return 0


if __name__ == "__main__":
    sys.exit(main())
