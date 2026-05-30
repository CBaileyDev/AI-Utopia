"""Identify the PROVEN M1B gatherer checkpoint by SIM-control rollout.

The hash-named run dirs lost track of which checkpoint is the proven 3/3 policy
(several non-decision-core runs were collapsed reward-tuning dead-ends; the eval
auto-picked the newest, which is bad in sim). The clean, over-claim-proof
discriminator is the SIM control: a proven gatherer collects 64/64 oak_log in the
headless sim (pure Python, no live MC). Sweep every NON-decision-core checkpoint,
greedy-rollout the M1 scenario in sim, report oak_log. The 64/64 ones are the
candidates to pin for transfer / promotion.

Run: PYTHONPATH=src AIUTOPIA_ROOT=... AIUTOPIA_DATA_DIR=... \
       py -3.11 scripts/find_proven_checkpoint.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_REPO = Path(__file__).resolve().parent.parent


def _p(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _is_decision_core(run_dir: Path) -> bool:
    try:
        return '"decision_core": true' in (run_dir / "params.json").read_text()
    except OSError:
        return False


def _sim_oak(module) -> tuple[int, int]:
    """Greedy sim rollout of the M1 scenario (seed 1); return (oak_log, steps)."""
    from transfer_eval import run_instrumented

    from aiutopia.train.scenario_runner import M1_SCENARIOS
    from aiutopia.train.sim_env_factory import make_aiutopia_sim_env

    scn = M1_SCENARIOS[0]
    r = run_instrumented(
        scn,
        env_factory=lambda cfg: make_aiutopia_sim_env(cfg),
        env_config={
            "stage": 1,
            "active_roles": ["gatherer"],
            "seed_strategy": "fixed_easy",
            "per_worker_seed_offset": False,
        },
        rl_module=module,
        device="cpu",
        wall_budget_s=120,
    )
    return r["oak_log"], r["steps_used"]


def main() -> int:
    from ray.rllib.core.rl_module.rl_module import RLModule

    import aiutopia.rl_module.role_rl_module  # noqa: F401 (concrete module for load)

    ckpts = sorted(
        (
            c
            for c in (_REPO / "runs" / "aiutopia_M1_seed1").glob(
                "PPO_aiutopia_sim_*/checkpoint_*"
            )
            if not _is_decision_core(c.parent)
        ),
        key=lambda p: p.stat().st_mtime,
    )
    _p(f"=== SIM-control sweep over {len(ckpts)} non-decision-core checkpoints ===")
    proven: list[str] = []
    for c in ckpts:
        mod_dir = c / "learner_group" / "learner" / "rl_module" / "gatherer_policy"
        if not mod_dir.exists():
            _p(f"  {c.parent.name}/{c.name}: (no gatherer_policy module) — skip")
            continue
        try:
            m = RLModule.from_checkpoint(mod_dir)
            m.eval()
            m.to("cpu")
            oak, steps = _sim_oak(m)
        except Exception as exc:
            _p(f"  {c.parent.name}/{c.name}: LOAD/RUN ERROR {type(exc).__name__}: {exc}")
            continue
        tag = "  <-- PROVEN (64/64)" if oak >= 64 else ""
        _p(f"  {c.parent.name}/{c.name}: sim oak_log={oak}/64 steps={steps}{tag}")
        if oak >= 64:
            proven.append(f"{c.parent.name}/{c.name}")
    _p("")
    if proven:
        _p(f">>> PROVEN (sim 64/64) checkpoints: {proven}")
        _p(">>> pin one of these for transfer_eval (TRANSFER_CKPT) / promotion.")
    else:
        _p(">>> NO non-decision-core checkpoint clears sim 64/64 — the proven M1B "
           "policy is not among the current run dirs (or all collapsed). "
           "A re-train of the proven HARVEST-spam config is needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
