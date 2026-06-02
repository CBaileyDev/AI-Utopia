#!/usr/bin/env python
"""Variance-controlled real-MC transfer eval for a bare state_dict checkpoint.

THE M1 BLOCKER this addresses (HANDOFF.md S3): real-MC HARVEST is non-deterministic
-- back-to-back identical dispatches on the SAME seed collect different oak counts
(measured 46-64 swings regardless of policy or instance freshness). A single greedy
3-seed eval therefore CANNOT declare a clean pass/fail at n=1. This harness runs each
gate scenario N>=repeats times and gates on a per-scenario success RATE (>= a
threshold), reporting the per-seed oak distribution so the variance is visible.

Each repeat resets the SAME scenario seed (identical byte-faithful arena), so the
only thing that varies between repeats is the real HARVEST non-determinism -- exactly
the quantity we are characterizing. --warmup absorbs the documented cold-start spawn
race (the first reset on a freshly-launched server can strand the agent at origin ->
0/64) by discarding one reset before measuring.

Run: PYTHONPATH=src py -3.11 scripts/transfer_eval_bc.py \
        --weights weights/bc_gatherer.pt --port 25001 --repeats 5 --pass-threshold 0.6
"""

from __future__ import annotations

import argparse
import importlib
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def summarize(scenario_runs: dict[str, list[dict]], pass_threshold: float) -> dict:
    """Aggregate per-scenario repeats into rates + a gate verdict (pure, testable).

    ``scenario_runs`` maps scenario name -> list of per-repeat dicts, each with keys
    ``success`` (bool) and ``oak`` (int). A scenario PASSES when its success rate is
    >= ``pass_threshold``; the overall gate PASSES only when EVERY scenario passes.
    Returns a structured summary (no I/O) so it can be unit-tested without a server.
    """
    per_scenario = {}
    for name, runs in scenario_runs.items():
        n = len(runs)
        successes = sum(1 for r in runs if r["success"])
        oaks = [int(r["oak"]) for r in runs]
        rate = (successes / n) if n > 0 else 0.0
        per_scenario[name] = {
            "n": n,
            "successes": successes,
            "rate": rate,
            "oak_min": min(oaks) if oaks else 0,
            "oak_max": max(oaks) if oaks else 0,
            "oak_mean": (sum(oaks) / n) if n > 0 else 0.0,
            "oaks": oaks,
            "passed": n > 0 and rate >= pass_threshold,
        }
    overall = bool(per_scenario) and all(s["passed"] for s in per_scenario.values())
    return {"per_scenario": per_scenario, "passed": overall, "pass_threshold": pass_threshold}


def _format_summary(summary: dict) -> str:
    lines = []
    for name, s in summary["per_scenario"].items():
        verdict = "PASS" if s["passed"] else "FAIL"
        lines.append(
            f"  {name}: rate {s['successes']}/{s['n']} = {s['rate']:.2f} [{verdict}] "
            f"| oak min/mean/max {s['oak_min']}/{s['oak_mean']:.1f}/{s['oak_max']} "
            f"| oaks {s['oaks']}"
        )
    overall = "PASS" if summary["passed"] else "FAIL"
    lines.append(
        f"GATE (threshold {summary['pass_threshold']:.2f} success-rate per seed): {overall}"
    )
    return "\n".join(lines)


def main() -> None:
    import torch  # noqa: PLC0415
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec  # noqa: PLC0415
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec  # noqa: PLC0415

    sp = importlib.import_module("aiutopia." + "env.spaces")
    rw = importlib.import_module("aiutopia." + "env.reward")
    from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule  # noqa: PLC0415

    sr = importlib.import_module("aiutopia.train.scenario_runner")

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--port", type=int, default=25001)
    ap.add_argument(
        "--repeats", type=int, default=5, help="evals per scenario (variance control; >=1)"
    )
    ap.add_argument(
        "--pass-threshold",
        type=float,
        default=0.6,
        help="min per-scenario success RATE to pass (0.6 == >=3/5)",
    )
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="discard one seed_1 reset first to absorb the cold-start spawn race",
    )
    args = ap.parse_args()

    spec = MultiRLModuleSpec(
        rl_module_specs={
            "gatherer_policy": RLModuleSpec(
                module_class=AiUtopiaRoleRLModule,
                observation_space=sp.build_role_observation_space("gatherer", stage=1),
                action_space=sp.build_role_action_space("gatherer"),
                model_config={
                    "role": "gatherer",
                    "max_seq_len": 32,
                    "actor_hidden": [256],
                    "mask_comm": True,
                    "core_encoder": {"core_hidden": [512, 256]},
                    "shared_backbone": {"lstm_hidden": 256},
                    "ctde_critic": {"critic_hidden": 256},
                },
            )
        }
    ).build()
    mod = spec["gatherer_policy"]
    sd = torch.load(args.weights, map_location="cpu")
    missing, unexpected = mod.load_state_dict(sd, strict=False)
    print(f"[load] {args.weights} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    mod.eval()

    env_config = {
        "active_roles": ["gatherer"],
        "stage": 1,
        "py4j_ports": [args.port],
        "enable_memory_writes": False,
    }

    def _run_once(sc) -> dict:
        r = sr.run_scenario(sc, env_config=env_config, rl_module=mod)
        inv = rw._inventory_from_obs(r["final_inventory"].get("gatherer_0", {}))
        oak = sum(c for k, c in inv.items() if k == "oak_log")
        return {"success": bool(r["success"]), "oak": int(oak)}

    if args.warmup:
        warm_sc = next(s for s in sr.M1_SCENARIOS if s.seed == 1)
        wr = _run_once(warm_sc)
        print(f"[warmup] discarded seed_1 reset (oak={wr['oak']})", flush=True)

    scenario_runs: dict[str, list[dict]] = {}
    for sc in sr.M1_SCENARIOS:
        runs = []
        for i in range(max(1, args.repeats)):
            res = _run_once(sc)
            runs.append(res)
            print(
                f"REAL {sc.name} rep {i + 1}/{args.repeats}: "
                f"success={res['success']} oak={res['oak']}",
                flush=True,
            )
        scenario_runs[sc.name] = runs

    summary = summarize(scenario_runs, args.pass_threshold)
    print(_format_summary(summary), flush=True)


if __name__ == "__main__":
    main()
