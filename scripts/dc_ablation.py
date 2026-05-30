"""Oracle-ablation CHARACTERIZATION of the decision-core gatherer (sim-only).

Question this answers (NEXT_SESSION step 1, advisor-framed): how much of the
decision-core's blind-explore clearance is the POLICY learning vs the env oracle?
The decision-core feeds the policy two crutches:
  - perception HARVEST MASK  -> tells it WHEN (mask forces NAVIGATE when blind),
  - ground-truth bearing CUE -> tells it WHICH WAY (g_hostiles_nearby[0]).
A scripted ZERO-LEARNING follower that just reads those already clears 5/5
(scripts/_dc_follower.py) — so "5/5" measured the scaffolding, not the net.

This harness trains PPO under each crutch ablation and, for EVERY cell, reports
the gap between the trained policy and a per-cell scripted follower (the same
discriminator, restricted to the crutches present). A near-zero gap == the policy
learned ~nothing beyond reactive obs-reading; a positive gap in the no-crutch
cell == genuine learned search/selection.

  full      : mask ON , cue ON   (current decision_core baseline)
  mask_only : mask ON , cue OFF
  cue_only  : mask OFF, cue ON
  neither   : mask OFF, cue OFF

This is the OPPOSITE of the v1-v7 loop (which ADDED assists to win a number):
here we REMOVE assists to MEASURE. No claim is made without the follower control.

Run (sim-only, no live MC needed):
  PYTHONPATH=src AIUTOPIA_ROOT=C:/Users/Carte/OneDrive/Desktop/AiUtopia \
    AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data \
    DC_ITERS=100 DC_SEEDS=1,2,3 py -3.11 scripts/dc_ablation.py

Env knobs: DC_ITERS (train iters/run), DC_SEEDS (train seeds), DC_HELDOUT
(held-out eval seeds), DC_ABLATIONS (subset, comma-sep), DC_OUT (json path).
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

_REPO = Path(__file__).resolve().parent.parent

# Crutch matrix. decision_core stays ON for all (it's the MECHANISM under test);
# we toggle only the two oracle crutches.
ABLATIONS: dict[str, dict] = {
    "full": {"resource_bearing_cue": True, "harvest_mask_off": False},
    "mask_only": {"resource_bearing_cue": False, "harvest_mask_off": False},
    "cue_only": {"resource_bearing_cue": True, "harvest_mask_off": True},
    "neither": {"resource_bearing_cue": False, "harvest_mask_off": True},
}

ITERS = int(os.environ.get("DC_ITERS", "100"))
TRAIN_SEEDS = [int(s) for s in os.environ.get("DC_SEEDS", "1,2,3").split(",")]
HELDOUT_SEEDS = [
    int(s)
    for s in os.environ.get("DC_HELDOUT", "90001,90002,90003,90004,90005").split(",")
]
WHICH = [
    a.strip()
    for a in os.environ.get("DC_ABLATIONS", ",".join(ABLATIONS)).split(",")
    if a.strip()
]
OUT = Path(os.environ.get("DC_OUT", str(_REPO / "Research" / "dc_ablation_results.json")))
ARENA_HALF = 34.0
EVAL_MAX_TICKS = 200


def _p(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


# ───────────────────────── training ─────────────────────────
def _build_cfg(ablation: str, seed: int):
    """PPOConfig for one (ablation, seed): the proven decision_core sim config
    with the ablation crutch-knobs layered onto env_config, num_env_runners=0
    (local, fast, no worker spawn) and the Windows libuv learner override."""
    from aiutopia.train.config import m1_gatherer_config

    # num_envs_per_env_runner=1 matches the proven sim path (scripts/train.py);
    # the config default of 2 vectorizes the local runner and breaks RLlib's
    # remove-single-ts-time-rank connector (squeeze axis-0 size!=1).
    cfg = m1_gatherer_config(
        backend="sim", seed=seed, num_env_runners=0,
        num_envs_per_env_runner=1, decision_core=True,
    )
    # Merge the ablation knobs into the decision_core env_config (single source).
    env_config = dict(cfg.env_config)
    env_config.update(ABLATIONS[ablation])
    cfg = cfg.environment(env_config=env_config)
    cfg = cfg.learners(num_learners=0, num_gpus_per_learner=1.0)
    return cfg


def _train_one(ablation: str, seed: int, iters: int) -> tuple[object, list[float], float]:
    """Train in-process; return (module on CPU, return curve, wall_s)."""
    cfg = _build_cfg(ablation, seed)
    algo = cfg.build_algo()
    curve: list[float] = []
    t0 = time.time()
    for i in range(iters):
        res = algo.train()
        rm = float(res.get("env_runners", {}).get("episode_return_mean", 0.0) or 0.0)
        curve.append(round(rm, 2))
        if (i + 1) % 20 == 0:
            _p(f"      [{ablation} s{seed}] iter {i + 1}/{iters}  ret_mean={rm:.2f}")
    wall = time.time() - t0
    module = algo.get_module("gatherer_policy")
    if hasattr(module, "eval"):
        module.eval()
    if hasattr(module, "to"):
        module.to("cpu")
    # Keep a reference; stop the algo AFTER pulling the module to CPU (copies weights).
    import copy

    import torch

    with torch.no_grad():
        module_cpu = copy.deepcopy(module)
    algo.stop()
    return module_cpu, curve, wall


# ───────────────────────── eval ─────────────────────────
def _eval_env_config(ablation: str) -> dict:
    """Eval config: same crutches as training, but fixed layout + no shaping
    (eval is pure reward, novel geometries from held-out seeds)."""
    ec = {
        "stage": 1,
        "active_roles": ["gatherer"],
        "decision_core": True,
        "arena_mode": "clusters",
        "arena_half": ARENA_HALF,
        "randomize_layout": False,
        "distance_shaping": False,
        "max_episode_ticks": EVAL_MAX_TICKS,
    }
    ec.update(ABLATIONS[ablation])
    return ec


def _greedy_eval(module, ablation: str, seeds: list[int]) -> dict:
    """Greedy clearance on held-out seeds via the production run_instrumented loop."""
    from transfer_eval import run_instrumented

    from aiutopia.train.scenario_runner import (
        Scenario,
        _gatherer_collected_64_oak_log,
    )
    from aiutopia.train.sim_env_factory import make_aiutopia_sim_env

    ec = _eval_env_config(ablation)
    per = []
    cleared = 0
    for s in seeds:
        scn = Scenario(
            name=f"heldout_{s}", seed=s, max_ticks=EVAL_MAX_TICKS,
            success=_gatherer_collected_64_oak_log,
        )
        r = run_instrumented(
            scn, env_factory=lambda cfg: make_aiutopia_sim_env(cfg),
            env_config=ec, rl_module=module, device="cpu", wall_budget_s=120,
        )
        hist = Counter(e["skill"] for e in r["trace"])
        last = r["trace"][-1] if r["trace"] else {}
        end = ("SUCCESS" if last.get("term") else
               ("TICK_LIMIT" if last.get("trunc") else "?"))
        ok = int(r["oak_log"] >= 64)
        cleared += ok
        per.append({
            "seed": s, "oak": r["oak_log"], "steps": r["steps_used"],
            "NAVIGATE": hist.get("NAVIGATE", 0), "HARVEST": hist.get("HARVEST", 0),
            "end": end, "cleared": bool(ok),
        })
    return {"cleared": cleared, "n": len(seeds), "per_seed": per}


# ───────────────────────── scripted-follower discriminator ─────────────────────────
def _act(skill: int, sp=None) -> dict:
    return {
        "skill_type": skill, "target_class": 0,
        "spatial_param": (sp if sp is not None else np.zeros(3, np.float32)),
        "scalar_param": np.array([1.0], np.float32),
        "comm_payload": np.zeros(128, np.float32), "should_broadcast": 0,
        "comm_target_mask": np.zeros(4, np.int8),
    }


# Fixed blind-explore heading used when NO cue is available (the honest "no
# information" baseline): the clusters arena places B in a randomized direction,
# so a single fixed heading only sometimes finds it — exactly the search problem.
_BLIND_DIR = np.array([0.0, 0.0, -1.0], np.float32)  # toward -z ("south")


def _scripted_follower(ablation: str, seeds: list[int]) -> dict:
    """Per-cell ZERO-LEARNING baseline. Reads OBS directly (no network):
      - HARVEST nearest visible trunk (obs g_nearest_resources[0] != 0),
      - else explore: toward the cue if THIS cell has it, else a fixed heading.
    The trained policy's clearance MINUS this == its learned contribution."""
    from aiutopia.env.reward import _inventory_from_obs
    from aiutopia.sim.sim_env import AiUtopiaSimEnv

    has_cue = ABLATIONS[ablation]["resource_bearing_cue"]
    ec = _eval_env_config(ablation)
    per = []
    cleared = 0
    for s in seeds:
        env = AiUtopiaSimEnv(ec)
        obs, _ = env.reset(seed=s)
        o = obs["gatherer_0"]
        nav = 0
        for _ in range(EVAL_MAX_TICKS):
            nr = np.asarray(o["g_nearest_resources"], np.float32)
            if np.any(nr[0] != 0.0):  # a trunk is visible -> mine nearest
                act = _act(1)
            else:  # blind: explore toward cue if present, else fixed heading
                if has_cue:
                    bearing = np.asarray(o["g_hostiles_nearby"], np.float32)[0]
                    sp = np.array([bearing[0], 0.0, bearing[1]], np.float32)
                else:
                    sp = _BLIND_DIR
                act = _act(0, sp)
                nav += 1
            obs, _r, term, trunc, _i = env.step({"gatherer_0": act})
            o = obs["gatherer_0"]
            if not env.agents or term.get("gatherer_0") or trunc.get("gatherer_0"):
                break
        oak = int(sum(c for n, c in _inventory_from_obs(o).items() if n == "oak_log"))
        ok = int(oak >= 64)
        cleared += ok
        per.append({"seed": s, "oak": oak, "NAVIGATE": nav, "cleared": bool(ok)})
    return {"cleared": cleared, "n": len(seeds), "per_seed": per}


# ───────────────────────── driver ─────────────────────────
def main() -> int:
    import ray

    _p("=" * 76)
    _p("ORACLE-ABLATION CHARACTERIZATION — decision-core gatherer (sim-only)")
    _p(f"  iters/run={ITERS}  train_seeds={TRAIN_SEEDS}  heldout={HELDOUT_SEEDS}")
    _p(f"  ablations={WHICH}")
    _p("=" * 76)

    # Quiet, single-node Ray (local sampling; num_env_runners=0).
    ray.init(num_cpus=8, num_gpus=1, include_dashboard=False, logging_level="ERROR",
             ignore_reinit_error=True)
    # Concrete module must be importable for get_module / from_checkpoint.
    import aiutopia.rl_module.role_rl_module  # noqa: F401

    results: dict = {"meta": {"iters": ITERS, "train_seeds": TRAIN_SEEDS,
                              "heldout_seeds": HELDOUT_SEEDS}, "cells": {}}
    for ablation in WHICH:
        results["cells"][ablation] = {}
        # Scripted-follower baseline is policy-independent -> compute once per cell.
        foll = _scripted_follower(ablation, HELDOUT_SEEDS)
        _p(f"\n## {ablation}  (cue={ABLATIONS[ablation]['resource_bearing_cue']}, "
           f"mask_off={ABLATIONS[ablation]['harvest_mask_off']})")
        _p(f"   scripted follower (zero-learning baseline): "
           f"{foll['cleared']}/{foll['n']} cleared")
        results["cells"][ablation]["follower"] = foll
        results["cells"][ablation]["policy"] = {}
        for seed in TRAIN_SEEDS:
            _p(f"   -- training seed {seed} ({ITERS} iters) --")
            module, curve, wall = _train_one(ablation, seed, ITERS)
            ev = _greedy_eval(module, ablation, HELDOUT_SEEDS)
            ev["train_wall_s"] = round(wall, 1)
            ev["final_ret_mean"] = curve[-1] if curve else None
            ev["ret_curve_tail"] = curve[-5:]
            results["cells"][ablation]["policy"][seed] = ev
            _p(f"      policy seed{seed}: greedy held-out {ev['cleared']}/{ev['n']} "
               f"cleared  final_ret={ev['final_ret_mean']}  ({wall:.0f}s)")
            # incremental save (overnight-safe: partial results survive a crash)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=2))

    # ── summary table ──
    _p("\n" + "=" * 76)
    _p("SUMMARY  (policy greedy held-out vs scripted-follower baseline)")
    _p("=" * 76)
    _p(f"  {'ablation':<10} {'cue':<4} {'mask':<5} {'follower':<10} {'policy(mean/seed)':<20} gap")
    for ablation in WHICH:
        cell = results["cells"][ablation]
        fol = cell["follower"]["cleared"]
        n = cell["follower"]["n"]
        pol_vals = [cell["policy"][s]["cleared"] for s in TRAIN_SEEDS if s in cell["policy"]]
        pol_mean = sum(pol_vals) / len(pol_vals) if pol_vals else 0.0
        gap = pol_mean - fol
        cue = "ON" if ABLATIONS[ablation]["resource_bearing_cue"] else "off"
        mask = "off" if ABLATIONS[ablation]["harvest_mask_off"] else "ON"
        _p(f"  {ablation:<10} {cue:<4} {mask:<5} {fol}/{n:<8} "
           f"{pol_mean:.1f}/{n} {pol_vals!s:<12} gap={gap:+.1f}")
    _p(f"\n  wrote {OUT}")
    _p("  READ: gap≈0 => policy learned ~nothing beyond reactive obs-reading;")
    _p("        gap>0 (esp. 'neither') => genuine learned search/selection.")
    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
