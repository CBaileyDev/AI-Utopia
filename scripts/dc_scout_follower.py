"""DECISIVE Fork-A test (no training, no RLModule surgery).

Fork A = thin reactive controller + a good bearing PRODUCER. The scripted follower
already clears these arenas given a GOOD bearing (oracle = 5/5). The open question
the whole fork rides on: does a REAL partial-information scout (sim/scout.py
FrontierScout — frontier exploration over only what's been perceived, never ground
truth) produce good-enough bearings for the same follower to clear held-out
geometries?

Runs the identical scripted follower under three bearing sources, on N held-out
novel cluster geometries:
  - oracle      : g_hostiles[0] = ground-truth direction to nearest log (KNOWN 5/5 ceiling)
  - real-scout  : g_hostiles[0] = FrontierScout bearing from perceived state only (THE TEST)
  - fixed       : ignore the cue, always head -z (the no-information floor)

Reading: real-scout ~= oracle  -> Fork A validated end-to-end, FOR FREE (no training).
         real-scout ~= fixed    -> the scout's bearings are the hard part (the real open
                                    question); PPO stabilization would not have helped.

Run: PYTHONPATH=src AIUTOPIA_ROOT=... AIUTOPIA_DATA_DIR=... \
       DC_SEEDS=90001,...,90010  py -3.11 scripts/dc_scout_follower.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

from aiutopia.env.reward import _inventory_from_obs
from aiutopia.sim.sim_env import AiUtopiaSimEnv

SEEDS = [
    int(s)
    for s in os.environ.get(
        "DC_SEEDS", "90001,90002,90003,90004,90005,90006,90007,90008,90009,90010"
    ).split(",")
]
MAX_TICKS = int(os.environ.get("DC_MAX_TICKS", "200"))
ARENA = os.environ.get("DC_ARENA", "clusters")  # "clusters" (degenerate) | "clusters_omni"
ARENA_HALF = float(os.environ.get("DC_HALF", "40.0" if "omni" in ARENA else "34.0"))
_FIXED_DIR = np.array([0.0, 0.0, -1.0], np.float32)


def _p(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _act(skill: int, sp=None) -> dict:
    return {
        "skill_type": skill,
        "target_class": 0,
        "spatial_param": (sp if sp is not None else np.zeros(3, np.float32)),
        "scalar_param": np.array([1.0], np.float32),
        "comm_payload": np.zeros(128, np.float32),
        "should_broadcast": 0,
        "comm_target_mask": np.zeros(4, np.int8),
    }


def _env_config(mode: str) -> dict:
    # mode in {"oracle","real","sweep","fixed"}; "fixed" uses scout off + constant heading.
    scout_mode = {
        "oracle": "oracle", "real": "real", "sweep": "sweep", "fixed": "off"
    }[mode]
    ec = {
        "stage": 1,
        "active_roles": ["gatherer"],
        "decision_core": True,
        "arena_mode": ARENA,
        "arena_half": ARENA_HALF,
        "randomize_layout": False,
        "distance_shaping": False,
        "max_episode_ticks": MAX_TICKS,
        "scout_mode": scout_mode,
    }
    if mode == "oracle":
        ec["resource_bearing_cue"] = True  # ground-truth oracle into g_hostiles[0]
    return ec


def _run_one(mode: str, seed: int) -> dict:
    env = AiUtopiaSimEnv(_env_config(mode))
    obs, _ = env.reset(seed=seed)
    o = obs["gatherer_0"]
    nav = 0
    for _ in range(MAX_TICKS):
        nr = np.asarray(o["g_nearest_resources"], np.float32)
        mask_harvest = int(np.asarray(o["action_mask"]["skill_type"])[1])
        if mask_harvest and np.any(nr[0] != 0.0):
            act = _act(1)  # HARVEST nearest visible
        else:
            if mode == "fixed":
                sp = _FIXED_DIR
            else:
                bearing = np.asarray(o["g_hostiles_nearby"], np.float32)[0]
                sp = np.array([bearing[0], 0.0, bearing[1]], np.float32)
                # a zero bearing (scout fully explored / none) falls back to fixed
                if not np.any(sp):
                    sp = _FIXED_DIR
            act = _act(0, sp)
            nav += 1
        obs, _r, term, trunc, _i = env.step({"gatherer_0": act})
        o = obs["gatherer_0"]
        if not env.agents or term.get("gatherer_0") or trunc.get("gatherer_0"):
            break
    oak = int(sum(c for n, c in _inventory_from_obs(o).items() if n == "oak_log"))
    return {"seed": seed, "oak": oak, "nav": nav, "cleared": oak >= 64}


def main() -> int:
    _p("=" * 74)
    _p("DECISIVE Fork-A test — scripted follower under 3 bearing sources")
    _p(f"  held-out seeds={SEEDS}  max_ticks={MAX_TICKS}")
    _p("=" * 74)
    summary = {}
    modes = [m.strip() for m in os.environ.get("DC_MODES", "oracle,real,sweep,fixed").split(",")]
    for mode in modes:
        rows = [_run_one(mode, s) for s in SEEDS]
        cleared = sum(r["cleared"] for r in rows)
        summary[mode] = (cleared, len(rows))
        _p(f"\n## {mode}")
        for r in rows:
            _p(
                f"   seed={r['seed']}  oak={r['oak']:>2}/64  NAV={r['nav']:>3}  "
                f"{'CLEARED' if r['cleared'] else 'stuck'}"
            )
        _p(f"   -> {cleared}/{len(rows)} cleared")
    _p("\n" + "=" * 74)
    _p(f"SUMMARY (follower clearance by bearing source, arena={ARENA})")
    for mode in modes:
        c, n = summary[mode]
        _p(f"   {mode:<7} {c}/{n}")
    o = summary.get("oracle", (None, 0))[0]
    f = summary.get("fixed", (None, 0))[0]
    _p("")
    # Report each scout (real/sweep) relative to the fixed-heading floor + oracle ceiling.
    for mode in modes:
        if mode in ("oracle", "fixed"):
            continue
        s, n = summary[mode]
        verdict = (
            "~= oracle: PRODUCER VALIDATED" if (o is not None and s >= o - 1)
            else "~= fixed-floor: no better than no-info" if (f is not None and s <= f + 1)
            else "PARTIAL: between floor and oracle"
        )
        _p(f">>> {mode}-scout {s}/{n}  ({verdict}; floor={f}, ceiling={o})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
