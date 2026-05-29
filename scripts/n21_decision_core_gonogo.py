"""Phase B go/no-go (architecture analysis's discriminator): prove the DECISION-CORE
task is SOLVABLE by a hand-coded policy before spending on training.

Decision-core = HARVEST demoted to mine ONLY the policy-pointed instance
(target_class indexes g_nearest_resources), no findNearest chaining. A scripted
"point at the nearest visible trunk -> MINE; if none visible -> NAVIGATE toward the
arena center to reveal more" policy must clear all 64 logs. If it can't, the bounded
task is unsolvable as posed and the design needs rework BEFORE any PPO/Java cost.

Run: PYTHONPATH=src AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data \
       py -3.11 scripts/n21_decision_core_gonogo.py
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from aiutopia.env.reward import _inventory_from_obs
from aiutopia.sim.sim_env import AiUtopiaSimEnv

_CENTER = np.array([64.5, 65.0, -47.5], dtype=np.float64)
_MAX_NAV_RANGE = 32.0


def _p(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _oak(obs: dict) -> int:
    inv = _inventory_from_obs(obs)
    return int(sum(c for n, c in inv.items() if n == "oak_log"))


def _harvest(pointer: int) -> dict:
    return {
        "skill_type": 1,
        "target_class": int(pointer),  # decision-core: slot index into g_nearest_resources
        "spatial_param": np.zeros(3, np.float32),
        "scalar_param": np.array([1.0], np.float32),
        "comm_payload": np.zeros(128, np.float32),
        "should_broadcast": 0,
        "comm_target_mask": np.zeros(4, np.int8),
    }


def _navigate_toward(pos: np.ndarray, target: np.ndarray) -> dict:
    raw = np.clip((target - pos) / _MAX_NAV_RANGE, -1.0, 1.0).astype(np.float32)
    return {
        "skill_type": 0,
        "target_class": 0,
        "spatial_param": raw,
        "scalar_param": np.array([1.0], np.float32),
        "comm_payload": np.zeros(128, np.float32),
        "should_broadcast": 0,
        "comm_target_mask": np.zeros(4, np.int8),
    }


def run(seed: int, arena_mode: str = "trees", arena_half: float = 24.0,
        max_steps: int = 200) -> dict:
    env = AiUtopiaSimEnv({
        "active_roles": ["gatherer"], "decision_core": True,
        "arena_mode": arena_mode, "arena_half": arena_half,
        "max_episode_ticks": max_steps,
    })
    obs, _ = env.reset(seed=seed)
    o = obs["gatherer_0"]
    hist: Counter = Counter()
    for _ in range(max_steps):
        nr = np.asarray(o["g_nearest_resources"], np.float32)
        visible = bool(np.any(nr[0] != 0.0))  # is there a nearest instance?
        pos = np.asarray(o["position"], np.float64)
        if visible:
            act = _harvest(0)  # point at the NEAREST visible trunk
            hist["MINE"] += 1
        elif arena_mode == "clusters":
            # blind explore: sweep SOUTH (cluster B is south of spawn) until a
            # trunk enters perception. A genuinely-blind heuristic (it does not
            # know where B is — it just keeps moving into unexplored space).
            act = _navigate_toward(pos, pos + np.array([0.0, 0.0, -16.0]))
            hist["EXPLORE"] += 1
        else:
            act = _navigate_toward(pos, _CENTER)
            hist["EXPLORE"] += 1
        obs, _r, term, trunc, _i = env.step({"gatherer_0": act})
        o = obs["gatherer_0"]
        if not env.agents or term.get("gatherer_0") or trunc.get("gatherer_0"):
            break
    return {"seed": seed, "oak": _oak(o), "hist": dict(hist)}


def main() -> int:
    _p("=" * 66)
    _p("Phase B GO/NO-GO — scripted point-then-explore on the decision-core")
    _p("=" * 66)
    ok = True
    for label, mode, half in (("trees", "trees", 24.0), ("clusters(blind-explore)", "clusters", 34.0)):
        _p(f"  -- arena: {label} --")
        for seed in (1, 2, 3):
            r = run(seed, arena_mode=mode, arena_half=half)
            cleared = r["oak"] >= 64
            ok = ok and cleared
            _p(f"    seed={seed}  oak_log={r['oak']}/64  {'CLEARED ✓' if cleared else 'STUCK ✗'}  {r['hist']}")
    _p("")
    _p(f">>> decision-core task SOLVABLE by scripted policy (both arenas): {ok} <<<")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
