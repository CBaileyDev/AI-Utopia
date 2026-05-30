"""Discriminator: does the decision-core POLICY actually learn, or does it just
follow the env-fed oracle (perception-mask = WHEN, ground-truth bearing cue =
WHICH WAY)? A scripted ZERO-LEARNING follower — HARVEST target_class=0 when a
trunk is visible, else NAVIGATE toward g_hostiles_nearby[0] (the cue) — run on the
held-out cluster seeds. If it ALSO clears 5/5, PPO learned ~nothing; the mask+cue
do the deciding."""
from __future__ import annotations

import sys

import numpy as np

from aiutopia.env.reward import _inventory_from_obs
from aiutopia.sim.sim_env import AiUtopiaSimEnv


def _p(m):
    print(m, file=sys.stderr, flush=True)


def _act(skill, sp=None):
    return {"skill_type": skill, "target_class": 0,
            "spatial_param": (sp if sp is not None else np.zeros(3, np.float32)),
            "scalar_param": np.array([1.0], np.float32),
            "comm_payload": np.zeros(128, np.float32), "should_broadcast": 0,
            "comm_target_mask": np.zeros(4, np.int8)}


def main():
    n_full = 0
    for s in (90001, 90002, 90003, 90004, 90005):
        env = AiUtopiaSimEnv({"active_roles": ["gatherer"], "decision_core": True,
                              "resource_bearing_cue": True, "arena_mode": "clusters",
                              "arena_half": 34.0, "max_episode_ticks": 200})
        obs, _ = env.reset(seed=s)
        o = obs["gatherer_0"]
        nav = 0
        for _ in range(200):
            nr = np.asarray(o["g_nearest_resources"], np.float32)
            mask_harvest = int(np.asarray(o["action_mask"]["skill_type"])[1])  # HARVEST bit
            if mask_harvest and np.any(nr[0] != 0.0):
                act = _act(1)  # HARVEST nearest
            else:
                bearing = np.asarray(o["g_hostiles_nearby"], np.float32)[0]  # [dx,dz,dist,valid]
                sp = np.array([bearing[0], 0.0, bearing[1]], np.float32)  # follow the cue
                act = _act(0, sp)
                nav += 1
            obs, _r, term, trunc, _i = env.step({"gatherer_0": act})
            o = obs["gatherer_0"]
            if not env.agents or term.get("gatherer_0") or trunc.get("gatherer_0"):
                break
        oak = int(sum(c for n, c in _inventory_from_obs(o).items() if n == "oak_log"))
        n_full += int(oak >= 64)
        _p(f"  seed={s}  oak={oak}/64  NAVIGATE={nav}  {'CLEARED ✓' if oak >= 64 else 'stuck'}")
    _p(f">>> scripted ZERO-LEARNING follower (mask+cue only): {n_full}/5 cleared <<<")


if __name__ == "__main__":
    main()
