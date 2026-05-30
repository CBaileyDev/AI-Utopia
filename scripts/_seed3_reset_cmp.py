"""THROWAWAY (investigation-only). Reset-only fidelity check: compare the REAL
seed-3 arena (port 25001, fresh reset, NO harvest) against the sim seed-3 arena
(SimWorld). Confirms whether the remaining-log dy=+3 / 'no oak_log within 16'
stall is an ARENA-FIDELITY mismatch or a pure harvest-displacement artifact.

Dumps: real g_resource_grid nonzero cells (decoded to dx,dz offsets + channel),
real g_nearest_resources top-8 decoded, vs sim's. If reset grids match, the
arena is faithful and the stall is a walk/displacement effect; if they differ
(esp. dy), the real arena has vertical structure the flat-sim lacks.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def decode_grid(grid):
    """g_resource_grid (32,32,6): cell [dx+16][dz+16][channel]. Return list of
    (dx,dz,channel) nonzero. Note grid has NO y info (flattened to a plane)."""
    g = np.asarray(grid).reshape(32, 32, 6)
    out = []
    for ix in range(32):
        for iz in range(32):
            for c in range(6):
                if g[ix, iz, c] > 0:
                    out.append((ix - 16, iz - 16, c))
    return out


def decode_near(gnr):
    g = np.asarray(gnr).reshape(8, 6)
    rows = []
    for r in g:
        if np.count_nonzero(r) == 0:
            continue
        rows.append((round(float(r[0]) * 16, 1), round(float(r[1]) * 8, 1),
                     round(float(r[2]) * 16, 1), [round(float(x), 3) for x in r]))
    return rows


def main():
    from aiutopia.sim.world import SimWorld
    from aiutopia.sim.obs_adapter import build_gatherer_obs

    # ── SIM seed-3 reset obs ──
    w = SimWorld(); w.reset(3)
    sim_obs = build_gatherer_obs(w)
    sim_grid = decode_grid(sim_obs["g_resource_grid"])
    sim_near = decode_near(sim_obs["g_nearest_resources"])
    print("=== SIM seed-3 reset ===")
    print(f"  g_resource_grid nonzero cells: {len(sim_grid)} (expect ~60 in window)")
    print(f"  nearest top-8 (dx,dy,dz, rawrow):")
    for r in sim_near[:8]:
        print(f"    {r}")
    sim_dys = sorted({c for _, _, c in []})  # grid has no y; print channels
    chans = sorted({c for _, _, c in sim_grid})
    print(f"  grid channels present: {chans}")

    # ── REAL seed-3 reset obs ──
    from aiutopia.env.wrapper import AiUtopiaPettingZooEnv
    cfg = {"stage": 1, "active_roles": ["gatherer"], "seed_strategy": "fixed_easy",
           "py4j_ports": [25001], "tick_warp": True, "max_episode_ticks": 1000,
           "per_worker_seed_offset": False, "enable_memory_writes": False,
           "aiutopia_root_per_worker": False}
    env = AiUtopiaPettingZooEnv(cfg)
    try:
        obs, _ = env.reset(seed=3)
        a = obs["gatherer_0"]
        real_grid = decode_grid(a["g_resource_grid"])
        real_near = decode_near(a["g_nearest_resources"])
        print("\n=== REAL seed-3 reset (port 25001) ===")
        print(f"  pos={np.asarray(a['position']).tolist()}")
        print(f"  g_resource_grid nonzero cells: {len(real_grid)}")
        print(f"  nearest top-8 (dx,dy,dz, rawrow):")
        for r in real_near[:8]:
            print(f"    {r}")
        rchans = sorted({c for _, _, c in real_grid})
        print(f"  grid channels present: {rchans}")

        # ── Compare ──
        sset = {(dx, dz) for dx, dz, c in sim_grid}
        rset = {(dx, dz) for dx, dz, c in real_grid}
        print("\n=== COMPARE (grid is x-z plane; no y) ===")
        print(f"  sim cells={len(sset)} real cells={len(rset)} "
              f"intersection={len(sset & rset)}")
        print(f"  in SIM only (dx,dz): {sorted(sset - rset)[:20]}")
        print(f"  in REAL only (dx,dz): {sorted(rset - sset)[:20]}")
        # Critical: do real nearest rows carry dy != +1 (sim's flat value)?
        real_dys = sorted({r[1] for r in real_near})
        sim_dys2 = sorted({r[1] for r in sim_near})
        print(f"  SIM nearest dy values: {sim_dys2}")
        print(f"  REAL nearest dy values: {real_dys}  "
              f"(sim arena is flat y=66 -> dy should be +1.0 only)")
    finally:
        env.close()
        print("[cmp] env closed (JVM left running)")


if __name__ == "__main__":
    main()
