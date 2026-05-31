"""Parity: batched gatherer_nearest_columns must match the scalar one per-env.

The scalar gatherer_nearest_columns (obs_adapter.py) is 65% of sim step time and is
byte-faithful to the Java GathererOverlayBuilder. The vectorized fast-trainer must
reproduce it EXACTLY (grid occupancy, nearest_res_dist, richness, top-8 nearest rows)
or sim->real parity breaks. This test builds B envs from real SimWorld seeds and asserts
the batched outputs equal the scalar outputs env-by-env.
"""

import numpy as np
from aiutopia.sim.vec_obs import gatherer_nearest_columns_batched

from aiutopia.sim.obs_adapter import GRID_RADIUS, SCAN_RADIUS, gatherer_nearest_columns
from aiutopia.sim.world import SimWorld


def _scalar_grid_and_nearest(world):
    """Derive (grid32x32 occupancy, nearest8, nearest_dist, richness) from the scalar fn."""
    import math

    col_top, nearby = gatherer_nearest_columns(world)
    grid = np.zeros((2 * GRID_RADIUS, 2 * GRID_RADIUS), dtype=np.float32)
    for (dx, dz), _dy in col_top.items():
        grid[dx + GRID_RADIUS][dz + GRID_RADIUS] = 1.0
    nearest8 = np.zeros((8, 6), dtype=np.float32)
    for k in range(min(8, len(nearby))):
        _x, _z, dx, dy, dz, _ = nearby[k]
        nearest8[k] = [dx / SCAN_RADIUS, dy / 8.0, dz / SCAN_RADIUS, 0.0, 1.0, 1.0]
    nearest_dist = math.sqrt(nearby[0][5]) if nearby else 999.0
    richness = min(1.0, len(nearby) / 64.0)
    return grid, nearest8, np.float32(nearest_dist), np.float32(richness)


def test_batched_nearest_columns_matches_scalar() -> None:
    seeds = [1, 2, 3, 7]
    worlds = []
    for s in seeds:
        w = SimWorld(s)
        w.reset(s, arena_mode="trees")
        worlds.append(w)

    logs = np.stack([w.logs for w in worlds]).astype(np.int64)  # (B, n, 3)
    alive = np.stack([w.log_alive for w in worlds]).astype(bool)  # (B, n)
    agent = np.stack([w.agent_pos for w in worlds]).astype(np.float64)  # (B, 3)

    grid_b, near_b, dist_b, rich_b = gatherer_nearest_columns_batched(logs, alive, agent)

    for i, w in enumerate(worlds):
        g, n8, d, r = _scalar_grid_and_nearest(w)
        assert np.array_equal(grid_b[i], g), f"grid mismatch env {i}"
        assert np.allclose(near_b[i], n8), f"nearest8 mismatch env {i}"
        assert np.isclose(dist_b[i], d), f"nearest_dist mismatch env {i}: {dist_b[i]} vs {d}"
        assert np.isclose(rich_b[i], r), f"richness mismatch env {i}"
