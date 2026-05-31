"""Vectorized (batched-over-envs) obs primitives for the fast trainer.

The scalar ``gatherer_nearest_columns`` (obs_adapter.py) is the sim's hot path
(~65% of step time): a per-(dx,dz)-column TOPMOST-log scan within a +/-3 dy window,
sorted by (distSq, dx, dz), byte-faithful to the Java GathererOverlayBuilder. This
module reproduces it for B envs at once with pure numpy so a batched sim can build
all observations in one vectorized pass. Verified equal per-env to the scalar
function in tests/unit/test_vec_obs_parity.py.
"""
from __future__ import annotations

import numpy as np

from aiutopia.sim.obs_adapter import GRID_RADIUS, SCAN_RADIUS, SENTINEL_NO_TARGET

_G = GRID_RADIUS          # 16
_W = 2 * GRID_RADIUS      # 32
_NCELL = _W * _W          # 1024
_EMPTY_DY = -10_000
_BIG_KEY = np.int64(1 << 60)


def gatherer_nearest_columns_batched(logs, log_alive, agent_pos):
    """Batched topmost-per-column scan. Returns (grid, nearest8, nearest_dist, richness)."""
    B, n, _ = logs.shape
    b = np.floor(agent_pos).astype(np.int64)
    d = logs - b[:, None, :]
    dx = d[..., 0]; dy = d[..., 1]; dz = d[..., 2]

    valid = (
        log_alive
        & (dx >= -_G) & (dx < _G)
        & (dz >= -_G) & (dz < _G)
        & (dy >= -3) & (dy <= 3)
    )

    env_idx = np.broadcast_to(np.arange(B)[:, None], (B, n))
    cell = (dx + _G) * _W + (dz + _G)
    flat = (env_idx * _NCELL + cell).astype(np.int64)
    col_top = np.full(B * _NCELL, _EMPTY_DY, dtype=np.int64)
    vmask = valid.reshape(-1)
    np.maximum.at(col_top, flat.reshape(-1)[vmask], dy.reshape(-1)[vmask].astype(np.int64))
    col_top = col_top.reshape(B, _W, _W)

    occupied = col_top > _EMPTY_DY
    grid = occupied.astype(np.float32)

    cx = np.arange(_W)[None, :, None]
    cz = np.arange(_W)[None, None, :]
    cdx = (cx - _G).astype(np.int64)
    cdz = (cz - _G).astype(np.int64)
    cdy = np.where(occupied, col_top, 0)
    distsq = (cdx * cdx + cdy * cdy + cdz * cdz).astype(np.int64)

    in_scan = occupied & (distsq <= int(SCAN_RADIUS * SCAN_RADIUS))

    big = np.where(in_scan, distsq, _BIG_KEY)
    min_distsq = big.reshape(B, -1).min(axis=1)
    has_any = in_scan.reshape(B, -1).any(axis=1)
    nearest_dist = np.where(
        has_any, np.sqrt(min_distsq.astype(np.float64)), SENTINEL_NO_TARGET
    ).astype(np.float32)
    richness = np.minimum(1.0, in_scan.reshape(B, -1).sum(axis=1) / 64.0).astype(np.float32)

    cellkey = (
        np.broadcast_to(cx, (1, _W, _W)) * _W + np.broadcast_to(cz, (1, _W, _W))
    ).astype(np.int64)
    key = np.where(in_scan, distsq * _NCELL + cellkey, _BIG_KEY).reshape(B, -1)
    order = np.argsort(key, axis=1, kind="stable")[:, :8]

    cdx_flat = np.broadcast_to(cdx, (B, _W, _W)).reshape(B, -1)
    cdz_flat = np.broadcast_to(cdz, (B, _W, _W)).reshape(B, -1)
    cdy_flat = cdy.reshape(B, -1)
    sel_dx = np.take_along_axis(cdx_flat, order, axis=1)
    sel_dy = np.take_along_axis(cdy_flat, order, axis=1)
    sel_dz = np.take_along_axis(cdz_flat, order, axis=1)
    sel_valid = np.take_along_axis(key, order, axis=1) < _BIG_KEY

    nearest8 = np.zeros((B, 8, 6), dtype=np.float32)
    nearest8[..., 0] = sel_dx / SCAN_RADIUS
    nearest8[..., 1] = sel_dy / 8.0
    nearest8[..., 2] = sel_dz / SCAN_RADIUS
    nearest8[..., 4] = 1.0
    nearest8[..., 5] = 1.0
    nearest8 *= sel_valid[..., None]

    return grid, nearest8, nearest_dist, richness
