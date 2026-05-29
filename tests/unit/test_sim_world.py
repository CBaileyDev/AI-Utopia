import numpy as np

from aiutopia.sim.world import LOG_Y, TREES, TRUNK_H, SimWorld


def test_reset_places_16_trunks_of_height_4_totaling_64_logs():
    # N21 Inc2: the arena is now 16 vertical bare oak trunks (4 logs each,
    # Y=66..69), not a flat 8x8 single-log grid. Total stays 64 (gate unchanged).
    assert TREES * TRUNK_H == 64
    w = SimWorld()
    w.reset(seed=1)
    assert w.logs.shape == (64, 3)  # (x, y, z) per log
    # Group logs by (x, z) column -> exactly TREES trunks, each a full stack.
    cols: dict[tuple[int, int], list[int]] = {}
    for x, y, z in w.logs.tolist():
        cols.setdefault((int(x), int(z)), []).append(int(y))
    assert len(cols) == TREES, f"expected {TREES} trunks, got {len(cols)}"
    for (x, z), ys in cols.items():
        assert sorted(ys) == [LOG_Y + d for d in range(TRUNK_H)]  # 66,67,68,69


def test_reset_logs_in_bounds_and_off_spawn():
    w = SimWorld()
    w.reset(seed=1)
    xs, zs = w.logs[:, 0], w.logs[:, 2]
    assert np.all((xs >= 48) & (xs <= 80))  # arena x-bounds
    assert np.all((zs >= -64) & (zs <= -32))  # arena z-bounds
    # trunk tops (Y=69) are within the ground-standing reach (REACH=4.5)
    assert np.all(w.logs[:, 1] <= LOG_Y + TRUNK_H - 1)
    assert not np.any((xs == 64) & (zs == -48))  # no trunk on the spawn tile


def test_reset_agent_at_spawn_and_empty_inventory():
    w = SimWorld()
    w.reset(seed=1)
    assert np.allclose(w.agent_pos, [64.5, 65.0, -47.5])  # matches real spawn obs
    assert w.inventory.get("oak_log", 0) == 0


def test_reset_is_seed_deterministic_and_seed_varies_layout():
    a = SimWorld()
    a.reset(seed=1)
    b = SimWorld()
    b.reset(seed=1)
    c = SimWorld()
    c.reset(seed=2)
    assert np.array_equal(a.logs, b.logs)  # same seed -> same layout
    assert not np.array_equal(a.logs, c.logs)  # different seed -> different
