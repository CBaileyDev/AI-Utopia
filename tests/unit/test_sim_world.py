import numpy as np

from aiutopia.sim.world import SimWorld


def test_reset_places_64_logs_flat_at_y66_in_bounds():
    w = SimWorld()
    w.reset(seed=1)
    assert w.logs.shape == (64, 3)  # (x, y, z) per log
    assert np.all(w.logs[:, 1] == 66)  # all flat at y=66
    xs, zs = w.logs[:, 0], w.logs[:, 2]
    assert np.all((xs >= 48) & (xs <= 80))  # arena x-bounds
    assert np.all((zs >= -64) & (zs <= -32))  # arena z-bounds
    assert len({(int(x), int(z)) for x, z in zip(xs, zs, strict=False)}) == 64  # all distinct (x,z)
    assert not np.any((xs == 64) & (zs == -48))  # none on spawn tile


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
