"""Lock the Fork-A producer validation (12h autonomous run, 2026-05-30):
the committed out-and-back SweepScout + the non-degenerate clusters_omni arena.

Findings these guard:
  - clusters_omni places cluster B in UNIFORM directions (not always south like the
    degenerate `clusters`), so a fixed heading cannot trivially clear it.
  - SweepScout alternates OUT (dirs[k], a 32-block hop) and BACK (toward spawn),
    rotating direction each cycle — matched to the NAVIGATE fixed-32-block dynamics.
  - The scout is partial-info: it holds no world reference (structurally cannot read
    ground truth), same invariant as FrontierScout.
"""

import numpy as np

from aiutopia.sim.scout import SweepScout
from aiutopia.sim.world import SimWorld


def test_sweep_is_out_and_back_and_rotates():
    s = SweepScout(n_dirs=12)
    b0 = s.bearing(64, -48)  # OUT dir0 (east), from spawn
    assert b0 is not None
    assert abs(b0[0] - 1.0) < 1e-9 and abs(b0[1]) < 1e-9  # +x
    assert b0[2] == 32.0  # fixed hop magnitude
    # now the agent is out at ~ (96,-48); next call is BACK toward spawn
    b1 = s.bearing(96, -48)
    assert b1 is not None
    assert b1[0] < -0.9  # heading back -x toward spawn
    # after the back phase, the direction index has rotated to dir1
    b2 = s.bearing(64, -48)  # OUT dir1 (30 deg)
    assert b2 is not None
    assert b2[0] > 0 and b2[1] > 0  # NE-ish, not east again


def test_sweep_holds_no_world_reference():
    s = SweepScout()
    # only bookkeeping state; no SimWorld / logs / log_alive anywhere.
    assert set(vars(s)) == {"_dirs", "_idx", "_phase_out", "_spawn", "observed", "resource_seen"}


def test_clusters_omni_b_direction_varies():
    """The degenerate `clusters` put B always south (dz<0); omni must span both
    north and south across seeds (the whole point of the non-degenerate arena)."""
    dzs = []
    for seed in range(90001, 90021):
        w = SimWorld()
        w.reset(seed, arena_mode="clusters_omni")
        # farthest trunk from spawn ~ cluster B
        d2 = (w.logs[:, 0] - 64) ** 2 + (w.logs[:, 2] + 48) ** 2
        far = w.logs[int(np.argmax(d2))]
        dzs.append(int(far[2]) + 48)
    assert any(dz > 8 for dz in dzs), "no northward B — arena still south-biased"
    assert any(dz < -8 for dz in dzs), "no southward B"


def test_clusters_omni_reproducible_per_seed():
    a = SimWorld(); a.reset(90001, arena_mode="clusters_omni")
    b = SimWorld(); b.reset(90001, arena_mode="clusters_omni")
    assert np.array_equal(a.logs, b.logs)  # seeded => deterministic
