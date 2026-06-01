"""Tests for the NAVIGATE-then-HARVEST scripted demonstrator (BC oracle).

The demonstrator must, on a HARVEST-MASKED spawn, emit a NAVIGATE action toward
the nearest visible trunk (using ONLY the obs bearing the net sees), and HARVEST
once the trunk is in reach. Cloning a HARVEST-spam demonstrator would teach the
same degeneracy that broke seed_1, so these tests pin the navigate-then-harvest
behavior the clone must inherit.
"""

from __future__ import annotations

import numpy as np

from aiutopia.sim.bc_demonstrator import demonstrate
from aiutopia.sim.skills import SKILL_HARVEST, SKILL_NAVIGATE
from aiutopia.sim.vec_sim import VecGathererSim


def test_unmasked_spawn_emits_harvest() -> None:
    """When HARVEST is unmasked the demonstrator presses HARVEST (class 0, scalar 1)."""
    sim = VecGathererSim(num_envs=4, max_episode_ticks=300)
    obs = sim.reset(np.array([2, 3, 2, 3], dtype=np.int64))  # seeds 2/3 spawn in-reach
    unmasked = obs["action_mask"]["skill_type"][:, SKILL_HARVEST] == 1
    assert unmasked.any(), "expected at least one unmasked spawn on seeds 2/3"
    act = demonstrate(obs)
    sk = np.asarray(act["skill_type"])
    assert (sk[unmasked] == SKILL_HARVEST).all()
    assert (np.asarray(act["target_class"])[unmasked] == 0).all()
    assert np.allclose(np.asarray(act["scalar_param"])[unmasked].reshape(-1, 1)[:, 0], 1.0)


def test_masked_spawn_emits_navigate_toward_trunk() -> None:
    """Forced-masked spawn -> NAVIGATE, horizontal step toward nearest trunk."""
    sim = VecGathererSim(
        num_envs=8,
        max_episode_ticks=300,
        force_masked_spawn=True,
        randomize_layout=True,
    )
    obs = sim.reset(np.arange(1, 9, dtype=np.int64))
    masked = obs["action_mask"]["skill_type"][:, SKILL_HARVEST] == 0
    assert masked.any(), "force_masked_spawn should produce masked starts"
    act = demonstrate(obs)
    sk = np.asarray(act["skill_type"])
    assert (sk[masked] == SKILL_NAVIGATE).all()
    sp = np.asarray(act["spatial_param"]).reshape(-1, 3)
    # vertical nav component must be zero (do not chase the topmost log up)
    assert np.allclose(sp[masked][:, 1], 0.0)
    # horizontal component points toward the bearing of nearest trunk (g_nearest[0])
    bearing = np.asarray(obs["g_nearest_resources"])[:, 0, :]  # dx/16, dy/8, dz/16
    for i in np.nonzero(masked)[0]:
        if abs(bearing[i, 0]) > 1e-6:
            assert np.sign(sp[i, 0]) == np.sign(bearing[i, 0])
        if abs(bearing[i, 2]) > 1e-6:
            assert np.sign(sp[i, 2]) == np.sign(bearing[i, 2])


def test_demonstrator_solves_masked_spawns() -> None:
    """Oracle: greedy on forced-masked spawns terminates episodes at the 64-oak goal."""
    B = 64
    sim = VecGathererSim(
        num_envs=B,
        max_episode_ticks=300,
        force_masked_spawn=True,
        randomize_layout=True,
    )
    obs = sim.reset(np.arange(1, B + 1, dtype=np.int64))
    successes = 0
    for _ in range(40):
        act = demonstrate(obs)
        obs, _rew, term, _trunc = sim.step(act)
        successes += int(np.asarray(term, dtype=bool).sum())
    # auto-reset lets each env finish multiple episodes; demand the oracle actually
    # reaches the goal at a healthy rate (>= B terminations over 40 macro-steps).
    assert successes >= B, f"oracle terminated only {successes} episodes over {B} envs"


def test_no_trunk_visible_emits_nonzero_explore_move() -> None:
    """No trunk perceived (validity flag 0) -> masked NAVIGATE is a non-zero explore move."""
    # Hand-craft a 1-env obs: HARVEST masked, and g_nearest_resources all-zero
    # (no visible trunk -> validity flag [..,5]==0).
    obs = {
        "action_mask": {"skill_type": np.array([[1, 0, 1, 1, 1, 1]], dtype=np.int8)},
        "g_nearest_resources": np.zeros((1, 8, 6), dtype=np.float32),
    }
    act = demonstrate(obs)
    assert int(np.asarray(act["skill_type"])[0]) == SKILL_NAVIGATE
    sp = np.asarray(act["spatial_param"]).reshape(1, 3)
    assert float(np.abs(sp[0]).sum()) > 0.0, "no-trunk NAVIGATE must move, not stall"
