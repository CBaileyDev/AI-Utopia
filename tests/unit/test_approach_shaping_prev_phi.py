"""Regression: approach_shaping must survive co-enabled distance_shaping.

Bug (sim_env.py): both the distance_shaping (blind-only) and approach_shaping
branches wrote the SAME `self._prev_phi[agent]`. The distance branch's write runs
unconditionally BEFORE the approach branch reads it in the same step, so with both
flags on the approach term computes `phi_a - phi == 0` every step — silently zeroing
approach shaping. Fix: a separate `_prev_phi_approach` store.

This test drives a deterministic NAVIGATE-toward-nearest-log rollout from a
HARVEST-masked (but trunk-visible) spawn and asserts the both-flags shaped return is
NOT less than the approach-only shaped return (i.e. the approach signal is preserved).
"""

import numpy as np
import pytest

from aiutopia.sim.sim_env import AiUtopiaSimEnv
from aiutopia.sim.skills import MAX_NAV_RANGE, NAV_VERT_RANGE


def _nav_action_toward(world):
    """Gatherer NAVIGATE action pointing spatial_param at the nearest alive log."""
    alive = np.asarray(world.log_alive, bool)
    logs = np.asarray(world.logs, float)[alive]
    d = logs - world.agent_pos
    j = int(np.argmin((d * d).sum(axis=1)))
    delta = d[j]
    horiz = float(np.hypot(delta[0], delta[2])) or 1.0
    unit = delta * (1.5 / horiz)  # ~1.5b horizontal step keeps agent masked while approaching
    sp = np.clip(unit / np.array([MAX_NAV_RANGE, NAV_VERT_RANGE, MAX_NAV_RANGE]), -1, 1)
    return {
        "skill_type": np.int64(0),  # NAVIGATE
        "target_class": np.int64(0),
        "spatial_param": sp.astype(np.float32),
        "scalar_param": np.array([0.0], np.float32),
        "comm_payload": np.zeros(256, np.float32),
        "should_broadcast": np.int64(0),
        "comm_target_mask": np.zeros(4, np.int8),
    }


def _masked_seed(both: bool):
    """Find a fixed seed whose jittered spawn starts HARVEST-masked, return env+seed."""
    cfg = {
        "stage": 1,
        "active_roles": ["gatherer"],
        "max_episode_ticks": 400,
        "backend": "sim",
        "randomize_layout": True,
        "spawn_jitter": 9.0,
        "approach_shaping": True,
        "distance_shaping": both,
    }
    env = AiUtopiaSimEnv(cfg)
    for _ in range(60):
        obs, _ = env.reset()
        if int(np.asarray(obs["gatherer_0"]["action_mask"]["skill_type"]).reshape(-1)[1]) == 0:
            return env
    env.close()
    pytest.skip("no HARVEST-masked spawn found in 60 resets")


def _shaped_return(both: bool) -> float:
    env = _masked_seed(both)
    total = 0.0
    for _ in range(20):
        act = _nav_action_toward(env.worlds["gatherer_0"])
        _o, rew, term, trunc, _i = env.step({"gatherer_0": act})
        total += float(rew["gatherer_0"])
        if term.get("gatherer_0") or trunc.get("gatherer_0"):
            break
    env.close()
    return total


def test_approach_shaping_not_zeroed_by_distance_shaping() -> None:
    approach_only = _shaped_return(both=False)
    both_flags = _shaped_return(both=True)
    # Co-enabling distance_shaping must not DESTROY the approach signal. Before the
    # fix the clobber zeroed it, making both_flags strictly smaller.
    assert both_flags >= approach_only - 1e-6, (
        f"approach shaping zeroed by distance_shaping clobber: "
        f"both={both_flags:.4f} < approach_only={approach_only:.4f}"
    )
