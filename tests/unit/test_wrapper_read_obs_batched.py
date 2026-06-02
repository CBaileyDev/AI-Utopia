"""Tests for env/wrapper helpers + the batched-observation invariant.

The batched-obs invariant (spec §4.6 / bridge.py): observationsAll() returns every
agent in ONE Py4J round-trip and must be called AT MOST ONCE per tick. The fallback
path in _read_all_obs previously called it INSIDE the per-agent loop, so on
gatherer-only Java it fired N times/tick (the forbidden per-agent-roundtrip pattern).
"""

import numpy as np
import pytest

# wrapper imports the heavy dep chain (chroma, pettingzoo, planner); skip cleanly if
# any is unavailable in the runner rather than erroring at collection.
wrapper = pytest.importorskip("aiutopia.env.wrapper")


def test_as_int_scalar_handles_scalar_array_and_bad_values():
    f = wrapper._as_int_scalar
    assert f(3) == 3
    assert f(np.int64(5)) == 5
    assert f(np.array([2])) == 2  # size-1 array via .item()
    assert f(np.float32(4.0)) == 4
    assert f(None) == -1  # default
    assert f("nope", default=9) == 9
    assert f({}, default=0) == 0


class _CountingEntryPoint:
    """A Java entry_point stand-in WITHOUT role-specific obs methods, so
    _read_all_obs takes the observations_all() fallback (current gatherer-only Java)."""


class _CountingBridge:
    def __init__(self, player_names):
        self.entry_point = _CountingEntryPoint()
        self._player_names = player_names
        self.calls = 0

    def observations_all(self):
        self.calls += 1
        # Empty raw per player; _decode_obs zero-fills, compute_gatherer_action_mask
        # tolerates missing keys — we only care about the call COUNT here.
        return {name: {} for name in self._player_names}


def test_read_all_obs_calls_observations_all_once_for_many_agents():
    env = object.__new__(wrapper.AiUtopiaPettingZooEnv)  # bypass heavy __init__
    agents = ["gatherer_0", "gatherer_1", "gatherer_2", "gatherer_3"]
    env.agents = agents
    env.agent_id_to_player_name = {a: a for a in agents}
    env.stage = 1
    env._stub_goal_embed = np.zeros(384, dtype=np.float32)
    env.bridge = _CountingBridge(agents)

    out = env._read_all_obs()

    assert set(out.keys()) == set(agents)
    assert env.bridge.calls == 1, (
        f"batched-obs invariant violated: observations_all() called {env.bridge.calls} "
        f"times for {len(agents)} agents (must be exactly 1)"
    )
