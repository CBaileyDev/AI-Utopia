import numpy as np
import pytest
from gymnasium.spaces import Dict as DictSpace

from aiutopia.env.spaces import (
    build_role_observation_space, build_role_action_space, CORE_KEYS, GATHERER_KEYS,
)


def test_gatherer_obs_space_has_core_plus_gatherer_keys() -> None:
    space = build_role_observation_space("gatherer", stage=1)
    assert isinstance(space, DictSpace)
    for k in CORE_KEYS:
        assert k in space.spaces, f"missing core key {k}"
    for k in GATHERER_KEYS:
        assert k in space.spaces, f"missing gatherer key {k}"


def test_other_role_keys_NOT_in_gatherer_obs() -> None:
    space = build_role_observation_space("gatherer", stage=1)
    for k in space.spaces:
        assert not k.startswith("b_"), f"builder key {k} leaked into gatherer obs"
        assert not k.startswith("f_"), f"farmer key {k} leaked into gatherer obs"
        assert not k.startswith("d_"), f"defender key {k} leaked into gatherer obs"


def test_comm_buffer_is_32_slots_not_8() -> None:
    space = build_role_observation_space("gatherer", stage=1)
    assert space.spaces["comm_payloads"].shape == (32, 128)
    assert space.spaces["comm_metadata"].shape == (32, 8)


def test_goal_ticks_left_capped_at_12000() -> None:
    space = build_role_observation_space("gatherer", stage=1)
    assert space.spaces["goal_ticks_left"].high.max() == 12_000


def test_action_space_has_universal_header() -> None:
    space = build_role_action_space("gatherer")
    for k in ("skill_type", "target_class", "spatial_param", "scalar_param",
              "comm_payload", "should_broadcast", "comm_target_mask"):
        assert k in space.spaces


def test_action_space_sample_roundtrip() -> None:
    space = build_role_action_space("gatherer")
    sample = space.sample()
    assert space.contains(sample)


def test_observation_space_contains_action_mask() -> None:
    space = build_role_observation_space("gatherer", stage=1)
    assert "action_mask" in space.spaces
    mask = space.spaces["action_mask"]
    assert "skill_type" in mask.spaces
    assert "target_per_skill" in mask.spaces
