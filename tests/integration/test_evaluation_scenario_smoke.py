import numpy as np
import pytest
import torch

from aiutopia.train.scenario_runner import (
    M1_SCENARIOS, _greedy_decode, aggregate_success_rate,
)


def test_m1_scenarios_present() -> None:
    assert len(M1_SCENARIOS) >= 3
    for s in M1_SCENARIOS:
        assert s.max_ticks == 1000


def test_greedy_decode_returns_valid_action() -> None:
    # OUTPUT_DIM is 344 post-T7.5 (MultiBinary slice emits 2*N logits under
    # TorchMultiCategorical, not N). See actor_head.py.
    flat = torch.randn(344)
    action = _greedy_decode(flat)
    assert "skill_type" in action
    assert 0 <= action["skill_type"] < 6
    assert action["spatial_param"].shape == (3,)
    assert action["spatial_param"].dtype == np.float32


def test_aggregate_success_rate() -> None:
    assert aggregate_success_rate([]) == 0.0
    assert aggregate_success_rate([{"success": True}, {"success": False},
                                     {"success": True}]) == 2/3
