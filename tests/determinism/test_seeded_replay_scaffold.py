"""§7.8 scaffold. Will NOT pass in M0 (no real weights yet). Tests that
the harness logic itself is correct using a synthetic deterministic agent;
M1+ replaces the dummy with a real RLlib policy."""
from __future__ import annotations

import numpy as np
import pytest

from aiutopia.determinism.harness import (
    compute_divergence, EPS_ARGMAX, EPS_L2,
)


pytestmark = pytest.mark.determinism


def _synthetic_trace(seed: int, length: int = 1000) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{
        "action_argmax":     int(rng.integers(0, 6)),
        "continuous_params": rng.uniform(-1, 1, size=4).astype(np.float32),
    } for _ in range(length)]


def test_identical_traces_pass() -> None:
    a = _synthetic_trace(seed=42)
    b = _synthetic_trace(seed=42)
    div = compute_divergence(a, b)
    assert div.passes
    assert div.action_argmax_divergence == 0.0
    assert div.continuous_param_l2 == 0.0


def test_different_seeds_fail() -> None:
    a = _synthetic_trace(seed=1)
    b = _synthetic_trace(seed=2)
    div = compute_divergence(a, b)
    assert not div.passes


def test_thresholds_match_spec() -> None:
    assert EPS_ARGMAX == 0.05
    assert EPS_L2     == 0.10
