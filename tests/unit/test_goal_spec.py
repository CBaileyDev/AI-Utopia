"""Verifies §3.1 invariants without loading the actual BGE model
(the model is loaded lazily by the adapter; tests inject a fake encoder)."""
from __future__ import annotations

import numpy as np
import pytest

from aiutopia.planner.goal_spec import (
    GoalSpecAdapter, build_structured_features, build_nl_summary,
)
from aiutopia.schemas.plan import (
    Constraints, GoalSpecification, Subgoal, TargetState, TerminationConditions,
)


class _FakeBGE:
    """Deterministic stand-in: returns a constant 384-d vector regardless of text."""
    def encode(self, text: str) -> np.ndarray:
        # Different texts → slightly different vectors so we can distinguish.
        h = hash(text) % 1000
        v = np.full(384, h / 1000.0, dtype=np.float32)
        return v


def _sg() -> Subgoal:
    return Subgoal(
        role="gatherer",
        priority=7,
        goal_specification=GoalSpecification(
            target_state=TargetState(inventory_delta={"oak_log": 32}),
            termination_conditions=TerminationConditions(
                success_criteria=["inventory_meets_delta"],
                timeout_ticks=6000,
            ),
        ),
        constraints=Constraints(),
        nl_summary="collect 32 oak_log",
    )


def test_structured_features_shape_is_128() -> None:
    sg = _sg()
    feat = build_structured_features(sg)
    assert feat.shape == (128,)
    assert feat.dtype == np.float32


def test_role_onehot_set_correctly() -> None:
    sg = _sg()
    feat = build_structured_features(sg)
    # role_one_hot occupies indices 0-3 in our layout: gatherer=0
    assert feat[0] == 1.0
    assert feat[1] == 0.0


def test_goal_embedding_is_512_d() -> None:
    sg = _sg()
    adapter = GoalSpecAdapter(bge=_FakeBGE())
    emb = adapter.embed(sg)
    assert emb.shape == (512,)
    assert emb.dtype == np.float32


def test_role_dispatch_returns_correct_policy_name() -> None:
    sg = _sg()
    adapter = GoalSpecAdapter(bge=_FakeBGE())
    assert adapter.policy_name_for(sg) == "gatherer_policy"


def test_invalid_role_raises() -> None:
    """If subgoal.role were ever forged outside the Literal, dispatch raises."""
    adapter = GoalSpecAdapter(bge=_FakeBGE())
    class _Fake:
        role = "not_a_real_role"
    with pytest.raises(KeyError):
        adapter.policy_name_for(_Fake())


def test_build_nl_summary_uses_subgoal_nl_summary_verbatim() -> None:
    sg = _sg()
    s = build_nl_summary(sg)
    assert "collect 32 oak_log" in s
    assert "gatherer" in s
