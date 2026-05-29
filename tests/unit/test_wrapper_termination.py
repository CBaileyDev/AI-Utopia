"""Phase-0 fix #5 — success-termination predicate.

The episode previously only terminated on death; collecting the M1B goal
(64 oak_log) never ended it, so TIME_PENALTY/PBRS kept accruing past the
goal, the policy got no episodic "win" signal, and every eval burned the
full step budget.

`_goal_success(goal_spec, inventory)` is the pure predicate extracted from
`step()` so it can be unit-tested without the Py4J/Fabric bridge. It takes a
fully-reconstructed `{item_name: count}` inventory dict (built at the call
site by reward._inventory_from_obs, the same function the reward path uses,
so success and reward never disagree about what's in the bag).

These tests feed hand-built inventory dicts directly, so they are insulated
from the obs decode path, reward.py, and the Java side entirely.
"""

from __future__ import annotations

from aiutopia.env.wrapper import _goal_success
from aiutopia.schemas.plan import (
    GoalSpecification,
    TargetState,
    TerminationConditions,
)


def _goal(
    inventory_delta: dict[str, int], success_criteria: list[str] | None = None
) -> GoalSpecification:
    """The M1B goal spec by default: collect 64 oak_log via inventory_meets_delta."""
    return GoalSpecification(
        target_state=TargetState(inventory_delta=inventory_delta),
        termination_conditions=TerminationConditions(
            success_criteria=success_criteria or ["inventory_meets_delta"],
            timeout_ticks=6000,
        ),
    )


# ───── the M1B goal: collect 64 oak_log ─────


def test_exactly_target_is_success() -> None:
    g = _goal({"oak_log": 64})
    assert _goal_success(g, {"oak_log": 64}) is True


def test_one_below_target_is_not_success() -> None:
    g = _goal({"oak_log": 64})
    assert _goal_success(g, {"oak_log": 63}) is False


def test_over_target_is_success() -> None:
    """>= semantics: more than the target still counts as met."""
    g = _goal({"oak_log": 64})
    assert _goal_success(g, {"oak_log": 65}) is True


def test_target_met_with_extra_offtask_items_is_success() -> None:
    """Extra cobblestone in the bag does not disqualify a met oak_log goal."""
    g = _goal({"oak_log": 64})
    assert _goal_success(g, {"oak_log": 64, "cobblestone": 30}) is True


def test_wrong_item_is_not_success() -> None:
    """Plenty of the wrong item, none of the target → not success."""
    g = _goal({"oak_log": 64})
    assert _goal_success(g, {"cobblestone": 256}) is False


def test_missing_target_item_is_not_success() -> None:
    g = _goal({"oak_log": 64})
    assert _goal_success(g, {}) is False


# ───── multi-item goal (every item must meet its target) ─────


def test_multi_item_all_met_is_success() -> None:
    g = _goal({"oak_log": 32, "cobblestone": 16})
    assert _goal_success(g, {"oak_log": 40, "cobblestone": 16}) is True


def test_multi_item_one_short_is_not_success() -> None:
    g = _goal({"oak_log": 32, "cobblestone": 16})
    assert _goal_success(g, {"oak_log": 40, "cobblestone": 15}) is False


# ───── robustness guards ─────


def test_non_inventory_criterion_is_not_success() -> None:
    """The helper only implements `inventory_meets_delta`. A spatial/blueprint
    success criterion must NOT fire off the inventory delta alone."""
    # TargetState requires at least one field set, so we keep the delta but
    # declare a criterion the helper does not implement.
    g = _goal({"oak_log": 64}, success_criteria=["reached_spatial_target"])
    assert _goal_success(g, {"oak_log": 999}) is False


def test_empty_delta_is_not_instant_success() -> None:
    """`all([])` is True; an empty inventory_delta must NOT read as an instant
    win. (Constructed directly to dodge the TargetState non-empty validator.)"""
    g = _goal({"oak_log": 1})  # valid spec...
    object.__setattr__(g.target_state, "inventory_delta", {})  # ...then empty the delta
    assert _goal_success(g, {"oak_log": 100}) is False
