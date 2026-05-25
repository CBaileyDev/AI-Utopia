import pytest

from aiutopia.schemas.state_machine import (
    can_transition, SubgoalTransitionError,
)


def test_pending_to_active_allowed() -> None:
    assert can_transition("pending", "active")


def test_active_to_completed_allowed() -> None:
    assert can_transition("active", "completed")


def test_active_to_failed_allowed() -> None:
    assert can_transition("active", "failed")


def test_active_to_paused_allowed() -> None:
    assert can_transition("active", "paused")


def test_paused_to_active_allowed() -> None:
    assert can_transition("paused", "active")


def test_failed_to_pending_allowed_fallback() -> None:
    assert can_transition("failed", "pending")


def test_completed_terminal() -> None:
    assert not can_transition("completed", "active")
    assert not can_transition("completed", "pending")


def test_pending_to_completed_forbidden() -> None:
    assert not can_transition("pending", "completed")


def test_assert_helper_raises_on_invalid() -> None:
    from aiutopia.schemas.state_machine import assert_transition
    with pytest.raises(SubgoalTransitionError):
        assert_transition("completed", "active")
