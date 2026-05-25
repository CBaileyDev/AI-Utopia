"""§6.6 — Subgoal DAG state machine: allowed transitions only.
Runtime hooks (Py4J calls, episodic memory writes, EventQueue puts)
land in M5 alongside the real planner."""
from __future__ import annotations

from aiutopia.schemas.enums import SubgoalState


class SubgoalTransitionError(ValueError):
    pass


_ALLOWED: dict[SubgoalState, frozenset[SubgoalState]] = {
    "pending":   frozenset({"active"}),
    "active":    frozenset({"completed", "failed", "paused"}),
    "paused":    frozenset({"active"}),
    "failed":    frozenset({"pending"}),   # fallback substitution
    "completed": frozenset(),              # terminal
}


def can_transition(src: SubgoalState, dst: SubgoalState) -> bool:
    return dst in _ALLOWED.get(src, frozenset())


def assert_transition(src: SubgoalState, dst: SubgoalState) -> None:
    if not can_transition(src, dst):
        raise SubgoalTransitionError(f"invalid transition: {src!r} → {dst!r}")
