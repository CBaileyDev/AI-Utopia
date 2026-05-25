import pytest

from aiutopia.schemas.plan import (
    Constraints, Dependency, GoalSpecification, LlmPlanOutput,
    Subgoal, TargetState, TerminationConditions,
)


def _sg(role: str = "gatherer", sgid: str | None = None,
        fallbacks: list[str] | None = None) -> Subgoal:
    return Subgoal(
        subgoal_id=sgid or "01J0SG0000123456789ABCDE0Y",
        role=role,
        goal_specification=GoalSpecification(
            target_state=TargetState(inventory_delta={"oak_log": 32}),
            termination_conditions=TerminationConditions(
                success_criteria=["inventory_meets_delta"],
                timeout_ticks=6000,
            ),
        ),
        constraints=Constraints(),
        fallback_subgoals=fallbacks or [],
        nl_summary="collect 32 oak_log",
    )


def test_minimal_plan_validates() -> None:
    plan = LlmPlanOutput(
        high_level_goal="build a small village",
        subgoals=[_sg()],
        created_at=1,
        created_by="stub-planner",
    )
    assert plan.schema_version == "1.0.0"
    assert plan.max_fallback_chain_depth == 3


def test_target_state_requires_at_least_one_field() -> None:
    with pytest.raises(Exception):
        TargetState()   # all fields empty/None


def test_timeout_ticks_capped_at_12000() -> None:
    with pytest.raises(Exception):
        TerminationConditions(
            success_criteria=["x"],
            timeout_ticks=12_001,
        )


def test_dependency_missing_subgoal_id_raises() -> None:
    a = _sg(sgid="01J0SG0001000000000000000A")
    with pytest.raises(Exception):
        LlmPlanOutput(
            high_level_goal="g",
            subgoals=[a],
            dependencies=[Dependency(before=a.subgoal_id,
                                      after="01J0SG0002000000000000000B")],
            created_at=1, created_by="stub-planner",
        )


def test_dependency_self_loop_raises() -> None:
    a = _sg(sgid="01J0SG0001000000000000000A")
    with pytest.raises(Exception):
        LlmPlanOutput(
            high_level_goal="g",
            subgoals=[a],
            dependencies=[Dependency(before=a.subgoal_id, after=a.subgoal_id)],
            created_at=1, created_by="stub-planner",
        )


def test_dag_cycle_detection_via_kahn() -> None:
    a = _sg(sgid="01J0SG0001000000000000000A")
    b = _sg(sgid="01J0SG0002000000000000000B")
    c = _sg(sgid="01J0SG0003000000000000000C")
    with pytest.raises(Exception):
        LlmPlanOutput(
            high_level_goal="g",
            subgoals=[a, b, c],
            dependencies=[
                Dependency(before=a.subgoal_id, after=b.subgoal_id),
                Dependency(before=b.subgoal_id, after=c.subgoal_id),
                Dependency(before=c.subgoal_id, after=a.subgoal_id),  # cycle
            ],
            created_at=1, created_by="stub-planner",
        )


def test_fallback_pointing_to_unknown_subgoal_raises() -> None:
    a = _sg(sgid="01J0SG0001000000000000000A",
            fallbacks=["01J0SG0099000000000000000Z"])
    with pytest.raises(Exception):
        LlmPlanOutput(
            high_level_goal="g",
            subgoals=[a],
            created_at=1, created_by="stub-planner",
        )


def test_nl_summary_max_length_1500() -> None:
    with pytest.raises(Exception):
        Subgoal(
            subgoal_id="01J0SG0000123456789ABCDE0Y",
            role="gatherer",
            goal_specification=GoalSpecification(
                target_state=TargetState(inventory_delta={"x": 1}),
                termination_conditions=TerminationConditions(
                    success_criteria=["x"], timeout_ticks=1,
                ),
            ),
            nl_summary="x" * 1501,
        )
