import pytest

from aiutopia.schemas.failure import (
    ExecutionTraceEntry, FailureDetails, FailureReport, PartialProgress,
)


def _make_report(**overrides) -> FailureReport:
    base = dict(
        plan_id="01J0CABCDEFGHJKMNPQRSTVWXY",
        subgoal_id="01J0CABCDEFGHJKMNPQRSTVWXY",
        role="gatherer",
        agent_uuid="01J0CABCDEFGHJKMNPQRSTVWXY",
        failure_details=FailureDetails(
            failure_type="timeout",
            failure_tick=1000,
            final_state_summary={"hunger": 4},
            descriptor_summary="ran out of time fetching wood",
        ),
        partial_progress=PartialProgress(progress_fraction=0.6),
        reported_at=1,
    )
    base.update(overrides)
    return FailureReport(**base)


def test_minimal_failure_report_validates() -> None:
    r = _make_report()
    assert r.status == "failed"
    assert r.failure_details.failure_type == "timeout"


def test_failure_type_must_be_in_closed_vocab() -> None:
    with pytest.raises(Exception):
        _make_report(failure_details=FailureDetails(
            failure_type="not_a_real_failure",   # type: ignore[arg-type]
            failure_tick=1, final_state_summary={},
            descriptor_summary="x",
        ))


def test_partial_progress_fraction_bounded() -> None:
    with pytest.raises(Exception):
        PartialProgress(progress_fraction=1.5)
    with pytest.raises(Exception):
        PartialProgress(progress_fraction=-0.1)


def test_execution_trace_capped_at_200() -> None:
    long_trace = [
        ExecutionTraceEntry(tick=i, action_summary="a",
                             observation_summary="o", reward=0.0)
        for i in range(201)
    ]
    with pytest.raises(Exception):
        FailureDetails(
            failure_type="timeout", failure_tick=1,
            final_state_summary={}, descriptor_summary="x",
            execution_trace=long_trace,
        )
