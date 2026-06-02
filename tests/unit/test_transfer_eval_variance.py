"""Gating logic for the variance-controlled real-MC transfer eval.

HANDOFF.md S3: real-MC HARVEST is non-deterministic, so a single greedy 3-seed
eval can't declare a clean pass/fail at n=1. scripts/transfer_eval_bc.py now runs
each gate scenario N times and gates on a per-scenario success RATE. The decision
logic (`summarize`) is a pure function so it is unit-testable without a live Fabric
server -- these tests pin the gate semantics that the (server-bound) eval depends on.

`scripts/` is not on pythonpath (src-only); the script's module top is import-light
(argparse/os only -- the heavy ray/torch imports live inside main), so loading it by
file location is cheap.
"""

import importlib.util
import pathlib

_PY = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "transfer_eval_bc.py"
_spec = importlib.util.spec_from_file_location("_transfer_eval_bc", _PY)
_te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_te)


def _runs(successes_oaks):
    return [{"success": s, "oak": o} for s, o in successes_oaks]


def test_rate_gate_passes_above_threshold_despite_variance():
    """A noisy seed (oak swings 46-64) still PASSES when the success rate clears it."""
    scenario_runs = {
        "m1_oak_log_seed_1": _runs([(True, 64), (True, 64), (False, 46), (True, 64), (True, 58)]),
    }
    summary = _te.summarize(scenario_runs, pass_threshold=0.6)
    s = summary["per_scenario"]["m1_oak_log_seed_1"]
    assert s["successes"] == 4 and s["n"] == 5
    assert abs(s["rate"] - 0.8) < 1e-9
    assert s["oak_min"] == 46 and s["oak_max"] == 64
    assert s["passed"] is True
    assert summary["passed"] is True


def test_rate_gate_fails_below_threshold():
    """Below-threshold success rate fails the scenario AND the overall gate."""
    scenario_runs = {
        "m1_oak_log_seed_1": _runs([(True, 64), (False, 46), (False, 50), (False, 48), (True, 64)]),
    }
    summary = _te.summarize(scenario_runs, pass_threshold=0.6)
    s = summary["per_scenario"]["m1_oak_log_seed_1"]
    assert s["successes"] == 2
    assert abs(s["rate"] - 0.4) < 1e-9
    assert s["passed"] is False
    assert summary["passed"] is False


def test_overall_gate_requires_every_scenario_to_pass():
    """One failing seed sinks the whole gate even if the others are perfect."""
    scenario_runs = {
        "m1_oak_log_seed_1": _runs([(True, 64)] * 5),
        "m1_oak_log_seed_2": _runs([(True, 64)] * 5),
        "m1_oak_log_seed_3": _runs(
            [(False, 50), (False, 48), (True, 64), (False, 46), (False, 52)]
        ),
    }
    summary = _te.summarize(scenario_runs, pass_threshold=0.6)
    assert summary["per_scenario"]["m1_oak_log_seed_1"]["passed"] is True
    assert summary["per_scenario"]["m1_oak_log_seed_2"]["passed"] is True
    assert summary["per_scenario"]["m1_oak_log_seed_3"]["passed"] is False
    assert summary["passed"] is False


def test_threshold_boundary_is_inclusive():
    """rate == threshold passes (>=, not >): exactly 3/5 clears a 0.6 gate."""
    scenario_runs = {"s": _runs([(True, 64), (True, 64), (True, 64), (False, 46), (False, 50)])}
    summary = _te.summarize(scenario_runs, pass_threshold=0.6)
    assert abs(summary["per_scenario"]["s"]["rate"] - 0.6) < 1e-9
    assert summary["per_scenario"]["s"]["passed"] is True
    assert summary["passed"] is True


def test_empty_runs_do_not_falsely_pass():
    """No data must never report a pass (guards a vacuous-truth gate)."""
    assert _te.summarize({}, pass_threshold=0.6)["passed"] is False
    one_empty = _te.summarize({"s": []}, pass_threshold=0.6)
    assert one_empty["per_scenario"]["s"]["passed"] is False
    assert one_empty["passed"] is False
