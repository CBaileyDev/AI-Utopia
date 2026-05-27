import json
from pathlib import Path

import pytest

from aiutopia.promotion.checklist import run_checklist


def _write_metrics(ckpt: Path, **overrides) -> Path:
    defaults = {
        "last_50k_steps_return_trend":      1.0,
        "evaluation_scenario_success_rate": 0.92,
        "exploit_penalty_ratio":            0.02,
        "per_role_entropy":                 {"gatherer": 1.8},
        "q_variance_ratio":                 3.5,
        "trajectory_cosine":                1.0,
        "determinism_argmax_div":           0.02,
        "determinism_l2":                   0.04,
    }
    defaults.update(overrides)
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "aiutopia_metrics.json").write_text(json.dumps(defaults))
    return ckpt


def test_all_gates_pass(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    from aiutopia.common.config import Paths
    paths = Paths.from_env(); paths.ensure()
    ckpt = _write_metrics(tmp_path / "ckpt")
    report = run_checklist(role="gatherer", checkpoint=ckpt, paths=paths)
    assert report.passes, report.issues
    assert len(report.gates_passed) == 5


def test_missing_metrics_file_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    from aiutopia.common.config import Paths
    paths = Paths.from_env(); paths.ensure()
    report = run_checklist(role="gatherer",
                            checkpoint=tmp_path / "nonexistent",
                            paths=paths)
    assert not report.passes


def test_failing_scenario_rate_blocks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    from aiutopia.common.config import Paths
    paths = Paths.from_env(); paths.ensure()
    ckpt = _write_metrics(tmp_path / "ckpt", evaluation_scenario_success_rate=0.5)
    report = run_checklist(role="gatherer", checkpoint=ckpt, paths=paths)
    assert not report.passes
    assert any("Gate 2" in i for i in report.issues)


def test_failing_determinism_blocks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    from aiutopia.common.config import Paths
    paths = Paths.from_env(); paths.ensure()
    ckpt = _write_metrics(tmp_path / "ckpt",
                           determinism_argmax_div=0.2,
                           determinism_l2=0.5)
    report = run_checklist(role="gatherer", checkpoint=ckpt, paths=paths)
    assert not report.passes
    assert any("Gate 5" in i for i in report.issues)
