"""Section 5.10 promotion checklist — 5 gates."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from aiutopia.common.config import Paths


@dataclass
class ChecklistReport:
    passes:        bool
    gates_passed:  list[str] = field(default_factory=list)
    issues:        list[str] = field(default_factory=list)


def run_checklist(*, role: str, checkpoint: Path, paths: Paths) -> ChecklistReport:
    metrics_file = checkpoint / "aiutopia_metrics.json"
    report = ChecklistReport(passes=False)
    if not metrics_file.exists():
        report.issues.append(f"aiutopia_metrics.json not found at {metrics_file}")
        return report
    metrics = json.loads(metrics_file.read_text())

    if metrics.get("last_50k_steps_return_trend", -1) >= 0:
        report.gates_passed.append("1_return_trend_nonneg")
    else:
        report.issues.append("Gate 1: episodic return is collapsing")

    rate = metrics.get("evaluation_scenario_success_rate", 0)
    if rate >= 0.80:
        report.gates_passed.append(f"2_scenario_success_{rate:.0%}")
    else:
        report.issues.append(f"Gate 2: scenario success rate {rate:.0%} < 80%")

    ratio = metrics.get("exploit_penalty_ratio", 1.0)
    if ratio < 0.05:
        report.gates_passed.append(f"3_exploit_ratio_{ratio:.2%}")
    else:
        report.issues.append(f"Gate 3: exploit penalty ratio {ratio:.2%} >= 5%")

    per_role = metrics.get("per_role_entropy", {})
    entropy = per_role.get(role, 0.0)
    qvar    = metrics.get("q_variance_ratio", 999.0)
    tcos    = metrics.get("trajectory_cosine", 1.0)
    single_agent = len(per_role) == 1
    failure_mode_ok = (entropy > 1.5 and qvar < 5.0
                        and (single_agent or tcos < 0.8))
    if failure_mode_ok:
        report.gates_passed.append(
            f"4_failure_modes_entropy_{entropy:.2f}_qvar_{qvar:.1f}")
    else:
        report.issues.append(
            f"Gate 4: entropy={entropy:.2f} (need >1.5), qvar={qvar:.1f} "
            f"(need <5), traj_cos={tcos:.2f}")

    argmax_div = metrics.get("determinism_argmax_div", 1.0)
    l2_div     = metrics.get("determinism_l2", 999.0)
    if argmax_div < 0.05 and l2_div < 0.1:
        report.gates_passed.append(
            f"5_determinism_argmax_{argmax_div:.3f}_l2_{l2_div:.3f}")
    else:
        report.issues.append(
            f"Gate 5: argmax_div={argmax_div:.3f} (need <0.05), "
            f"l2={l2_div:.3f} (need <0.1)")

    report.passes = len(report.issues) == 0
    return report
