"""Pytest wrapper for the RLlib 1-iter smoke test.

Runs scripts/rllib_smoke.py as a subprocess to keep Ray init isolated from
the pytest process and avoid interaction with other integration tests.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("ray")


@pytest.mark.integration
@pytest.mark.slow
def test_rllib_smoke_runs_one_iteration() -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    out = subprocess.run(
        [sys.executable, "scripts/rllib_smoke.py"],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert out.returncode == 0, (
        f"STDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}"
    )
    assert "RLLIB SMOKE OK" in out.stdout
