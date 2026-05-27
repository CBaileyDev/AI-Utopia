import os
import subprocess
import sys


def test_determinism_check_missing_weights_exits_2(tmp_path) -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    out = subprocess.run(
        [sys.executable, "-m", "aiutopia.cli.app", "determinism", "check",
         "--weights", str(tmp_path / "nonexistent")],
        capture_output=True, text=True, env=env,
    )
    assert out.returncode == 2
    combined = (out.stdout + out.stderr).lower()
    assert "weights not found" in combined
