"""End-to-end test of `aiutopia agent spawn`. Skips when no live Fabric
server is reachable. The identity-only path (--no-fabric) is always tested."""
from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest


pytestmark = pytest.mark.integration


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _subprocess_env(tmp_path) -> dict:
    """Ensure subprocess can import aiutopia (PYTHONPATH=src) when the
    package isn't editable-installed. Harmless when it IS installed."""
    env = {**os.environ, "AIUTOPIA_ROOT": str(tmp_path), "PYTHONPATH": "src"}
    return env


def test_spawn_no_fabric_creates_identity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    env = _subprocess_env(tmp_path)
    out = subprocess.run(
        [sys.executable, "-m", "aiutopia.cli.app", "agent", "spawn",
         "--role", "gatherer", "--no-fabric"],
        capture_output=True, text=True, check=True, env=env,
    )
    assert "spawned" in out.stdout
    list_out = subprocess.run(
        [sys.executable, "-m", "aiutopia.cli.app", "agent", "list"],
        capture_output=True, text=True, check=True, env=env,
    )
    assert "gatherer" in list_out.stdout


@pytest.mark.skipif(not _port_open("127.0.0.1",
                                     int(os.environ.get("PY4J_SMOKE_PORT", "25099"))),
                    reason="no live Fabric server on PY4J_SMOKE_PORT")
def test_spawn_with_fabric_calls_carpet(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    env = _subprocess_env(tmp_path)
    port = int(os.environ.get("PY4J_SMOKE_PORT", "25099"))
    out = subprocess.run(
        [sys.executable, "-m", "aiutopia.cli.app", "agent", "spawn",
         "--role", "gatherer", "--py4j-port", str(port)],
        capture_output=True, text=True, check=True, env=env,
    )
    assert "carpet: /player" in out.stdout
    assert "→ ok" in out.stdout
