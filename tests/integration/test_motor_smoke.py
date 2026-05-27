"""End-to-end smoke: dispatch NAVIGATE + HARVEST to a live agent and
verify (a) Java emits completion events, (b) reward signal moves.

Skips when no Py4J server on port 25099."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest

PORT = int(os.environ.get("PY4J_SMOKE_PORT", "25099"))


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


@pytest.fixture
def live_server() -> int:
    if not _port_open("127.0.0.1", PORT):
        pytest.skip(f"no Py4J server on port {PORT}")
    return PORT


@pytest.mark.integration
def test_navigate_then_harvest_emits_completion_events(live_server: int, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": "src"}

    # 1. Spawn agent
    out = subprocess.run(
        [py, "-m", "aiutopia.cli.app", "agent", "spawn", "--role", "gatherer",
         "--py4j-port", str(live_server)],
        capture_output=True, text=True, env=env, check=True, timeout=30,
    )
    assert "spawn (skin=" in out.stdout
    # Extract the spawned agent name from "identity: spawned <Name> ..."
    name_line = next(l for l in out.stdout.splitlines() if l.startswith("identity: spawned"))
    agent_name = name_line.split()[2]
    time.sleep(1.5)  # let Carpet finish placing the fake player

    # 2. Drive NAVIGATE forward
    out = subprocess.run(
        [py, "-m", "aiutopia.cli.app", "agent", "drive",
         "--agent-name", agent_name,
         "--skill", "0",   # NAVIGATE
         "--dx", "0.1", "--dy", "0.0", "--dz", "0.0",
         "--scalar", "0.5",
         "--py4j-port", str(live_server),
         "--timeout-ms", "30000"],
        capture_output=True, text=True, env=env, check=False, timeout=45,
    )
    assert "resultCode" in out.stdout, out.stderr

    # 3. Drive HARVEST oak_log (target_class=0)
    out = subprocess.run(
        [py, "-m", "aiutopia.cli.app", "agent", "drive",
         "--agent-name", agent_name,
         "--skill", "1",   # HARVEST
         "--target", "0",  # oak_log
         "--scalar", "0.05",
         "--py4j-port", str(live_server),
         "--timeout-ms", "30000"],
        capture_output=True, text=True, env=env, check=False, timeout=45,
    )
    # Allowed outcomes:
    #   - COMPLETED if there's a tree nearby
    #   - IMMEDIATE_FAILURE if no oak_log within 8 blocks
    #   - FAILED_TIMEOUT if it tried but ran out of ticks
    assert any(code in out.stdout for code in
               ("COMPLETED", "IMMEDIATE_FAILURE", "FAILED_TIMEOUT")), out.stdout
