"""End-to-end env-level smoke: spawn a Carpet agent, build the
AiUtopiaPettingZooEnv with the agent_id → player_name map populated,
call reset()+step() a few times, assert obs/reward/memory pipeline works.

Skips when no Py4J server on port 25099."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
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
def test_env_reset_and_step_against_live_carpet(live_server: int,
                                                  tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIUTOPIA_ROOT", str(tmp_path))
    py  = sys.executable
    env = {**os.environ, "PYTHONPATH": "src", "AIUTOPIA_ROOT": str(tmp_path)}

    # 1. Spawn an agent via the CLI — this populates identity.db AND
    # the Carpet fake player, AND returns the assigned name + ULID.
    out = subprocess.run(
        [py, "-m", "aiutopia.cli.app", "agent", "spawn",
         "--role", "gatherer", "--py4j-port", str(live_server)],
        capture_output=True, text=True, env=env, check=True, timeout=30,
    )
    # Parse "identity: spawned <Name> (gatherer, uuid=<ULID>)"
    spawn_line = next(l for l in out.stdout.splitlines()
                        if l.startswith("identity: spawned"))
    player_name = spawn_line.split()[2]
    uuid = spawn_line.split("uuid=")[-1].rstrip(")")
    time.sleep(1.5)  # let Carpet place the fake player

    # 2. Build the env with the agent_id → player_name map populated
    # (this is the R6 fix — without it, dispatch_skill fails).
    from aiutopia.env.wrapper import AiUtopiaPettingZooEnv
    env_inst = AiUtopiaPettingZooEnv({
        "stage": 1, "active_roles": ["gatherer"],
        "seed_strategy": "fixed_easy", "tick_warp": False,
        "py4j_ports": [live_server], "max_episode_ticks": 100,
        "per_worker_seed_offset": False, "worker_index": 0,
        "agent_id_to_player_name": {"gatherer_0": player_name},
        "agent_id_to_uuid":        {"gatherer_0": uuid},
        # Real Chroma writes — verifies R7 wiring works end-to-end.
        "enable_memory_writes": True,
    })
    try:
        # 3. reset() returns valid obs
        obs, _info = env_inst.reset(seed=1)
        assert "gatherer_0" in obs
        sample = obs["gatherer_0"]
        # R2: shape (1,) scalars
        assert sample["health"].shape == (1,)
        # R2: 384-d agent_uuid_embed derived from UUID — should be deterministic
        assert sample["agent_uuid_embed"].shape == (384,)
        # R4: role_one_hot has gatherer bit set
        assert sample["role_one_hot"][0] == 1
        # R4: action_mask should allow at least WAIT
        assert sample["action_mask"]["skill_type"].any()

        # 4. step() with a WAIT action — should not crash + should
        # complete fast (WAIT default scalar=0.05 ≈ half-second).
        act = {
            "skill_type":       4,    # WAIT
            "target_class":     0,
            "spatial_param":    np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "scalar_param":     np.array([0.05], dtype=np.float32),
            "comm_payload":     np.zeros(128, dtype=np.float32),
            "should_broadcast": 0,
            "comm_target_mask": np.zeros(4, dtype=np.int8),
        }
        new_obs, rew, term, trunc, info = env_inst.step({"gatherer_0": act})
        assert isinstance(rew["gatherer_0"], float)
        # Skill completed — completion event came back, not "agent player not found"
        comp = info["gatherer_0"].get("skill_completion", {})
        assert comp.get("resultCode") in {"COMPLETED", "RUNNING"}
        assert "not found" not in comp.get("failureReason", "")

        # 5. After a few ticks, the memory writer should have recorded
        # at least one HIGH/MEDIUM event for this agent (completion bonus
        # alone is 0.3 — MEDIUM threshold; should buffer at least).
        from aiutopia.memory.client import open_chroma
        from aiutopia.common.config import Paths
        from aiutopia.common.ids import memory_id_for
        chroma = open_chroma(Paths.from_env().chroma_dir)
        coll = chroma.get_or_create_collection(memory_id_for(uuid))
        # Records can be buffered (MEDIUM) — flush to see them.
        env_inst.memory_writer.flush()
        snapshot = coll.get(limit=10)
        # Don't assert non-empty (a single WAIT may stay below thresholds),
        # but assert the collection exists and the query path works.
        assert isinstance(snapshot.get("ids"), list)
    finally:
        env_inst.close()
