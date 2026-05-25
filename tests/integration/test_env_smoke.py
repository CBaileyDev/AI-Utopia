"""End-to-end smoke: spin up the wrapper against a running Fabric server
on port PY4J_SMOKE_PORT (default 25099) and verify reset() + 1 step().

Skip if PY4J_SMOKE_PORT is not reachable — most contributors won't have a
server running, and that's fine."""
from __future__ import annotations

import os
import socket

import numpy as np
import pytest

from aiutopia.env.wrapper import AiUtopiaPettingZooEnv


pytestmark = pytest.mark.integration


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


@pytest.fixture
def smoke_port() -> int:
    return int(os.environ.get("PY4J_SMOKE_PORT", "25099"))


def test_env_reset_returns_valid_obs(smoke_port: int) -> None:
    if not _port_open("127.0.0.1", smoke_port):
        pytest.skip(f"no Py4J server on port {smoke_port} (set PY4J_SMOKE_PORT)")
    env = AiUtopiaPettingZooEnv({
        "stage": 1, "active_roles": ["gatherer"],
        "seed_strategy": "fixed_easy", "tick_warp": True,
        "py4j_ports": [smoke_port], "max_episode_ticks": 100,
        "per_worker_seed_offset": False, "worker_index": 0,
    })
    try:
        obs, info = env.reset(seed=1)
        assert "gatherer_0" in obs
        sample = obs["gatherer_0"]
        assert sample["goal_embedding"].shape == (512,)
        assert sample["comm_payloads"].shape == (32, 128)
        assert "action_mask" in sample

        # 1 step
        act = env.action_space("gatherer_0").sample()
        new_obs, rew, term, trunc, _info = env.step({"gatherer_0": act})
        assert "gatherer_0" in new_obs
        assert isinstance(rew["gatherer_0"], float)
    finally:
        env.close()
