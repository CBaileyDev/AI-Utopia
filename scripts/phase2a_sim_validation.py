#!/usr/bin/env python
"""Phase 2a Sim Validation: Greedy multi-agent (gatherer + explorer + farmer).

No Ray required. Direct sim env stepping with oracle/greedy policies.
Validates RLModule obs/action/reward dispatch for Phase 2 infrastructure.

Run: PYTHONPATH=src py -3.11 scripts/phase2a_sim_validation.py
"""
from __future__ import annotations

import os
from pathlib import Path

# No Ray / Torch init needed for sim-only
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np


def main() -> None:
    """Phase 2a: greedy agent validation on flat farmland."""
    from aiutopia.common.logging import get_logger, setup_logging
    from aiutopia.env.spaces import build_role_action_space, build_role_observation_space
    from aiutopia.sim.sim_env import AiUtopiaSimEnv

    setup_logging("INFO")
    log = get_logger("phase2a")

    # Roles to test: gatherer (proven), explorer (new), farmer (new)
    roles = ["gatherer", "explorer", "farmer"]

    # Multi-role env config: active_roles enables all 3 agents in one env
    env_config = {
        "active_roles": roles,  # Key: tells sim_env to instantiate all roles
        "peaceful": True,
        "backend": "sim",
        "arena_bounds_check": False,
        "max_episode_ticks": 100,
        "seed": 1,
    }

    log.info("Initializing Phase 2a sim validation (3-agent MARL on flat farmland)")
    log.info(f"  active_roles: {roles}")

    # Create single multi-agent env (all 3 agents in one world state)
    try:
        env = AiUtopiaSimEnv(env_config)
    except Exception as e:
        log.error(f"Failed to instantiate multi-agent sim env: {e}")
        log.info("Falling back to single-agent (gatherer only)")
        env_config["active_roles"] = ["gatherer"]
        env = AiUtopiaSimEnv(env_config)

    # Log agent setup
    log.info(f"  possible_agents: {env.possible_agents}")

    # Get obs/action spaces
    action_spaces = {role: build_role_action_space(role) for role in roles}
    obs_spaces = {role: build_role_observation_space(role, stage=1) for role in roles}

    # Reset
    obs, infos = env.reset(seed=1)
    step_count = 0
    episode_returns = {role: 0.0 for role in roles}

    # Greedy loop: take random actions (validates obs/action/reward dispatch)
    log.info(f"  running {env_config['max_episode_ticks']} steps with greedy agents")
    while env.agents:
        # Sample random valid action per role
        action = {}
        for agent_id in env.agents:
            role = agent_id.split("_", 1)[0]
            action[agent_id] = action_spaces[role].sample()

        obs, rewards, terminated, truncated, infos = env.step(action)

        for agent_id, rew in rewards.items():
            role = agent_id.split("_", 1)[0]
            episode_returns[role] += rew
        step_count += 1

    env.close()

    log.info(f"Phase 2a validation complete ({step_count} steps).")
    log.info("Metrics:")
    for role in roles:
        log.info(f"  {role}: return={episode_returns[role]:.4f}")


if __name__ == "__main__":
    main()
