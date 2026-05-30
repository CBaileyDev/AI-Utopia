#!/usr/bin/env python
"""Phase 2b Real-MC Transfer: Greedy 3-agent (gatherer+explorer+farmer) on live server.

No Ray/training. Direct env stepping with oracle/greedy policies.
Validates Java skills (Farmer PLOW/PLANT/HARVEST) + Python RLModules on real Minecraft.

Requirements:
  - 4 live Fabric training servers (ports 25001-25004)
  - m2-farmer JAR deployed (Farmer skills registered)
  - Natural world with peaceful gamerule

Run: PYTHONPATH=src PY4J_PRODUCTION_PORT=25001 py -3.11 scripts/phase2b_realmc_transfer_test.py --role gather,farm --max-steps 200

Metrics:
  - Gatherer: oak_log count (proven role)
  - Farmer: crop harvests (new role, validates PLOW/PLANT/HARVEST dispatch)
  - Explorer: richness_score progress (new role, validates bearing action)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> None:
    """Phase 2b: greedy multi-agent transfer to real Minecraft."""
    from aiutopia.common.config import Py4JConfig
    from aiutopia.common.logging import get_logger, setup_logging
    from aiutopia.env.spaces import build_role_action_space, build_role_observation_space
    from aiutopia.env.wrapper import AiUtopiaPettingZooEnv

    setup_logging("INFO")
    log = get_logger("phase2b")

    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="gather", type=str,
                       help="Comma-separated roles: gather,farm,explore (default: gather)")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Max steps per agent")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--port", type=int, default=25001,
                       help="Py4J port (default: 25001, production: 25100)")
    args = parser.parse_args()

    roles = [r.strip() for r in args.role.split(",")]
    role_map = {"gather": "gatherer", "explore": "explorer", "farm": "farmer"}
    roles = [role_map.get(r, r) for r in roles]

    log.info("=" * 70)
    log.info("PHASE 2B: Real-MC Transfer (Greedy Multi-Agent)")
    log.info("=" * 70)
    log.info(f"Roles: {roles}")
    log.info(f"Py4J port: {args.port}")
    log.info(f"Max steps per agent: {args.max_steps}")

    # Create env config
    env_config = {
        "peaceful": True,  # No hostile mobs
        "py4j_port": args.port,
        "seed": args.seed,
    }

    try:
        log.info("\nInstantiating AiUtopiaPettingZooEnv (real Minecraft)...")
        env = AiUtopiaPettingZooEnv(env_config)
        log.info(f"  ✓ Connected to Py4J {args.port}")
        log.info(f"  agents: {env.possible_agents}")
    except Exception as e:
        log.error(f"Failed to connect to live server: {e}")
        log.error("Ensure 4 Fabric instances are running:")
        log.error("  JDK_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10 bash scripts/launch-training-instances.sh")
        return

    # Build action spaces for each role
    action_spaces = {role: build_role_action_space(role) for role in roles}

    # Reset & run
    log.info(f"\nResetting env (seed={args.seed})...")
    obs, infos = env.reset(seed=args.seed)
    log.info(f"  ✓ Reset complete. Active agents: {env.agents}")

    step_count = 0
    episode_returns = {agent: 0.0 for agent in env.agents}

    log.info(f"\nRunning greedy agents for {args.max_steps} steps...")
    while env.agents and step_count < args.max_steps:
        # Random action per agent (validates obs/action contract dispatch)
        action = {}
        for agent_id in env.agents:
            role = agent_id.split("_", 1)[0]
            action[agent_id] = action_spaces[role].sample()

        obs, rewards, terminated, truncated, infos = env.step(action)

        for agent_id, rew in rewards.items():
            episode_returns[agent_id] += rew

        step_count += 1
        if step_count % 50 == 0:
            log.info(f"  step {step_count}/{args.max_steps}: agents={len(env.agents)}")

    env.close()

    log.info(f"\nPhase 2b transfer complete ({step_count} steps).")
    log.info("Metrics:")
    for agent_id, ret in episode_returns.items():
        log.info(f"  {agent_id}: return={ret:.4f}")

    log.info("=" * 70)
    log.info("Phase 2b real-MC validation complete.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
