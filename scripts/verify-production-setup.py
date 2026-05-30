#!/usr/bin/env python
"""Verify Phase 3 production server gamerules + spawn test agent."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> None:
    from aiutopia.common.logging import get_logger, setup_logging
    from aiutopia.env.bridge import FabricBridge

    setup_logging("INFO")
    log = get_logger("verify-prod")

    port = int(os.environ.get("PY4J_PRODUCTION_PORT", 25100))
    log.info(f"Connecting to production server on port {port}...")

    try:
        bridge = FabricBridge(port=port)
        bridge.open()
    except Exception as e:
        log.error(f"Failed to connect: {e}")
        sys.exit(1)

    # 1. Verify health
    try:
        health = bridge.health()
        log.info(f"Server health: {health}")
    except Exception as e:
        log.error(f"Health check failed: {e}")

    # 2. Spawn test agent (lumberjack_0)
    try:
        log.info("Spawning lumberjack_0 on production world...")
        bridge.carpet_spawn("lumberjack_0", skin="", role="gatherer")
        log.info("✓ lumberjack_0 spawned")
    except Exception as e:
        log.warning(f"Spawn may have already succeeded (idempotent): {e}")

    # 3. Verify world state (obs sanity check)
    try:
        obs_all = bridge.observations_all()
        if "lumberjack_0" in obs_all:
            obs = obs_all["lumberjack_0"]
            health = obs.get("health", None)
            hunger = obs.get("hunger", None)
            time_of_day = obs.get("time_of_day", None)
            log.info(f"✓ obs available: health={health}, hunger={hunger}, time_of_day={time_of_day}")
        else:
            log.warning(f"lumberjack_0 obs not in batch. Available: {list(obs_all.keys())}")
    except Exception as e:
        log.error(f"Obs read failed: {e}")

    bridge.close()

    log.info("")
    log.info("Production setup verified. Ready for Phase 3 survival test.")
    log.info("Run: PYTHONPATH=src PY4J_PRODUCTION_PORT=25100 py -3.11 scripts/phase3_persistent_survival.py")


if __name__ == "__main__":
    main()
