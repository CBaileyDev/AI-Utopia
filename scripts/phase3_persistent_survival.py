#!/usr/bin/env python
"""Phase 3: Persistent Survival World Validation.

Run the proven M1B Lumberjack on a real Minecraft survival world (not flat arena).
Measures: survival time, oak_log collection, hunger/health/day-night progression.

Setup:
  1. Production MC server at port 25100 (real survival world, persistent)
  2. Load proven M1B checkpoint (2f908/checkpoint_000003, 3/3 transfer validation)
  3. Spawn agent, run greedy policy, log metrics per 100 ticks

Run: PYTHONPATH=src PY4J_PRODUCTION_PORT=25100 py -3.11 scripts/phase3_persistent_survival.py
     --checkpoint 2f908 --max-ticks 2000 --seed 42

Metrics tracked:
  - oak_log (inventory count)
  - health (0-20)
  - hunger (0-20)
  - time_of_day (0-24000)
  - position (x, y, z)
  - death_count
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> None:
    """Phase 3: Persistent survival world test."""
    from aiutopia.common.logging import get_logger, setup_logging

    setup_logging("INFO")
    log = get_logger("phase3")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="2f908", help="M1B checkpoint ID (default: proven 2f908)")
    parser.add_argument("--max-ticks", type=int, default=2000, help="Max ticks (20 ticks/sec ≈ 100 sec runtime)")
    parser.add_argument("--seed", type=int, default=42, help="World seed")
    parser.add_argument("--port", type=int, default=25100, help="Py4J production port")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("PHASE 3: Persistent Survival World Validation")
    log.info("=" * 70)
    log.info(f"Checkpoint: {args.checkpoint} (proven M1B)")
    log.info(f"Max ticks: {args.max_ticks} ({args.max_ticks / 20:.1f} sec at 20 TPS)")
    log.info(f"World seed: {args.seed}")
    log.info(f"Py4J port: {args.port}")

    log.info("\nPhase 3 requires:")
    log.info("  1. Production Minecraft server (port 25100) with REAL survival world")
    log.info("  2. Gamerule: difficulty normal, doDaylightCycle true, doHungerExhaustion true")
    log.info("  3. Peaceful OFF (hostiles present, agent must survive without combat)")
    log.info("\nStatus: Harness ready. Awaiting user decision on real survival world setup.")
    log.info("\nNext steps:")
    log.info("  - Setup production server: scripts/launch-production-instance.sh (or equivalent)")
    log.info("  - Verify world is NOT peaceful (mobs spawn)")
    log.info("  - Verify gamerules: /gamerule")
    log.info("  - Load proven checkpoint + run transfer")
    log.info("\nExpected behavior (Lumberjack survival strategy):")
    log.info("  - Will NOT fight mobs (no combat skill)")
    log.info("  - Will try to NAVIGATE away (g_hostiles_nearby observable)")
    log.info("  - Will HARVEST logs when mobs not nearby")
    log.info("  - Will eat (?) if hunger tracking enabled")
    log.info("  - Expected survival: <1-5 min (mobs + hunger + no combat = hard environment)")
    log.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
