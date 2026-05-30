#!/usr/bin/env python
"""Diagnostic: test single step on Phase 3 server."""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from aiutopia.common.logging import get_logger, setup_logging
from aiutopia.env.bridge import FabricBridge
from aiutopia.env.spaces import build_role_action_space

setup_logging("INFO")
log = get_logger("test-step")

port = int(os.environ.get("PY4J_PRODUCTION_PORT", 25100))
log.info(f"Testing single step on port {port}")

try:
    bridge = FabricBridge(port=port)
    bridge.open()
    log.info("Bridge opened")

    # One step
    agent = "lumberjack_0"
    space = build_role_action_space("gatherer")
    action = space.sample()

    log.info(f"Dispatching action to {agent}...")
    bridge.dispatch_skill(agent, action, f"{agent}-0")
    log.info("Dispatch OK")

    log.info("Awaiting events (5s timeout)...")
    events = bridge.advance_tick_await_events(timeout_ms=5000)
    log.info(f"Events received: {len(events)} items")

    # Read obs
    log.info("Reading obs...")
    obs_all = bridge.observations_all()
    log.info(f"Obs keys: {list(obs_all.keys())}")

    bridge.close()
    log.info("✓ Single step works")
except Exception as e:
    log.error(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
