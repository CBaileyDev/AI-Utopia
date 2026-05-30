#!/usr/bin/env python
"""Diagnostic: test 10-step loop on Phase 3 server."""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from aiutopia.common.logging import get_logger, setup_logging
from aiutopia.env.bridge import FabricBridge
from aiutopia.env.spaces import build_role_action_space

setup_logging("INFO")
log = get_logger("test-loop")

port = int(os.environ.get("PY4J_PRODUCTION_PORT", 25100))
log.info(f"Testing 10-step loop on port {port}")

try:
    bridge = FabricBridge(port=port)
    bridge.open()

    agent = "lumberjack_0"
    space = build_role_action_space("gatherer")

    for step in range(10):
        action = space.sample()
        bridge.dispatch_skill(agent, action, f"{agent}-{step}")
        events = bridge.advance_tick_await_events(timeout_ms=10000)
        obs_all = bridge.observations_all()
        obs = obs_all.get(agent, {})
        health = obs.get("health", [20])[0] if obs.get("health") else 20
        oak_log = obs.get("oak_log", [0])[0] if obs.get("oak_log") else 0
        log.info(f"[{step}] health={health}, oak_log={oak_log}")

    bridge.close()
    log.info("✓ 10-step loop works")
except Exception as e:
    log.error(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
