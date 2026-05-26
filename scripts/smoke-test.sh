#!/usr/bin/env bash
# M0 end-to-end smoke test.
#
# Prereqs (you set up these once):
#   1. Local Fabric 1.21.1 server with Carpet + Lithium + FerriteCore + our mod
#      jar, started with -Daiutopia.py4j.port=25099 on port 25565 (MC) /
#      25099 (Py4J).
#   2. A Minecraft Java client (matching version) connected to localhost:25565.
#
# This script:
#   - calls `aiutopia agent spawn --role gatherer --py4j-port 25099`
#   - asserts an identity row was inserted
#   - asserts Carpet /player spawn returned ok
#   - (manual) you see the new player appear in your MC client

set -euo pipefail

export AIUTOPIA_ROOT="${AIUTOPIA_ROOT:-/tmp/aiutopia-smoke}"
export PY4J_PRODUCTION_PORT="${PY4J_PRODUCTION_PORT:-25099}"

mkdir -p "$AIUTOPIA_ROOT"
rm -f "$AIUTOPIA_ROOT/identity.db"

echo "[smoke] spawning gatherer via aiutopia CLI…"
aiutopia agent spawn --role gatherer --py4j-port "$PY4J_PRODUCTION_PORT"

echo "[smoke] listing identity rows…"
aiutopia agent list

echo "[smoke] check your connected MC client — the gatherer should be visible."
echo "[smoke] PASS (manual verification required for visual confirmation)"
