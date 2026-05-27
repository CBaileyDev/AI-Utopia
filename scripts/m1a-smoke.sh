#!/usr/bin/env bash
# M1-Pipeline end-to-end smoke test.
#
# Prereqs:
#   1. Fabric server running on port 25565 with our mod + Carpet + Lithium + FerriteCore
#   2. -Daiutopia.py4j.port=25099 system property set on server launch
#   3. (Optional) MC client connected to localhost:25565 for visual confirmation
#
# This script:
#   - spawns a gatherer agent
#   - drives NAVIGATE forward (dx=0.5 = 16 blocks)
#   - drives HARVEST oak_log (scalar=0.1 = ~6 blocks cap)
#   - drives WAIT
#   - prints each completion event

set -euo pipefail

export AIUTOPIA_ROOT="${AIUTOPIA_ROOT:-/tmp/aiu-m1a-smoke}"
export PYTHONPATH="${PYTHONPATH:-src}"
PORT="${PY4J_PRODUCTION_PORT:-25099}"

rm -rf "$AIUTOPIA_ROOT"

echo "[1/4] spawning gatherer…"
SPAWN_OUT=$(python -m aiutopia.cli.app agent spawn --role gatherer --py4j-port "$PORT")
echo "$SPAWN_OUT"
AGENT=$(echo "$SPAWN_OUT" | awk '/identity: spawned/ {print $3}')
echo "[*] agent name: $AGENT"
sleep 1.5

echo "[2/4] NAVIGATE forward (dx=0.5)…"
python -m aiutopia.cli.app agent drive \
    --agent-name "$AGENT" --skill 0 \
    --dx 0.5 --dy 0.0 --dz 0.0 --scalar 0.5 \
    --py4j-port "$PORT" --timeout-ms 30000

echo "[3/4] HARVEST oak_log (target=0, scalar=0.1)…"
python -m aiutopia.cli.app agent drive \
    --agent-name "$AGENT" --skill 1 --target 0 --scalar 0.1 \
    --py4j-port "$PORT" --timeout-ms 60000

echo "[4/4] WAIT 1 sec…"
python -m aiutopia.cli.app agent drive \
    --agent-name "$AGENT" --skill 4 --scalar 0.1 \
    --py4j-port "$PORT" --timeout-ms 30000

echo "smoke PASS — check connected MC client to verify agent moved + chopped"
