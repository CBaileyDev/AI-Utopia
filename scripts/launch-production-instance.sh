#!/usr/bin/env bash
# Launch Phase 3 production server (persistent survival world).
# Single instance, port 25100, real survival gamerules (difficulty normal, day/night, hunger).
# No setup_training_scene — world is persistent and managed manually.
#
# Usage: JDK_HOME=/path/to/jdk bash scripts/launch-production-instance.sh
# Then verify world setup via /gamerule, spawn agent, run Phase 3 test harness.
set -euo pipefail
: "${JDK_HOME:?must be set}"
export JAVA_HOME="$JDK_HOME"
export PATH="$JDK_HOME/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRODUCTION_DIR="$REPO_ROOT/server-runtime"
PROD_INST="$REPO_ROOT/server-runtime/production"
MOD_JAR="$REPO_ROOT/fabric_mod/build/libs/aiutopia-mod-0.0.0-m2-farmer.jar"
MOD_JAR_NAME="$(basename "$MOD_JAR")"

PY4J_PORT=25100
MC_PORT=25599
HEAP_MAX="${HEAP_MAX:-2g}"

if [[ ! -f "$MOD_JAR" ]]; then
    echo "ERROR: $MOD_JAR not found — run gradlew build first"
    exit 1
fi

# --- helper: is a Py4J port already bound? ---
port_in_use() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -i ":$port" >/dev/null 2>&1 && return 0
    fi
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$port\$" && return 0
    fi
    if command -v netstat >/dev/null 2>&1; then
        netstat -an 2>/dev/null | grep -E "[:.]$port[[:space:]].*LISTEN" -q && return 0
    fi
    return 1
}

# --- helper: is instance running? ---
instance_running() {
    local pidfile="$PROD_INST/production.pid"
    [[ -f "$pidfile" ]] || return 1
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" >/dev/null 2>&1
}

mkdir -p "$PROD_INST/mods" "$PROD_INST/world"

# Setup (one-time)
if [[ ! -f "$PROD_INST/fabric-server-launcher.jar" ]]; then
    echo "[setup] production instance directory"
    cp "$PRODUCTION_DIR/fabric-server-launcher.jar" "$PROD_INST/"
    # Same mod stack as training (no Lithium)
    for m in fabric-api fabric-carpet ferritecore; do
        cp "$PRODUCTION_DIR/mods/$m"-*.jar "$PROD_INST/mods/" 2>/dev/null || true
    done
    cp "$MOD_JAR" "$PROD_INST/mods/$MOD_JAR_NAME"
    echo "eula=true" > "$PROD_INST/eula.txt"

    # Survival world gamerules (NOT peaceful, NOT flat arena)
    cat > "$PROD_INST/server.properties" <<PROPS
server-port=$MC_PORT
online-mode=false
white-list=false
gamemode=survival
difficulty=normal
spawn-protection=0
max-players=5
view-distance=10
simulation-distance=10
level-name=world
motd=AI Utopia production (persistent survival)
PROPS

    echo "[setup] created $PROD_INST with survival world config"
fi

if port_in_use "$PY4J_PORT" || instance_running "$PROD_INST"; then
    echo "[launch] production already RUNNING on MC:$MC_PORT Py4J:$PY4J_PORT"
    echo "To restart: kill the Java process and delete $PROD_INST/production.pid"
    exit 0
fi

echo "[launch] production instance on MC:$MC_PORT Py4J:$PY4J_PORT (heap $HEAP_MAX)"
(
    cd "$PROD_INST"
    nohup java -Daiutopia.py4j.port=$PY4J_PORT \
               -Xms1g -Xmx"$HEAP_MAX" -XX:+UseG1GC \
               -jar fabric-server-launcher.jar nogui \
               > "production.log" 2>&1 &
    echo $! > "production.pid"
)

echo "Production instance launching. Waiting for 'Done (' (120s timeout)..."
WAIT_TIMEOUT=120
waited=0
while (( waited < WAIT_TIMEOUT )); do
    if grep -q "Done (" "$PROD_INST/production.log" 2>/dev/null; then
        echo "[wait] READY (${waited}s)"
        break
    fi
    sleep 2
    waited=$((waited + 2))
done

if (( waited >= WAIT_TIMEOUT )); then
    echo "ERROR: production did not reach Done() — check $PROD_INST/production.log"
    exit 3
fi

cat <<BANNER
============================================================
Production instance READY:
  - MC port: $MC_PORT
  - Py4J port: $PY4J_PORT
  - World: Real survival (persists, difficulty normal)
  - Config: $PROD_INST/server.properties

Next steps:
  1. Connect to MC on port $MC_PORT (localhost:$MC_PORT)
  2. Verify gamerules: /gamerule
     Should show: doDaylightCycle=true, doHungerExhaustion=true, difficulty=normal
  3. Run Phase 3 test: PYTHONPATH=src PY4J_PRODUCTION_PORT=$PY4J_PORT py -3.11 scripts/phase3_persistent_survival.py

============================================================
BANNER
