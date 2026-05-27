#!/usr/bin/env bash
# Launch 4 parallel Fabric training instances for M1B training, wait for them
# to be Done!, then bootstrap each Py4J port via setup_training_scene().
#
# One command -> training-ready cluster.
#
# On Windows under MSYS/Git-Bash, use cp not ln -sf to avoid symlink perms.
#
# Idempotency: re-running the script when servers are already up will SKIP the
# Java launch for any instance whose Py4J port is bound (or whose .pid file
# points at a live process) and will still re-run the bootstrap step. That
# makes this script safe to use as a "fix-up after partial start" tool.
#
# DRY_RUN=1: skip the actual java + py bootstrap invocations (validation mode
# only — used by `bash -x DRY_RUN=1 ...` to exercise control flow without
# touching real servers). Useful when training is already running and we
# only want to syntax-check the script.
set -euo pipefail
: "${JDK_HOME:?must be set}"
export JAVA_HOME="$JDK_HOME"
export PATH="$JDK_HOME/bin:$PATH"

DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRODUCTION_DIR="$REPO_ROOT/server-runtime"
TRAINING_DIR="$REPO_ROOT/server-runtime/training"
MOD_JAR="$REPO_ROOT/fabric_mod/build/libs/aiutopia-mod-0.0.0-m1b.jar"

if [[ ! -f "$MOD_JAR" ]]; then
    echo "ERROR: $MOD_JAR not found — run T22's gradle build first"
    exit 1
fi

# --- helper: is a Py4J port already bound? -----------------------------------
port_in_use() {
    local port="$1"
    # lsof works on MSYS/Git-Bash when available; fall back to netstat/ss.
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

# --- helper: is an instance's recorded pid still alive? ----------------------
instance_running() {
    local inst_dir="$1"
    local pidfile="$inst_dir/instance-$(basename "$inst_dir" | sed 's/instance-//').pid"
    [[ -f "$pidfile" ]] || return 1
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    [[ -n "$pid" ]] || return 1
    # On MSYS, `kill -0` works for native windows PIDs spawned by bash.
    kill -0 "$pid" >/dev/null 2>&1
}

mkdir -p "$TRAINING_DIR"
for i in 1 2 3 4; do
    INST="$TRAINING_DIR/instance-$i"
    PY4J_PORT=$((25000 + i))
    MC_PORT=$((25565 + i))

    if [[ ! -d "$INST" ]]; then
        echo "[setup] creating $INST"
        mkdir -p "$INST/mods" "$INST/world"
        cp "$PRODUCTION_DIR/fabric-server-launcher.jar" "$INST/"
        # N11: lithium removed — its forEachInBox optimization is not
        # concurrent-safe with Carpet fake players modifying entity lists
        # mid-tick, crashes server with ConcurrentModificationException
        # under tick warp. Vanilla server perf is fine for headless training.
        for m in fabric-api fabric-carpet ferritecore; do
            cp "$PRODUCTION_DIR/mods/$m"-*.jar "$INST/mods/" 2>/dev/null || true
        done
        cp "$MOD_JAR" "$INST/mods/aiutopia-mod-0.0.0-m1b.jar"
        echo "eula=true" > "$INST/eula.txt"
        cat > "$INST/server.properties" <<PROPS
server-port=$MC_PORT
online-mode=false
white-list=false
gamemode=survival
difficulty=peaceful
spawn-protection=0
max-players=5
view-distance=10
simulation-distance=10
level-name=world
motd=AI Utopia training instance $i
PROPS
    fi

    if port_in_use "$PY4J_PORT" || instance_running "$INST"; then
        echo "[launch] instance-$i ALREADY RUNNING on MC:$MC_PORT Py4J:$PY4J_PORT — skipping java launch"
        continue
    fi

    echo "[launch] instance-$i on MC:$MC_PORT Py4J:$PY4J_PORT"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] would: cd $INST && nohup java -Daiutopia.py4j.port=$PY4J_PORT ... &"
        continue
    fi
    (
      cd "$INST"
      nohup java -Daiutopia.py4j.port=$PY4J_PORT \
                  -Xms1g -Xmx2g -XX:+UseG1GC \
                  -jar fabric-server-launcher.jar nogui \
                  > "instance-$i.log" 2>&1 &
      echo $! > "instance-$i.pid"
    )
done

echo "All 4 instances launching. Waiting for 'Done (' in each log (90s/instance timeout)..."

# --- wait-for-ready: poll each instance-N.log for the "Done (" line ----------
WAIT_TIMEOUT=90
WAIT_FAILED=0
for i in 1 2 3 4; do
    INST="$TRAINING_DIR/instance-$i"
    LOG="$INST/instance-$i.log"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] would poll $LOG for 'Done (' up to ${WAIT_TIMEOUT}s"
        continue
    fi
    echo -n "[wait] instance-$i: "
    waited=0
    ready=0
    while (( waited < WAIT_TIMEOUT )); do
        if [[ -f "$LOG" ]] && grep -q "Done (" "$LOG" 2>/dev/null; then
            ready=1
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done
    if (( ready == 1 )); then
        echo "READY (${waited}s)"
    else
        echo "TIMEOUT after ${WAIT_TIMEOUT}s — check $LOG"
        WAIT_FAILED=1
    fi
done

if (( WAIT_FAILED == 1 )); then
    echo "ERROR: one or more instances did not reach Done() — aborting bootstrap."
    exit 3
fi

# --- bootstrap: setup_training_scene on each port via FabricBridge -----------
echo "[bootstrap] invoking setup_training_scene() on ports 25001-25004..."
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] would: PYTHONPATH=src py -3.11 -c '<bootstrap snippet>'"
else
    (
      cd "$REPO_ROOT"
      PYTHONPATH=src py -3.11 -c "
from aiutopia.env.bridge import FabricBridge
ok_all = True
for port in (25001, 25002, 25003, 25004):
    try:
        with FabricBridge(port=port) as b:
            h = b.health()
            ok = b.setup_training_scene()
            print(f'[bootstrap] port {port}: health={h} setup={ok}')
            ok_all = ok_all and (h == 'ok') and ok
    except Exception as e:
        print(f'[bootstrap] port {port}: ERROR {e}')
        ok_all = False
if not ok_all:
    exit(2)
"
    )
fi

cat <<'BANNER'
============================================================
4 training instances are READY:
  - MC ports: 25566-25569
  - Py4J ports: 25001-25004
  - Arena: flat grass at Y=65, log ring at Y=66, tick rate 300
You can now launch training:
  PYTHONPATH=src py -3.11 scripts/train.py --milestone M1 ...
============================================================
BANNER
