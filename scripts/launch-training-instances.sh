#!/usr/bin/env bash
# Launch N parallel Fabric training instances for M1B training, wait for them
# to be Done!, then bootstrap each Py4J port via setup_training_scene().
#
# One command -> training-ready cluster.
#
# Count is parameterized: NUM_INSTANCES (default 4). Instance i uses
# Py4J port 25000+i and MC port 25565+i. e.g. NUM_INSTANCES=12 -> Py4J
# 25001-25012, MC 25566-25577. (P0: scaled from a hardcoded 4 to N so the
# now-single-attractor reward can use more on-task rollouts per PPO update.)
#
# On Windows under MSYS/Git-Bash, use cp not ln -sf to avoid symlink perms.
#
# Idempotency: re-running the script when servers are already up will SKIP the
# Java launch for any instance whose Py4J port is bound (or whose .pid file
# points at a live process) and will still re-run the bootstrap step. That
# makes this script safe to use as a "fix-up after partial start" tool.
#
# DRY_RUN=1: skip the actual java + py bootstrap invocations (validation mode).
set -euo pipefail
: "${JDK_HOME:?must be set}"
export JAVA_HOME="$JDK_HOME"
export PATH="$JDK_HOME/bin:$PATH"

DRY_RUN="${DRY_RUN:-0}"
NUM_INSTANCES="${NUM_INSTANCES:-4}"
HEAP_MAX="${HEAP_MAX:-3g}"          # per-instance -Xmx (tiny forceloaded arena)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRODUCTION_DIR="$REPO_ROOT/server-runtime"
TRAINING_DIR="$REPO_ROOT/server-runtime/training"
# P0: deploy the Phase-0 jar (64-reachable-log arena + reward/term fixes), NOT
# the old m1b jar the script originally hardcoded.
MOD_JAR="$REPO_ROOT/fabric_mod/build/libs/aiutopia-mod-0.0.0-m1c-p0.jar"
MOD_JAR_NAME="$(basename "$MOD_JAR")"

if [[ ! -f "$MOD_JAR" ]]; then
    echo "ERROR: $MOD_JAR not found — run the gradle build first"
    exit 1
fi

# --- helper: is a Py4J port already bound? -----------------------------------
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

# --- helper: is an instance's recorded pid still alive? ----------------------
instance_running() {
    local inst_dir="$1"
    local pidfile="$inst_dir/instance-$(basename "$inst_dir" | sed 's/instance-//').pid"
    [[ -f "$pidfile" ]] || return 1
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" >/dev/null 2>&1
}

mkdir -p "$TRAINING_DIR"
for i in $(seq 1 "$NUM_INSTANCES"); do
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
        cp "$MOD_JAR" "$INST/mods/$MOD_JAR_NAME"
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

    echo "[launch] instance-$i on MC:$MC_PORT Py4J:$PY4J_PORT (heap $HEAP_MAX)"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] would: cd $INST && nohup java -Daiutopia.py4j.port=$PY4J_PORT ... &"
        continue
    fi
    (
      cd "$INST"
      # Per-instance heap. The arena is a tiny forceloaded 33x33 region and P0
      # success-termination keeps episodes short (end at 64 logs), so the
      # ItemEntity/chunk buildup that drove the old 4g bump is much reduced;
      # 3g x N keeps total RAM modest while scaling instance count.
      nohup java -Daiutopia.py4j.port=$PY4J_PORT \
                  -Xms2g -Xmx"$HEAP_MAX" -XX:+UseG1GC \
                  -jar fabric-server-launcher.jar nogui \
                  > "instance-$i.log" 2>&1 &
      echo $! > "instance-$i.pid"
    )
done

echo "All $NUM_INSTANCES instances launching. Waiting for 'Done (' in each log (90s/instance timeout)..."

# --- wait-for-ready: poll each instance-N.log for the "Done (" line ----------
WAIT_TIMEOUT=90
WAIT_FAILED=0
for i in $(seq 1 "$NUM_INSTANCES"); do
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
echo "[bootstrap] invoking setup_training_scene() on ports 25001-$((25000 + NUM_INSTANCES))..."
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] would: PYTHONPATH=src py -3.11 -c '<bootstrap snippet>'"
else
    (
      cd "$REPO_ROOT"
      PYTHONPATH=src AIUTOPIA_NUM="$NUM_INSTANCES" py -3.11 -c "
import os
from aiutopia.env.bridge import FabricBridge
n = int(os.environ['AIUTOPIA_NUM'])
ok_all = True
for port in range(25001, 25001 + n):
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

cat <<BANNER
============================================================
$NUM_INSTANCES training instances are READY:
  - MC ports: 25566-$((25565 + NUM_INSTANCES))
  - Py4J ports: 25001-$((25000 + NUM_INSTANCES))
  - Arena: flat grass at Y=65, 64 oak_log flat grid at Y=66, tick rate 60
Launch training with matching runner count:
  PYTHONPATH=src py -3.11 scripts/train.py --milestone M1 --num-env-runners $NUM_INSTANCES ...
============================================================
BANNER
