#!/usr/bin/env bash
# Launch 4 parallel Fabric training instances for M1B training.
# On Windows under MSYS/Git-Bash, use cp not ln -sf to avoid symlink perms.
set -euo pipefail
: "${JDK_HOME:?must be set}"
export JAVA_HOME="$JDK_HOME"
export PATH="$JDK_HOME/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRODUCTION_DIR="$REPO_ROOT/server-runtime"
TRAINING_DIR="$REPO_ROOT/server-runtime/training"
MOD_JAR="$REPO_ROOT/fabric_mod/build/libs/aiutopia-mod-0.0.0-m1b.jar"

if [[ ! -f "$MOD_JAR" ]]; then
    echo "ERROR: $MOD_JAR not found — run T22's gradle build first"
    exit 1
fi

mkdir -p "$TRAINING_DIR"
for i in 1 2 3 4; do
    INST="$TRAINING_DIR/instance-$i"
    PY4J_PORT=$((25000 + i))
    MC_PORT=$((25565 + i))

    if [[ ! -d "$INST" ]]; then
        echo "[setup] creating $INST"
        mkdir -p "$INST/mods" "$INST/world"
        cp "$PRODUCTION_DIR/fabric-server-launcher.jar" "$INST/"
        for m in fabric-api fabric-carpet lithium ferritecore; do
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

    echo "[launch] instance-$i on MC:$MC_PORT Py4J:$PY4J_PORT"
    (
      cd "$INST"
      nohup java -Daiutopia.py4j.port=$PY4J_PORT \
                  -Xms1g -Xmx2g -XX:+UseG1GC \
                  -jar fabric-server-launcher.jar nogui \
                  > "instance-$i.log" 2>&1 &
      echo $! > "instance-$i.pid"
    )
done

echo "All 4 instances launching. Wait for 'Done (X.Xs)!' in each log before training."
