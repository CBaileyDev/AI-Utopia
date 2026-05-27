#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:?usage: $0 <checkpoint-dir>}"
export PYTHONPATH="${PYTHONPATH:-src}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

echo "=== section 5.10 Gate 5: determinism check (writes metrics) ==="
python -m aiutopia.cli.app determinism check \
    --weights "$CKPT" --episodes 3 --py4j-port 25001 \
    || { echo "determinism FAILED"; exit 3; }

echo
echo "=== section 5.10 checklist gates 1-5 ==="
python -m aiutopia.cli.app promote-weights promote \
    --role gatherer --checkpoint "$CKPT" \
    --notes "M1B-Training initial promotion via evaluation-gate script"
