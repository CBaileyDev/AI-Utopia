#!/usr/bin/env bash
# Weekly full tarball of /var/lib/aiutopia.
# Retention: 4 weeklies.
#
# Usage:
#   AIUTOPIA_ROOT=/var/lib/aiutopia \
#   AIUTOPIA_BACKUP_DIR=/mnt/nas/aiutopia/weekly \
#   scripts/backup-weekly.sh

set -euo pipefail

: "${AIUTOPIA_ROOT:?must be set}"
: "${AIUTOPIA_BACKUP_DIR:?must be set}"

STAMP="$(date +%Y-W%V)"
TARGET="$AIUTOPIA_BACKUP_DIR/aiutopia-$STAMP.tar.zst"

mkdir -p "$AIUTOPIA_BACKUP_DIR"

tar --use-compress-program=zstd \
    -cf "$TARGET" \
    -C "$(dirname "$AIUTOPIA_ROOT")" "$(basename "$AIUTOPIA_ROOT")"

# Trim to 4 weeklies
ls -1t "$AIUTOPIA_BACKUP_DIR"/aiutopia-*.tar.zst 2>/dev/null \
  | tail -n +5 | xargs -r rm -f

echo "weekly backup → $TARGET"
