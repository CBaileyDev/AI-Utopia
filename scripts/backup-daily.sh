#!/usr/bin/env bash
# Daily incremental backup of /var/lib/aiutopia to a date-stamped directory
# on the configured backup target (AIUTOPIA_BACKUP_DIR).
#
# Retention: 7 dailies (script trims old ones).
#
# Usage:
#   AIUTOPIA_ROOT=/var/lib/aiutopia \
#   AIUTOPIA_BACKUP_DIR=/mnt/nas/aiutopia/daily \
#   scripts/backup-daily.sh

set -euo pipefail

: "${AIUTOPIA_ROOT:?must be set}"
: "${AIUTOPIA_BACKUP_DIR:?must be set}"

STAMP="$(date +%Y-%m-%d)"
TARGET="$AIUTOPIA_BACKUP_DIR/$STAMP"
LATEST_LINK="$AIUTOPIA_BACKUP_DIR/latest"

mkdir -p "$AIUTOPIA_BACKUP_DIR"

if [[ -e "$LATEST_LINK" ]]; then
  LINK_ARG=(--link-dest="$LATEST_LINK")
else
  LINK_ARG=()
fi

rsync -a --delete "${LINK_ARG[@]}" "$AIUTOPIA_ROOT/" "$TARGET/"

ln -sfn "$TARGET" "$LATEST_LINK"

# Trim to 7 dailies
ls -1d "$AIUTOPIA_BACKUP_DIR"/20* 2>/dev/null \
  | sort | head -n -7 | xargs -r rm -rf

echo "daily backup → $TARGET"
