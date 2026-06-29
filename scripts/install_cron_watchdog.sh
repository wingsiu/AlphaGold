#!/bin/bash
# Install minute cron watchdog for v16 bots (backup to launchd KeepAlive).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WATCH="$ROOT/watchdog_bots.sh"
chmod +x "$WATCH"
LINE="* * * * * /bin/bash $WATCH"
MARK="# AlphaGold v16 bot watchdog"
TMP="$(mktemp)"
( crontab -l 2>/dev/null | grep -v "watchdog_bots.sh" | grep -v "$MARK" || true ) > "$TMP"
echo "$MARK" >> "$TMP"
echo "$LINE" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "Installed crontab entry:"
crontab -l | grep -A1 "AlphaGold"
