#!/bin/bash
# Install AlphaGold launchd services (API, hybrid bot, health watchdog).
#
# Usage:
#   ./v14/scripts/install_launch_services.sh              # LaunchAgents (needs login)
#   ./v14/scripts/install_launch_services.sh --boot-api   # API also as system daemon (sudo)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$ROOT/v14/scripts"
AGENTS_DIR="$HOME/Library/LaunchAgents"
GUI_DOMAIN="gui/$(id -u)"
USE_BOOT_API=0

for arg in "$@"; do
  case "$arg" in
    --boot-api) USE_BOOT_API=1 ;;
    -h|--help)
      sed -n '2,6p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

chmod +x "$SCRIPT_DIR/run_mobile_api.sh" "$SCRIPT_DIR/run_hybrid_bot.sh" "$SCRIPT_DIR/alphagold_watchdog.sh"
mkdir -p "$ROOT/runtime" "$AGENTS_DIR"

bootout_if_loaded() {
  local domain="$1"
  local label="$2"
  local plist="$AGENTS_DIR/${label}.plist"
  launchctl bootout "${domain}/${label}" 2>/dev/null || true
  if [[ -f "$plist" ]]; then
    launchctl bootout "${domain}" "$plist" 2>/dev/null || true
  fi
}

bootstrap_agent() {
  local plist="$1"
  local label
  label="$(/usr/libexec/PlistBuddy -c 'Print :Label' "$plist")"
  bootout_if_loaded "$GUI_DOMAIN" "$label"
  cp "$plist" "$AGENTS_DIR/"
  launchctl bootstrap "$GUI_DOMAIN" "$AGENTS_DIR/$(basename "$plist")"
  echo "  loaded $label (LaunchAgent)"
}

echo "AlphaGold launchd install → $ROOT"
echo ""

if [[ "$USE_BOOT_API" -eq 1 ]]; then
  echo "Mode: system daemon for mobile-api (starts at boot, no login required)"
  bootout_if_loaded "$GUI_DOMAIN" "com.alphagold.mobile-api"
  rm -f "$AGENTS_DIR/com.alphagold.mobile-api.plist"
  sudo cp "$SCRIPT_DIR/com.alphagold.mobile-api.daemon.plist" /Library/LaunchDaemons/com.alphagold.mobile-api.plist
  sudo chown root:wheel /Library/LaunchDaemons/com.alphagold.mobile-api.plist
  bootout_if_loaded "system" "com.alphagold.mobile-api"
  sudo launchctl bootstrap system /Library/LaunchDaemons/com.alphagold.mobile-api.plist
  echo "  loaded com.alphagold.mobile-api (LaunchDaemon)"
  echo "  removed GUI LaunchAgent copy (only one API job allowed)"
else
  bootout_if_loaded "system" "com.alphagold.mobile-api"
  if [[ -f /Library/LaunchDaemons/com.alphagold.mobile-api.plist ]]; then
    echo "WARN  Found system daemon plist — boot-api and LaunchAgent conflict."
    echo "      Run: sudo launchctl bootout system/com.alphagold.mobile-api"
    echo "      Then: sudo rm /Library/LaunchDaemons/com.alphagold.mobile-api.plist"
  fi
  bootstrap_agent "$SCRIPT_DIR/com.alphagold.mobile-api.plist"
fi

bootstrap_agent "$SCRIPT_DIR/com.alphagold.hybrid-bot.plist"
bootstrap_agent "$SCRIPT_DIR/com.alphagold.watchdog.plist"

sleep 2
if curl -sf -m 5 "http://127.0.0.1:${MOBILE_API_PORT:-8765}/api/v1/health" >/dev/null; then
  echo ""
  echo "OK  http://127.0.0.1:${MOBILE_API_PORT:-8765}/api/v1/health"
else
  echo ""
  echo "WARN  API not healthy yet — check runtime/launchd_mobile_api.err"
fi

echo ""
launchctl list | grep alphagold || true
echo ""
echo "Done. Watchdog runs every 90s; logs: runtime/launchd_watchdog.log"
