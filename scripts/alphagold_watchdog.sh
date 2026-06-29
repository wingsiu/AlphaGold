#!/bin/bash
# Restart AlphaGold launchd jobs if the mobile API stops responding.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/runtime/launchd_watchdog.log"
PORT="${MOBILE_API_PORT:-8765}"
HEALTH_URL="http://127.0.0.1:${PORT}/api/v1/health"
TIMEOUT_SEC="${ALPHAGOLD_WATCHDOG_TIMEOUT:-5}"

mkdir -p "$ROOT/runtime"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

if curl -sf -m "$TIMEOUT_SEC" "$HEALTH_URL" >/dev/null 2>&1; then
  exit 0
fi

echo "$(ts) WARN health failed ($HEALTH_URL) — restarting services" >>"$LOG"

GUI_DOMAIN="gui/$(id -u)"
restart_job() {
  local domain="$1"
  local label="$2"
  if launchctl print "${domain}/${label}" &>/dev/null; then
    launchctl kickstart -k "${domain}/${label}" 2>>"$LOG" || true
    return 0
  fi
  return 1
}

# Prefer system daemon when --boot-api was used (no GUI login required).
if launchctl print "system/com.alphagold.mobile-api" &>/dev/null; then
  restart_job "system" "com.alphagold.mobile-api"
else
  restart_job "$GUI_DOMAIN" "com.alphagold.mobile-api" || {
    if [[ -f "$HOME/Library/LaunchAgents/com.alphagold.mobile-api.plist" ]]; then
      launchctl bootstrap "$GUI_DOMAIN" "$HOME/Library/LaunchAgents/com.alphagold.mobile-api.plist" 2>>"$LOG" || true
    fi
  }
fi
sleep 2
if ! curl -sf -m "$TIMEOUT_SEC" "$HEALTH_URL" >/dev/null 2>&1; then
  restart_job "$GUI_DOMAIN" "com.alphagold.gold-bot-v16" || {
    if [[ -f "$HOME/Library/LaunchAgents/com.alphagold.gold-bot-v16.plist" ]]; then
      launchctl bootstrap "$GUI_DOMAIN" "$HOME/Library/LaunchAgents/com.alphagold.gold-bot-v16.plist" 2>>"$LOG" || true
    fi
  }
  restart_job "$GUI_DOMAIN" "com.alphagold.oil-bot-v16" || {
    if [[ -f "$HOME/Library/LaunchAgents/com.alphagold.oil-bot-v16.plist" ]]; then
      launchctl bootstrap "$GUI_DOMAIN" "$HOME/Library/LaunchAgents/com.alphagold.oil-bot-v16.plist" 2>>"$LOG" || true
    fi
  }
fi

if curl -sf -m "$TIMEOUT_SEC" "$HEALTH_URL" >/dev/null 2>&1; then
  echo "$(ts) OK API recovered" >>"$LOG"
else
  echo "$(ts) ERROR API still down after restart" >>"$LOG"
  exit 1
fi
