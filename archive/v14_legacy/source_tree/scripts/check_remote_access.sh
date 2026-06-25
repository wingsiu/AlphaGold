#!/bin/bash
# Quick remote-access diagnostics (SSH + mobile API).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${MOBILE_API_PORT:-8765}"
LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
PUBLIC_IP="${ALPHAGOLD_PUBLIC_IP:-123.203.51.164}"

echo "=== AlphaGold remote access ==="
echo "Host: $(hostname)  LAN: ${LAN_IP:-unknown}  Port: $PORT"
echo ""

echo "--- SSH (Remote Login) ---"
if nc -z -w 2 127.0.0.1 22 2>/dev/null; then
  echo "OK  SSH listening on port 22 (localhost)"
else
  echo "FAIL  SSH not listening — enable System Settings → General → Sharing → Remote Login"
fi
if command -v systemsetup >/dev/null 2>&1; then
  systemsetup -getremotelogin 2>/dev/null || true
fi
echo ""

echo "--- Mobile API ---"
if curl -sf -m 3 "http://127.0.0.1:${PORT}/api/v1/health" >/dev/null; then
  echo "OK  http://127.0.0.1:${PORT}/api/v1/health"
else
  echo "FAIL  API not responding locally"
fi
if [[ -n "$LAN_IP" ]] && curl -sf -m 3 "http://${LAN_IP}:${PORT}/api/v1/health" >/dev/null; then
  echo "OK  http://${LAN_IP}:${PORT}/api/v1/health (LAN)"
else
  echo "WARN  LAN URL not reachable (Wi‑Fi / firewall / API down)"
fi
if curl -sf -m 5 "http://${PUBLIC_IP}:${PORT}/api/v1/health" >/dev/null 2>&1; then
  echo "OK  http://${PUBLIC_IP}:${PORT}/api/v1/health (port-forward)"
else
  echo "WARN  Public URL not reachable (router forward ${PORT} or ISP CGNAT)"
fi
echo ""

echo "--- launchd (avoid duplicate API jobs) ---"
launchctl print "system/com.alphagold.mobile-api" 2>/dev/null | grep -E "state =|active count" || echo "  system daemon: not loaded"
launchctl print "gui/$(id -u)/com.alphagold.mobile-api" 2>/dev/null | grep -E "state =|active count" || echo "  GUI agent: not loaded"
if launchctl print "system/com.alphagold.mobile-api" &>/dev/null \
  && launchctl print "gui/$(id -u)/com.alphagold.mobile-api" &>/dev/null; then
  echo "FAIL  BOTH system daemon and GUI agent loaded — causes port ${PORT} bind loops"
  echo "      Fix: ./scripts/install_launch_services.sh   OR   --boot-api (not both)"
fi
launchctl print "gui/$(id -u)/com.alphagold.watchdog" 2>/dev/null | grep -E "state =|active count" || echo "  watchdog: not loaded"
echo ""

ERR="$ROOT/runtime/launchd_mobile_api.err"
if [[ -f "$ERR" ]]; then
  n="$(grep -c "address already in use" "$ERR" 2>/dev/null || echo 0)"
  if [[ "$n" -gt 0 ]]; then
    echo "WARN  $n 'address already in use' lines in launchd_mobile_api.err (duplicate API starters)"
  fi
fi

echo ""
echo "iPhone (home): http://${LAN_IP:-<lan-ip>}:${PORT}"
echo "iPhone (away): http://${PUBLIC_IP}:${PORT}  + router TCP ${PORT} forward"
echo ""
echo "Screen Sharing / Remote Desktop is separate — run: ./scripts/check_remote_desktop.sh"
