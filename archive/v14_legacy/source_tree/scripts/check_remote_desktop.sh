#!/bin/bash
# macOS Screen Sharing / VNC / Remote Desktop diagnostics (not the mobile API).
set -uo pipefail

LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
PUBLIC_IP="${ALPHAGOLD_PUBLIC_IP:-123.203.51.164}"

# macOS: -G = connect timeout (seconds). Avoid long hangs on closed ports / NAT hairpin.
port_open() {
  local host="$1" port="$2" secs="${3:-2}"
  if [[ "$(uname -s)" == Darwin ]]; then
    nc -z -G "$secs" "$host" "$port" 2>/dev/null
  else
    nc -z -w "$secs" "$host" "$port" 2>/dev/null
  fi
}

echo "=== macOS Remote Desktop (Screen Sharing / VNC) ==="
echo "LAN IP: ${LAN_IP:-unknown}   (mobile API uses port 8765 — different service)"
echo ""

echo "--- Screen Sharing service ---"
if pgrep -x screensharingd >/dev/null; then
  echo "OK  screensharingd is running"
else
  echo "FAIL  screensharingd not running — turn on Screen Sharing in System Settings"
fi
if launchctl print system/com.apple.screensharing 2>/dev/null | grep -q "state = running"; then
  echo "OK  launchd com.apple.screensharing = running"
else
  echo "WARN  com.apple.screensharing launch job not running"
fi
echo ""

echo "--- VNC port 5900 (required for most Remote Desktop clients) ---"
if port_open 127.0.0.1 5900 2; then
  echo "OK  localhost:5900 listening"
  VNC_OK=1
else
  echo "FAIL  nothing listening on port 5900 — Remote Desktop cannot connect"
  echo "      Fix: System Settings → General → Sharing → Screen Sharing → ON"
  echo "      Then allow user '$(whoami)' (or All users), then re-run this script"
  VNC_OK=0
fi
if [[ "$VNC_OK" == 1 && -n "$LAN_IP" ]] && port_open "$LAN_IP" 5900 2; then
  echo "OK  ${LAN_IP}:5900 (LAN)"
elif [[ "$VNC_OK" == 0 ]]; then
  echo "      (LAN/public skipped until localhost:5900 is OK)"
  echo "      Run: ~/AlphaGold/scripts/fix_remote_desktop.sh  (admin password)"
fi
if [[ "$VNC_OK" == 1 && "${CHECK_PUBLIC_RD:-0}" == 1 ]]; then
  if port_open "$PUBLIC_IP" 5900 2; then
    echo "OK  ${PUBLIC_IP}:5900 (router port-forward)"
  else
    echo "WARN  ${PUBLIC_IP}:5900 not reachable — add router TCP 5900 → ${LAN_IP}"
  fi
elif [[ "$VNC_OK" == 1 ]]; then
  echo "      Away: forward TCP 5900 → ${LAN_IP} (set CHECK_PUBLIC_RD=1 to probe public IP)"
fi
echo ""

echo "--- Third-party VNC clients (Microsoft Remote Desktop, TightVNC, etc.) ---"
legacy="$(defaults read /Library/Preferences/com.apple.RemoteManagement VNCLegacyConnectionsEnabled 2>/dev/null || echo "unknown")"
echo "VNCLegacyConnectionsEnabled = $legacy"
if [[ "$legacy" == "0" || "$legacy" == "false" ]]; then
  echo "WARN  Legacy VNC is OFF — many Remote Desktop apps will fail"
  echo "      Apple Screen Sharing app / Finder vnc:// may still work on LAN"
  echo "      To allow standard VNC (needs admin password):"
  echo "        sudo defaults write /Library/Preferences/com.apple.RemoteManagement VNCLegacyConnectionsEnabled -bool true"
  echo "        sudo launchctl kickstart -k system/com.apple.screensharing"
fi
echo ""

echo "--- SSH (works without VNC) ---"
if port_open 127.0.0.1 22 2; then
  echo "OK  SSH port 22 — ssh $(whoami)@${LAN_IP:-<mac-ip>}"
  if [[ "$VNC_OK" == 1 ]]; then
    echo "    Tunnel: ssh -L 5900:127.0.0.1:5900 $(whoami)@${PUBLIC_IP}  → VNC to localhost:5900"
  fi
else
  echo "FAIL  SSH off — enable Remote Login in Sharing"
fi
echo ""

echo "--- Mobile API (separate) ---"
if curl -sf -m 2 "http://127.0.0.1:8765/api/v1/health" >/dev/null; then
  echo "OK  mobile API :8765 (AlphaGold iPhone app)"
else
  echo "WARN  mobile API not up — see check_remote_access.sh"
fi
echo ""

echo "--- Hang while connected? (CPU / RAM) ---"
load="$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')"
cores="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
ram_gb="$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1e9}')"
echo "  Load avg: ${load:-?}  cores: ${cores}  RAM: ${ram_gb:-?} GB"
if [[ -n "${load:-}" ]] && awk -v l="$load" -v c="$cores" 'BEGIN { exit (l > c * 1.5) ? 0 : 1 }'; then
  echo "  WARN  Mac is busy — VNC often freezes (quit Cursor on mini while remoting in)"
fi
top_cpu="$(ps -axo %cpu,comm 2>/dev/null | sort -nrk 1 | head -4 | tail -3 | awk '{printf "    %s%% %s\n", $1, $2}')"
[[ -n "$top_cpu" ]] && echo "$top_cpu"
if pgrep -x replayd >/dev/null; then
  echo "  WARN  replayd (screen recording) running — disable if not needed"
fi
if pgrep -f "Cursor" >/dev/null && port_open 127.0.0.1 5900 2; then
  echo "  TIP   Cursor + active VNC on ${ram_gb}GB mini → use SSH for dev, iPhone app for monitoring"
fi
echo "  See: docs/ops/remote_desktop.md"
