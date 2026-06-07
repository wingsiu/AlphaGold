#!/bin/bash
# Restore Screen Sharing / VNC (port 5900). Run on the Mac mini (needs admin password).
set -euo pipefail

port_open() {
  nc -z -G 2 127.0.0.1 5900 2>/dev/null
}

echo "=== Fix Remote Desktop (VNC port 5900) ==="
if port_open; then
  echo "OK  port 5900 already open — nothing to do"
  exit 0
fi

echo "Port 5900 closed — restarting Screen Sharing…"

# Enable Screen Sharing at launchd level (same as Sharing toggle ON).
if [[ ! -f /Library/Preferences/com.apple.ScreenSharing.launchd ]] \
  || [[ "$(cat /Library/Preferences/com.apple.ScreenSharing.launchd 2>/dev/null)" != "enabled" ]]; then
  echo "enabled" | sudo tee /Library/Preferences/com.apple.ScreenSharing.launchd >/dev/null
  echo "  wrote ScreenSharing.launchd = enabled"
fi

sudo launchctl kickstart -k system/com.apple.screensharing

KICKSTART="/System/Library/CoreServices/RemoteManagement/ARDAgent.app/Contents/Resources/kickstart"
if [[ -x "$KICKSTART" ]]; then
  sudo "$KICKSTART" -activate -configure -access -on -restart -agent -privs -all 2>/dev/null || true
fi

# Legacy VNC for Microsoft Remote Desktop etc.
sudo defaults write /Library/Preferences/com.apple.RemoteManagement VNCLegacyConnectionsEnabled -bool true 2>/dev/null || true
sudo launchctl kickstart -k system/com.apple.screensharing 2>/dev/null || true

sleep 2
if port_open; then
  echo "OK  port 5900 is open — try Remote Desktop again"
  LAN="$(ipconfig getifaddr en0 2>/dev/null || true)"
  [[ -n "$LAN" ]] && echo "    LAN: vnc://${LAN}:5900"
  echo "    Away: forward router TCP 5900 → ${LAN:-Mac} (8765 is only for the iPhone app)"
else
  echo "FAIL  still no listener on 5900"
  echo "  Open System Settings → General → Sharing → Screen Sharing → ON"
  echo "  Allow user $(whoami), then run this script again."
  exit 1
fi
