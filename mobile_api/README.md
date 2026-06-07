# AlphaGold Mobile — iOS / Watch + Mac Mini Setup

## Architecture

```
Mac Mini
├── trading_bot_hybrid_v14.py   → trades + signals → runtime/mobile/alphagold.db
└── mobile_api/server.py        → REST API :8765 for iPhone / Watch

iPhone / Watch app  →  http://<mac-mini-ip>:8765/api/v1/...
```

## 1. Install API dependencies

```bash
cd ~/AlphaGold
.venv/bin/pip install -r requirements-mobile.txt
```

Add to `.env`:
```
MOBILE_API_KEY=choose-a-long-random-secret
MOBILE_API_PORT=8765
```

## 2. Run API (manual)

```bash
MOBILE_API_KEY=your-secret .venv/bin/python3 mobile_api/server.py
```

## 3. Auto-start on Mac Mini boot (launchd)

One-shot install (API + hybrid bot + **health watchdog**):

```bash
cd ~/AlphaGold
./v14/scripts/install_launch_services.sh
```

**API at boot without login** (optional; needs sudo — do not also load the mobile-api LaunchAgent):

```bash
./v14/scripts/install_launch_services.sh --boot-api
```

Manual install (same as before):

```bash
chmod +x v14/scripts/run_mobile_api.sh v14/scripts/run_hybrid_bot.sh v14/scripts/alphagold_watchdog.sh
mkdir -p ~/Library/LaunchAgents
cp v14/scripts/com.alphagold.*.plist ~/Library/LaunchAgents/

launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alphagold.mobile-api.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alphagold.hybrid-bot.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alphagold.watchdog.plist
```

### Reliability (avoid “can’t reach Mac mini”)

| Measure | What it does |
|--------|----------------|
| **Auto-login** (`alpha`) | GUI LaunchAgents start after reboot |
| **Watchdog** (`com.alphagold.watchdog`) | Every 90s: `GET /api/v1/health`; restarts API/bot if down |
| **Bot startup wait** | Hybrid bot waits for API + 45s settle after reboot (less CPU pile-up) |
| **Bot Nice + Background** | Lower scheduler priority so SSH/UI stay responsive |
| **`--boot-api`** | Mobile API as system LaunchDaemon (port 8765 even before user login) |

Check watchdog: `tail -f runtime/launchd_watchdog.log`

iPhone URLs: home `http://192.168.0.4:8765` or `http://192.168.0.57:8765`, away `http://123.203.51.164:8765` (router port-forward **8765** must stay enabled).

**Note:** If the repo lives under `Desktop/`, macOS may block launchd with `Operation not permitted`. Move the project to e.g. `~/AlphaGold` or grant **Full Disk Access** to `/bin/bash` in System Settings → Privacy.

**Do not load both** the mobile-api LaunchAgent and the `--boot-api` LaunchDaemon — they fight for port 8765 and break remote access. Use one mode only.

Diagnostics:

```bash
./v14/scripts/check_remote_access.sh
./v14/scripts/check_remote_desktop.sh   # VNC port 5900; hang / CPU notes
```

Remote Desktop freezes while connected: see `v14/docs/remote_desktop.md`.

Stop a manually started bot before loading the hybrid launchd agent (only one instance).

Check:
```bash
launchctl list | grep alphagold
tail -f runtime/launchd_hybrid_bot.log
tail -f runtime/launchd_mobile_api.log
```

Unload:
```bash
launchctl bootout gui/$(id -u)/com.alphagold.watchdog.plist
launchctl bootout gui/$(id -u)/com.alphagold.hybrid-bot.plist
launchctl bootout gui/$(id -u)/com.alphagold.mobile-api.plist
sudo launchctl bootout system/com.alphagold.mobile-api 2>/dev/null || true
```

## 4. API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/health` | No auth — ping |
| `GET /api/v1/status` | Bot state + today summary |
| `GET /api/v1/signals?minutes=30` | Last N minutes of scored bars |
| `GET /api/v1/trades/today` | All trades this NY trading day |
| `GET /api/v1/compare/today?refresh=false` | Live vs hybrid backtest |

Header (when `MOBILE_API_KEY` set):
```
X-API-Key: your-secret
```

## 5. iOS app

Open `ios/AlphaGoldMonitor/` in Xcode (create new iOS App project, replace generated files with these sources).

Set **API base URL** in app Settings to `http://192.168.x.x:8765` (Mac Mini LAN IP).

For Apple Watch: share `APIClient` via Watch App target; use simplified `WatchStatusView`.

## 6. Compare / backtest note

`/compare/today` runs hybrid backtest for the current trading day (cached 1 hour). First call after market open may take 1–3 minutes.

Trading day = **22:00 UTC** cutoff (= **06:00 HKT**).
