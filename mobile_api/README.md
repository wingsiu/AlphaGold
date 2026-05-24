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
cd /Users/alpha/Desktop/python/AlphaGold
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

Edit paths in plists if your user/home differs, then:

```bash
cp v14/scripts/com.alphagold.hybrid-bot.plist ~/Library/LaunchAgents/
cp v14/scripts/com.alphagold.mobile-api.plist ~/Library/LaunchAgents/

launchctl load ~/Library/LaunchAgents/com.alphagold.hybrid-bot.plist
launchctl load ~/Library/LaunchAgents/com.alphagold.mobile-api.plist
```

Check:
```bash
launchctl list | grep alphagold
tail -f runtime/launchd_hybrid_bot.log
tail -f runtime/launchd_mobile_api.log
```

Unload:
```bash
launchctl unload ~/Library/LaunchAgents/com.alphagold.hybrid-bot.plist
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

Trading day = **NY 17:00 cutoff** (same as bot).
