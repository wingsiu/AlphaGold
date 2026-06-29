#!/bin/bash
# AlphaGold Bot Watchdog — checks and restarts crashed v16 bots every minute via cron.
# Install:  ./scripts/install_cron_watchdog.sh

cd /Users/alpha/AlphaGold || exit 1
LOG="/Users/alpha/AlphaGold/runtime/watchdog.log"
VENV_PYTHON="/Users/alpha/AlphaGold/.venv/bin/python3"
[[ -x "$VENV_PYTHON" ]] || VENV_PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
export PATH="/Users/alpha/AlphaGold/.venv/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="/Users/alpha/AlphaGold"
export V14_HYBRID=1

write_log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" >> "$LOG"; }

# ── v16 Gold Bot ─────────────────────────────────────────────────────
if ! pgrep -f "trading_bot_gold_v16.py" > /dev/null; then
    write_log "v16 gold bot DOWN — restarting..."
    nohup "$VENV_PYTHON" trading_bot_gold_v16.py >> runtime/trading_bot_gold_v16.log 2>&1 &
    write_log "v16 gold bot started (PID $!)"
fi

# ── v16 Oil Bot ──────────────────────────────────────────────────────
if ! pgrep -f "trading_bot_oil_v16.py" > /dev/null; then
    write_log "v16 oil bot DOWN — restarting..."
    rm -f runtime/oil_live_bot_v16.pid
    IG_REQUEST_CONSUMER=bot_oil nohup "$VENV_PYTHON" trading_bot_oil_v16.py >> runtime/oil_live_bot_v16.log 2>&1 &
    write_log "v16 oil bot started (PID $!)"
fi

# ── Mobile API ───────────────────────────────────────────────────────
if ! pgrep -f "mobile_api/server.py" > /dev/null; then
    write_log "mobile API DOWN — restarting..."
    nohup "$VENV_PYTHON" mobile_api/server.py --host 0.0.0.0 --port 8765 >> runtime/mobile_api.log 2>&1 &
    write_log "mobile API started (PID $!)"
fi

# Telegram agent runs via LaunchAgent (com.alphagold.telegram-agent) for Cursor keychain auth.
