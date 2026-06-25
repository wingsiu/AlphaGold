#!/bin/bash
# AlphaGold Bot Watchdog — checks and restarts crashed bots every minute via cron.
# Add to crontab:  * * * * * /bin/bash /Users/alpha/AlphaGold/watchdog_bots.sh

cd /Users/alpha/AlphaGold || exit 1
LOG="/Users/alpha/AlphaGold/runtime/watchdog.log"
VENV_PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python"
export PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin:/usr/local/bin:/usr/bin:/bin"

write_log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" >> "$LOG"; }

# ── v15 Hybrid Gold Bot ──────────────────────────────────────────────
if ! pgrep -f "trading_bot_hybrid_v15.py" > /dev/null; then
    write_log "v15 gold bot DOWN — restarting..."
    nohup "$VENV_PYTHON" trading_bot_hybrid_v15.py >> /dev/null 2>&1 &
    write_log "v15 gold bot started (PID $!)"
fi

# ── Oil Live Bot ─────────────────────────────────────────────────────
if ! pgrep -f "oil_live_bot.py" > /dev/null; then
    write_log "oil bot DOWN — restarting..."
    rm -f runtime/oil_live_bot.pid
    IG_REQUEST_CONSUMER=bot_oil nohup "$VENV_PYTHON" oil_live_bot.py >> runtime/oil_live_bot.log 2>&1 &
    write_log "oil bot started (PID $!)"
fi

# ── Mobile API ───────────────────────────────────────────────────────
if ! pgrep -f "mobile_api/server.py" > /dev/null; then
    write_log "mobile API DOWN — restarting..."
    nohup "$VENV_PYTHON" mobile_api/server.py --host 0.0.0.0 --port 8765 >> /dev/null 2>&1 &
    write_log "mobile API started (PID $!)"
fi
