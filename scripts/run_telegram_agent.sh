#!/bin/bash
# Start Telegram ↔ Cursor agent with runtime state under AlphaGold (not Desktop).
set -euo pipefail
CLAW="/Users/alpha/Desktop/python/cursor-claw"
RUNTIME="/Users/alpha/AlphaGold/runtime/telegram"
export PATH="${CLAW}/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"
export TELEGRAM_BOT_RUNTIME="${RUNTIME}"
mkdir -p "${RUNTIME}/logs"
exec /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 "${CLAW}/telegram-bot/agent_bot.py"
