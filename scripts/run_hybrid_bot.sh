#!/bin/bash
# Production gold bot (v15). Legacy v14 launcher: archive/v14_legacy/launchd/run_hybrid_bot.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export IG_REQUEST_CONSUMER=bot_trade
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
exec "$ROOT/.venv/bin/python3" "$ROOT/trading_bot_hybrid_v15.py"
