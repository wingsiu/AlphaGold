#!/bin/bash
# Production gold bot (v16). Legacy v15: trading_bot_hybrid_v15.py
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
exec "$ROOT/.venv/bin/python3" "$ROOT/trading_bot_gold_v16.py"
