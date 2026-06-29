#!/bin/bash
# Production gold bot (v16 hybrid patterns + energetic).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export IG_REQUEST_CONSUMER=bot_trade
export V14_HYBRID=1
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
exec "$ROOT/.venv/bin/python3" "$ROOT/trading_bot_gold_v16.py"
