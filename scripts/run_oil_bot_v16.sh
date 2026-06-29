#!/bin/bash
# Production oil bot (v16 five-leg portfolio).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export IG_REQUEST_CONSUMER=bot_oil
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
exec "$ROOT/.venv/bin/python3" "$ROOT/trading_bot_oil_v16.py"
