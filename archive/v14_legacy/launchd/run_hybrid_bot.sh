#!/bin/bash
# Launchd wrapper — loads .env then starts the hybrid live bot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
# launchd cannot use login keychain reliably; always login with .env credentials.
export IG_SKIP_KEYRING="${IG_SKIP_KEYRING:-1}"

# After reboot: let the API come up first and avoid stacking CPU with login items.
if [[ "${ALPHAGOLD_BOT_STARTUP_WAIT:-1}" == "1" ]]; then
  PORT="${MOBILE_API_PORT:-8765}"
  for _ in $(seq 1 45); do
    if curl -sf -m 2 "http://127.0.0.1:${PORT}/api/v1/health" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  sleep "${ALPHAGOLD_BOT_SETTLE_SEC:-45}"
fi

exec "$ROOT/.venv/bin/python3" "$ROOT/trading_bot_hybrid_v14.py"
