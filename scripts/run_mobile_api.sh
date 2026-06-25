#!/bin/bash
# Launchd wrapper — loads .env then starts the mobile API.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
HOST="${MOBILE_API_HOST:-0.0.0.0}"
PORT="${MOBILE_API_PORT:-8765}"
export IG_REQUEST_CONSUMER=mobile_api
HEALTH="http://127.0.0.1:${PORT}/api/v1/health"
# Avoid launchd KeepAlive restart storms when another instance already owns the port.
if curl -sf -m 3 "$HEALTH" >/dev/null 2>&1; then
  exit 0
fi
exec "$ROOT/.venv/bin/python3" "$ROOT/mobile_api/server.py" --host "$HOST" --port "$PORT"
