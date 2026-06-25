#!/usr/bin/env python3
"""
AlphaGold Mobile API — serves iPhone / Apple Watch app on local network.

Usage:
  .venv/bin/python3 mobile_api/server.py
  MOBILE_API_KEY=your-secret .venv/bin/python3 mobile_api/server.py --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

STATE_PATH = PROJECT_ROOT / "runtime" / "trading_bot_hybrid_v15_state.json"

from mobile_api.compare import compare_today  # noqa: E402
from mobile_api.journal import SignalJournal  # noqa: E402

app = FastAPI(title="AlphaGold Mobile API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

journal = SignalJournal()


@app.on_event("startup")
def _startup_warm_caches() -> None:
    """Warm caches only when bot bridge files are missing (avoid IG rate limits)."""
    from mobile_api.market_price import _read_bot_bridge as _gold_bridge

    bridge_missing = not (
        (PROJECT_ROOT / "runtime" / "live_price.json").exists()
        and _gold_bridge()
    )
    if not bridge_missing:
        return
    import threading
    import time

    from mobile_api.market_price import warm_gold_price_cache

    def _run() -> None:
        warm_gold_price_cache()
        time.sleep(3)

    threading.Thread(target=_run, daemon=True).start()


def _load_hybrid_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("MOBILE_API_KEY", "").strip()
    if not expected:
        return  # dev mode — no key required
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/api/v1/health")
def health():
    return {"ok": True}


@app.get("/api/v1/status", dependencies=[Depends(verify_api_key)])
def status():
    state = _load_hybrid_state()
    snap = journal.status_snapshot(state)
    return snap


@app.get("/api/v1/signals", dependencies=[Depends(verify_api_key)])
def signals(minutes: int = Query(default=30, ge=1, le=240)):
    from mobile_api.trading_window import get_trading_window_status

    view = journal.resolve_signals_view(minutes)
    view["trading_window"] = get_trading_window_status()
    return view


@app.get("/api/v1/trades/today", dependencies=[Depends(verify_api_key)])
def trades_today():
    view = journal.resolve_trades_view(hybrid_state=_load_hybrid_state())
    return {
        "summary": view["summary"],
        "trades": view["trades"],
        "trading_day": view["trading_day"],
        "trading_day_start_utc": view["trading_day_start_utc"],
        "is_fallback": view["is_fallback"],
        "source": view["source"],
    }


@app.get("/api/v1/compare/today", dependencies=[Depends(verify_api_key)])
def compare(refresh: bool = Query(default=False)):
    """Live journal vs hybrid backtest for current trading day (22:00 UTC / 06:00 HKT)."""
    cached = compare_today(refresh_backtest=refresh)
    return cached


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("MOBILE_API_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("MOBILE_API_PORT", "8765")))
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
