"""Latest gold price for mobile API (cached, non-blocking)."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

_CACHE: dict[str, Any] = {}
_CACHE_TS = 0.0
_LOCK = threading.Lock()
_FETCH_LOCK = threading.Lock()
TTL_SECONDS = 30


def _fetch_and_cache() -> dict[str, Any]:
    from ig_scripts.ig_data_api import API_CONFIG, IGService, Price, fetch_market_snapshot

    service = IGService(
        api_key=API_CONFIG["api_key"],
        username=API_CONFIG["username"],
        password=API_CONFIG["password"],
        base_url=API_CONFIG["base_url"],
    )
    snap = fetch_market_snapshot(service, Price.Gold)
    mid = snap.get("mid")
    payload = {
        "close": mid,
        "bid": snap.get("bid"),
        "offer": snap.get("offer"),
        "market_status": snap.get("market_status"),
        "updated_at_utc": snap.get("update_time_utc") or snap.get("fetch_time_utc"),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with _LOCK:
        global _CACHE, _CACHE_TS
        _CACHE = payload
        _CACHE_TS = time.time()
    return payload


def refresh_gold_price_background() -> None:
    if not _FETCH_LOCK.acquire(blocking=False):
        return

    def _run() -> None:
        global _CACHE, _CACHE_TS
        try:
            _fetch_and_cache()
        except Exception as exc:
            with _LOCK:
                if _CACHE:
                    _CACHE = {**_CACHE, "stale": True, "error": str(exc)}
                else:
                    _CACHE = {"status": "error", "error": str(exc)}
                _CACHE_TS = time.time()
        finally:
            _FETCH_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()


def warm_gold_price_cache() -> None:
    refresh_gold_price_background()


def _read_bot_bridge() -> dict[str, Any] | None:
    """Read live price from the bot's bridge file (zero extra IG calls)."""
    from pathlib import Path

    bridge_path = Path(__file__).resolve().parent.parent / "runtime" / "live_price.json"
    if not bridge_path.exists():
        return None
    try:
        import json

        data = json.loads(bridge_path.read_text(encoding="utf-8"))
        return data
    except Exception:
        return None


def get_gold_price_summary(*, refresh: bool = False) -> dict[str, Any]:
    # Prefer bot bridge (always fresh, zero IG calls)
    bridge = _read_bot_bridge()
    if bridge and bridge.get("close"):
        now = time.time()
        bridge["cached"] = True
        bridge["market_status"] = "TRADEABLE"
        try:
            from datetime import datetime, timezone

            fetched = bridge.get("fetched_at_utc")
            if fetched:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(fetched)).total_seconds()
                bridge["age_seconds"] = int(age)
            else:
                bridge["age_seconds"] = 0
        except Exception:
            bridge["age_seconds"] = 0
        return bridge

    now = time.time()
    with _LOCK:
        cached = dict(_CACHE) if _CACHE else {}
        age = now - _CACHE_TS if _CACHE_TS else None

    if cached and not refresh and age is not None and age < TTL_SECONDS:
        cached["cached"] = True
        cached["age_seconds"] = int(age)
        return cached

    if cached and not refresh:
        cached["cached"] = True
        cached["stale"] = True
        if age is not None:
            cached["age_seconds"] = int(age)
        refresh_gold_price_background()
        return cached

    # Cold cache with explicit refresh requested, or completely empty cache.
    # Do a synchronous fetch so callers never get {"status": "loading"}.
    try:
        payload = _fetch_and_cache()
        payload["cached"] = True
        payload["age_seconds"] = 0
        return payload
    except Exception as exc:
        if cached:
            cached["cached"] = True
            cached["stale"] = True
            return cached
        return {"status": "error", "error": str(exc)}
