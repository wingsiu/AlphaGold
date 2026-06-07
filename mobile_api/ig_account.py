"""Fetch IG account summary for mobile API (cached, non-blocking on status polls)."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

_CACHE: dict[str, Any] = {}
_CACHE_TS = 0.0
_LOCK = threading.Lock()
_FETCH_LOCK = threading.Lock()
TTL_SECONDS = 60


def _fetch_and_cache() -> dict[str, Any]:
    from ig_scripts.ig_data_api import API_CONFIG, IGService, fetch_primary_account_summary

    service = IGService(
        api_key=API_CONFIG["api_key"],
        username=API_CONFIG["username"],
        password=API_CONFIG["password"],
        base_url=API_CONFIG["base_url"],
    )
    summary = fetch_primary_account_summary(service)
    payload = {
        **summary,
        "cached": False,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with _LOCK:
        global _CACHE, _CACHE_TS
        _CACHE = payload
        _CACHE_TS = time.time()
    return payload


def refresh_ig_account_background() -> None:
    """Refresh IG cache in a background thread (never block API handlers)."""
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


def warm_ig_cache() -> None:
    refresh_ig_account_background()


def get_ig_account_summary(*, refresh: bool = False) -> dict[str, Any]:
    """Return cached IG account; refresh in background if stale (fast for /status)."""
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
        refresh_ig_account_background()
        return cached

    refresh_ig_account_background()
    if cached:
        cached["cached"] = True
        cached["stale"] = True
        cached["status"] = cached.get("status") or "loading"
        return cached
    return {"status": "loading"}
