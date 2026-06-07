"""Trading window status for mobile UI (IG market + v14 time filter)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from xgboost_filter_model.time_slot_filter import (
    SESSION_PERIOD_SPECS,
    is_blocked_entry,
    load_weak_filter,
    resolve_v14_time_filter_path,
)

HKT = ZoneInfo("Asia/Hong_Kong")


def _session_active(ts: pd.Timestamp, session_key: str) -> bool:
    spec = SESSION_PERIOD_SPECS[session_key]
    local = ts.tz_convert(spec["timezone"])
    minute_of_day = local.hour * 60 + local.minute
    start_min = int(spec["start_hour"]) * 60 + int(spec["start_minute"])
    end_min = int(spec["end_hour"]) * 60 + int(spec["end_minute"])
    return start_min <= minute_of_day < end_min


def _active_sessions(ts: pd.Timestamp) -> list[str]:
    return [k for k in SESSION_PERIOD_SPECS if _session_active(ts, k)]


def _session_label(session_key: str) -> str:
    return {"hkt": "HKT", "london": "London", "ny": "NY"}.get(session_key, session_key.upper())


def _fmt_hkt(ts: pd.Timestamp) -> str:
    return ts.tz_convert(HKT).strftime("%a %H:%M HKT")


def _next_session_start(ts: pd.Timestamp) -> tuple[pd.Timestamp | None, str | None]:
    """Earliest start of any major session (HKT / London / NY), labeled in HKT."""
    best: pd.Timestamp | None = None
    best_label: str | None = None
    for session_key, spec in SESSION_PERIOD_SPECS.items():
        tz = spec["timezone"]
        local = ts.tz_convert(tz)
        start_min = int(spec["start_hour"]) * 60 + int(spec["start_minute"])
        candidate_local = local.replace(
            hour=int(spec["start_hour"]),
            minute=int(spec["start_minute"]),
            second=0,
            microsecond=0,
        )
        if local.hour * 60 + local.minute >= start_min:
            candidate_local = candidate_local + timedelta(days=1)
        candidate_utc = pd.Timestamp(candidate_local).tz_convert("UTC")
        if best is None or candidate_utc < best:
            best = candidate_utc
            best_label = f"{_session_label(session_key)} {_fmt_hkt(candidate_utc)}"
    return best, best_label


def _next_unblocked_minute(
    now: pd.Timestamp,
    weak_cells: list[dict],
    *,
    require_session: bool,
    market_open: bool,
) -> tuple[pd.Timestamp | None, str | None]:
    probe = now.floor("min") + pd.Timedelta(minutes=1)
    limit = now + pd.Timedelta(days=2)
    while probe < limit:
        blocked = is_blocked_entry(probe, weak_cells) if weak_cells else False
        sessions = _active_sessions(probe)
        in_session = bool(sessions) if require_session else True
        if market_open and not blocked and in_session:
            label = ", ".join(_session_label(s) for s in sessions) if sessions else "Open"
            return probe, f"{label} · {_fmt_hkt(probe)}"
        probe += pd.Timedelta(minutes=1)
    return None, None


def _load_weak_cells() -> list[dict]:
    path = resolve_v14_time_filter_path()
    if not path:
        return []
    try:
        return load_weak_filter(path)
    except Exception:
        return []


def get_trading_window_status() -> dict[str, Any]:
    from mobile_api.market_price import get_gold_price_summary

    now = pd.Timestamp.now(tz="UTC")
    gold = get_gold_price_summary()
    market_status = str(gold.get("market_status") or "unknown").upper()
    market_open = market_status == "TRADEABLE"

    weak_cells = _load_weak_cells()
    filter_blocked = is_blocked_entry(now, weak_cells) if weak_cells else False
    sessions = _active_sessions(now)

    in_window = market_open and not filter_blocked
    reasons: list[str] = []
    if not market_open:
        reasons.append("market_closed")
    if filter_blocked:
        reasons.append("time_filter")
    if not sessions:
        reasons.append("outside_session")

    next_start: pd.Timestamp | None = None
    next_label: str | None = None

    if not in_window:
        if not market_open:
            next_start, next_label = _next_session_start(now)
            if next_label:
                next_label = f"Market closed · next session {next_label}"
        else:
            next_start, next_label = _next_unblocked_minute(
                now, weak_cells, require_session=False, market_open=market_open
            )
            if next_start is None:
                next_start, next_label = _next_session_start(now)

    return {
        "in_window": in_window,
        "market_status": market_status,
        "market_open": market_open,
        "time_filter_blocked": filter_blocked,
        "active_sessions": sessions,
        "blocked_reasons": reasons,
        "next_window_start_utc": next_start.isoformat() if next_start is not None else None,
        "next_window_label": next_label,
        "display_timezone": "HKT",
        "checked_at_utc": now.isoformat(),
        "checked_at_hkt": _fmt_hkt(now),
    }
