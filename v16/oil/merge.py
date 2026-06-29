"""Oil v16 portfolio merge — single-slot by entry time."""
from __future__ import annotations

import pandas as pd

LEG_ENTRY_PRIORITY: dict[str, int] = {
    "short_impulse": 0,
    "si": 0,
    "oil_rip_short": 0,
    "rip": 0,
    "ret_short": 0,
    "oil_retrace_short": 0,
    "wr90": 1,
    "wr90_long": 1,
    "ret": 2,
    "oil_retrace": 2,
    "long_retrace": 2,
    "oil_long_retrace": 2,
}


def _utc_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def merge_single_position(trades: list[dict]) -> list[dict]:
    """One open trade at a time — skip entries before prior exit; tie-break by leg priority."""
    if not trades:
        return []

    def sort_key(tr):
        entry = _utc_ts(tr["entry"])
        typ = str(tr.get("type", ""))
        return (entry, LEG_ENTRY_PRIORITY.get(typ, 9))

    taken: list[dict] = []
    busy_until = None
    for tr in sorted(trades, key=sort_key):
        entry = _utc_ts(tr["entry"])
        if busy_until is not None and entry < busy_until:
            continue
        taken.append(tr)
        busy_until = _utc_ts(tr["exit"])
    return taken
