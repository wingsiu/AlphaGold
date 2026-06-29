"""Gold v16 portfolio merge — single-slot by entry time."""
from __future__ import annotations

import pandas as pd

from config.pattern_registry import PATTERN_REGISTRY, PRODUCTION_PATTERNS
from v16.config.gold_config import LEG_PRIORITY


def _utc_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def build_leg_priority() -> dict[str, int]:
    pri = dict(LEG_PRIORITY)
    for name, spec in PATTERN_REGISTRY.items():
        if name in PRODUCTION_PATTERNS:
            pri.setdefault(name, int(spec.get("priority", 9)))
    pri.setdefault("energetic", 25)
    pri.setdefault("v16_momentum", 14)
    pri.setdefault("v16_dip_short", 0)
    pri.setdefault("dip_short_rip", 0)
    return pri


GOLD_LEG_PRIORITY = build_leg_priority()


def trade_row(
    entry,
    exit,
    pnl: float,
    leg: str,
    *,
    side: int = 1,
    typ: str | None = None,
) -> dict:
    return {
        "entry": _utc_ts(entry),
        "exit": _utc_ts(exit),
        "pnl": float(pnl),
        "type": typ or leg,
        "_leg": leg,
        "side": side,
    }


def df_to_trades(tdf: pd.DataFrame, leg: str, typ: str | None = None) -> list[dict]:
    if tdf.empty:
        return []
    out = []
    for _, r in tdf.iterrows():
        side = int(r["side"]) if "side" in r and pd.notna(r["side"]) else (1 if leg != "v16_dip_short" else -1)
        et = r.get("entry_time", r.get("entry"))
        xt = r.get("exit_time", r.get("exit"))
        out.append(trade_row(et, xt, r["pnl"], leg, side=side, typ=typ or leg))
    return out


def merge_gold_trades(trades: list[dict]) -> list[dict]:
    """Single slot — sort by entry time, tie-break by leg priority."""
    if not trades:
        return []

    def sort_key(tr):
        entry = _utc_ts(tr["entry"])
        typ = str(tr.get("type", tr.get("_leg", "")))
        pri = GOLD_LEG_PRIORITY.get(typ, GOLD_LEG_PRIORITY.get(str(tr.get("_leg", "")), 9))
        return (entry, pri)

    taken: list[dict] = []
    busy_until = None
    for tr in sorted(trades, key=sort_key):
        entry = _utc_ts(tr["entry"])
        if busy_until is not None and entry < busy_until:
            continue
        taken.append(tr)
        busy_until = _utc_ts(tr["exit"])
    return taken
