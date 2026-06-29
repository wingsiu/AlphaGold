"""Protective swing levels from 15m zigzag (no lookahead)."""
from __future__ import annotations

import pandas as pd

from v16.structure.swing_zigzag import build_15m_ohlc, build_swing_table


def swing_table_for_cfg(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sc = cfg.get("structure", {})
    ohlc = build_15m_ohlc(df)
    return build_swing_table(
        ohlc,
        atr_mult=float(sc.get("atr_mult", 3.0)),
        atr_period=int(sc.get("atr_period", 14)),
    )


def last_swing_price(swings: pd.DataFrame, ts: pd.Timestamp, kind: str) -> float | None:
    """Last confirmed H or L swing at or before ts."""
    if swings.empty:
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    past = swings.loc[:t]
    sub = past[past["kind"] == kind]
    if sub.empty:
        return None
    return float(sub.iloc[-1]["price"])


def protective_stop(side: int, swings: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    """Long → last L; short → last H."""
    kind = "L" if side == 1 else "H"
    return last_swing_price(swings, ts, kind)
