"""
Structure trend hold — enter on with-trend retrace, exit when structure breaks.

Designed for multi-hour trends with 30–90 min pullbacks (2–6 × 15m legs).
"""
from __future__ import annotations

import pandas as pd

from v16.backtest.features import session_mask
from v16.config import v16_config
from v16.structure.swing_zigzag import build_15m_ohlc, build_structure_context


def _cfg() -> dict:
    return v16_config.STRUCTURE_TREND_HOLD


def build_structure_retrace_signals(
    df: pd.DataFrame,
    *,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Signal at 15m slot close when:
      - struct_trend ±1 (HH/HL or LH/LL)
      - pullback into [min, max] % of prior leg
      - leg age in [min, max] 15m bars (default 2–6 → 30–90 min)
      - last pivot opposes trade direction (pullback phase)
    One signal per swing confirmation (debounced).
    """
    c = cfg or _cfg()
    sc = c.get("structure", {})
    ent = c.get("entry", {})
    sessions = tuple(c.get("sessions", ("london", "ny")))

    struct = build_structure_context(
        df,
        rule=str(sc.get("rule", "15min")),
        atr_mult=float(sc.get("atr_mult", 3.0)),
        atr_period=int(sc.get("atr_period", 14)),
    )
    if struct.empty:
        return pd.DataFrame()

    ohlc = build_15m_ohlc(df)
    sess_1m = session_mask(df.index, sessions)
    min_pb = float(ent.get("min_pullback_pct", 0.15))
    max_pb = float(ent.get("max_pullback_pct", 0.65))
    min_age = int(ent.get("min_leg_age_15m", 2))
    max_age = int(ent.get("max_leg_age_15m", 6))

    rows: list[dict] = []
    last_swing_ts: pd.Timestamp | None = None

    for slot_ts in ohlc.index:
        if slot_ts not in struct.index:
            continue
        # map to last 1m in slot for session check
        loc = df.index.searchsorted(slot_ts, side="right") - 1
        if loc < 0 or loc >= len(df):
            continue
        if not bool(sess_1m.iloc[loc]):
            continue

        row = struct.loc[slot_ts]
        raw_trend = row.get("struct_trend", 0)
        if pd.isna(raw_trend):
            continue
        trend = int(raw_trend)
        if trend == 0:
            continue

        raw_age = row.get("struct_leg_age_15m", 0)
        age = 0 if pd.isna(raw_age) else int(raw_age)
        pb = 0.0 if pd.isna(row.get("struct_pullback_pct")) else float(row["struct_pullback_pct"])
        raw_kind = row.get("struct_last_kind", 0)
        last_kind = 0 if pd.isna(raw_kind) else int(raw_kind)

        if age < min_age or age > max_age:
            continue
        if pb < min_pb or pb > max_pb:
            continue

        side = int(trend)
        # pullback: long needs last pivot high; short needs last pivot low
        if side == 1 and last_kind != 1:
            continue
        if side == -1 and last_kind != -1:
            continue

        swing_ts = slot_ts
        if last_swing_ts is not None and swing_ts == last_swing_ts:
            continue
        last_swing_ts = swing_ts

        entry_loc = min(loc + 1, len(df) - 1)
        entry_ts = df.index[entry_loc]
        ep = float(df.iloc[entry_loc]["open_ask"] if side == 1 else df.iloc[entry_loc]["open_bid"])

        rows.append(
            {
                "signal_ts": slot_ts,
                "entry_time": entry_ts,
                "entry_idx": entry_loc,
                "entry_price": ep,
                "side": side,
                "struct_trend": trend,
                "pullback_pct": pb,
                "leg_age_15m": age,
                "struct_last_swing_price": float(row.get("struct_last_swing_price", 0.0)),
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("signal_ts")
    return out.sort_index()
