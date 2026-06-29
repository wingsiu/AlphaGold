"""Attach v16 zigzag structure context to oil 15m bars."""
from __future__ import annotations

import pandas as pd

from v16.structure.swing_zigzag import build_structure_context

STRUCT_COLS = (
    "struct_trend",
    "struct_dist_pts",
    "struct_leg_pts",
    "struct_pullback_pct",
    "struct_leg_age_15m",
    "struct_aligned",
    "struct_hh",
    "struct_hl",
    "struct_lh",
    "struct_ll",
)


def _normalize_1m_for_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Oil live cache uses close_ask/bid; structure resampler expects mid/high_ask/low_bid."""
    out = df.copy()
    if "mid" not in out.columns:
        if "close_ask" in out.columns and "close_bid" in out.columns:
            out["mid"] = (out["close_ask"] + out["close_bid"]) / 2.0
        elif "close" in out.columns:
            out["mid"] = out["close"]
        else:
            raise KeyError("Need mid or close_ask/close_bid for structure context")
    if "high_ask" not in out.columns:
        out["high_ask"] = out["high"]
    if "low_bid" not in out.columns:
        out["low_bid"] = out["low"]
    return out


def structure_on_d15(d1m_v16: pd.DataFrame, d15: pd.DataFrame) -> pd.DataFrame:
    """Map 1m structure context onto 15m slot-start index."""
    struct = build_structure_context(
        _normalize_1m_for_structure(d1m_v16),
        rule="15min",
        atr_mult=3.0,
        atr_period=14,
    )
    if struct.empty:
        return pd.DataFrame(index=d15.index)

    out = pd.DataFrame(index=d15.index)
    for col in STRUCT_COLS:
        if col not in struct.columns:
            continue
        s = struct[col].copy()
        mapped = []
        for ts in d15.index:
            # last 1m bar at or before 15m slot close (slot + 14m)
            slot_end = ts + pd.Timedelta(minutes=14)
            sub = s.loc[:slot_end]
            mapped.append(float(sub.iloc[-1]) if len(sub) else 0.0)
        out[col] = mapped
    return out.fillna(0.0)


def apply_long_structure_gate(
    d15: pd.DataFrame,
    struct_frame: pd.DataFrame,
    mask: pd.Series,
) -> pd.Series:
    """Keep longs when structure is up/neutral (HH/HL)."""
    if struct_frame.empty or "struct_trend" not in struct_frame.columns:
        return mask
    trend = struct_frame["struct_trend"].reindex(d15.index).fillna(0)
    return mask & (trend >= 0)


def apply_short_structure_gate(
    d15: pd.DataFrame,
    struct_frame: pd.DataFrame,
    mask: pd.Series,
) -> pd.Series:
    """Keep shorts when structure is down/neutral (LH/LL)."""
    if struct_frame.empty or "struct_trend" not in struct_frame.columns:
        return mask
    trend = struct_frame["struct_trend"].reindex(d15.index).fillna(0)
    return mask & (trend <= 0)
