"""Intraday rolling day high/low on 1m gold bars."""
from __future__ import annotations

import pandas as pd


def attach_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_low_rolling / day_high_rolling (UTC calendar day, no lookahead)."""
    out = df.copy()
    day_key = out.index.normalize()
    out["day_low_rolling"] = out.groupby(day_key)["low_bid"].cummin()
    out["day_high_rolling"] = out.groupby(day_key)["high_ask"].cummax()
    return out


def day_level_entry_allowed(
    side: int,
    entry_price: float,
    day_low: float,
    day_high: float,
    offset: float,
) -> bool:
    """
    Long: entry must be below day_low + offset (room to run up toward target).
    Short: entry must be above day_high - offset (room to fall toward target).
    """
    off = float(offset)
    if side == 1:
        return float(entry_price) < float(day_low) + off
    if side == -1:
        return float(entry_price) > float(day_high) - off
    return False


def filter_entries_by_day_level(
    df: pd.DataFrame,
    entries: pd.DataFrame,
    offset: float,
) -> pd.DataFrame:
    """Keep only fills with room to a day_low+offset (long) or day_high-offset (short) target."""
    if entries.empty:
        return entries
    if "day_low_rolling" not in df.columns or "day_high_rolling" not in df.columns:
        raise ValueError("filter_entries_by_day_level requires attach_day_levels(df) first")

    keep_idx: list[pd.Timestamp] = []
    for ts, row in entries.iterrows():
        j = int(row["entry_idx"])
        if day_level_entry_allowed(
            int(row["side"]),
            float(row["entry_price"]),
            float(df["day_low_rolling"].iloc[j]),
            float(df["day_high_rolling"].iloc[j]),
            offset,
        ):
            keep_idx.append(ts)
    return entries.loc[keep_idx]
