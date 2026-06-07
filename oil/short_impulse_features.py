"""Features for oil_trader short_impulse rule port (1m + shifted 15m)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_short_impulse_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror oil_trader define_signals() inputs on AlphaGold `prices` bars."""
    df = df.copy()
    if "day_high_rolling" not in df.columns:
        raise ValueError("add_short_impulse_features requires day_high_rolling from prepare_data_v14")

    df["bar_change"] = df["close"] - df["open"]
    df["prev_bar_change"] = df["bar_change"].shift(1)
    df["bar_lower_wick"] = df["close"] - df["low"]
    df["dist_from_day_high"] = df["day_high_rolling"] - df["close"]

    if "is_london" in df.columns and "is_ny" in df.columns:
        df["oil_session"] = ((df["is_london"] > 0.5) | (df["is_ny"] > 0.5)).astype(int)
    else:
        df["oil_session"] = 0

    idx = df.index.sort_values()
    base = df.loc[idx]
    m15 = pd.DataFrame(
        {
            "open_15": base["open"].resample("15min", label="right", closed="right").first(),
            "close_15": base["close"].resample("15min", label="right", closed="right").last(),
        }
    ).dropna(how="all")
    m15["up_15"] = np.select(
        [m15["close_15"] > m15["open_15"], m15["close_15"] < m15["open_15"]],
        [1, -1],
        default=0,
    )
    m15["up_count3_15min"] = m15["up_15"].rolling(3, min_periods=1).sum()
    m15_feat = m15[["up_count3_15min"]].shift(1)

    left = pd.DataFrame(index=base.index)
    merged = pd.merge_asof(
        left.sort_index(),
        m15_feat.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    df["up_count3_15min"] = merged["up_count3_15min"].reindex(df.index)

    if "closePrice_ask" in df.columns and "closePrice_bid" in df.columns:
        df["oil_spread"] = df["closePrice_ask"] - df["closePrice_bid"]
    elif "spread" not in df.columns:
        df["oil_spread"] = np.nan

    if "atr" in df.columns:
        df["oil_atr"] = df["atr"]
    elif "ATR" in df.columns:
        df["oil_atr"] = df["ATR"]

    if "day_range_so_far_rel" in df.columns and "day_open" in df.columns:
        df["oil_day_range_db"] = df["day_range_so_far_rel"] * df["day_open"]
    else:
        df["oil_day_range_db"] = df["day_high_rolling"] - df["day_low_rolling"]

    base_imp = (
        (df["bar_change"] < -16)
        & (df["prev_bar_change"] < 10)
        & (df["prev_bar_change"] > -16)
        & (df["bar_lower_wick"] < 35)
        & (df["volume"] > 1100)
        & (df["up_count3_15min"] != -3)
        & (df["dist_from_day_high"] < 180)
    )
    df["impulse_recent_60"] = base_imp.astype(int).rolling(60, min_periods=1).sum().shift(1).fillna(0)

    return df
