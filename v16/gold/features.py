"""v16 gold deterministic energetic features for S1/S2 models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_energetic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "bar_move" in df.columns:
        df["en_prev_body"] = df["bar_move"].shift(1)
    else:
        df["en_prev_body"] = (df["close"].shift(1) - df["open"].shift(1)).abs()

    bm = df["bar_move"] if "bar_move" in df.columns else (df["close"] - df["open"]).abs()
    df["en_rolling_bm_5"] = bm.rolling(5, min_periods=1).mean()

    rvol = df["volume"].rolling(20, min_periods=1).mean()
    df["en_vol_ratio_20"] = df["volume"] / rvol.replace(0, np.nan)

    bar_range = df["high"] - df["low"]
    rolling_range = bar_range.rolling(5, min_periods=1).mean()
    df["en_range_ratio_5"] = bar_range / rolling_range.replace(0, np.nan)

    ema_20 = df["close"].ewm(span=20, min_periods=1).mean()
    df["en_above_ema_20"] = (df["close"] > ema_20).astype(int)

    bar_dir = np.sign(df["close"] - df["open"])
    streak = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        streak[i] = streak[i - 1] + 1 if bar_dir.iloc[i] == bar_dir.iloc[i - 1] else 1
    df["en_dir_streak"] = streak

    if "er_30" not in df.columns:
        delta_30 = (df["close"] - df["close"].shift(30)).abs()
        path_30 = df["close"].diff().abs().rolling(30, min_periods=5).sum()
        df["en_er_30"] = delta_30 / path_30.replace(0, np.nan)
    else:
        df["en_er_30"] = df["er_30"]

    return df
