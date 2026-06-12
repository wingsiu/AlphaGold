"""v15 Deterministic Energetic Features — Model Input Columns (No HMM).

These features replace the removed hmm_regime column with deterministic,
live/backtest-consistent alternatives that the S1/S2 XGBoost models can
learn to weight during training.

Add to any DataFrame with OHLCV columns via add_v15_energetic_features().
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_v15_energetic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add v15 deterministic energetic features (live ↔ backtest consistent).

    These are computed from raw OHLCV only — no HMM, no MySQL-specific data.
    Added columns:
      - en_prev_body         : previous bar's |close - open|
      - en_rolling_bm_5      : 5-bar rolling mean of bar_move
      - en_vol_ratio_20      : volume / 20-bar rolling mean volume
      - en_range_ratio_5     : (high-low) / 5-bar rolling mean range
      - en_above_ema_20      : 1 if close > ema(20), else 0
      - en_dir_streak        : consecutive bars in same direction (sign of body)
      - en_er_30             : efficiency ratio over 30 bars (if not already present)

    Parameters
    ----------
    df : DataFrame
        Must contain 'open', 'high', 'low', 'close', 'volume'.

    Returns
    -------
    DataFrame with new columns added.
    """
    df = df.copy()

    # Previous bar body (same as bar_move shifted by 1)
    if "bar_move" in df.columns:
        df["en_prev_body"] = df["bar_move"].shift(1)
    else:
        df["en_prev_body"] = (df["close"].shift(1) - df["open"].shift(1)).abs()

    # Rolling bar_move mean (5 bars)
    bm = df["bar_move"] if "bar_move" in df.columns else (df["close"] - df["open"]).abs()
    df["en_rolling_bm_5"] = bm.rolling(5, min_periods=1).mean()

    # Volume ratio vs 20-bar mean
    rvol = df["volume"].rolling(20, min_periods=1).mean()
    df["en_vol_ratio_20"] = df["volume"] / rvol.replace(0, np.nan)

    # Range expansion (current bar range / 5-bar mean range)
    bar_range = df["high"] - df["low"]
    rolling_range = bar_range.rolling(5, min_periods=1).mean()
    df["en_range_ratio_5"] = bar_range / rolling_range.replace(0, np.nan)

    # Price vs EMA(20)
    ema_20 = df["close"].ewm(span=20, min_periods=1).mean()
    df["en_above_ema_20"] = (df["close"] > ema_20).astype(int)

    # Directional streak (consecutive same-direction bars)
    bar_dir = np.sign(df["close"] - df["open"])
    df["en_dir_streak"] = 1  # at minimum, current bar is 1
    streak = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        if bar_dir.iloc[i] == bar_dir.iloc[i - 1]:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 1
    df["en_dir_streak"] = streak

    # ER 30 (if not already present)
    if "er_30" not in df.columns:
        delta_30 = (df["close"] - df["close"].shift(30)).abs()
        path_30 = df["close"].diff().abs().rolling(30, min_periods=5).sum()
        df["en_er_30"] = delta_30 / path_30.replace(0, np.nan)
    else:
        df["en_er_30"] = df["er_30"]

    return df
