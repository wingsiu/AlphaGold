"""v16 gold energetic bar filter — deterministic (no HMM). Ported from legacy v15 gate."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def energetic_bar_mask(
    df: pd.DataFrame,
    *,
    min_bar_move: float = 3.0,
    min_volume: float = 200,
) -> pd.Series:
    """Production: bar_move > 3.0 AND volume > 200."""
    bar_move = df["bar_move"] if "bar_move" in df.columns else (df["close"] - df["open"]).abs()
    mask = (bar_move > min_bar_move) & (df["volume"] > min_volume)

    prev_move_pct = _env_float("V16_EN_PREV_MOVE", 0)
    if prev_move_pct > 0:
        prev_body = (df["close"].shift(1) - df["open"].shift(1)).abs()
        mask &= prev_body > prev_move_pct

    rolling_bm_min = _env_float("V16_EN_ROLLING_BM_MIN", 0)
    if rolling_bm_min > 0:
        win = int(_env_float("V16_EN_ROLLING_BM_WIN", 5))
        mask &= bar_move.rolling(win, min_periods=1).mean() > rolling_bm_min

    vol_ratio_min = _env_float("V16_EN_VOL_RATIO_MIN", 0)
    if vol_ratio_min > 0:
        win = int(_env_float("V16_EN_VOL_RATIO_WIN", 20))
        rvol = df["volume"].rolling(win, min_periods=1).mean()
        mask &= (df["volume"] / rvol.replace(0, np.nan)) > vol_ratio_min

    range_exp_min = _env_float("V16_EN_RANGE_EXP_MIN", 0)
    if range_exp_min > 0:
        win = int(_env_float("V16_EN_RANGE_EXP_WIN", 5))
        bar_range = df["high"] - df["low"]
        rr = bar_range.rolling(win, min_periods=1).mean()
        mask &= (bar_range / rr.replace(0, np.nan)) > range_exp_min

    if _env_bool("V16_EN_EMA_CROSS", False):
        ema_20 = df["close"].ewm(span=20, min_periods=1).mean()
        mask &= df["close"] > ema_20

    dir_persist = int(_env_float("V16_EN_DIR_PERSIST", 0))
    if dir_persist > 0:
        bar_dir = np.sign(df["close"] - df["open"])
        streak = np.ones(len(df), dtype=int)
        for i in range(1, len(df)):
            streak[i] = streak[i - 1] + 1 if bar_dir.iloc[i] == bar_dir.iloc[i - 1] else 1
        mask &= pd.Series(streak, index=df.index) >= dir_persist

    er_min = _env_float("V16_EN_ER30_MIN", 0)
    if er_min > 0:
        col = "er_30" if "er_30" in df.columns else "en_er_30"
        if col in df.columns:
            mask &= df[col] > er_min

    return mask
