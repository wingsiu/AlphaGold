"""v15 Energetic Bar Filter — Deterministic (No HMM).

Replaces v14's HMM-based trend-regime gate with a simple, deterministic filter
that is identical between live bot and backtest.

Production gate:  bar_move > 3.0 AND volume > 200  (no HMM, no env config)
Research gates:  additional deterministic filters via env vars for sweeps.

Research env vars (0 = disabled):
  V15_EN_PREV_MOVE       — prev_bar_abs_change > this (default 0)
  V15_EN_ROLLING_BM_MIN  — rolling bar_move mean > this (default 0)
  V15_EN_VOL_RATIO_MIN   — volume / rolling_volume > this (default 0)
  V15_EN_RANGE_EXP_MIN   — range / rolling_range > this (default 0)
  V15_EN_EMA_CROSS       — 1 = price > ema_20
  V15_EN_DIR_PERSIST     — N consecutive same-direction bars (default 0)
  V15_EN_ER30_MIN        — er_30 > this (default 0)
"""
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


def energetic_bar_mask_v15(
    df: pd.DataFrame,
    *,
    min_bar_move: float = 3.0,
    min_volume: float = 200,
) -> pd.Series:
    """Deterministic energetic bar mask — no HMM, consistent live ↔ backtest.

    Production: bar_move > 3.0 AND volume > 200.
    Research gates can be enabled via V15_EN_* env vars.
    """
    if "bar_move" in df.columns:
        bar_move = df["bar_move"]
    else:
        bar_move = (df["close"] - df["open"]).abs()

    mask = (bar_move > min_bar_move) & (df["volume"] > min_volume)

    # ---- Research gates (all off by default) ----

    prev_move_pct = _env_float("V15_EN_PREV_MOVE", 0)
    if prev_move_pct > 0:
        prev_body = (df["close"].shift(1) - df["open"].shift(1)).abs()
        mask &= prev_body > prev_move_pct

    rolling_bm_min = _env_float("V15_EN_ROLLING_BM_MIN", 0)
    if rolling_bm_min > 0:
        win = int(_env_float("V15_EN_ROLLING_BM_WIN", 5))
        mask &= bar_move.rolling(win, min_periods=1).mean() > rolling_bm_min

    vol_ratio_min = _env_float("V15_EN_VOL_RATIO_MIN", 0)
    if vol_ratio_min > 0:
        win = int(_env_float("V15_EN_VOL_RATIO_WIN", 20))
        rvol = df["volume"].rolling(win, min_periods=1).mean()
        mask &= (df["volume"] / rvol.replace(0, np.nan)) > vol_ratio_min

    range_exp_min = _env_float("V15_EN_RANGE_EXP_MIN", 0)
    if range_exp_min > 0:
        win = int(_env_float("V15_EN_RANGE_EXP_WIN", 5))
        bar_range = df["high"] - df["low"]
        rr = bar_range.rolling(win, min_periods=1).mean()
        mask &= (bar_range / rr.replace(0, np.nan)) > range_exp_min

    if _env_bool("V15_EN_EMA_CROSS"):
        ema_20 = df["close"].ewm(span=20, min_periods=1).mean()
        direction = os.environ.get("V15_EN_EMA_CROSS_DIR", "up").strip()
        mask &= df["close"] > ema_20 if direction == "up" else df["close"] < ema_20

    dir_persist = int(_env_float("V15_EN_DIR_PERSIST", 0))
    if dir_persist > 0:
        bar_dir = np.sign(df["close"] - df["open"])
        for lag in range(1, dir_persist):
            prev_dir = np.sign(df["close"].shift(lag) - df["open"].shift(lag))
            mask &= bar_dir == prev_dir

    er30_min = _env_float("V15_EN_ER30_MIN", 0)
    if er30_min > 0 and "er_30" in df.columns:
        mask &= df["er_30"] > er30_min

    return mask
