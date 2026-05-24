"""15m candle shape and pattern features (causal: last completed 15m bar only)."""
import os
import pandas as pd
import numpy as np

EPS = 1e-9


def _enabled() -> bool:
    return os.environ.get("V14_CANDLE_15M", "").strip().lower() in ("1", "true", "yes", "on")


def add_candle_pattern_15m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enable with V14_CANDLE_15M=1.

    Shape (last closed 15m candle, forward-filled to 1m):
      c15_body_pct, c15_upper_wick_pct, c15_lower_wick_pct, c15_close_loc, c15_bullish

    Patterns (0/1 flags on that same candle):
      c15_doji, c15_hammer, c15_shooting_star, c15_marubozu
      c15_bull_engulf, c15_bear_engulf, c15_inside_bar
    """
    if not _enabled():
        return df

    df = df.copy()
    df_15m = df.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    o, h, l, c = df_15m["open"], df_15m["high"], df_15m["low"], df_15m["close"]
    rng = (h - l).clip(lower=0)
    body = (c - o).abs()
    body_top = pd.concat([o, c], axis=1).max(axis=1)
    body_bottom = pd.concat([o, c], axis=1).min(axis=1)
    upper = h - body_top
    lower = body_bottom - l

    safe_rng = rng.replace(0, np.nan)
    df_15m["c15_body_pct"] = body / (safe_rng + EPS)
    df_15m["c15_upper_wick_pct"] = upper / (safe_rng + EPS)
    df_15m["c15_lower_wick_pct"] = lower / (safe_rng + EPS)
    df_15m["c15_close_loc"] = (c - l) / (safe_rng + EPS)
    df_15m["c15_bullish"] = (c > o).astype(float)

    df_15m["c15_doji"] = ((body / (safe_rng + EPS)) < 0.10) & (rng > 0)
    df_15m["c15_hammer"] = (
        (lower / (safe_rng + EPS) >= 0.55)
        & (upper / (safe_rng + EPS) <= 0.20)
        & (rng > 0)
    )
    df_15m["c15_shooting_star"] = (
        (upper / (safe_rng + EPS) >= 0.55)
        & (lower / (safe_rng + EPS) <= 0.20)
        & (rng > 0)
    )
    df_15m["c15_marubozu"] = (body / (safe_rng + EPS) >= 0.85) & (rng > 0)

    po, pc = o.shift(1), c.shift(1)
    df_15m["c15_bull_engulf"] = (
        (c > o)
        & (pc < po)
        & (c >= po)
        & (o <= pc)
    )
    df_15m["c15_bear_engulf"] = (
        (c < o)
        & (pc > po)
        & (c <= po)
        & (o >= pc)
    )
    df_15m["c15_inside_bar"] = (h < h.shift(1)) & (l > l.shift(1))

    pattern_cols = [
        "c15_doji",
        "c15_hammer",
        "c15_shooting_star",
        "c15_marubozu",
        "c15_bull_engulf",
        "c15_bear_engulf",
        "c15_inside_bar",
    ]
    for col in pattern_cols:
        df_15m[col] = df_15m[col].astype(float)

    shape_cols = [
        "c15_body_pct",
        "c15_upper_wick_pct",
        "c15_lower_wick_pct",
        "c15_close_loc",
        "c15_bullish",
    ]
    out_cols = shape_cols + pattern_cols

    df_15m_last = df_15m[out_cols].shift(1)
    mapped = df_15m_last.reindex(df.index, method="ffill")
    for col in out_cols:
        df[col] = mapped[col]

    df[out_cols] = df[out_cols].fillna(0)
    return df
