"""Volatility-adaptive probability threshold for pattern entry.

Computes a scaling factor based on recent bar range vs. long-term baseline.
In low volatility: lower threshold (let more signals through).
In high volatility: higher threshold (be more selective).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


# Configurable parameters (env-overridable for sweeps)
RANGE_COL = os.environ.get("V14_ADAPTIVE_RANGE_COL", "range_15m")
LOOKBACK_BARS = int(os.environ.get("V14_ADAPTIVE_LOOKBACK", 3000))  # ~1 month of 15m bars
BASELINE_LOOKBACK = int(os.environ.get("V14_ADAPTIVE_BASELINE", 12000))  # ~4 months
MIN_SCALE = float(os.environ.get("V14_ADAPTIVE_MIN_SCALE", 0.85))
MAX_SCALE = float(os.environ.get("V14_ADAPTIVE_MAX_SCALE", 1.15))
# If disabled, always returns 1.0 (no scaling)
ENABLED = os.environ.get("V14_ADAPTIVE_PROB", "0") not in ("0", "no", "false")


def _volatility_scale(ts_series: pd.Series, ref_median: float) -> pd.Series:
    """Return scaling factor per bar based on volatility regime.

    1.0 = normal vol, <1.0 = low vol (lower prob threshold), >1.0 = high vol
    """
    if not ENABLED:
        return pd.Series(1.0, index=ts_series.index)

    recent_med = ts_series.rolling(LOOKBACK_BARS, min_periods=LOOKBACK_BARS // 4).median()
    ratio = recent_med / ref_median

    # Map ratio to scale: low vol -> lower threshold, high vol -> higher threshold
    # ratio < 1 (quiet): scale = MAX_SCALE - delta (pulls threshold down)
    # ratio > 1 (active): scale = MIN_SCALE + delta (pushes threshold up)
    # Clamp to [MIN_SCALE, MAX_SCALE]
    scale = 1.0 / ratio  # invert: quiet -> scale > 1, active -> scale < 1
    scale = scale.clip(MIN_SCALE, MAX_SCALE)
    return scale.fillna(1.0)


def adaptive_prob_threshold(
    base_prob: float,
    df: pd.DataFrame,
    range_col: str = RANGE_COL,
) -> pd.Series:
    """Return a per-bar probability threshold.

    Args:
        base_prob: The static threshold from config (e.g. 0.55).
        df: Feature DataFrame (must contain range_col).
        range_col: Column name to use as volatility proxy.

    Returns:
        Series of per-bar thresholds, same index as df.
    """
    if not ENABLED or range_col not in df.columns:
        return pd.Series(base_prob, index=df.index)

    # Baseline: long-term median of the range column
    baseline = df[range_col].iloc[:BASELINE_LOOKBACK].median()
    if pd.isna(baseline) or baseline <= 0:
        return pd.Series(base_prob, index=df.index)

    scale = _volatility_scale(df[range_col], baseline)
    # Invert: scale > 1 means LOW vol -> LOWER threshold -> more trades
    # base_prob / scale  (or equivalently base_prob * (2 - scale) for symmetry)
    # Using: threshold = base_prob * (2 - scale) to keep values in reasonable range
    threshold = base_prob * (2.0 - scale)
    # Clamp to not go below 0.35 or above 0.70
    threshold = threshold.clip(0.35, 0.70)
    return threshold


def adaptive_prob_for_df(
    base_prob: float,
    df: pd.DataFrame,
    range_col: str = RANGE_COL,
) -> pd.Series:
    """Convenience wrapper returning per-bar thresholds."""
    return adaptive_prob_threshold(base_prob, df, range_col)
