"""Oil pattern registry — add one pattern at a time (start with downtrend retrace)."""

from __future__ import annotations

from pathlib import Path

from oil.config import (
    PATTERN_MODEL_DIR,
    SPOT_CONTRACT_MULTIPLIER,
    TARGET_ATR_SL_MULT,
    TARGET_ATR_TP_MULT,
    THRESHOLD_SCALE,
)

PROJECT_ROOT = PATTERN_MODEL_DIR.parent.parent.parent

# Pattern rules: absolute $ in DB units (÷100 ≈ spot $). Match gold rule sizes when SCALE=1.
_S = THRESHOLD_SCALE

PATTERN_REGISTRY: dict[str, dict] = {
    "oil_short_impulse": {
        "direction_bias": "short",
        "priority": 0,
        "feature_set": "short_impulse",
        # Rule-only default (oil_trader v2 filters). Raise prob (e.g. 0.55) to enable XGB filter.
        "thresholds": {"prob": 0.0},
        "context": [
            {"feat": "oil_session", "op": "==", "val": 1.0},
        ],
        "pattern": [
            {"feat": "bar_change", "op": "<", "val": -16.0 * _S},
            {"feat": "prev_bar_change", "op": "<", "val": 10.0 * _S},
            {"feat": "prev_bar_change", "op": ">", "val": -14.0 * _S},
            {"feat": "bar_lower_wick", "op": "<", "val": 35.0 * _S},
            {"feat": "volume", "op": ">", "val": 1100.0},
            {"feat": "up_count3_15min", "op": "!=", "val": -3.0},
            {"feat": "dist_from_day_high", "op": "<", "val": 180.0 * _S},
            {"feat": "bar_change", "op": ">", "val": -50.0 * _S},
            {"feat": "oil_spread", "op": "<=", "val": 4.25},
            {"feat": "oil_atr", "op": "<=", "val": 8.0},
            {"feat": "impulse_recent_60", "op": "<=", "val": 8.0},
        ],
        "exclude": [],
        "execution": {
            "horizon": 90,
            "target_mode": "fixed",
            "tp": 70.0,
            "sl": 40.0,
        },
    },
    "oil_bar_drop_short": {
        "direction_bias": "short",
        "priority": 1,
        "feature_set": "current",
        "thresholds": {"prob": 0.55},
        "context": [],
        "pattern": [
            {"feat": "bar_bear_drop", "op": ">", "val": 15.0 * _S},
            {"feat": "volume", "op": ">", "val": 900.0},
        ],
        "exclude": [],
        "execution": {
            "horizon": 120,
            "target_mode": "fixed",
            "tp": 80.0,
            "sl": 50.0,
        },
    },
    "oil_downtrend_retrace": {
        "direction_bias": "short",
        "priority": 2,
        "feature_set": "current",
        "thresholds": {"prob": 0.55},
        "context": [],
        "pattern": [
            {"feat": "drop_from_high_240", "op": ">=", "val": 25.0 * _S},
            {"feat": "rise_from_low_240", "op": ">=", "val": 5.0 * _S},
        ],
        "exclude": [
            {"feat": "near_low_zone", "op": "==", "val": 1.0},
        ],
        "execution": {
            "horizon": 15,
            "target_mode": "atr",
            "tp": TARGET_ATR_TP_MULT,
            "sl": TARGET_ATR_SL_MULT,
        },
    },
}

PRODUCTION_PATTERNS: tuple[str, ...] = tuple(PATTERN_REGISTRY.keys())

EXCLUDE_COLS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "timestamp",
    "trend_label",
    "target_v10",
    "target_v14",
    "target_pattern",
    "is_trend",
    "atr",
    "day_utc2",
    "future_max_move",
    "future_min_move",
    "future_er",
    "bar_move",
    "hour",
    "day_id",
    "day_high",
    "day_low",
    "high_90",
    "low_90",
    "day_open",
    "day_high_rolling",
    "day_low_rolling",
    "openPrice_ask",
    "openPrice_bid",
    "closePrice_ask",
    "closePrice_bid",
    "highPrice_ask",
    "highPrice_bid",
    "lowPrice_ask",
    "lowPrice_bid",
    "closePrice",
    "lowPrice",
    "open_price",
    "highPrice",
    "openPrice",
    "ma_60m",
    "high_60m",
    "low_60m",
    "high_15m",
    "low_15m",
    "hmm_regime",
    "daily_poc",
    "daily_vwap",
    "rolling_poc_4h",
    "dynamic_tp",
    "dynamic_sl",
    "fvg_bull_bottom",
    "fvg_bull_top",
    "fvg_bear_top",
    "fvg_bear_bottom",
}


def pattern_feature_set_for(name: str) -> str:
    return str(PATTERN_REGISTRY[name].get("feature_set", "current"))


def enrich_pattern_features(df, pattern_names: list[str] | None = None):
    """Add feature-set-specific columns (e.g. short_impulse) before routing."""
    import pandas as pd

    names = pattern_names if pattern_names is not None else list(PATTERN_REGISTRY.keys())
    need_si = any(
        PATTERN_REGISTRY.get(n, {}).get("feature_set") == "short_impulse" for n in names
    )
    if need_si:
        from oil.short_impulse_features import add_short_impulse_features

        return add_short_impulse_features(df)
    return df


def collect_pa_groups(pattern_names: list[str] | None = None) -> tuple[str, ...]:
    from xgboost_filter_model.price_action_features import ALL_GROUPS

    names = pattern_names if pattern_names is not None else list(PATTERN_REGISTRY.keys())
    out: set[str] = set()
    for name in names:
        spec = PATTERN_REGISTRY.get(name, {})
        for g in spec.get("pa_groups", ()):
            if g in ALL_GROUPS:
                out.add(g)
    if "wick" in ALL_GROUPS:
        out.add("wick")
    return tuple(sorted(out))
