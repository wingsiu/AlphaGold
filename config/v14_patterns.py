"""Pattern-specialist config for v14 (no HMM / bar_move / volume gate on pattern path)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_HYGIENE = {
    "min_warmup_bars": 240,
}

PATTERN_MODEL_DIR = PROJECT_ROOT / "runtime" / "bot_assets" / "wf_models_v14_patterns"

# Single-stage: one model per pattern predicts P(directional TP hit).
# Direction comes from direction_bias; entry when prob >= thresholds.prob.
#
# pa_groups: optional 15m price-action features for the pattern model
#   fvg  — dist/time from bull/bear FVG
#   wick — time from long upper/lower wick (50% of 15m range)
#   fake — time from fake up/down (>=10 bars vs 15m open)

# Baseline reproduction: docs/v14_2398_baseline.md
# Per-pattern feature_set: retrace=v2398 (93 feats), breakthrough=current (96 feats).
# Backtest/inference always builds the current (96-col) matrix — retrace models use a subset.

PATTERN_REGISTRY = {
    "uptrend_retrace": {
        "direction_bias": "long",
        "priority": 1,
        "feature_set": "v2398",
        "thresholds": {"prob": 0.45},
        # 240m: rallied ≥$30 off 240m low, pulled back ≥$10 from 240m high
        "context": [],
        "pattern": [
            {"feat": "rise_from_low_240", "op": ">=", "val": 30.0},
            {"feat": "drop_from_high_240", "op": ">=", "val": 10.0},
        ],
        "exclude": [
            {"feat": "near_high_zone", "op": "==", "val": 1.0},
        ],
        "execution": {"horizon": 15, "tp": 20.0, "sl": 15.0},
    },
    "downtrend_retrace": {
        "direction_bias": "short",
        "priority": 2,
        "feature_set": "v2398",
        "thresholds": {"prob": 0.45},
        # 240m: fell ≥$25 from 240m high, bounced ≥$5 off 240m low
        "context": [],
        "pattern": [
            {"feat": "drop_from_high_240", "op": ">=", "val": 25.0},
            {"feat": "rise_from_low_240", "op": ">=", "val": 5.0},
        ],
        "exclude": [
            {"feat": "near_low_zone", "op": "==", "val": 1.0},
        ],
        "execution": {"horizon": 15, "tp": 40.0, "sl": 30.0},
    },
    "breakthrough_long": {
        "direction_bias": "long",
        "priority": 10,
        "feature_set": "current",
        "thresholds": {"prob": 0.45},
        # WR(90) > -30 and ret_3m > 4
        "context": [],
        "pattern": [
            {"feat": "c15_breakthrough_up", "op": "==", "val": 1.0},
        ],
        "exclude": [],
        "execution": {"horizon": 15, "tp": 40.0, "sl": 20.0},
    },
    "breakthrough_short": {
        "direction_bias": "short",
        "priority": 11,
        "feature_set": "current",
        "thresholds": {"prob": 0.45},
        # WR(90) < -70 and ret_3m < -10
        "context": [],
        "pattern": [
            {"feat": "c15_breakthrough_down", "op": "==", "val": 1.0},
        ],
        "exclude": [],
        "execution": {"horizon": 30, "tp": 40.0, "sl": 30.0},
    },
    # --- Reversal trials (15m PA: wick / FVG within 30m) — add one at a time via try_add_pattern.py
    "reversal_wick_long": {
        "direction_bias": "long",
        "priority": 12,
        "feature_set": "current",
        "pa_groups": ("wick",),
        "thresholds": {"prob": 0.55},
        "context": [],
        "pattern": [
            {"feat": "time_from_long_lower_wick", "op": "<", "val": 30.0},
        ],
        "router": [
            {"feat": "time_from_long_lower_wick", "op": "<", "val": 45.0},
        ],
        "exclude": [],
        "execution": {"horizon": 15, "tp": 20.0, "sl": 15.0},
    },
    "reversal_fvg_long": {
        "direction_bias": "long",
        "priority": 6,
        "feature_set": "current",
        "pa_groups": ("fvg",),
        "thresholds": {"prob": 0.45},
        "context": [],
        "pattern": [
            {"feat": "time_from_fvg_bull", "op": "<", "val": 30.0},
        ],
        "router": [
            {"feat": "time_from_fvg_bull", "op": "<", "val": 45.0},
        ],
        "exclude": [],
        "execution": {"horizon": 15, "tp": 20.0, "sl": 15.0},
    },
    "reversal_wick_short": {
        "direction_bias": "short",
        "priority": 13,
        "feature_set": "current",
        "pa_groups": ("wick",),
        "thresholds": {"prob": 0.55},
        "context": [],
        "pattern": [
            {"feat": "time_from_long_upper_wick", "op": "<", "val": 30.0},
        ],
        "router": [
            {"feat": "time_from_long_upper_wick", "op": "<", "val": 60.0},
        ],
        "exclude": [],
        "execution": {"horizon": 15, "tp": 40.0, "sl": 30.0},
    },
    "reversal_fvg_short": {
        "direction_bias": "short",
        "priority": 8,
        "feature_set": "current",
        "pa_groups": ("fvg",),
        "thresholds": {"prob": 0.45},
        "context": [],
        "pattern": [
            {"feat": "time_from_fvg_bear", "op": "<", "val": 30.0},
        ],
        "router": [
            {"feat": "time_from_fvg_bear", "op": "<", "val": 60.0},
        ],
        "exclude": [],
        "execution": {"horizon": 15, "tp": 40.0, "sl": 30.0},
    },
}

BASELINE_PATTERNS: tuple[str, ...] = (
    "uptrend_retrace",
    "downtrend_retrace",
    "breakthrough_long",
    "breakthrough_short",
)

# Production router: 4-pattern baseline + both FVG reversals (see docs/v14_6pattern_baseline.md).
PRODUCTION_PATTERNS: tuple[str, ...] = (
    *BASELINE_PATTERNS,
    "reversal_fvg_long",
    "reversal_fvg_short",
)

REVERSAL_TRIAL_PATTERNS: tuple[str, ...] = (
    "reversal_wick_long",
    "reversal_fvg_long",
    "reversal_wick_short",
    "reversal_fvg_short",
)

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
    "atr_threshold",
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
    "c15_breakthrough_up",
    "c15_breakthrough_down",
    "pattern_name",
    "pattern_id",
}


def pattern_prob_override() -> float | None:
    """Env override for pattern probability threshold (e.g. V14_PATTERN_PROB_BASE=0.45)."""
    raw = os.environ.get("V14_PATTERN_PROB_BASE", "").strip()
    return float(raw) if raw else None


def pattern_feature_set() -> str:
    """Env override for single-set training/sweeps. Prefer per-pattern feature_set in registry."""
    return os.environ.get("V14_PATTERN_FEATURE_SET", "current").strip().lower()


def pattern_feature_set_for(pattern_name: str) -> str:
    """Per-pattern training feature set (v2398=93 cols, current=96 cols)."""
    spec = PATTERN_REGISTRY.get(pattern_name, {})
    env = os.environ.get("V14_PATTERN_FEATURE_SET", "").strip().lower()
    if env:
        return env
    return str(spec.get("feature_set", "current")).strip().lower()


def backtest_feature_set() -> str:
    """Inference matrix must be the widest set (current=96) for mixed-model backtests."""
    return os.environ.get("V14_BACKTEST_FEATURE_SET", "current").strip().lower()


def collect_pa_groups(pattern_names: list[str] | None = None) -> tuple[str, ...]:
    """Union of pa_groups across active patterns (for feature prep)."""
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
