"""
v15 Configuration — extends v14 with regime adaptation + Kelly position sizing.

Layers on top of v14's pattern-first architecture:
  - Regime-aware thresholds, stops, targets
  - Kelly-based position sizing
  - Feature normalization by regime vol
"""

from __future__ import annotations

import os
from datetime import date as _date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# 1. Energetic Gate (unchanged from v14)
# =============================================================================
FILTER_CONFIG = {
    "min_bar_move": 3.0,
    "min_volume": 200,
}

# =============================================================================
# 2. Target Definition (unchanged from v14)
# =============================================================================
TARGET_CONFIG = {
    "horizon": 30,
    "tp": 20.0,
    "sl": 15.0,
    "tp_ratio": 2.0,
    "sl_ratio": 1.5,
    "min_tp": 5.0,
    "min_sl": 5.0,
}

# =============================================================================
# 3. Walk-Forward Training (unchanged from v14)
# =============================================================================
WF_CONFIG = {
    "retrain_days": 14,
    "anchor_date": _date(2025, 1, 3),
    "min_train_bars": 5000,
    "gap_bars": 500,
    "min_trades_per_cycle": 3,
}

# =============================================================================
# 4. Model Paths
# =============================================================================
MODEL_DIR = PROJECT_ROOT / "runtime" / "bot_assets" / "v15_models"
PATTERN_MODEL_DIR = PROJECT_ROOT / "runtime" / "bot_assets" / "v15_pattern_models"
S1_MODEL_PATH = MODEL_DIR / "s1_filter_v15.joblib"
S2_MODEL_PATH = MODEL_DIR / "s2_directional_v15.joblib"
HMM_MODEL_PATH = PROJECT_ROOT / "runtime" / "bot_assets" / "hmm_regime_v15.joblib"

# =============================================================================
# 5. HMM Regime Configuration
# =============================================================================
REGIME_CONFIG = {
    # Number of HMM hidden states (3: low_vol, trending, high_vol)
    "n_states": 3,
    # Features used for HMM fitting
    "hmm_features": ["ret_5m", "ret_15m", "range_ratio_20", "volume_ratio_20"],
    # Lookback bars for HMM probability computation
    "hmm_lookback": 500,
    # Re-fit HMM every N bars (live) / every WF cycle (backtest)
    "hmm_refit_interval": 2000,
    # Minimum bars before HMM regime is considered valid
    "hmm_min_warmup": 500,
}

# =============================================================================
# 6. Per-Regime Adaptation
# =============================================================================
# Regime labels: 0=low_vol, 1=trending, 2=high_vol
# These are calibrated via calibrate_regime.py sweep
REGIME_THRESHOLDS = {
    # Low volatility: tighter stops, lower prob threshold (fewer but cleaner signals)
    0: {
        "label": "low_vol",
        "s1_prob_threshold": 0.45,
        "pattern_prob_threshold": 0.40,
        "tp_multiplier": 0.7,      # scale TP distance by 0.7x
        "sl_multiplier": 0.8,      # scale SL distance by 0.8x
        "max_holding_bars": 25,    # tighter timeout
        "kelly_fraction": 0.15,    # slightly more aggressive in quiet markets
        "close_on_reverse": True,  # get out fast on reversal
        "description": "Quiet, range-bound. Small moves, lower thresholds, tighter stops.",
    },
    # Trending: wider stops, higher threshold (quality over quantity)
    1: {
        "label": "trending",
        "s1_prob_threshold": 0.50,
        "pattern_prob_threshold": 0.45,
        "tp_multiplier": 1.2,      # let winners run further
        "sl_multiplier": 1.0,      # standard stop
        "max_holding_bars": 35,    # longer timeout — trends take time
        "kelly_fraction": 0.13,    # standard half-Kelly
        "close_on_reverse": False,  # let winners run, trail stop instead
        "description": "Directional trend. Wider targets, standard sizing.",
    },
    # High volatility: wide stops, highest threshold (avoid chop)
    2: {
        "label": "high_vol",
        "s1_prob_threshold": 0.55,
        "pattern_prob_threshold": 0.50,
        "tp_multiplier": 1.5,      # bigger targets in volatile moves
        "sl_multiplier": 1.4,      # wider stops to avoid noise-shakeout
        "max_holding_bars": 20,    # shorter timeout — vol moves are fast
        "kelly_fraction": 0.08,    # reduced size in chaotic markets
        "close_on_reverse": True,  # get out fast
        "description": "High vol, choppy. Highest thresholds, wider stops, reduced size.",
    },
}

# =============================================================================
# 7. Kelly Position Sizing
# =============================================================================
KELLY_CONFIG = {
    # Default Kelly fraction (half-Kelly = 0.5 * f*)
    # Override with V15_KELLY_FRACTION env var
    "default_fraction": 0.13,
    # Window for trailing edge estimation (number of recent trades)
    "trailing_window": 20,
    # Minimum trades before Kelly sizing activates (else use fixed)
    "min_trades_for_kelly": 10,
    # Fixed size fallback (set >0 to disable Kelly entirely)
    "fixed_size": float(os.environ.get("V15_FIXED_SIZE", "0")),
    # Size bounds
    "max_size": 5.0,
    "min_size": 0.5,
    # Default account equity (points, not currency)
    "initial_equity": 500.0,
    # Maximum drawdown before halving Kelly fraction
    "max_dd_kelly_cut": 0.25,  # 25% DD → half the Kelly fraction
}

# =============================================================================
# 8. Execution Defaults (per-pattern overrides in v14_patterns.py)
# =============================================================================
EXECUTION_DEFAULTS = {
    "tp": 30.0,
    "sl": 25.0,
    "horizon": 30,
    "s1_threshold": 0.50,
    # v15 additions
    "trailing_stop_activation": 10.0,  # activate trailing stop after N pts profit
    "trailing_stop_distance": 5.0,     # trail by N pts
}

# =============================================================================
# 9. Time Filter (unchanged from v14)
# =============================================================================
TIME_FILTER = {
    "hkt_only": True,
    "skip_weekends": True,
    "weak_slots_path": PROJECT_ROOT / "runtime" / "v14_weak_time_slots.json",
    "min_hour_hkt": 8,
    "max_hour_hkt": 23,
}

# =============================================================================
# 10. Feature Engineering
# =============================================================================
FEATURE_CONFIG = {
    "lookback_bars": 150,
    "channels": [
        "open_rel", "high_rel", "low_rel", "close_rel",
        "volume", "spread", "range",
        "ret_1m", "ret_3m", "ret_5m", "ret_15m", "ret_30m",
    ],
    # Regime-normalized: these features get divided by regime_vol
    "normalize_by_regime": [
        "bar_move", "range", "ret_1m", "ret_3m", "ret_5m",
        "high_rel", "low_rel",
    ],
    # Price action features from 15m bars
    "pa_groups": ("fvg", "wick"),
}

# =============================================================================
# 11. Logging & Output
# =============================================================================
LOG_DIR = PROJECT_ROOT / "runtime"
BACKTEST_OUTPUT = LOG_DIR / "v15_backtest_trades.csv"
BACKTEST_LOG = LOG_DIR / "v15_backtest.log"
REGIME_LOG = LOG_DIR / "v15_regime.log"


def get_regime_config(regime_id: int) -> dict:
    """Get config dict for a given regime, with env override support."""
    base = REGIME_THRESHOLDS.get(regime_id, REGIME_THRESHOLDS[1])
    cfg = dict(base)

    # Env overrides for threshold sweeping
    for key in ("s1_prob_threshold", "pattern_prob_threshold"):
        env_key = f"V15_REGIME{regime_id}_{key.upper()}"
        raw = os.environ.get(env_key, "").strip()
        if raw:
            cfg[key] = float(raw)

    for key in ("tp_multiplier", "sl_multiplier", "kelly_fraction"):
        env_key = f"V15_REGIME{regime_id}_{key.upper()}"
        raw = os.environ.get(env_key, "").strip()
        if raw:
            cfg[key] = float(raw)

    return cfg


def get_kelly_fraction() -> float:
    """Get active Kelly fraction (env overridable).
    
    Returns 0.0 if V15_FIXED_SIZE is set (signals fixed sizing mode).
    Otherwise returns V15_KELLY_FRACTION env var or default.
    """
    fixed = float(os.environ.get("V15_FIXED_SIZE", "0"))
    if fixed > 0:
        return 0.0  # signal to use fixed sizing
    raw = os.environ.get("V15_KELLY_FRACTION", "").strip()
    return float(raw) if raw else KELLY_CONFIG["default_fraction"]
