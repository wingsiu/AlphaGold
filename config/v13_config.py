#!/usr/bin/env python3
"""
Central configuration for AlphaGold v13 models and backtesting.
Consolidates all hyperparameters, thresholds, and trade logic.
"""
from datetime import date as _date

# 1. Data Filtering Threshholds (Energetic Segments)
FILTER_CONFIG = {
    "min_bar_move": 3.0,
    "min_volume": 250,
}

# 2. Target Definition (Stage 1 / Filter Model)
# These define what the model tries to learn (Ground Truth)
TARGET_CONFIG = {
    "tp": 20.0,
    "sl": 10.0,
    "horizon": 45,      # minutes
    "er_threshold": 0.3, # Updated from 0.1 to 0.3 as requested
    "move_threshold": 10.0
}

# 3. Model Hyperparameters (XGBoost)
MODEL_CONFIG = {
    "s1": {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    "s2": {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
}

# 4. Trading / Execution parameters
EXECUTION_CONFIG = {
    "tp": 40.0,         # Final execution TP (absolute points)
    "sl": 25.0,         # Final execution SL (absolute points)
    "horizon": 45,      # Timeout in minutes
    "s1_threshold": 0.5,
    "s2_threshold": 0.55,       # Base S2 threshold for entries
    "s2_loss_increment": 0.01,  # Raise entry bar by this after each consecutive loss
    "s2_max_threshold": 0.70,   # Cap on dynamic S2 threshold
    "spread_default": 0.25,
    "size": 2.0,        # Trade size (lots)
    "take_profit_pct": 0.8,  # IG API take-profit as % of entry price (~40 pts at 4700)
}

# 5. Walk-Forward settings
WF_CONFIG = {
    "retrain_days": 14,
    "full_start": "2020-01-01",
    # Align to first Friday after 2025-01-01 at 17:00 NY time (UTC: 22:00)
    "wf_start": "2025-01-03T22:00:00Z",  # 2025-01-03 is a Friday
    "feature_warmup_days": 120,
    "wf_end": _date.today().strftime("%Y-%m-%d"),   # auto-updates to today
    "model_output_dir": "runtime/bot_assets/wf_models_v13"
}
