from datetime import date as _date

# 1. Data Filtering Threshholds (Energetic Segments)
FILTER_CONFIG = {
    "min_bar_move": 3.0,
    "min_volume": 200,
}

# 2. Target Definition (Stage 1 / Filter Model)
TARGET_CONFIG = {
    "tp": 30.0,
    "sl": 25.0,
    "horizon": 30,      # minutes
    "er_threshold": 0.3, 
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

# 4. Trading / Execution parameters (shared thresholds + pattern-router defaults)
_EXEC_BASE = {
    "tp": 30.0,
    "sl": 25.0,
    "horizon": 30,
    "s1_threshold": 0.50,
    "s2_threshold": 0.55,
    "s2_loss_increment": 0.01,
    "s2_max_threshold": 0.70,
    "spread_default": 0.25,
    "size": 3.0,
    "take_profit_pct": 0.6,
}

# Pattern router (2398 / 2764 baseline): no reverse exit, entry-level refresh.
EXECUTION_CONFIG = {
    **_EXEC_BASE,
    # "entry" = 2398 baseline: trail target / extend timeout using entry TP+H only.
    # "exec"  = upgrade TP/SL/H from new signal (max exec_*); use with upgrade_stop.
    # "global" = legacy global tp/horizon on refresh.
    # "none"  = ignore same-direction signals while open.
    "same_dir_refresh": "entry",
    "upgrade_stop": False,
    "close_on_reverse": False,
}

# Pure energetic S1/S2 (backtest_v14.py, hybrid fallback, live v14 bot).
# close_on_reverse + global refresh ≈ +2.2k–2.4k on 2025-06→2026-05 with time filter.
ENERGETIC_EXECUTION_CONFIG = {
    **_EXEC_BASE,
    "same_dir_refresh": "global",
    "upgrade_stop": False,
    "close_on_reverse": True,
}

# 5. Walk-Forward settings
WF_CONFIG = {
    "retrain_days": 14,
    "full_start": "2020-01-01",
    "wf_start": "2025-01-03T22:00:00Z",  # 2025-01-03 is a Friday
    "feature_warmup_days": 120,
    "wf_end": _date.today().strftime("%Y-%m-%d"),   # auto-updates to today
    "model_output_dir": "runtime/bot_assets/wf_models",
    # Train the NEXT cycle model only after the current 14d cycle ends (+ grace).
    # e.g. cycle 37 ends 2026-06-05 → train cycle_38 on/after 2026-06-06, not mid-cycle on 5/30.
    "wf_train_grace_days": 1,
}

# 6. Time-slot filter (v10-style session heatmaps; see run_hybrid_time_filter.py)
# Applies to pattern, energetic, and hybrid backtests when JSON exists.
# Disable at runtime: V14_NO_TIME_FILTER=1
#
# Weak cell rule (per HKT / London / NY session × weekday × hour):
#   trades > min_trades  AND  total_pnl < max_total_pnl
TIME_FILTER_CONFIG = {
    "enabled": True,
    "weak_slots_json": "runtime/hybrid_weak_time_slots.json",
    "min_trades": 3,              # block when trades > this (strictly)
    "min_trades_exclusive": True,  # True → > min_trades; False → >= min_trades (v10)
    "max_total_pnl": 0.0,         # block when total_pnl < this
    "max_win_rate": 40.0,         # only used if require_low_win_rate=True
    "require_low_win_rate": False,
}

# 7. Pattern router gates — stack energetic filter + S1 on pattern specialists.
# Override at runtime: V14_PATTERN_ENERGETIC_GATE=1  V14_PATTERN_S1_GATE=1
PATTERN_GATE_CONFIG = {
    "energetic_filter": False,
    "s1_gate": False,
    "s1_threshold": None,
}

# 8. Hybrid router — pattern entries first; energetic S1/S2 fallback when no pattern signal.
# Enable: V14_HYBRID=1
HYBRID_CONFIG = {
    "enabled": False,
    # Pattern open-position management (2398 baseline)
    "pattern_close_on_reverse": False,
    "pattern_same_dir_refresh": "entry",
    "pattern_upgrade_stop": False,
    # Energetic open-position management (classic S1/S2)
    "energetic_close_on_reverse": True,
    "energetic_same_dir_refresh": "global",
    "energetic_upgrade_stop": False,
}
