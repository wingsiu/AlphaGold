"""v16 oil — WR90 + retrace short + SI (rip-short disabled)."""

from pathlib import Path

from v16._paths import PROJECT_ROOT

OIL_MODEL_DIR = PROJECT_ROOT / "v16" / "oil" / "wf_models"

# Walk-forward ML (aligned with gold v16 / v15 hybrid)
OIL_ML_CONFIG = {
    "train_days": 45,
    "min_train_rows": 80,
    "retrain_days": 14,
    "retrain_freq": "14D",  # 14D | M
    "wf_start": "2024-07-01",  # first OOS month (like v15 oil)
}

# Default models per leg (overridden after model_search)
OIL_LEG_MODELS = {
    "wr90": {"model": "lgb", "ml_th": 0.55},
    "ret": {"model": "xgb", "ml_th": 0.55},
    "ret_short": {"model": "xgb", "ml_th": 0.55},
    "long_ret": {"model": "xgb", "ml_th": 0.50},
    "si": {"model": "et", "ml_th": 0.50},
    "rip": {"model": "et", "ml_th": 0.65},
}

# --- WR90 long (Option 1 strict prod) ---
WR90 = {
    "entry": -80,
    "cv": 15000,
    "ep_min": 3,
    "tp": 80,
    "sl": 30,
    "max_bars": 60,
    "ny_close_h": 14,
    "ny_close_m": 28,
    "default_exit": "struct_hold",  # struct_hold | fixed_tpsl — struct_hold +1.7k OOS
}

WR90_STRUCT_HOLD = {
    **WR90,
    "exit_mode": "struct_hold",  # struct_hold | fixed_tpsl
    "tp_enabled": False,
    "horizon_minutes": 720,  # 12h safety (48 × 15m bars)
    "exit_on_structure_change": True,
    "exit_on_swing_break": True,
    "structure": {"rule": "15min", "atr_mult": 3.0, "atr_period": 14},
}

# --- Oil retrace (15m long fade — red bar after rally from Dlow; prod "ret" leg) ---
RETRACE = {
    "dlow": 20,
    "rng": 30,
    "chg": -10,
    "wick": 16,
    "tp": 30,
    "sl": 15,
}

# --- Long retrace 15m (mirror: green bar after pullback from Dhigh) ---
LONG_RETRACE_15M = {
    "dhigh": 20,       # Dhigh - close >= 20 (pulled back from day high)
    "rng": 30,
    "chg": 10,         # close - open > 10 (green bounce bar)
    "wick": 16,        # upper wick < 16
    "tp": 30,
    "sl": 15,
    "max_bars": 60,
}

LONG_RETRACE_15M_FEATS = [
    "cah", "avg_r3", "bc", "uw", "range", "ret_1b", "ret_3b", "ret_5b",
    "vol_r", "h_dhigh", "l_dhigh", "body", "up", "up_p1", "up_p2", "body_p1", "range_p1",
]

# --- Gold-style dip long 15m on 1m (prev 15m down, slot down, early dip) ---
OIL_DIP_LONG_15M = {
    "sessions": ("ny",),
    "dip_min_below_open_pts": 0.50,
    "dip_min_slot_low_pts": 0.40,
    "dip_max_minute_in_slot": 10,
    "dip_require_slot_low": True,
    "dip_require_prev_down": True,
    "dip_require_two_prev_down": False,
    "dip_min_prev_body_pts": 0.0,
    "dip_min_prev_range_pts": 0.0,
    "execution": {"tp": 30.0, "sl": 20.0, "horizon": 45},
}

# --- Short impulse (1m) ---
SHORT_IMPULSE = {
    "change_max": -14.0,
    "vol_min": 800,
    "tp": 120,
    "sl": 80,
    "max_bars": 90,
}

# --- Oil rip short (v16 dip_short_rip port, oil-scaled pts) ---
OIL_RIP_SHORT = {
    "name": "oil_rip_short",
    "priority": 0,
    "sessions": ("ny",),  # oil bot is NY-focused
    "router": [
        ("prev_15m_dir", ">=", 1.0),
        ("slot_up", ">=", 1.0),
        ("slot_rip_pts", ">=", 0.50),
        ("minute_in_15m", "<", 10.0),
    ],
    "ml_prob": 0.65,
    "execution": {"tp": 25.0, "sl": 20.0, "horizon": 45},
    "execution_mechanical": {"tp": 30.0, "sl": 20.0, "horizon": 60},
    "ml_label_source": "execution",
    "same_dir_refresh": "entry",
    "model_subdir": "rip",
}

# --- Retrace short 15m (fade green bar when extended from Dlow) ---
RET_SHORT = {
    "dlow": 20,       # close - Dlow >= 20 (elevated from day low)
    "rng": 30,
    "chg": 10,        # close - open > 10 (green extension bar)
    "wick": 16,       # upper wick < 16
    "tp": 30,
    "sl": 15,
}

RET_SHORT_FEATS = [
    "cad", "avg_r3", "bc", "uw", "range", "ret_1b", "ret_3b", "ret_5b",
    "vol_r", "h_dlow", "l_dlow", "body", "up", "up_p1", "up_p2", "body_p1", "range_p1",
]

# Structure gates per direction
STRUCTURE_GATE = {
    "enabled": True,
    "long_min_trend": 0,   # ret / long_ret: struct_trend >= 0
    "short_max_trend": 0,  # ret_short / SI: struct_trend <= 0
}

BACKTEST = {
    "default_start": "2024-01-01",
    "default_end": "2026-06-30",
    "trades_csv": "runtime/oil_v16_combined_trades.csv",
    "stats_txt": "runtime/oil_v16_full_statistics.txt",
    "model_search_csv": "runtime/oil_v16_model_search.csv",
    "include_rip_short": False,  # experimental; ~9 signals OOS — skip
}
