"""v16 — standalone scalp system (no v15 hybrid)."""

DATA_CONFIG = {
    "instrument": "gold",
    "spread_pts": 0.25,
}

# Platform-style scale-out ladder
EXIT_CONFIG = {
    "first_scale_pnl": 5.0,
    "first_scale_frac": 0.5,
    "final_scale_pnl": 10.0,
    "initial_sl": 20.0,
    "runner_lock_pnl": 5.0,
    "horizon_minutes": 10,
}

# Signal mode: "burst" | "fade_15m" | "dip_long_15m" | "both"
SIGNAL_CONFIG = {
    "mode": "burst",
    "min_range_pts": 4.0,
    "min_volume_ratio": 1.4,
    "min_body_pts": 1.5,
    "sample_stride": 3,
    "sessions": ("london", "ny"),
    # Fade prior large 15m bar at start of new slot
    "fade_open_minutes": (0, 1, 2),
    "fade_minutes_strict": (0,),
    "fade_min_prev_body_pts": 10.0,
    "fade_min_prev_range_pts": 12.0,
    # Double-down dip long: prev 15m down, slot down, price >5 below slot open
    "dip_min_below_open_pts": 5.0,
    "dip_min_slot_low_pts": 10.0,
    "dip_max_minute_in_slot": 10,
    "dip_require_slot_low": True,
    "dip_require_prev_down": True,
    "dip_require_two_prev_down": False,  # set True for two consecutive 15m down bars
    "dip_min_prev_body_pts": 0.0,   # |prev down move| >= N (0 = off)
    "dip_min_prev_range_pts": 0.0,
    "dip_min_prev2_body_pts": 0.0,  # 2 bars ago filters (0 = off)
    "dip_min_prev2_range_pts": 0.0,
    # Symmetric short: two 15m up + slot up + rip above open
    "dip_short_min_above_open_pts": 5.0,
    "dip_short_min_slot_high_pts": 10.0,
    "dip_require_two_prev_up": False,
}

# Walk-forward ML — align retrain grid with v15 production (config/hybrid_config.py WF_CONFIG)
ML_CONFIG = {
    "train_days": 45,
    "min_train_rows": 120,
    "prob_threshold": 0.58,
    "min_edge": 0.05,
    "retrain_days": 14,       # v15 WF_CONFIG["retrain_days"]
    "retrain_freq": "14D",    # 14D=v15 anchored grid | M=monthly | 2W=rolling biweekly
}

# Research winner vs v15 (Jun 2025→Jun 2026): pattern + energetic + burst ML union
# See v16/research/pattern_burst_ml_hybrid.py — +4413 vs v15 +4282
UNION_FALLBACK_CONFIG = {
    "burst_prob_threshold": 0.65,
    "burst_min_edge": 0.08,
    "burst_exit_tp": 30.0,
    "burst_exit_sl": 25.0,
    "burst_exit_horizon": 30,
}

BACKTEST_CONFIG = {
    "default_start": "2025-06-01",
    "trades_csv": "runtime/v16_scalp_trades.csv",
}

# dip_short_rip — v16 pattern lane (not in production v15)
# Sweep: v16/research/dip_short_rip_tpsl_sweep.py (Jun 2025→Jun 2026)
DIP_SHORT_RIP = {
    "name": "dip_short_rip",
    "priority": 0,
    "sessions": ("london", "ny"),
    "router": [
        ("prev_15m_dir", ">=", 1.0),
        ("slot_up", ">=", 1.0),
        ("slot_rip_pts", ">=", 5.0),
        ("minute_in_15m", "<", 10.0),
    ],
    "dip_require_two_prev_up": False,
    "dip_min_prev_body_pts": 0.0,
    "ml_prob": 0.70,
    # scaleout labels for WF; execution labels underperform in sweep
    "ml_label_source": "scaleout",
    # ML sweep winner p>=0.70 (~292 tr, +918) — v16/research/dip_short_rip_tpsl_sweep.py
    "execution": {"tp": 35.0, "sl": 35.0, "horizon": 45},
    # Mechanical sweep winner (~797 tr, +1216)
    "execution_mechanical": {"tp": 40.0, "sl": 30.0, "horizon": 60},
    # v15 pattern behaviour: one position; same-dir signal refreshes target + horizon
    "same_dir_refresh": "entry",
    "upgrade_stop": False,
    "model_dir": "runtime/v16_models/dip_short_rip",
}

# Structure trend hold — enter on with-trend retrace, exit on structure/swing break
# See v16/V16_WINNERS.md §3 and v16/research/structure_trend_hold_backtest.py
STRUCTURE_TREND_HOLD = {
    "name": "structure_trend_hold",
    "sessions": ("london", "ny"),
    "structure": {
        "enabled": True,
        "rule": "15min",
        "atr_mult": 3.0,
        "atr_period": 14,
    },
    "entry": {
        "min_pullback_pct": 0.15,
        "max_pullback_pct": 0.65,
        "min_leg_age_15m": 2,   # 30 min
        "max_leg_age_15m": 6,   # 90 min
    },
    "exit": {
        "on_structure_change": True,
        "on_swing_break": True,
        "horizon_minutes": 480,
        "min_pnl_on_structure_exit": -1e9,
    },
    "same_dir_refresh": "none",
}

# momentum_15m_hold — first 1m |body|>=5pt in 15m slot; LSTM filters at entry
MOMENTUM_15M_HOLD = {
    "name": "impulse_1m_15m",
    "sessions": ("london", "ny"),
    "min_move_pts": 3.0,          # sweep winner (body); was 5.0
    "change_mode": "body",        # "body" | "range"
    "entry_minute_in_slot": 0,    # first minute of new 15m slot (after prior slot closes)
    "entry_mode": "open",         # "open" | "pullback" | "breakout"
    "entry_pullback": {
        "fraction": 0.5,            # limit at 50% of impulse bar range
        "timeout_minutes": 5,       # skip if not filled within N 1m bars
        "cancel_on_stop_touch": True,  # skip if impulse H/L breached before fill
    },
    "entry_breakout": {
        "buffer_pts": 0.0,          # stop entry above/below impulse bar
        "timeout_minutes": 10,
        "fill": "next_open",        # trigger = at break level | next_open = bar after break
    },
    "entry_fill": {
        "mode": "ideal",            # ideal | conservative | pessimistic
        "slippage_pts": 0.25,
        "intrabar_stop_first": True,
        "cancel_on_stop_during_wait": False,
    },
    "same_dir_refresh": "entry",
    "upgrade_stop": False,
    "exit_mode": "scaleout",
    # impulse_stop: SL @ impulse bar H/L, TP = tp_multiple × SL (see impulse_stop_ml research)
    "impulse_stop": {
        "tp_multiple": 3.0,
        "horizon": 120,
        "min_sl_pts": 1.0,
        "max_sl_pts": 80.0,
    },
    # 15m ATR zigzag — ML + research (v16/structure/swing_zigzag.py)
    "structure": {
        "enabled": True,
        "rule": "15min",
        "atr_mult": 3.0,
        "atr_period": 14,
        "gate": {
            "enabled": False,
            "require_with_trend": True,
            "max_leg_age_15m": 2,       # None = off; fresh-leg filter
        },
    },
    "scaleout": {
        "first_scale_pnl": 5.0,
        "first_scale_frac": 0.5,
        "final_scale_pnl": 10.0,
        "initial_sl": 20.0,
        "runner_lock_pnl": 5.0,
        "horizon_minutes": 10,
    },
}

# breakout + 15m with-trend gate — v16/research/momentum_15m_hold_breakout_structure.py
# OOS Jun 2025→Jun 2026: ~963 tr, WR 44%, net ~+4740, avg +4.9 (vs breakout alone +11078 / 2478 tr)
MOMENTUM_BREAKOUT_STRUCTURE = {
    **MOMENTUM_15M_HOLD,
    "entry_mode": "breakout",
    "entry_breakout": {
        **MOMENTUM_15M_HOLD["entry_breakout"],
        "fill": "next_open",
    },
    "structure": {
        **MOMENTUM_15M_HOLD["structure"],
        "gate": {
            "enabled": True,
            "require_with_trend": True,
            "max_leg_age_15m": None,
        },
    },
}

# Stricter: fresh leg only (~191 tr, WR 51%, net ~+1220, avg +6.4)
MOMENTUM_BREAKOUT_STRUCTURE_FRESH = {
    **MOMENTUM_BREAKOUT_STRUCTURE,
    "structure": {
        **MOMENTUM_BREAKOUT_STRUCTURE["structure"],
        "gate": {
            "enabled": True,
            "require_with_trend": True,
            "max_leg_age_15m": 2,
        },
    },
}

# Walk-forward ML: breakout + with-trend + next-bar open (realistic fills)
MOMENTUM_BREAKOUT_ML = {
    **MOMENTUM_BREAKOUT_STRUCTURE,
    "entry_breakout": {
        **MOMENTUM_15M_HOLD["entry_breakout"],
        "fill": "next_open",
    },
    "ml_label_mode": "impulse_stop",
    "entry_fill": {
        "mode": "ideal",
        "slippage_pts": 0.0,
        "intrabar_stop_first": False,
        "cancel_on_stop_during_wait": False,
    },
}

# Ideal-fill breakout + with-trend — ML research (labels match perfect trigger fills)
MOMENTUM_BREAKOUT_IDEAL_ML = {
    **MOMENTUM_BREAKOUT_STRUCTURE,
    "entry_breakout": {
        **MOMENTUM_15M_HOLD["entry_breakout"],
        "fill": "trigger",
    },
    "ml_label_mode": "impulse_stop",
    "entry_fill": {
        "mode": "ideal",
        "slippage_pts": 0.0,
        "intrabar_stop_first": False,
        "cancel_on_stop_during_wait": False,
    },
}

# Ideal-fill breakout all signals (no structure gate) — upper-bound mechanical ~+11k OOS
# entry_breakout.fill = "trigger" (stop at impulse H/L level — optimistic)
MOMENTUM_BREAKOUT_ALL_IDEAL = {
    **MOMENTUM_15M_HOLD,
    "entry_mode": "breakout",
    "entry_breakout": {
        **MOMENTUM_15M_HOLD["entry_breakout"],
        "fill": "trigger",
    },
    "entry_fill": {
        "mode": "ideal",
        "slippage_pts": 0.0,
        "intrabar_stop_first": False,
        "cancel_on_stop_during_wait": False,
    },
    "structure": {
        **MOMENTUM_15M_HOLD["structure"],
        "gate": {"enabled": False, "require_with_trend": True, "max_leg_age_15m": None},
    },
    "ml_label_mode": "impulse_stop",
}

# Realistic breakout: enter at next 1m open after break (default in MOMENTUM_15M_HOLD)
MOMENTUM_BREAKOUT_NEXT_OPEN = {
    **MOMENTUM_15M_HOLD,
    "entry_mode": "breakout",
    "entry_breakout": {
        **MOMENTUM_15M_HOLD["entry_breakout"],
        "fill": "next_open",
    },
    "ml_label_mode": "impulse_stop",
}

# Breakout: bar BEFORE break closed within 10pt of level → next_open entry
# OOS Jun 2025→Jun 2026: struct-hold H=720 ML +2,793 | H=480 +2,780 | baseline R=3 +1,777
# Portfolio w/ dip_short_rip ML: +3,711
MOMENTUM_BREAKOUT_PRECLOSE = {
    **MOMENTUM_BREAKOUT_STRUCTURE,
    "entry_breakout": {
        **MOMENTUM_15M_HOLD["entry_breakout"],
        "fill": "next_open",
        "max_pre_break_close_dist_pts": 10.0,
        "max_close_dist_pts": None,
    },
    "impulse_stop": {
        **MOMENTUM_15M_HOLD["impulse_stop"],
        "horizon": 720,
        "tp_enabled": False,
        "tp_multiple": 3.0,
        "exit_on_structure_change": True,
        "exit_on_structure_change_min_pnl": -1e9,
    },
    "ml_label_mode": "impulse_stop",
    "ml_prob": 0.50,
    "ml_model": "et",
}

# |body|>=3pt + impulse volume>=200 + pre-close breakout + structure gate + ML
# OOS Jun 2025→Jun 2026 mech +1019 (572 tr); best ML LogReg +820 — below PRECLOSE winner
MOMENTUM_VOL3_PRECLOSE = {
    **MOMENTUM_BREAKOUT_PRECLOSE,
    "min_move_pts": 3.0,
    "change_mode": "body",
    "min_impulse_volume": 200.0,
    "ml_prob": 0.50,
    "ml_model": "lgb",
}

# OOS Jun 2025→Jun 2026: mech ~+1138 (779 tr) | ML ~+1150 (299 tr) WR 42%
MOMENTUM_OPEN_STRUCTURE_ML = {
    **MOMENTUM_15M_HOLD,
    "entry_mode": "open",
    "ml_label_mode": "impulse_stop",
    "impulse_stop": {
        **MOMENTUM_15M_HOLD["impulse_stop"],
        "tp_multiple": 3.0,
        "horizon": 90,
    },
    "structure": {
        **MOMENTUM_15M_HOLD["structure"],
        "gate": {
            "enabled": True,
            "require_with_trend": True,
            "max_leg_age_15m": None,
        },
    },
    "ml_prob": 0.50,
    "ml_model": "lgb",
}

# Same vol/change filters, open entry (no breakout / pre-close gate)
MOMENTUM_VOL3_OPEN = {
    **MOMENTUM_OPEN_STRUCTURE_ML,
    "min_move_pts": 3.0,
    "change_mode": "body",
    "min_impulse_volume": 200.0,
}

# ---------------------------------------------------------------------------
# V16 research winners — two lanes (see v16/V16_WINNERS.md)
# ---------------------------------------------------------------------------
V16_RESEARCH_WINNERS = {
    "momentum_preclose": "MOMENTUM_V16_WINNER_PRECLOSE",
    "momentum_open": "MOMENTUM_V16_WINNER",
    "dip_short_rip": "DIP_SHORT_RIP",
}

# ---------------------------------------------------------------------------
# v16 momentum winners — OOS Jun 2025 → Jun 2026, realistic next_open fills
# ---------------------------------------------------------------------------
#   1. MOMENTUM_V16_WINNER_PRECLOSE  ET ML struct-hold H=720  +2,793 (174 tr)
#   2. DIP_SHORT_RIP                 ML p≥0.70  ~+918 (292 tr)  — run together → +3,711
#   3. MOMENTUM baseline R=3 H=120 (legacy exit)  +1,777 ML
MOMENTUM_V16_WINNER = MOMENTUM_OPEN_STRUCTURE_ML
MOMENTUM_V16_WINNER_PRECLOSE = MOMENTUM_BREAKOUT_PRECLOSE

# Breakout, with-trend gate, vol≥200, ≥5 bars after impulse, SL @ 15m slot H/L, TP=3R
MOMENTUM_SLOT_BREAKOUT = {
    **MOMENTUM_15M_HOLD,
    "min_move_pts": 3.0,
    "change_mode": "body",
    "min_impulse_volume": 200.0,
    "min_bars_after_impulse": 5,
    "entry_mode": "breakout",
    "entry_breakout": {
        **MOMENTUM_15M_HOLD["entry_breakout"],
        "fill": "next_open",
        "max_pre_break_close_dist_pts": None,
        "max_close_dist_pts": None,
        "max_entry_gap_pts": None,
    },
    "ml_label_mode": "impulse_stop",
    "structure": {
        **MOMENTUM_15M_HOLD["structure"],
        "gate": {"enabled": True, "require_with_trend": True, "max_leg_age_15m": None},
    },
    "impulse_stop": {
        "stop_mode": "slot_15m",
        "tp_multiple": 3.0,
        "horizon": 120,
        "min_sl_pts": 1.0,
        "max_sl_pts": 150.0,
    },
}

# Fade impulse: uptrend + down impulse → long; downtrend + up impulse → short
MOMENTUM_SLOT_BREAKOUT_REVERSE = {
    **MOMENTUM_SLOT_BREAKOUT,
    "reverse_impulse": True,
}

# Oil v16 research — see v16/OIL_V16.md and v16/config/oil_config.py

DIP_SHORT_WINNER = {
    "mode": "dip_short_15m",
    "dip_require_two_prev_up": True,
    "dip_min_prev_body_pts": 8.0,
    "exit_tp": 30.0,
    "exit_sl": 25.0,
    "exit_horizon": 30,
    # ~425 trades, +1543
}

DIP_SHORT_ML_V15 = {
    "mode": "dip_short_15m",
    "dip_require_two_prev_up": False,
    "ml_prob": 0.55,
    "exit_tp": 30.0,
    "exit_sl": 25.0,
    "exit_horizon": 30,
    # ~1072 trades, +3536
}
