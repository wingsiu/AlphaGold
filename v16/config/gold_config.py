"""v16 gold production — max-PnL portfolio (hybrid patterns + momentum + dip short)."""

from pathlib import Path

from v16._paths import PROJECT_ROOT

GOLD_TRAIN_START = "2024-01-01"

# Momentum pre-close struct-hold (winner)
MOMENTUM = {
    "model": "et",
    "ml_prob": 0.50,
    "horizon": 720,
    "retrain_freq": "14D",
}

# Dip short rip ML winner
DIP_SHORT = {
    "ml_prob": 0.70,
}

BACKTEST = {
    "default_start": "2025-06-01",
    "default_end": "2026-06-25",
    "trades_csv": "runtime/gold_v16_combined_trades.csv",
    "parity_csv": "runtime/gold_v16_parity_latest.txt",
    "hybrid_cache": "runtime/gold_v16_hybrid_trades.csv",
    "mom_cache": "runtime/gold_v16_mom_trades.csv",
    "dip_cache": "runtime/gold_v16_dip_trades.csv",
}

# Merge tie-break (lower = first at same entry minute)
LEG_PRIORITY: dict[str, int] = {
    "v16_dip_short": 0,
    "dip_short_rip": 0,
    "v16_momentum": 14,
    "energetic": 25,
    "pattern": 10,
}
