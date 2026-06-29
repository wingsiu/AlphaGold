"""Oil v14 pattern pipeline — mirrors gold WF calendar, separate models."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MySQL table synced via ig_scripts Price.Oil → "prices" (CC.D.CL.BMU.IP)
PRICE_TABLE = "prices"
OIL_IG_EPIC = "CC.D.CL.BMU.IP"

# Model output (do not mix with gold wf_models_v14_patterns)
PATTERN_MODEL_DIR = PROJECT_ROOT / "runtime" / "bot_assets" / "oil_pattern_models"
TRADES_CSV = PROJECT_ROOT / "runtime" / "oil_pattern_backtest_trades.csv"

# Align with gold v14 calendar (same OOS test window as docs/v14_2398_baseline)
FULL_START = "2022-06-01"  # first ~3y oil rows in DB (~2022-05-11)
WF_START = "2025-01-03T22:00:00Z"
WF_END = "2026-05-23"

# Gold-style holdout for first oil experiments (prod-only, no WF cycles)
TEST_START = "2025-06-01"
TEST_END = "2026-05-23"

# Train prod model on all pattern bars strictly before TEST_START
PROD_TRAIN_END = TEST_START

# IG oil quotes are ~100× spot (100 barrels); move features are in DB $ units.
# Same rule $ thresholds as gold (e.g. drop 25 ⇒ ~$0.25 spot) — scale 1.0.
SPOT_CONTRACT_MULTIPLIER = 100
THRESHOLD_SCALE = 1.0

# Train/backtest labels and sim: TP/SL = ATR × mult (DB units; scales with volatility).
TARGET_ATR_TP_MULT = 2.0
TARGET_ATR_SL_MULT = 1.5
