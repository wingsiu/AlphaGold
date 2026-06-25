"""Point shared pattern imports at the oil pattern registry before training/backtest."""

from __future__ import annotations

import os

import config.pattern_registry as gold_patterns
import oil.patterns as oil_patterns
import xgboost_filter_model.pattern_router as pattern_router
import xgboost_filter_model.pattern_training as pattern_training

from oil.config import PRICE_TABLE


def apply_oil_registry() -> None:
    """Patch modules that imported PATTERN_REGISTRY at load time."""
    for mod in (gold_patterns, pattern_training, pattern_router):
        mod.PATTERN_REGISTRY = oil_patterns.PATTERN_REGISTRY
        mod.PATTERN_MODEL_DIR = oil_patterns.PATTERN_MODEL_DIR
        mod.PRODUCTION_PATTERNS = oil_patterns.PRODUCTION_PATTERNS
        if hasattr(mod, "BASELINE_PATTERNS"):
            mod.BASELINE_PATTERNS = oil_patterns.PRODUCTION_PATTERNS
        if hasattr(mod, "EXCLUDE_COLS"):
            mod.EXCLUDE_COLS = oil_patterns.EXCLUDE_COLS
    os.environ["V14_PRICE_TABLE"] = PRICE_TABLE
    os.environ.setdefault("V14_FVG_MIN_GAP", "0")
