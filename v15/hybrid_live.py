"""
v15 Live Hybrid Scorer (ATR TP/SL version)
============================================
Extends HybridLiveScorer with:
  - v15 deterministic energetic gate (no HMM)
  - ATR-scaled pattern TP/SL at signal time
  - v15 deterministic features
"""
from __future__ import annotations

import logging
import os as _os
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from xgboost_filter_model.hybrid_live import (
    HybridLiveScorer,
    LiveSignal,
    BarScoreSnapshot,
)
from xgboost_filter_model.energetic_gate import (
    s1_feature_columns,
    s2_feature_columns,
)
from config.hybrid_config import ENERGETIC_EXECUTION_CONFIG
from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold
from v15.energetic_gate import energetic_bar_mask_v15
from v15.config.v15_patterns import V15_PATTERN_REGISTRY


class V15HybridLiveScorer(HybridLiveScorer):
    """v15 live scorer with deterministic energetic gate (no HMM)."""

    def __init__(self, logger: logging.Logger | None = None):
        super().__init__(logger)
        self.logger.info("v15 scorer: deterministic energetic gate (no HMM)")

    def score_energetic(self, df, ts, consecutive_losses=0):
        """v15 deterministic energetic gate (no HMM). Duplicated from v15/hybrid_live.py."""
        if ts not in df.index:
            return None
        if not bool(energetic_bar_mask_v15(df.loc[[ts]]).iloc[0]):
            return None
        s1_feats = s1_feature_columns(df)
        s2_feats = s2_feature_columns(df)
        s1, s2 = self._energetic_models_at(ts)
        s1_p = float(s1.predict_proba(df.loc[[ts], s1_feats])[:, 1][0])
        s1_base = float(ENERGETIC_EXECUTION_CONFIG["s1_threshold"])
        if _os.environ.get("V14_ADAPTIVE_ENERGETIC", "0") not in ("0", "no", "false"):
            s1_adaptive = adaptive_prob_threshold(s1_base, df)
            s1_thresh = float(s1_adaptive.loc[ts]) if ts in s1_adaptive.index else s1_base
        else:
            s1_thresh = s1_base
        if s1_p < s1_thresh:
            return None
        s2_p = float(s2.predict_proba(df.loc[[ts], s2_feats])[:, 1][0])
        s2_base = float(ENERGETIC_EXECUTION_CONFIG["s2_threshold"])
        if _os.environ.get("V14_ADAPTIVE_ENERGETIC", "0") not in ("0", "no", "false"):
            s2_adaptive = adaptive_prob_threshold(s2_base, df)
            s2_vol_base = float(s2_adaptive.loc[ts]) if ts in s2_adaptive.index else s2_base
        else:
            s2_vol_base = s2_base
        s2_inc = float(ENERGETIC_EXECUTION_CONFIG["s2_loss_increment"])
        s2_max = float(ENERGETIC_EXECUTION_CONFIG["s2_max_threshold"])
        dynamic_s2 = min(s2_max, s2_vol_base + consecutive_losses * s2_inc)
        side = 0
        if s2_p >= dynamic_s2:
            side = 1
        elif s2_p <= (1.0 - dynamic_s2):
            side = -1
        if side == 0:
            return None
        return LiveSignal(
            source="energetic",
            side=side,
            tp=float(ENERGETIC_EXECUTION_CONFIG["tp"]),
            sl=float(ENERGETIC_EXECUTION_CONFIG["sl"]),
            horizon=int(ENERGETIC_EXECUTION_CONFIG["horizon"]),
            probability=s2_p if side == 1 else 1.0 - s2_p,
            s1_prob=s1_p,
            s2_prob=s2_p,
        )
