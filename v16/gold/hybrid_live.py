"""v16 gold hybrid live scorer — deterministic energetic gate (no v15/ imports)."""
from __future__ import annotations

import logging
import os as _os

import pandas as pd

from config.hybrid_config import ENERGETIC_EXECUTION_CONFIG
from v16.gold.energetic import energetic_bar_mask
from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold
from xgboost_filter_model.energetic_gate import s1_feature_columns, s2_feature_columns
from xgboost_filter_model.hybrid_live import HybridLiveScorer, LiveSignal


class V16HybridLiveScorer(HybridLiveScorer):
    """Hybrid pattern + energetic live scorer for gold v16 production."""

    def __init__(self, logger: logging.Logger | None = None):
        super().__init__(logger)
        self.logger.info("v16 scorer: deterministic energetic gate (no HMM)")

    def score_energetic(self, df, ts, consecutive_losses=0):
        if ts not in df.index:
            return None
        if not bool(energetic_bar_mask(df.loc[[ts]]).iloc[0]):
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
