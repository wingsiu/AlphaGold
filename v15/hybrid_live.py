"""v15 Live Hybrid Scorer — v14 patterns + v15 deterministic energetic gate.

Extends xgboost_filter_model.hybrid_live.HybridLiveScorer, overriding
the energetic_bar_mask call to use v15's deterministic gate (no HMM).

This ensures live ↔ backtest signal parity since both use identical,
deterministic features computed from raw OHLCV only.
"""
from __future__ import annotations

import logging
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
from config.v14_config import ENERGETIC_EXECUTION_CONFIG
from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold
from v15.energetic_gate import energetic_bar_mask_v15


class V15HybridLiveScorer(HybridLiveScorer):
    """v15 live scorer — v14 patterns with v15 deterministic energetic gate."""

    def __init__(self, logger: logging.Logger | None = None):
        super().__init__(logger)
        self.logger.info("v15 scorer: using deterministic energetic gate (no HMM)")

    def build_feature_df_from_ohlcv(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Build v15 feature matrix — v14 path + v15 deterministic features."""
        df = super().build_feature_df_from_ohlcv(ohlcv)
        from v15.features import add_v15_energetic_features
        return add_v15_energetic_features(df)

    def score_energetic(
        self,
        df: pd.DataFrame,
        ts: pd.Timestamp,
        consecutive_losses: int = 0,
    ) -> Optional[LiveSignal]:
        """Override to use v15 deterministic gate instead of HMM."""
        import os

        if ts not in df.index:
            return None
        # v15: deterministic gate — no HMM dependency
        if not bool(energetic_bar_mask_v15(df.loc[[ts]]).iloc[0]):
            return None
        s1_feats = s1_feature_columns(df)
        s2_feats = s2_feature_columns(df)
        s1, s2 = self._energetic_models_at(ts)
        s1_p = float(s1.predict_proba(df.loc[[ts], s1_feats])[:, 1][0])

        # Volatility-adaptive S1 threshold
        s1_base = float(ENERGETIC_EXECUTION_CONFIG["s1_threshold"])
        if os.environ.get("V14_ADAPTIVE_ENERGETIC", "0") not in ("0", "no", "false"):
            s1_adaptive = adaptive_prob_threshold(s1_base, df)
            s1_thresh = float(s1_adaptive.loc[ts]) if ts in s1_adaptive.index else s1_base
        else:
            s1_thresh = s1_base
        if s1_p < s1_thresh:
            return None

        s2_p = float(s2.predict_proba(df.loc[[ts], s2_feats])[:, 1][0])
        s2_base = float(ENERGETIC_EXECUTION_CONFIG["s2_threshold"])
        if os.environ.get("V14_ADAPTIVE_ENERGETIC", "0") not in ("0", "no", "false"):
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

    def bar_score_snapshot(
        self,
        df: pd.DataFrame,
        ts: pd.Timestamp,
        consecutive_losses: int = 0,
    ) -> BarScoreSnapshot:
        """Override to use v15 deterministic gate."""
        import os

        snap = BarScoreSnapshot()
        if ts not in df.index:
            return snap

        row = df.loc[ts]
        pname = row.get("pattern_name")
        if not pd.isna(pname) and str(pname) in self.pattern_models:
            pname = str(pname)
            snap.routed_pattern = pname
            raw, conf, side, thresh = self._pattern_prob_at(df, ts, pname)
            snap.pattern_side = side
            snap.pattern_prob = round(conf, 4)
            snap.pattern_threshold = thresh
            snap.pattern_passes = raw >= thresh

        # v15: deterministic gate
        if bool(energetic_bar_mask_v15(df.loc[[ts]]).iloc[0]):
            snap.energetic_on_bar = True
            s1_feats = s1_feature_columns(df)
            s2_feats = s2_feature_columns(df)
            s1, s2 = self._energetic_models_at(ts)
            s1_p = float(s1.predict_proba(df.loc[[ts], s1_feats])[:, 1][0])
            s2_p = float(s2.predict_proba(df.loc[[ts], s2_feats])[:, 1][0])
            snap.s1_prob = round(s1_p, 4)
            snap.s2_prob = round(s2_p, 4)

            s1_base = float(ENERGETIC_EXECUTION_CONFIG["s1_threshold"])
            s2_base = float(ENERGETIC_EXECUTION_CONFIG["s2_threshold"])
            if os.environ.get("V14_ADAPTIVE_ENERGETIC", "0") not in ("0", "no", "false"):
                s1_adaptive = adaptive_prob_threshold(s1_base, df)
                s2_adaptive = adaptive_prob_threshold(s2_base, df)
                s1_thresh = float(s1_adaptive.loc[ts]) if ts in s1_adaptive.index else s1_base
                s2_vol_base = float(s2_adaptive.loc[ts]) if ts in s2_adaptive.index else s2_base
            else:
                s1_thresh = s1_base
                s2_vol_base = s2_base
            s2_inc = float(ENERGETIC_EXECUTION_CONFIG["s2_loss_increment"])
            s2_max = float(ENERGETIC_EXECUTION_CONFIG["s2_max_threshold"])
            dynamic_s2 = min(s2_max, s2_vol_base + consecutive_losses * s2_inc)
            if s1_p >= s1_thresh:
                if s2_p >= dynamic_s2:
                    snap.energetic_side = 1
                    snap.energetic_prob = round(s2_p, 4)
                    snap.energetic_passes = True
                elif s2_p <= (1.0 - dynamic_s2):
                    snap.energetic_side = -1
                    snap.energetic_prob = round(1.0 - s2_p, 4)
                    snap.energetic_passes = True

        return snap
