"""Live scoring for hybrid router — pattern-first, energetic fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config.v14_config import ENERGETIC_EXECUTION_CONFIG, EXECUTION_CONFIG, WF_CONFIG
from config.v14_patterns import PATTERN_MODEL_DIR, PRODUCTION_PATTERNS, backtest_feature_set, collect_pa_groups
from xgboost_filter_model.energetic_gate import energetic_bar_mask, s1_feature_columns, s2_feature_columns
from xgboost_filter_model.pattern_router import assign_patterns
from xgboost_filter_model.pattern_training import (
    cycle_model_path,
    pattern_variant_tag,
    prod_model_path,
    wf_cycle_at,
)
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LiveSignal:
    source: str  # "pattern" | "energetic"
    side: int
    tp: float
    sl: float
    horizon: int
    probability: float = 0.0
    pattern_name: Optional[str] = None
    s1_prob: Optional[float] = None
    s2_prob: Optional[float] = None


class HybridLiveScorer:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("HybridLiveScorer")
        self.pattern_names = list(PRODUCTION_PATTERNS)
        self.pattern_models: dict[str, dict] = {}
        self.s1_model = None
        self.s2_model = None
        self.wf_dir = PROJECT_ROOT / WF_CONFIG.get("model_output_dir", "runtime/bot_assets/wf_models_v14")
        self._load_all()

    def _load_all(self) -> None:
        self._load_pattern_models()
        self._load_energetic_models()

    def reload(self) -> None:
        self._load_all()

    def _load_pattern_models(self) -> None:
        from config.v14_patterns import PATTERN_REGISTRY

        self.pattern_models = {}
        for name in self.pattern_names:
            spec = PATTERN_REGISTRY[name]
            ex = spec["execution"]
            variant = pattern_variant_tag(ex["horizon"], ex["tp"], ex["sl"])
            pdir = PATTERN_MODEL_DIR / name / variant
            mp = prod_model_path(pdir)
            if not mp.exists():
                raise FileNotFoundError(f"Missing pattern model: {mp}")
            self.pattern_models[name] = {
                "prod": joblib.load(mp),
                "spec": spec,
                "dir": pdir,
                "variant": variant,
            }
            self.logger.info(f"Pattern model loaded: {name} ({variant})")

    def _load_energetic_models(self) -> None:
        now = pd.Timestamp.now(tz="UTC")
        cycle_start, cycle = wf_cycle_at(now)
        s1_path = self.wf_dir / f"filter_v14_cycle_{cycle}_{cycle_start.date()}.joblib"
        s2_path = self.wf_dir / f"directional_v14_cycle_{cycle}_{cycle_start.date()}.joblib"
        prod_s1 = PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v14_wf.joblib"
        prod_s2 = PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v14_wf.joblib"
        self.s1_model = joblib.load(s1_path if s1_path.exists() else prod_s1)
        self.s2_model = joblib.load(s2_path if s2_path.exists() else prod_s2)
        self.logger.info(
            f"Energetic models: S1={s1_path.name if s1_path.exists() else prod_s1.name} "
            f"S2={s2_path.name if s2_path.exists() else prod_s2.name}"
        )

    def build_feature_df(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = prepare_data_v14(
            start_date=start_date,
            end_date=end_date,
            energetic_filter=False,
            pa_groups=collect_pa_groups(self.pattern_names),
            pattern_feature_set=backtest_feature_set(),
        )
        df = prepare_directional_data_v14(df)
        return assign_patterns(df)

    def _pattern_model_at(self, name: str, ts: pd.Timestamp):
        m = self.pattern_models[name]
        cycle_start, cycle = wf_cycle_at(ts)
        path = cycle_model_path(m["dir"], cycle, cycle_start.date())
        return joblib.load(path) if path.exists() else m["prod"]

    def _energetic_models_at(self, ts: pd.Timestamp):
        cycle_start, cycle = wf_cycle_at(ts)
        s1_path = self.wf_dir / f"filter_v14_cycle_{cycle}_{cycle_start.date()}.joblib"
        s2_path = self.wf_dir / f"directional_v14_cycle_{cycle}_{cycle_start.date()}.joblib"
        prod_s1 = PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v14_wf.joblib"
        prod_s2 = PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v14_wf.joblib"
        s1 = joblib.load(s1_path if s1_path.exists() else prod_s1)
        s2 = joblib.load(s2_path if s2_path.exists() else prod_s2)
        return s1, s2

    def score_pattern(self, df: pd.DataFrame, ts: pd.Timestamp) -> Optional[LiveSignal]:
        if ts not in df.index:
            return None
        row = df.loc[ts]
        pname = row.get("pattern_name")
        if pd.isna(pname) or pname not in self.pattern_models:
            return None
        m = self.pattern_models[str(pname)]
        spec = m["spec"]
        model = self._pattern_model_at(str(pname), ts)
        feats = list(model.feature_names_in_)
        prob = float(model.predict_proba(df.loc[[ts], feats])[:, 1][0])
        thresh = float(spec["thresholds"]["prob"])
        if prob < thresh:
            return None
        bias = spec["direction_bias"]
        side = 1 if bias == "long" else -1
        ex = spec["execution"]
        return LiveSignal(
            source="pattern",
            side=side,
            tp=float(ex["tp"]),
            sl=float(ex["sl"]),
            horizon=int(ex["horizon"]),
            probability=prob if side == 1 else 1.0 - prob,
            pattern_name=str(pname),
        )

    def score_energetic(
        self,
        df: pd.DataFrame,
        ts: pd.Timestamp,
        consecutive_losses: int = 0,
    ) -> Optional[LiveSignal]:
        if ts not in df.index:
            return None
        if not bool(energetic_bar_mask(df.loc[[ts]]).iloc[0]):
            return None
        s1_feats = s1_feature_columns(df)
        s2_feats = s2_feature_columns(df)
        s1, s2 = self._energetic_models_at(ts)
        s1_p = float(s1.predict_proba(df.loc[[ts], s1_feats])[:, 1][0])
        s1_thresh = float(ENERGETIC_EXECUTION_CONFIG["s1_threshold"])
        if s1_p < s1_thresh:
            return None
        s2_p = float(s2.predict_proba(df.loc[[ts], s2_feats])[:, 1][0])
        s2_base = float(ENERGETIC_EXECUTION_CONFIG["s2_threshold"])
        s2_inc = float(ENERGETIC_EXECUTION_CONFIG["s2_loss_increment"])
        s2_max = float(ENERGETIC_EXECUTION_CONFIG["s2_max_threshold"])
        dynamic_s2 = min(s2_max, s2_base + consecutive_losses * s2_inc)
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

    def score_bar(
        self,
        df: pd.DataFrame,
        ts: pd.Timestamp,
        consecutive_losses: int = 0,
    ) -> tuple[Optional[LiveSignal], Optional[LiveSignal]]:
        return (
            self.score_pattern(df, ts),
            self.score_energetic(df, ts, consecutive_losses=consecutive_losses),
        )
