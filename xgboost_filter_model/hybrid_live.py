"""Live scoring for hybrid router — pattern-first, energetic fallback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config.v14_config import ENERGETIC_EXECUTION_CONFIG, EXECUTION_CONFIG, TARGET_CONFIG, WF_CONFIG
from config.v14_patterns import PATTERN_MODEL_DIR, PRODUCTION_PATTERNS, backtest_feature_set, collect_pa_groups
from xgboost_filter_model.energetic_gate import energetic_bar_mask, s1_feature_columns, s2_feature_columns
from xgboost_filter_model.pattern_router import assign_patterns
from xgboost_filter_model.pattern_training import (
    cycle_model_path,
    pattern_variant_tag,
    prod_model_path,
    wf_cycle_at,
)
from xgboost_filter_model.train_filter_v14 import add_v14_daily_features, add_v14_window_features, prepare_data_v14
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


@dataclass
class BarScoreSnapshot:
    """Per-bar scoring for journal / mobile UI (includes sub-threshold pattern reads)."""

    routed_pattern: Optional[str] = None
    pattern_side: int = 0
    pattern_prob: Optional[float] = None
    pattern_threshold: Optional[float] = None
    pattern_passes: bool = False
    energetic_on_bar: bool = False
    energetic_side: int = 0
    energetic_prob: Optional[float] = None
    s1_prob: Optional[float] = None
    s2_prob: Optional[float] = None
    energetic_passes: bool = False


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

    def build_feature_df_from_ohlcv(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Build hybrid feature matrix from in-memory OHLCV (live bot cache)."""
        from xgboost_filter_model.candle_pattern_15m import add_candle_pattern_15m
        from xgboost_filter_model.hmm_regime import add_hmm_regime
        from xgboost_filter_model.pattern_features import add_pattern_features
        from xgboost_filter_model.price_action_features import add_price_action_features
        from xgboost_filter_model.sudden_move_features import add_sudden_move_features
        from xgboost_filter_model.time_features import add_time_features
        from xgboost_filter_model.train_directional_model_v2 import add_directional_features
        from xgboost_filter_model.train_directional_model_v3 import add_ma_features
        from xgboost_filter_model.train_directional_model_v9 import add_momentum_features
        from xgboost_filter_model.train_filter_1min import prepare_features as prepare_base_features
        from xgboost_filter_model.train_filter_v10 import add_liquidity_indicators
        from xgboost_filter_model.volume_profile import add_volume_profile_features

        df = ohlcv.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = prepare_base_features(
            df,
            move_threshold=TARGET_CONFIG["move_threshold"],
            er_threshold=TARGET_CONFIG["er_threshold"],
            future_window=TARGET_CONFIG["horizon"],
            for_live_inference=True,
        )
        df = add_liquidity_indicators(df)
        df = add_hmm_regime(df)
        df = add_v14_daily_features(df)
        df = add_v14_window_features(df)
        if os.environ.get("V14_SUDDEN_RISE_A", "").strip() and os.environ.get(
            "V14_SUDDEN_DROP_B", ""
        ).strip():
            df = add_sudden_move_features(df)
        df = add_volume_profile_features(df)
        df = add_time_features(df)
        if os.environ.get("V14_CANDLE_15M", "").strip().lower() in ("1", "true", "yes", "on"):
            df = add_candle_pattern_15m(df)
        pa_groups = collect_pa_groups(self.pattern_names)
        if pa_groups:
            df = add_price_action_features(df, groups=pa_groups)

        pfs = backtest_feature_set()
        df = add_pattern_features(df, feature_set=pfs)
        df = add_directional_features(df)
        df = add_ma_features(df)
        df = add_momentum_features(df)
        df = self._trim_feature_warmup(df)
        return assign_patterns(df)

    @staticmethod
    def _trim_feature_warmup(df: pd.DataFrame) -> pd.DataFrame:
        """Drop leading warm-up NaN rows only; preserve the latest bars for live scoring."""
        if df.empty:
            return df
        core = [c for c in s1_feature_columns(df) if c in df.columns]
        if not core:
            return df
        ok = df[core].notna().all(axis=1)
        if not ok.any():
            return df.iloc[0:0]
        start = int(ok.argmax())
        return df.iloc[start:].copy()

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

    def _pattern_prob_at(
        self, df: pd.DataFrame, ts: pd.Timestamp, pname: str
    ) -> tuple[float, float, int, float]:
        """Return (model prob, same for UI, side, threshold).

        Target is P(pattern direction hits TP); class-1 prob is directional confidence
        for both long and short patterns (do not invert for short — that made UI show
        ~99% while raw was ~1% and pass=False).
        """
        m = self.pattern_models[str(pname)]
        spec = m["spec"]
        model = self._pattern_model_at(str(pname), ts)
        feats = list(model.feature_names_in_)
        raw = float(model.predict_proba(df.loc[[ts], feats])[:, 1][0])
        side = 1 if spec["direction_bias"] == "long" else -1
        base_thresh = float(spec["thresholds"]["prob"])
        # Env override for sweeps (e.g. V14_PATTERN_PROB_BASE=0.45)
        from config.v14_patterns import pattern_prob_override

        _override = pattern_prob_override()
        if _override is not None:
            base_thresh = _override

        # Adaptive threshold based on volatility regime
        from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold

        adaptive = adaptive_prob_threshold(base_thresh, df)
        thresh = float(adaptive.loc[ts]) if ts in adaptive.index else base_thresh
        return raw, raw, side, thresh

    def score_pattern(self, df: pd.DataFrame, ts: pd.Timestamp) -> Optional[LiveSignal]:
        if ts not in df.index:
            return None
        row = df.loc[ts]
        pname = row.get("pattern_name")
        if pd.isna(pname) or pname not in self.pattern_models:
            return None
        pname = str(pname)
        raw, _conf, side, thresh = self._pattern_prob_at(df, ts, pname)
        if raw < thresh:
            return None
        m = self.pattern_models[pname]
        spec = m["spec"]
        ex = spec["execution"]
        return LiveSignal(
            source="pattern",
            side=side,
            tp=float(ex["tp"]),
            sl=float(ex["sl"]),
            horizon=int(ex["horizon"]),
            probability=raw,
            pattern_name=pname,
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

        # Volatility-adaptive S1 threshold
        from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold
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

    def bar_score_snapshot(
        self,
        df: pd.DataFrame,
        ts: pd.Timestamp,
        consecutive_losses: int = 0,
    ) -> BarScoreSnapshot:
        """Full per-bar read for API journal (pattern route + probs even if below threshold)."""
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

        if bool(energetic_bar_mask(df.loc[[ts]]).iloc[0]):
            snap.energetic_on_bar = True
            s1_feats = s1_feature_columns(df)
            s2_feats = s2_feature_columns(df)
            s1, s2 = self._energetic_models_at(ts)
            s1_p = float(s1.predict_proba(df.loc[[ts], s1_feats])[:, 1][0])
            s2_p = float(s2.predict_proba(df.loc[[ts], s2_feats])[:, 1][0])
            snap.s1_prob = round(s1_p, 4)
            snap.s2_prob = round(s2_p, 4)

            # Volatility-adaptive S1/S2 thresholds
            from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold
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
