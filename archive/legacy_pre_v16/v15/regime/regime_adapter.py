"""Regime Adapter — normalizes features and adjusts thresholds/stops/sizing per regime.

Used by both backtest and live bot to adapt XGBoost model scoring and trade
parameters based on the current HMM regime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from v15.config.v15_config import REGIME_THRESHOLDS, FEATURE_CONFIG, get_regime_config

logger = logging.getLogger(__name__)


@dataclass
class AdaptedParams:
    """Output of regime adaptation for a single bar."""
    regime_id: int
    regime_label: str
    regime_confidence: float

    # Adjusted thresholds
    s1_prob_threshold: float
    pattern_prob_threshold: float

    # Adjusted trade params
    tp_multiplier: float
    sl_multiplier: float
    max_holding_bars: int

    # Sizing
    kelly_fraction: float

    # Position management
    close_on_reverse: bool


class RegimeAdapter:
    """Adapts model thresholds, stops, targets, and sizing to current regime.

    Two main functions:
      1. normalize_features(feature_matrix) — divide vol-sensitive features by regime vol
      2. adapt_params(regime_id, confidence) — return adapted thresholds/sizing
    """

    def __init__(self):
        self._regime_vol_cache: dict[int, float] = {0: 1.0, 1: 2.5, 2: 6.0}
        self._last_regime_id: int = 1

    # ------------------------------------------------------------------
    # Feature Normalization
    # ------------------------------------------------------------------

    def normalize_features(
        self,
        feature_matrix: np.ndarray,
        regime_id: int,
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """Divide regime-sensitive features by the current regime's volatility scale.

        This makes a 3-pt bar_move in low_vol (regime 0) comparable to a 6-pt bar_move
        in high_vol (regime 2), preventing the model from being biased toward high-vol regimes.

        Args:
            feature_matrix: shape (n_bars, n_features) — the full feature window
            regime_id: current regime 0/1/2
            feature_names: list of feature names matching columns. If None, uses
                          FEATURE_CONFIG["normalize_by_regime"] positional matching.

        Returns:
            Normalized feature matrix (copy, original unchanged).
        """
        normalized = feature_matrix.copy()

        if feature_names is None:
            # Simple positional approach: normalize first N vol-sensitive columns
            # which are at known positions in the feature matrix layout
            # Position 0: open_rel, 1: high_rel, 2: low_rel, 3: close_rel
            # Position ~6-10: ret_1m, ret_3m, ret_5m, volume, spread, range
            vol_scale = self._get_regime_vol(regime_id)
            if vol_scale <= 0:
                vol_scale = 1.0

            # Normalize returns (positions 0-3: open/high/low/close_rel, and ret columns)
            # These are percentage-scale values that need regime scaling
            price_cols = [0, 1, 2, 3]  # open_rel, high_rel, low_rel, close_rel
            for col in price_cols:
                if col < normalized.shape[1]:
                    normalized[:, col] = normalized[:, col] / max(vol_scale, 0.5)

            # Range/ret columns (typically past the OHLC rels)
            for col in range(6, min(12, normalized.shape[1])):
                normalized[:, col] = normalized[:, col] / max(vol_scale, 0.5)

            return normalized

        # Named feature approach
        normalize_keys = set(FEATURE_CONFIG.get("normalize_by_regime", []))
        vol_scale = self._get_regime_vol(regime_id)
        if vol_scale <= 0:
            vol_scale = 1.0

        for name in normalize_keys:
            if name in feature_names:
                idx = feature_names.index(name)
                normalized[:, idx] = normalized[:, idx] / max(vol_scale, 0.5)

        return normalized

    # ------------------------------------------------------------------
    # Parameter Adaptation
    # ------------------------------------------------------------------

    def adapt_params(
        self,
        regime_id: int,
        confidence: float = 0.5,
        base_tp: float = 30.0,
        base_sl: float = 25.0,
        base_horizon: int = 30,
    ) -> AdaptedParams:
        """Return adapted thresholds, stops, and sizing for the current regime.

        Args:
            regime_id: 0=low_vol, 1=trending, 2=high_vol
            confidence: HMM state probability [0, 1]
            base_tp: default TP distance (before regime multiplier)
            base_sl: default SL distance (before regime multiplier)
            base_horizon: default max holding bars (before regime multiplier)

        Returns:
            AdaptedParams with all adjusted values.
        """
        self._last_regime_id = regime_id
        cfg = get_regime_config(regime_id)

        # Blend threshold by confidence: low confidence → use neutral threshold
        # high confidence → use regime-specific threshold
        neutral_threshold = 0.50
        s1_thresh = (confidence * cfg["s1_prob_threshold"] +
                     (1 - confidence) * neutral_threshold)
        pattern_thresh = (confidence * cfg["pattern_prob_threshold"] +
                          (1 - confidence) * neutral_threshold)

        return AdaptedParams(
            regime_id=regime_id,
            regime_label=cfg["label"],
            regime_confidence=confidence,

            s1_prob_threshold=round(s1_thresh, 3),
            pattern_prob_threshold=round(pattern_thresh, 3),

            tp_multiplier=cfg["tp_multiplier"],
            sl_multiplier=cfg["sl_multiplier"],
            max_holding_bars=cfg["max_holding_bars"],

            kelly_fraction=cfg["kelly_fraction"],

            close_on_reverse=cfg["close_on_reverse"],
        )

    def adapt_stops(
        self,
        regime_id: int,
        base_tp: float,
        base_sl: float,
    ) -> tuple[float, float]:
        """Scale TP and SL distances by regime multipliers."""
        cfg = get_regime_config(regime_id)
        return (
            round(base_tp * cfg["tp_multiplier"], 1),
            round(base_sl * cfg["sl_multiplier"], 1),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_regime_vol(self, regime_id: int) -> float:
        """Get the volatility scaling factor for a regime (used in feature normalization).

        These values represent the expected bar range in points for each regime.
        Updated dynamically as more data arrives.
        """
        return self._regime_vol_cache.get(regime_id, 2.5)

    def update_regime_vol(self, regime_id: int, bar_range: float, alpha: float = 0.05) -> None:
        """Exponential moving update of regime volatility estimate."""
        old = self._regime_vol_cache.get(regime_id, 2.5)
        self._regime_vol_cache[regime_id] = old * (1 - alpha) + bar_range * alpha
        self._regime_vol_cache[regime_id] = max(self._regime_vol_cache[regime_id], 0.3)

    def get_regime_vol(self, regime_id: int) -> float:
        return self._regime_vol_cache.get(regime_id, 2.5)


# ------------------------------------------------------------------
# Convenience: create a regime pipeline for backtest
# ------------------------------------------------------------------

def create_regime_pipeline(model_path=None) -> tuple:
    """Create detector + adapter pair for backtest or live use.

    Returns (RegimeDetector, RegimeAdapter).
    HMM import is deferred — if hmmlearn is not installed, raises ImportError.
    """
    from v15.regime.regime_detector import RegimeDetector
    from v15.config.v15_config import REGIME_CONFIG

    detector = RegimeDetector(
        model_path=model_path,
        n_states=REGIME_CONFIG["n_states"],
        lookback=REGIME_CONFIG["hmm_lookback"],
        refit_interval=REGIME_CONFIG["hmm_refit_interval"],
        min_warmup=REGIME_CONFIG["hmm_min_warmup"],
    )
    adapter = RegimeAdapter()
    return detector, adapter
