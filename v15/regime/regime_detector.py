"""HMM Regime Detector — wraps hmmlearn for real-time regime classification.

Computes a HMM over rolling price features to label each bar as one of 3 regimes:
  0 = low_vol   (quiet, range-bound)
  1 = trending  (directional, smooth)
  2 = high_vol  (choppy, volatile)

Used by both live bot and backtest to adapt thresholds/stops/sizing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy import — hmmlearn may not be installed in all environments
HMM_AVAILABLE = False
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    pass


class RegimeDetector:
    """Online HMM-based regime detector for 1-min gold bars.

    Usage:
        detector = RegimeDetector(model_path=HMM_MODEL_PATH)
        detector.warmup(historical_bars_df)          # fit initial HMM
        regime_id, confidence = detector.predict(latest_features)
        detector.update(latest_features)             # rolling update
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        n_states: int = 3,
        lookback: int = 500,
        refit_interval: int = 2000,
        min_warmup: int = 500,
        random_state: int = 42,
    ):
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for RegimeDetector. "
                "Install with: pip install hmmlearn"
            )

        self.model_path = model_path
        self.n_states = n_states
        self.lookback = lookback
        self.refit_interval = refit_interval
        self.min_warmup = min_warmup
        self.random_state = random_state

        self._model: Optional[hmm.GaussianHMM] = None
        self._feature_buffer: list[np.ndarray] = []
        self._state_history: list[int] = []
        self._bars_since_refit: int = 0
        self._is_warm: bool = False
        self._current_state: int = 1  # default to trending
        self._state_confidence: float = 0.5

        # Try loading pre-trained model
        if self.model_path and self.model_path.exists():
            self._load_model()
            self._is_warm = True
            logger.info("Loaded pre-trained HMM from %s", self.model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warmup(self, bars: pd.DataFrame) -> None:
        """Fit initial HMM on historical bars. Called once at startup."""
        features = self._compute_features(bars)
        if len(features) < self.min_warmup:
            logger.warning(
                "Warmup: only %d feature rows (need %d), using simple vol heuristic",
                len(features), self.min_warmup,
            )
            self._fallback_warmup(bars)
            return

        self._fit_model(features)
        self._is_warm = True
        self._bars_since_refit = 0

        logger.info("HMM warmup complete: %d bars, %d states",
                     len(features), self.n_states)

    def predict(self, feature_row: np.ndarray) -> tuple[int, float]:
        """Predict regime for a single bar's feature vector.

        Returns (regime_id, confidence).
        regime_id: 0=low_vol, 1=trending, 2=high_vol
        confidence: probability of the predicted state [0, 1]
        """
        if not self._is_warm or self._model is None:
            return 1, 0.5  # default: trending, low confidence

        try:
            feature_2d = feature_row.reshape(1, -1)
            probs = self._model.predict_proba(feature_2d)[0]
            state = int(np.argmax(probs))
            confidence = float(probs[state])
            return state, confidence
        except Exception:
            return 1, 0.5

    def update(self, feature_row: np.ndarray) -> None:
        """Feed a new bar's features into the rolling buffer. May trigger refit."""
        self._feature_buffer.append(feature_row.copy())
        if len(self._feature_buffer) > self.lookback * 2:
            self._feature_buffer = self._feature_buffer[-self.lookback:]

        self._bars_since_refit += 1

        # Periodic refit on the rolling buffer
        if self._bars_since_refit >= self.refit_interval and len(self._feature_buffer) >= self.min_warmup:
            features = np.array(self._feature_buffer[-self.lookback:])
            self._fit_model(features)
            self._bars_since_refit = 0
            logger.debug("HMM refit: %d features, states=%s",
                         len(features),
                         np.bincount(self._model.predict(features)))

    @property
    def current_state(self) -> int:
        return self._current_state

    @property
    def is_warm(self) -> bool:
        return self._is_warm

    def save_model(self) -> None:
        if self._model is not None and self.model_path is not None:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._model, self.model_path)
            logger.info("Saved HMM to %s", self.model_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_features(self, bars: pd.DataFrame) -> np.ndarray:
        """Compute HMM input features from 1-min OHLCV DataFrame.

        Features:
          - ret_5m:  5-bar log return
          - ret_15m: 15-bar log return
          - range_ratio_20: (high-low) / rolling 20-bar mean(high-low)
          - volume_ratio_20: volume / rolling 20-bar mean volume
        """
        close = bars["close"].astype(float)
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)
        volume = bars.get("volume", pd.Series(1.0, index=bars.index)).astype(float)

        ret_5m = np.log(close / close.shift(5)).fillna(0).values
        ret_15m = np.log(close / close.shift(15)).fillna(0).values

        bar_range = high - low
        range_roll = bar_range.rolling(20, min_periods=5).mean()
        range_ratio_20 = (bar_range / range_roll.replace(0, np.nan)).fillna(1.0).values

        vol_roll = volume.rolling(20, min_periods=5).mean()
        volume_ratio_20 = (volume / vol_roll.replace(0, np.nan)).fillna(1.0).values

        features = np.column_stack([
            ret_5m, ret_15m, range_ratio_20, volume_ratio_20,
        ])

        # Clip extreme outliers
        features = np.clip(features, -10, 10)
        features = np.nan_to_num(features, nan=0.0)

        return features

    def _fit_model(self, features: np.ndarray) -> None:
        """Fit (or refit) a GaussianHMM on the feature matrix."""
        # Use diagonal covariance + regularization to avoid singular matrices
        self._model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",     # diagonal = more stable than full
            n_iter=200,
            random_state=self.random_state,
            tol=1e-4,
            min_covar=0.001,            # minimum covariance (prevents singular)
            params="mct",               # means, covars, transmat
            init_params="mct",
        )
        try:
            self._model.fit(features)
        except Exception as e:
            logger.warning("HMM fit failed: %s, using last model", e)
            return

        # Assign regime labels based on volatility ordering
        # State with highest mean |ret_5m| + range_ratio = high_vol (2)
        # State with lowest = low_vol (0)
        means = self._model.means_
        vol_score = np.abs(means[:, 0]) + means[:, 2]  # |ret_5m| + range_ratio
        order = np.argsort(vol_score)  # low → high vol
        mapping = {order[0]: 0, order[1]: 1, order[2]: 2}
        self._state_mapping = mapping

        # Update current state from latest bar
        if len(features) > 0:
            probs = self._model.predict_proba(features[-1:])[0]
            raw_state = int(np.argmax(probs))
            self._current_state = mapping.get(raw_state, 1)
            self._state_confidence = float(probs[raw_state])

    def _fallback_warmup(self, bars: pd.DataFrame) -> None:
        """Simple volatility-based regime guess when HMM can't fit."""
        if len(bars) < 10:
            self._current_state = 1
            return

        bar_range = (bars["high"] - bars["low"]).tail(50)
        if len(bar_range) < 10:
            self._current_state = 1
            return

        mean_range = bar_range.mean()
        std_range = bar_range.std()

        # Heuristic thresholds (points for XAU/USD 1-min bars)
        if mean_range < 2.0:
            self._current_state = 0  # low vol
        elif mean_range > 6.0 or std_range > 3.0:
            self._current_state = 2  # high vol
        else:
            self._current_state = 1  # trending

        self._is_warm = True
        logger.info("Fallback warmup: regime=%d (mean_range=%.1f)", self._current_state, mean_range)

    def _load_model(self) -> None:
        try:
            self._model = joblib.load(self.model_path)
            logger.info("Loaded HMM from %s", self.model_path)
        except Exception as e:
            logger.warning("Failed to load HMM: %s, will train from scratch", e)
            self._model = None


def compute_regime_features(bars: pd.DataFrame) -> np.ndarray:
    """Standalone: compute HMM features from a DataFrame (used by backtest)."""
    detector = RegimeDetector.__new__(RegimeDetector)
    return detector._compute_features(bars)
