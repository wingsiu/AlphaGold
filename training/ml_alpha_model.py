"""
Machine Learning Candle Pattern Recognition for Uptrend Detection
===================================================================

This module uses machine learning to identify candlestick patterns that
precede uptrends in gold prices.

Features:
- Extracts 50+ candlestick pattern features
- Trains Random Forest and Gradient Boosting models
- Identifies uptrend probability with confidence scores
- Provides feature importance analysis
- Visualizes pattern recognition results

Author: AI Assistant
Date: February 2026
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import clone
import matplotlib.pyplot as plt
import pickle
import os

warnings.filterwarnings('ignore')


def chronological_train_test_split(X, y, test_size=0.2):
    """Split features/labels into contiguous train and test tails."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if len(X) < 2:
        raise ValueError("Need at least 2 samples for chronological split")

    split_idx = int(len(X) * (1 - float(test_size)))
    split_idx = max(1, min(split_idx, len(X) - 1))

    return (
        X.iloc[:split_idx].copy(),
        X.iloc[split_idx:].copy(),
        y.iloc[:split_idx].copy(),
        y.iloc[split_idx:].copy(),
    )


class CandlePatternExtractor:
    """Extract candlestick pattern features for ML"""

    def __init__(self):
        self.feature_names = []
        self.utc_plus_2_tz = timezone(timedelta(hours=2))
        self.asia_tz = ZoneInfo("Asia/Hong_Kong")
        self.london_tz = ZoneInfo("Europe/London")
        self.ny_tz = ZoneInfo("America/New_York")
        self._context_cache = {}

    def _get_timestamp_series_utc(self, df):
        """Resolve all row timestamps as UTC for vectorized feature generation."""
        if isinstance(df.index, pd.DatetimeIndex):
            ts = pd.Series(df.index, index=df.index)
            if ts.dt.tz is None:
                return ts.dt.tz_localize("UTC")
            return ts.dt.tz_convert("UTC")

        for col in ["timestamp", "snapshotTimeUTC", "datetime", "time"]:
            if col in df.columns:
                if col == "timestamp" and pd.api.types.is_numeric_dtype(df[col]):
                    return pd.to_datetime(df[col], unit="ms", utc=True, errors="coerce")
                return pd.to_datetime(df[col], utc=True, errors="coerce")

        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    def _ensure_context_cache(self, df):
        """Build running day/session context values once per dataframe instance."""
        cache_key = (id(df), len(df))
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        ts_utc = self._get_timestamp_series_utc(df)
        utc2_dt = ts_utc.dt.tz_convert(self.utc_plus_2_tz)
        asia_dt = ts_utc.dt.tz_convert(self.asia_tz)
        london_dt = ts_utc.dt.tz_convert(self.london_tz)
        ny_dt = ts_utc.dt.tz_convert(self.ny_tz)

        # Session windows use local clocks (DST-aware for London/NY).
        is_asia = (asia_dt.dt.hour >= 8) & (asia_dt.dt.hour < 16)
        is_london = ((london_dt.dt.hour > 8) | ((london_dt.dt.hour == 8) & (london_dt.dt.minute >= 0))) & (
            (london_dt.dt.hour < 16) | ((london_dt.dt.hour == 16) & (london_dt.dt.minute < 30))
        )
        is_ny = ((ny_dt.dt.hour > 9) | ((ny_dt.dt.hour == 9) & (ny_dt.dt.minute >= 30))) & (ny_dt.dt.hour < 16)

        def running_session_ohl(session_mask, session_key):
            open_full = pd.Series(np.nan, index=df.index, dtype="float64")
            high_full = pd.Series(np.nan, index=df.index, dtype="float64")
            low_full = pd.Series(np.nan, index=df.index, dtype="float64")

            in_session_idx = session_mask.fillna(False)
            if in_session_idx.any():
                key_in = session_key[in_session_idx]
                open_in = df.loc[in_session_idx, "open"].groupby(key_in).transform("first")
                high_in = df.loc[in_session_idx, "high"].groupby(key_in).cummax()
                low_in = df.loc[in_session_idx, "low"].groupby(key_in).cummin()

                open_full.loc[in_session_idx] = open_in.to_numpy()
                high_full.loc[in_session_idx] = high_in.to_numpy()
                low_full.loc[in_session_idx] = low_in.to_numpy()

            return open_full, high_full, low_full

        day_key = utc2_dt.dt.strftime("%Y-%m-%d")
        day_open = df["open"].groupby(day_key).transform("first")
        day_high = df["high"].groupby(day_key).cummax()
        day_low = df["low"].groupby(day_key).cummin()

        # Resolve active session key with priority NY > London > Asia.
        asia_key = "asia_" + asia_dt.dt.strftime("%Y-%m-%d")
        london_key = "london_" + london_dt.dt.strftime("%Y-%m-%d")
        ny_key = "ny_" + ny_dt.dt.strftime("%Y-%m-%d")

        session_key = pd.Series(pd.NA, index=df.index, dtype="object")
        session_key = session_key.where(~is_asia, asia_key)
        session_key = session_key.where(~is_london, london_key)
        session_key = session_key.where(~is_ny, ny_key)

        in_session = session_key.notna()
        session_open = df["open"].where(in_session).groupby(session_key).transform("first")
        session_high = df["high"].where(in_session).groupby(session_key).cummax()
        session_low = df["low"].where(in_session).groupby(session_key).cummin()

        # Separate session OHLC features (Asia/London/NY).
        asia_open, asia_high, asia_low = running_session_ohl(is_asia, asia_key)
        london_open, london_high, london_low = running_session_ohl(is_london, london_key)
        ny_open, ny_high, ny_low = running_session_ohl(is_ny, ny_key)

        cache = {
            "is_asia_session": is_asia.fillna(False).astype(int).to_numpy(),
            "is_london_session": is_london.fillna(False).astype(int).to_numpy(),
            "is_ny_session": is_ny.fillna(False).astype(int).to_numpy(),
            "asia_hour": asia_dt.dt.hour.fillna(-1).astype(int).to_numpy(),
            "london_hour": london_dt.dt.hour.fillna(-1).astype(int).to_numpy(),
            "ny_hour": ny_dt.dt.hour.fillna(-1).astype(int).to_numpy(),
            "weekday_utc_plus_2": utc2_dt.dt.weekday.fillna(-1).astype(int).to_numpy(),
            "minute_in_15": (ts_utc.dt.minute % 15).fillna(-1).astype(int).to_numpy(),
            "day_open": day_open.to_numpy(),
            "day_high": day_high.to_numpy(),
            "day_low": day_low.to_numpy(),
            "session_open": session_open.to_numpy(),
            "session_high": session_high.to_numpy(),
            "session_low": session_low.to_numpy(),
            "asia_session_open": asia_open.to_numpy(),
            "asia_session_high": asia_high.to_numpy(),
            "asia_session_low": asia_low.to_numpy(),
            "london_session_open": london_open.to_numpy(),
            "london_session_high": london_high.to_numpy(),
            "london_session_low": london_low.to_numpy(),
            "ny_session_open": ny_open.to_numpy(),
            "ny_session_high": ny_high.to_numpy(),
            "ny_session_low": ny_low.to_numpy(),
        }
        self._context_cache = {cache_key: cache}
        return cache

    def _get_timestamp_utc(self, df, idx):
        """Resolve row timestamp as UTC from index or common timestamp columns."""
        timestamp = None

        if isinstance(df.index, pd.DatetimeIndex):
            timestamp = pd.Timestamp(df.index[idx])
        else:
            for col in ["timestamp", "snapshotTimeUTC", "datetime", "time"]:
                if col in df.columns:
                    raw = df[col].iloc[idx]
                    if col == "timestamp" and pd.api.types.is_number(raw):
                        timestamp = pd.to_datetime(raw, unit="ms", utc=True, errors="coerce")
                    else:
                        timestamp = pd.to_datetime(raw, utc=True, errors="coerce")
                    break

        if timestamp is None or pd.isna(timestamp):
            return None

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp

    def extract_session_time_features(self, df, idx):
        """Extract session flags and local calendar features for Asia/London/NY."""
        cache = self._ensure_context_cache(df)
        weekday_utc_plus_2 = int(cache["weekday_utc_plus_2"][idx])

        return {
            "is_asia_session": int(cache["is_asia_session"][idx]),
            "is_london_session": int(cache["is_london_session"][idx]),
            "is_ny_session": int(cache["is_ny_session"][idx]),
            "weekday_utc_plus_2": weekday_utc_plus_2,
            "asia_weekday": weekday_utc_plus_2,
            "london_weekday": weekday_utc_plus_2,
            "ny_weekday": weekday_utc_plus_2,
            "asia_hour": int(cache["asia_hour"][idx]),
            "london_hour": int(cache["london_hour"][idx]),
            "ny_hour": int(cache["ny_hour"][idx]),
            "minute_in_15": int(cache["minute_in_15"][idx]),
        }

    def extract_session_ohl_features(self, df, idx):
        """Extract day/session running OHLC features (bias-safe, no future data)."""
        cache = self._ensure_context_cache(df)

        current_close = float(df["close"].iloc[idx])
        day_open = float(cache["day_open"][idx]) if pd.notna(cache["day_open"][idx]) else 0.0
        day_high = float(cache["day_high"][idx]) if pd.notna(cache["day_high"][idx]) else 0.0
        day_low = float(cache["day_low"][idx]) if pd.notna(cache["day_low"][idx]) else 0.0
        session_open = float(cache["session_open"][idx]) if pd.notna(cache["session_open"][idx]) else 0.0
        session_high = float(cache["session_high"][idx]) if pd.notna(cache["session_high"][idx]) else 0.0
        session_low = float(cache["session_low"][idx]) if pd.notna(cache["session_low"][idx]) else 0.0

        def pct_from(reference):
            return ((current_close - reference) / reference) * 100 if reference else 0.0

        def pct_to(level):
            return ((level - current_close) / current_close) * 100 if current_close else 0.0

        def pos_in_range(low_val, high_val):
            span = high_val - low_val
            if span <= 0:
                return 0.5
            return (current_close - low_val) / span

        day_range = max(day_high - day_low, 0.0)
        session_range = max(session_high - session_low, 0.0)

        return {
            "day_open": day_open,
            "day_high": day_high,
            "day_low": day_low,
            "session_open": session_open,
            "session_high": session_high,
            "session_low": session_low,
            "asia_session_open": float(cache["asia_session_open"][idx]) if pd.notna(cache["asia_session_open"][idx]) else 0.0,
            "asia_session_high": float(cache["asia_session_high"][idx]) if pd.notna(cache["asia_session_high"][idx]) else 0.0,
            "asia_session_low": float(cache["asia_session_low"][idx]) if pd.notna(cache["asia_session_low"][idx]) else 0.0,
            "london_session_open": float(cache["london_session_open"][idx]) if pd.notna(cache["london_session_open"][idx]) else 0.0,
            "london_session_high": float(cache["london_session_high"][idx]) if pd.notna(cache["london_session_high"][idx]) else 0.0,
            "london_session_low": float(cache["london_session_low"][idx]) if pd.notna(cache["london_session_low"][idx]) else 0.0,
            "ny_session_open": float(cache["ny_session_open"][idx]) if pd.notna(cache["ny_session_open"][idx]) else 0.0,
            "ny_session_high": float(cache["ny_session_high"][idx]) if pd.notna(cache["ny_session_high"][idx]) else 0.0,
            "ny_session_low": float(cache["ny_session_low"][idx]) if pd.notna(cache["ny_session_low"][idx]) else 0.0,
            "day_range_so_far": day_range,
            "session_range_so_far": session_range,
            "session_range_vs_day_range": (session_range / day_range) if day_range > 0 else 0.0,
            "close_vs_day_open_pct": pct_from(day_open),
            "close_vs_session_open_pct": pct_from(session_open),
            "dist_day_high_pct": pct_to(day_high),
            "dist_day_low_pct": ((current_close - day_low) / current_close) * 100 if current_close else 0.0,
            "dist_session_high_pct": pct_to(session_high),
            "dist_session_low_pct": ((current_close - session_low) / current_close) * 100 if current_close else 0.0,
            "close_in_day_range": pos_in_range(day_low, day_high),
            "close_in_session_range": pos_in_range(session_low, session_high),
            "close_vs_asia_open_pct": pct_from(float(cache["asia_session_open"][idx]) if pd.notna(cache["asia_session_open"][idx]) else 0.0),
            "close_vs_london_open_pct": pct_from(float(cache["london_session_open"][idx]) if pd.notna(cache["london_session_open"][idx]) else 0.0),
            "close_vs_ny_open_pct": pct_from(float(cache["ny_session_open"][idx]) if pd.notna(cache["ny_session_open"][idx]) else 0.0),
        }

    def extract_single_candle_features(self, df, idx):
        """Extract features from a single candlestick"""
        features = {}

        o = df['open'].iloc[idx]
        h = df['high'].iloc[idx]
        l = df['low'].iloc[idx]
        c = df['close'].iloc[idx]

        # Basic measurements
        body = c - o
        body_abs = abs(body)
        range_val = h - l

        # Avoid division by zero
        if range_val == 0:
            range_val = 0.01

        # Body features
        features['body'] = body
        features['body_abs'] = body_abs
        features['body_pct'] = (body_abs / range_val) * 100
        features['is_bullish'] = 1 if body > 0 else 0
        features['is_bearish'] = 1 if body < 0 else 0

        # Wick features
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        features['upper_wick'] = upper_wick
        features['lower_wick'] = lower_wick
        features['upper_wick_pct'] = (upper_wick / range_val) * 100
        features['lower_wick_pct'] = (lower_wick / range_val) * 100

        # Pattern identifiers
        features['is_doji'] = 1 if (body_abs / range_val) < 0.1 else 0
        features['is_hammer'] = 1 if (lower_wick > 2 * body_abs and upper_wick < body_abs) else 0
        features['is_shooting_star'] = 1 if (upper_wick > 2 * body_abs and lower_wick < body_abs) else 0
        features['is_marubozu'] = 1 if (body_abs / range_val) > 0.9 else 0
        features['is_spinning_top'] = 1 if (0.3 < (body_abs / range_val) < 0.5) else 0

        # Range features
        features['range'] = range_val

        return features

    def extract_multi_candle_features(self, df, idx, lookback=5):
        """Extract features from multiple candles"""
        features = {}

        if idx < lookback:
            return features

        # Get recent candles
        window = df.iloc[idx - lookback:idx + 1]

        # Consecutive patterns
        consecutive_green = 0
        consecutive_red = 0

        for i in range(len(window)):
            candle_body = window['close'].iloc[i] - window['open'].iloc[i]
            if candle_body > 0:
                consecutive_green += 1
                consecutive_red = 0
            elif candle_body < 0:
                consecutive_red += 1
                consecutive_green = 0
            else:
                consecutive_green = 0
                consecutive_red = 0

        features['consecutive_green'] = consecutive_green
        features['consecutive_red'] = consecutive_red

        # Price momentum
        features['price_change_5'] = window['close'].iloc[-1] - window['close'].iloc[0]
        features['high_change_5'] = window['high'].iloc[-1] - window['high'].iloc[0]
        features['low_change_5'] = window['low'].iloc[-1] - window['low'].iloc[0]

        # Volatility
        features['range_std_5'] = window['high'].sub(window['low']).std()
        features['close_std_5'] = window['close'].std()

        # Trend strength
        features['higher_highs'] = sum(window['high'].iloc[i] > window['high'].iloc[i - 1]
                                       for i in range(1, len(window)))
        features['higher_lows'] = sum(window['low'].iloc[i] > window['low'].iloc[i - 1]
                                      for i in range(1, len(window)))
        features['lower_highs'] = sum(window['high'].iloc[i] < window['high'].iloc[i - 1]
                                      for i in range(1, len(window)))
        features['lower_lows'] = sum(window['low'].iloc[i] < window['low'].iloc[i - 1]
                                     for i in range(1, len(window)))

        # Average body size
        bodies = abs(window['close'] - window['open'])
        features['avg_body_size'] = bodies.mean()
        features['max_body_size'] = bodies.max()

        # Support/Resistance proximity
        recent_high = window['high'].max()
        recent_low = window['low'].min()
        current_close = window['close'].iloc[-1]

        if recent_high - recent_low > 0:
            features['price_position'] = (current_close - recent_low) / (recent_high - recent_low)
        else:
            features['price_position'] = 0.5

        return features

    def extract_technical_indicators(self, df, idx, lookback=20):
        """Extract technical indicator features"""
        features = {}

        if idx < lookback:
            return features

        window = df.iloc[idx - lookback:idx + 1]

        # Simple Moving Averages
        features['sma_5'] = window['close'].iloc[-5:].mean() if idx >= 5 else window['close'].mean()
        features['sma_10'] = window['close'].iloc[-10:].mean() if idx >= 10 else window['close'].mean()
        features['sma_20'] = window['close'].mean()

        current_price = df['close'].iloc[idx]
        features['price_vs_sma5'] = (current_price - features['sma_5']) / features['sma_5'] * 100
        features['price_vs_sma10'] = (current_price - features['sma_10']) / features['sma_10'] * 100
        features['price_vs_sma20'] = (current_price - features['sma_20']) / features['sma_20'] * 100

        # RSI-like momentum
        changes = window['close'].diff()
        gains = changes.clip(lower=0).mean()
        losses = -changes.clip(upper=0).mean()

        if losses != 0:
            rs = gains / losses
            features['rsi_approx'] = 100 - (100 / (1 + rs))
        else:
            features['rsi_approx'] = 100

        # Volume proxy (range as volume substitute)
        features['avg_range'] = (window['high'] - window['low']).mean()
        features['current_range_vs_avg'] = ((df['high'].iloc[idx] - df['low'].iloc[idx]) /
                                            features['avg_range']) if features['avg_range'] > 0 else 1

        # Real volume features when available.
        if 'volume' in df.columns:
            current_volume = float(df['volume'].iloc[idx]) if pd.notna(df['volume'].iloc[idx]) else 0.0
            volume_window = df['volume'].iloc[max(0, idx - lookback):idx + 1].fillna(0)
            avg_volume = float(volume_window.mean()) if len(volume_window) > 0 else 0.0

            features['volume'] = current_volume
            features['avg_volume_20'] = avg_volume
            features['volume_vs_avg_20'] = (current_volume / avg_volume) if avg_volume > 0 else 1.0
        else:
            features['volume'] = 0.0
            features['avg_volume_20'] = 0.0
            features['volume_vs_avg_20'] = 1.0

        # Williams %R features
        for period in (15, 30, 90, 120):
            start = max(0, idx - period + 1)
            period_window = df.iloc[start:idx + 1]
            highest_high = period_window['high'].max()
            lowest_low = period_window['low'].min()
            denom = highest_high - lowest_low
            if denom > 0:
                features[f'wr_{period}'] = -100.0 * ((highest_high - current_price) / denom)
            else:
                features[f'wr_{period}'] = -50.0

        # Multi-horizon momentum features.
        for period in (1, 3, 5, 10, 15, 30, 60):
            start_idx = max(0, idx - period)
            past_close = float(df['close'].iloc[start_idx])
            features[f'ret_{period}_pct'] = ((current_price - past_close) / past_close) * 100 if past_close else 0.0

        # Range / ATR-like features using trailing true-range proxy.
        for period in (14, 30):
            range_window = (df['high'].iloc[max(0, idx - period + 1):idx + 1] - df['low'].iloc[max(0, idx - period + 1):idx + 1]).astype(float)
            atr_proxy = float(range_window.mean()) if len(range_window) else 0.0
            features[f'atr_proxy_{period}'] = atr_proxy
            if period == 14:
                current_range = float(df['high'].iloc[idx] - df['low'].iloc[idx])
                features['range_vs_atr_14'] = (current_range / atr_proxy) if atr_proxy > 0 else 1.0

        # Rolling realized volatility and close-location features.
        for period in (10, 20, 60):
            close_window = df['close'].iloc[max(0, idx - period + 1):idx + 1].astype(float)
            if len(close_window) >= 2:
                returns = close_window.pct_change().dropna()
                features[f'realized_vol_{period}'] = float(returns.std()) * np.sqrt(len(returns)) * 100 if len(returns) else 0.0
            else:
                features[f'realized_vol_{period}'] = 0.0

            high_window = df['high'].iloc[max(0, idx - period + 1):idx + 1].astype(float)
            low_window = df['low'].iloc[max(0, idx - period + 1):idx + 1].astype(float)
            rolling_high = float(high_window.max()) if len(high_window) else current_price
            rolling_low = float(low_window.min()) if len(low_window) else current_price
            span = rolling_high - rolling_low
            features[f'close_pos_{period}'] = ((current_price - rolling_low) / span) if span > 0 else 0.5

        # Bias-safe breakout pressure versus previous rolling range (exclude current bar).
        for period in (20, 60):
            prev_window = df.iloc[max(0, idx - period):idx]
            if len(prev_window) == 0:
                features[f'dist_prev_high_{period}_pct'] = 0.0
                features[f'dist_prev_low_{period}_pct'] = 0.0
                features[f'breakout_above_prev_high_{period}'] = 0
                continue

            prev_high = float(prev_window['high'].max())
            prev_low = float(prev_window['low'].min())
            features[f'dist_prev_high_{period}_pct'] = ((current_price - prev_high) / prev_high) * 100 if prev_high else 0.0
            features[f'dist_prev_low_{period}_pct'] = ((current_price - prev_low) / prev_low) * 100 if prev_low else 0.0
            features[f'breakout_above_prev_high_{period}'] = 1 if current_price > prev_high else 0

        # Linear trend slope and fit quality.
        for period in (10, 20, 60):
            close_window = df['close'].iloc[max(0, idx - period + 1):idx + 1].astype(float).to_numpy()
            x = np.arange(len(close_window), dtype=float)
            if len(close_window) >= 2:
                slope, intercept = np.polyfit(x, close_window, 1)
                predicted = slope * x + intercept
                ss_res = float(np.sum((close_window - predicted) ** 2))
                ss_tot = float(np.sum((close_window - np.mean(close_window)) ** 2))
                features[f'slope_{period}_pct_per_bar'] = (slope / current_price) * 100 if current_price else 0.0
                features[f'trend_r2_{period}'] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                features[f'up_close_ratio_{period}'] = float(np.mean(np.diff(close_window) > 0))
            else:
                features[f'slope_{period}_pct_per_bar'] = 0.0
                features[f'trend_r2_{period}'] = 0.0
                features[f'up_close_ratio_{period}'] = 0.0

        return features

    def extract_all_features(self, df, idx):
        """Extract all features for a given candle index"""
        all_features = {}

        # Single candle features
        all_features.update(self.extract_single_candle_features(df, idx))

        # Multi-candle features
        all_features.update(self.extract_multi_candle_features(df, idx, lookback=5))

        # Technical indicators
        all_features.update(self.extract_technical_indicators(df, idx, lookback=20))

        # Session/time features
        all_features.update(self.extract_session_time_features(df, idx))

        # Session/day OHLC features
        all_features.update(self.extract_session_ohl_features(df, idx))

        return all_features

    def create_feature_dataframe(self, df, min_idx=120):
        """Create feature dataframe for entire dataset"""
        import sys

        # Validate we have enough data
        if len(df) < min_idx:
            print(f"⚠️  Not enough data for feature extraction!")
            print(f"   Need at least {min_idx} bars, but only have {len(df)} bars")
            return pd.DataFrame()  # Return empty dataframe

        feature_list = []
        total = len(df) - min_idx

        print(f"Extracting features from {total:,} candles...")
        sys.stdout.flush()

        for i, idx in enumerate(range(min_idx, len(df))):
            features = self.extract_all_features(df, idx)
            features['idx'] = idx
            feature_list.append(features)

            # Show progress every 10%
            if (i + 1) % max(1, total // 10) == 0 or i == 0:
                progress = (i + 1) / total * 100
                print(f"  Progress: {progress:.0f}% ({i + 1:,}/{total:,} candles)")
                sys.stdout.flush()

        feature_df = pd.DataFrame(feature_list)
        self.feature_names = [col for col in feature_df.columns if col != 'idx']

        print(f"✅ Extracted {len(self.feature_names)} features from {len(feature_df)} candles")

        return feature_df


class UptrendLabeler:
    """Label candles based on future price movement (uptrend detection)"""

    def __init__(self, forward_bars=10, threshold_pct=0.2, max_adverse_low_pct=None):
        """
        Parameters:
        -----------
        forward_bars : int
            Number of bars to look ahead
        threshold_pct : float
            Minimum price increase percentage to label as uptrend
        max_adverse_low_pct : float | None
            Optional maximum allowed drawdown percentage over the same future window.
            Example: 0.40 means the future low must stay above -0.40% from current close.
        """
        self.forward_bars = forward_bars
        self.threshold_pct = threshold_pct
        self.max_adverse_low_pct = max_adverse_low_pct

    def create_labels(self, df, feature_df):
        """Create uptrend labels based on future price movement"""
        import sys

        # Check if feature_df is empty
        if len(feature_df) == 0:
            print("⚠️  Cannot create labels: feature dataframe is empty")
            return pd.DataFrame()

        # Check if 'idx' column exists
        if 'idx' not in feature_df.columns:
            print("⚠️  Cannot create labels: 'idx' column missing from feature dataframe")
            print(f"   Available columns: {list(feature_df.columns)}")
            return pd.DataFrame()

        labels = []
        total = len(feature_df)

        print(f"Creating labels for {total:,} samples...")
        sys.stdout.flush()

        for i, idx in enumerate(feature_df['idx'].values):
            # Look ahead
            future_end_idx = min(idx + self.forward_bars, len(df) - 1)

            current_price = df['close'].iloc[idx]
            future_slice = df.iloc[idx + 1:future_end_idx + 1]
            future_high = future_slice['high'].max() if future_end_idx > idx else current_price
            future_low = future_slice['low'].min() if future_end_idx > idx else current_price

            # Calculate price increase
            price_increase_pct = ((future_high - current_price) / current_price) * 100
            price_drawdown_pct = ((future_low - current_price) / current_price) * 100

            # Label as uptrend if price increases by threshold
            is_uptrend = 1 if price_increase_pct >= self.threshold_pct else 0
            if is_uptrend and self.max_adverse_low_pct is not None:
                is_uptrend = 1 if price_drawdown_pct >= -float(self.max_adverse_low_pct) else 0

            labels.append({
                'idx': idx,
                'is_uptrend': is_uptrend,
                'future_gain_pct': price_increase_pct,
                'future_drawdown_pct': price_drawdown_pct,
            })

            # Show progress every 10%
            if (i + 1) % max(1, total // 10) == 0 or i == 0:
                progress = (i + 1) / total * 100
                print(f"  Progress: {progress:.0f}% ({i + 1:,}/{total:,} samples)")
                sys.stdout.flush()

        label_df = pd.DataFrame(labels)

        # Print label distribution
        if len(label_df) > 0:
            uptrends = label_df['is_uptrend'].sum()
            total_labels = len(label_df)
            print(f"✅ Created labels for {total_labels} samples")
            print(f"\nLabel distribution:")
            print(f"  Uptrends: {uptrends} ({uptrends / total_labels * 100:.1f}%)")
            print(f"  No uptrends: {total_labels - uptrends} ({(total_labels - uptrends) / total_labels * 100:.1f}%)")

        return label_df


class MLCandlePatternModel:
    """Machine Learning model for candle pattern recognition"""

    def __init__(self, model_type='random_forest'):
        """
        Parameters:
        -----------
        model_type : str
            'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.feature_names = None

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=140,
                max_depth=8,
                min_samples_split=30,
                min_samples_leaf=12,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=140,
                max_depth=4,
                learning_rate=0.05,
                min_samples_split=30,
                min_samples_leaf=12,
                subsample=0.8,
                n_iter_no_change=None,
                random_state=42,
                verbose=1,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X_train, y_train, feature_names):
        """Train the model"""
        import sys

        print(f"Training {self.model_type} model...")
        print(f"Training samples: {len(X_train):,}")
        print(f"Positive class (uptrend): {y_train.sum():,} ({y_train.sum() / len(y_train) * 100:.1f}%)")
        sys.stdout.flush()

        # Store feature names
        self.feature_names = feature_names

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        # Cross-validation with explicit per-fold progress output.
        cv_scores = []
        y_array = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
        cv_splits = 3
        cv = TimeSeriesSplit(n_splits=cv_splits)

        # Downsample CV workload for large datasets to speed up experiments while preserving chronology.
        max_cv_samples = 80000
        if len(X_train_scaled) > max_cv_samples:
            X_cv = X_train_scaled[-max_cv_samples:]
            y_cv = y_array[-max_cv_samples:]
            print(f"Using last {len(X_cv):,} chronological rows for CV speed-up")
        else:
            X_cv = X_train_scaled
            y_cv = y_array

        print(f"Running {cv_splits}-fold cross-validation...")
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_cv, y_cv), start=1):
            y_train_fold = y_cv[tr_idx]
            y_val_fold = y_cv[val_idx]
            if len(np.unique(y_train_fold)) < 2 or len(np.unique(y_val_fold)) < 2:
                print(f"  CV fold {fold}/{cv_splits}: skipped (single-class fold)")
                continue
            print(f"  CV fold {fold}/{cv_splits}: training...")
            fold_model = clone(self.model)
            fold_model.fit(X_cv[tr_idx], y_train_fold)
            fold_pred_proba = fold_model.predict_proba(X_cv[val_idx])[:, 1]
            fold_score = roc_auc_score(y_val_fold, fold_pred_proba)
            cv_scores.append(fold_score)
            print(f"  CV fold {fold}/{cv_splits}: ROC-AUC={fold_score:.3f}")

        if cv_scores:
            cv_scores_series = pd.Series(cv_scores)
            print(f"Cross-validation ROC-AUC: {cv_scores_series.mean():.3f} (+/- {cv_scores_series.std():.3f})")
        else:
            print("Cross-validation ROC-AUC: skipped (insufficient class variation across time folds)")
        sys.stdout.flush()

        return self

    def predict(self, X, cutoff=None):
        """Predict binary uptrend labels."""
        X_scaled = self.scaler.transform(X)
        if cutoff is None:
            return self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        return (proba >= float(cutoff)).astype(int)

    def predict_proba(self, X):
        """Predict uptrend probability scores"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_scaled = self.scaler.transform(X_test)

        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        print(f"\nTest samples: {len(X_test)}")
        print(f"Actual uptrends: {y_test.sum()} ({y_test.sum() / len(y_test) * 100:.1f}%)")
        print(f"Predicted uptrends: {y_pred.sum()} ({y_pred.sum() / len(y_pred) * 100:.1f}%)")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Uptrend', 'Uptrend']))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.3f}")

        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }

    def plot_feature_importance(self, top_n=20):
        """Plot top feature importances"""
        if self.feature_importance is None:
            print("No feature importance available")
            return

        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)

        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Most Important Features for Uptrend Detection')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        return plt.gcf()

    def save(self, filepath):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
        }

        Path(filepath).write_bytes(pickle.dumps(model_data, protocol=pickle.HIGHEST_PROTOCOL))

        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model from disk"""
        model_data = pickle.loads(Path(filepath).read_bytes())

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']

        print(f"Model loaded from {filepath}")
        return self


class UptrendRecognitionSystem:
    """Complete system for ML-based uptrend recognition"""

    def __init__(self, forward_bars=10, threshold_pct=0.2, max_adverse_low_pct=None):
        """
        Parameters:
        -----------
        forward_bars : int
            Number of bars to look ahead for uptrend detection
        threshold_pct : float
            Minimum price increase percentage to label as uptrend
        max_adverse_low_pct : float | None
            Optional maximum allowed drawdown percentage over the forward window.
        """
        self.extractor = CandlePatternExtractor()
        self.labeler = UptrendLabeler(
            forward_bars=forward_bars,
            threshold_pct=threshold_pct,
            max_adverse_low_pct=max_adverse_low_pct,
        )
        self.models = {}
        self.results = {}
        self.split_info = {}

    def prepare_data(self, df):
        """Prepare features and labels from price data"""
        print("\n" + "=" * 80)
        print("PREPARING DATA")
        print("=" * 80)

        print(f"\nExtracting candlestick features...")
        feature_df = self.extractor.create_feature_dataframe(df)
        print(f"✅ Extracted {len(self.extractor.feature_names)} features from {len(feature_df)} candles")

        print(f"\nCreating uptrend labels...")
        label_df = self.labeler.create_labels(df, feature_df)
        print(f"✅ Created labels for {len(label_df)} samples")

        # Merge features and labels
        data = feature_df.merge(label_df, on='idx')

        print(f"\nLabel distribution:")
        print(f"  Uptrends: {data['is_uptrend'].sum()} ({data['is_uptrend'].sum() / len(data) * 100:.1f}%)")
        print(
            f"  No uptrends: {(~data['is_uptrend'].astype(bool)).sum()} ({(~data['is_uptrend'].astype(bool)).sum() / len(data) * 100:.1f}%)")

        return data

    def train_models(self, data, test_size=0.2):
        """Train multiple ML models"""
        print("\n" + "=" * 80)
        print("TRAINING MODELS")
        print("=" * 80)

        # Prepare X and y
        X = data[self.extractor.feature_names].fillna(0)
        y = data['is_uptrend']

        # Train-test split
        X_train, X_test, y_train, y_test = chronological_train_test_split(X, y, test_size=test_size)
        self.split_info = {
            'train_start_idx': int(data['idx'].iloc[0]),
            'train_end_idx': int(data.loc[X_train.index, 'idx'].iloc[-1]),
            'test_start_idx': int(data.loc[X_test.index, 'idx'].iloc[0]),
            'test_end_idx': int(data.loc[X_test.index, 'idx'].iloc[-1]),
        }

        print(f"\nTrain size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")
        print(
            "Chronological split: "
            f"train idx {self.split_info['train_start_idx']}->{self.split_info['train_end_idx']}, "
            f"test idx {self.split_info['test_start_idx']}->{self.split_info['test_end_idx']}"
        )

        # Train Random Forest
        print("\n" + "-" * 80)
        rf_model = MLCandlePatternModel('random_forest')
        rf_model.train(X_train, y_train, self.extractor.feature_names)
        self.models['random_forest'] = rf_model

        # Train Gradient Boosting
        print("\n" + "-" * 80)
        gb_model = MLCandlePatternModel('gradient_boosting')
        gb_model.train(X_train, y_train, self.extractor.feature_names)
        self.models['gradient_boosting'] = gb_model


        # Evaluate both models
        print("\n" + "=" * 80)
        print("RANDOM FOREST RESULTS")
        self.results['random_forest'] = rf_model.evaluate(X_test, y_test)

        print("\n" + "=" * 80)
        print("GRADIENT BOOSTING RESULTS")
        self.results['gradient_boosting'] = gb_model.evaluate(X_test, y_test)

        return X_train, X_test, y_train, y_test

    def predict_uptrend(self, df, idx, model_type='random_forest', cutoff=None):
        """Predict if a candle is at the start of an uptrend"""
        features = self.extractor.extract_all_features(df, idx)
        X = pd.DataFrame([features])[self.extractor.feature_names].fillna(0)

        model = self.models[model_type]
        prediction = model.predict(X, cutoff=cutoff)[0]
        probability = model.predict_proba(X)[0]

        return {
            'is_uptrend': bool(prediction),
            'uptrend_probability': probability,
            'confidence': 'High' if probability > 0.7 else 'Medium' if probability > 0.5 else 'Low'
        }

    def scan_recent_patterns(self, df, model_type='random_forest', lookback=20, cutoff=None):
        """Scan recent candles for uptrend patterns"""
        start_idx = max(20, len(df) - lookback)
        indices = list(range(start_idx, len(df)))
        if not indices:
            return pd.DataFrame()

        feature_rows = []
        valid_indices = []
        for idx in indices:
            features = self.extractor.extract_all_features(df, idx)
            feature_rows.append(features)
            valid_indices.append(idx)

        X = pd.DataFrame(feature_rows)[self.extractor.feature_names].fillna(0)
        model = self.models[model_type]
        proba = model.predict_proba(X)
        pred = model.predict(X, cutoff=cutoff)

        results = []
        for i, idx in enumerate(valid_indices):
            if pred[i]:
                p = float(proba[i])
                results.append({
                    'idx': idx,
                    'timestamp': df.index[idx] if hasattr(df, 'index') else idx,
                    'close': df['close'].iloc[idx],
                    'probability': p,
                    'confidence': 'High' if p > 0.7 else 'Medium' if p > 0.5 else 'Low'
                })

        return pd.DataFrame(results)

    def save_models(self, directory='ml_models'):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)

        for name, model in self.models.items():
            filepath = os.path.join(directory, f'{name}_alpha_model.pkl')
            model.save(filepath)

    def load_models(self, directory='ml_models'):
        """Load saved models"""
        for model_type in ['random_forest', 'gradient_boosting']:
            filepath = os.path.join(directory, f'{model_type}_alpha_model.pkl')
            if os.path.exists(filepath):
                model = MLCandlePatternModel(model_type)
                model.load(filepath)
                self.models[model_type] = model


def plot_results(df, predictions_df, title="Uptrend Predictions"):
    """Plot price chart with uptrend predictions"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])

    # Plot price
    ax1.plot(df['close'], label='Close Price', alpha=0.7)

    # Mark uptrend predictions
    for _, pred in predictions_df.iterrows():
        idx = pred['idx']
        color = 'green' if pred['confidence'] == 'High' else 'orange'
        ax1.axvline(x=idx, color=color, alpha=0.3, linestyle='--')
        ax1.scatter(idx, df['close'].iloc[idx], color=color, s=100, zorder=5)

    ax1.set_xlabel('Bar Index')
    ax1.set_ylabel('Price')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot probability
    ax2.scatter(predictions_df['idx'], predictions_df['probability'],
                c=predictions_df['probability'], cmap='RdYlGn', s=50)
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax2.set_xlabel('Bar Index')
    ax2.set_ylabel('Uptrend Probability')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("ML Candle Pattern Recognition System")
    print("This module should be imported and used by other scripts")
    print("\nExample usage:")
    print("""
    from ml_alpha_model import UptrendRecognitionSystem
    from data_utils import load_standard_period

    # Load data
    df = load_standard_period()

    # Initialize system
    system = UptrendRecognitionSystem(forward_bars=10, threshold_pct=0.2)

    # Prepare and train
    data = system.prepare_data(df)
    system.train_models(data)

    # Predict on recent data
    predictions = system.scan_recent_patterns(df, lookback=50)
    print(predictions)

    # Save models
    system.save_models()
    """)

