#!/usr/bin/env python3
"""15-minute alpha model with a multi-bar asymmetric target.

This module reuses the existing feature extractor and model implementations,
but trains on 15-minute candles and labels each sample positive when, over the
next 4 bars, price reaches +0.60% while not violating a -0.40% adverse low.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

try:
    from training.ml_alpha_model import (
        CandlePatternExtractor,
        MLCandlePatternModel,
        UptrendRecognitionSystem as BaseUptrendRecognitionSystem,
        chronological_train_test_split,
        plot_results,
    )
except ModuleNotFoundError:
    from ml_alpha_model import (
        CandlePatternExtractor,
        MLCandlePatternModel,
        UptrendRecognitionSystem as BaseUptrendRecognitionSystem,
        chronological_train_test_split,
        plot_results,
    )


MODEL_SUFFIX = "_alpha_15m_nextbar_model.pkl"
TIMEFRAME_RULE = "15min"
DEFAULT_MAX_ADVERSE_LOW_PCT = 0.40
DEFAULT_FORWARD_BARS = 4
DEFAULT_THRESHOLD_PCT = 0.60


def _resolve_timestamp_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(df.index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(df.index)
        return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

    for col in ("snapshotTimeUTC", "timestamp", "datetime", "time"):
        if col not in df.columns:
            continue
        if col == "timestamp" and pd.api.types.is_numeric_dtype(df[col]):
            ts = pd.to_datetime(df[col], unit="ms", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
        return pd.DatetimeIndex(ts)

    raise ValueError("Could not resolve timestamps from dataframe index or known timestamp columns")


def _resample_ohlcv(df: pd.DataFrame, rule: str = TIMEFRAME_RULE) -> pd.DataFrame:
    agg_map: dict[str, str] = {}
    for base, func in (("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")):
        for suffix in ("", "_ask", "_bid"):
            col = f"{base}{suffix}"
            if col in df.columns:
                agg_map[col] = func
    if "volume" in df.columns:
        agg_map["volume"] = "sum"

    if not agg_map:
        raise ValueError("No OHLC columns available for resampling")

    resampled = df.sort_index().resample(rule, label="left", closed="left").agg(agg_map)
    required_cols = [col for col in ("open", "high", "low", "close") if col in resampled.columns]
    resampled = resampled.dropna(subset=required_cols)
    return resampled


def prepare_gold_data_15m(raw_df: pd.DataFrame, rule: str = TIMEFRAME_RULE) -> pd.DataFrame:
    """Convert raw IG/legacy candles into 15-minute OHLCV candles."""
    def pick_column(candidates: tuple[str, ...]) -> str | None:
        for col in candidates:
            if col in raw_df.columns:
                return col
        return None

    open_col = pick_column(("open", "openPrice"))
    high_col = pick_column(("high", "highPrice"))
    low_col = pick_column(("low", "lowPrice"))
    close_col = pick_column(("close", "closePrice"))
    volume_col = pick_column(("volume", "lastTradedVolume"))

    missing = [name for name, col in {
        "open": open_col,
        "high": high_col,
        "low": low_col,
        "close": close_col,
    }.items() if col is None]
    if missing:
        raise ValueError(f"Missing required price columns: {', '.join(missing)}")

    index_utc = _resolve_timestamp_index(raw_df)
    prepared = pd.DataFrame(
        {
            "open": raw_df[open_col].astype(float).to_numpy(),
            "high": raw_df[high_col].astype(float).to_numpy(),
            "low": raw_df[low_col].astype(float).to_numpy(),
            "close": raw_df[close_col].astype(float).to_numpy(),
            "volume": raw_df[volume_col].fillna(0.0).astype(float).to_numpy() if volume_col else 0.0,
        },
        index=index_utc,
    )

    for side in ("ask", "bid"):
        for base in ("open", "high", "low", "close"):
            side_col = f"{base}Price_{side}"
            out_col = f"{base}_{side}"
            if side_col in raw_df.columns:
                prepared[out_col] = raw_df[side_col].astype(float).to_numpy()
            else:
                prepared[out_col] = prepared[base].to_numpy()

    prepared = prepared[~prepared.index.isna()].sort_index()
    return _resample_ohlcv(prepared, rule=rule)


class NextBarHighPctLabeler:
    """Positive when next-N bars hit gain target without breaching adverse low."""

    def __init__(
        self,
        threshold_pct: float = DEFAULT_THRESHOLD_PCT,
        max_adverse_low_pct: float = DEFAULT_MAX_ADVERSE_LOW_PCT,
        forward_bars: int = DEFAULT_FORWARD_BARS,
    ):
        self.threshold_pct = float(threshold_pct)
        self.max_adverse_low_pct = float(max_adverse_low_pct)
        self.forward_bars = int(forward_bars)

    def create_labels(self, df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
        if feature_df.empty or "idx" not in feature_df.columns:
            return pd.DataFrame(columns=["idx", "is_uptrend", "future_gain_pct", "future_drawdown_pct"])

        labels: list[dict[str, float | int]] = []
        last_idx = len(df) - 1
        last_valid_idx = last_idx - self.forward_bars
        total = len(feature_df)
        print(f"Creating next-{self.forward_bars}-bar labels for {total:,} samples...")

        for i, raw_idx in enumerate(feature_df["idx"].astype(int).to_numpy(), start=1):
            if raw_idx > last_valid_idx:
                continue

            current_close = float(df["close"].iloc[raw_idx])
            future_slice = df.iloc[raw_idx + 1: raw_idx + self.forward_bars + 1]
            future_high = float(future_slice["high"].max())
            future_low = float(future_slice["low"].min())
            future_gain_pct = ((future_high - current_close) / current_close) * 100 if current_close else 0.0
            future_drawdown_pct = ((future_low - current_close) / current_close) * 100 if current_close else 0.0
            gain_hit = future_gain_pct >= self.threshold_pct
            adverse_ok = future_drawdown_pct >= -self.max_adverse_low_pct
            labels.append(
                {
                    "idx": raw_idx,
                    "is_uptrend": int(gain_hit and adverse_ok),
                    "future_gain_pct": future_gain_pct,
                    "future_drawdown_pct": future_drawdown_pct,
                }
            )

            if i == 1 or i % max(1, total // 10) == 0:
                progress = i / total * 100.0
                print(f"  Progress: {progress:.0f}% ({i:,}/{total:,} samples)")

        label_df = pd.DataFrame(labels)
        if not label_df.empty:
            positives = int(label_df["is_uptrend"].sum())
            total_labels = len(label_df)
            print(f"✅ Created labels for {total_labels:,} samples")
            print(f"  Positives: {positives:,} ({positives / total_labels * 100:.2f}%)")
            print(f"  Negatives: {total_labels - positives:,} ({(total_labels - positives) / total_labels * 100:.2f}%)")
        return label_df


class UptrendRecognitionSystem15mNextBar(BaseUptrendRecognitionSystem):
    """15-minute model using next-4-bars gain+adverse constraints as target."""

    def __init__(
        self,
        threshold_pct: float = DEFAULT_THRESHOLD_PCT,
        max_adverse_low_pct: float = DEFAULT_MAX_ADVERSE_LOW_PCT,
        forward_bars: int = DEFAULT_FORWARD_BARS,
    ):
        super().__init__(forward_bars=int(forward_bars), threshold_pct=float(threshold_pct))
        self.extractor = CandlePatternExtractor()
        self.labeler = NextBarHighPctLabeler(
            threshold_pct=threshold_pct,
            max_adverse_low_pct=max_adverse_low_pct,
            forward_bars=forward_bars,
        )

    def save_models(self, directory: str = "ml_models_15m_nextbar") -> None:
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            filepath = os.path.join(directory, f"{name}{MODEL_SUFFIX}")
            model.save(filepath)

    def load_models(self, directory: str = "ml_models_15m_nextbar") -> None:
        for model_type in ("random_forest", "gradient_boosting"):
            filepath = Path(directory) / f"{model_type}{MODEL_SUFFIX}"
            if filepath.exists():
                model = MLCandlePatternModel(model_type)
                model.load(str(filepath))
                self.models[model_type] = model

