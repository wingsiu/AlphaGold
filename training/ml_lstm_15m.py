#!/usr/bin/env python3
"""Core utilities for a 15-minute LSTM experiment.

This module builds a sequence dataset from engineered candle features and
labels positives with an asymmetric target:
- next N bars max high >= +threshold_pct
- next N bars min low >= -max_adverse_low_pct
"""

from __future__ import annotations

import json
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from training.ml_alpha_model import CandlePatternExtractor
except ModuleNotFoundError:
    from ml_alpha_model import CandlePatternExtractor


DEFAULT_THRESHOLD_PCT = 0.60
DEFAULT_MAX_ADVERSE_LOW_PCT = 0.40
DEFAULT_FORWARD_BARS = 4
DEFAULT_SEQUENCE_LENGTH = 32


@dataclass
class LabelConfig:
    threshold_pct: float = DEFAULT_THRESHOLD_PCT
    max_adverse_low_pct: float = DEFAULT_MAX_ADVERSE_LOW_PCT
    forward_bars: int = DEFAULT_FORWARD_BARS


@dataclass
class SequenceDataset:
    X: np.ndarray
    y: np.ndarray
    idx: np.ndarray
    feature_names: list[str]


@dataclass
class SequenceSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_test: np.ndarray


def create_labels(df: pd.DataFrame, feature_df: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    labels: list[dict[str, float | int]] = []
    last_idx = len(df) - 1
    last_valid_idx = last_idx - int(cfg.forward_bars)

    for raw_idx in feature_df["idx"].astype(int).to_numpy():
        if raw_idx > last_valid_idx:
            continue

        current_close = float(df["close"].iloc[raw_idx])
        if current_close <= 0:
            continue

        future_slice = df.iloc[raw_idx + 1: raw_idx + int(cfg.forward_bars) + 1]
        future_high = float(future_slice["high"].max())
        future_low = float(future_slice["low"].min())

        gain_pct = ((future_high - current_close) / current_close) * 100.0
        drawdown_pct = ((future_low - current_close) / current_close) * 100.0
        is_uptrend = int(gain_pct >= float(cfg.threshold_pct) and drawdown_pct >= -float(cfg.max_adverse_low_pct))

        labels.append(
            {
                "idx": raw_idx,
                "is_uptrend": is_uptrend,
                "future_gain_pct": gain_pct,
                "future_drawdown_pct": drawdown_pct,
            }
        )

    return pd.DataFrame(labels)


def build_labeled_feature_frame(
    df: pd.DataFrame,
    *,
    label_cfg: LabelConfig,
    min_feature_idx: int = 120,
) -> tuple[pd.DataFrame, list[str]]:
    extractor = CandlePatternExtractor()
    feature_df = extractor.create_feature_dataframe(df, min_idx=min_feature_idx)
    labels_df = create_labels(df, feature_df, label_cfg)

    merged = feature_df.merge(labels_df, on="idx", how="inner")
    merged = merged.sort_values("idx").reset_index(drop=True)
    feature_names = [col for col in feature_df.columns if col != "idx"]
    return merged, feature_names


def build_sequence_dataset(
    merged_df: pd.DataFrame,
    feature_names: list[str],
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> SequenceDataset:
    seq_len = max(2, int(sequence_length))
    feat = merged_df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    target = pd.to_numeric(merged_df["is_uptrend"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)
    idx_values = merged_df["idx"].astype(int).to_numpy()

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    idx_list: list[int] = []

    for end_pos in range(seq_len - 1, len(merged_df)):
        start_pos = end_pos - seq_len + 1
        X_list.append(feat[start_pos: end_pos + 1])
        y_list.append(int(target[end_pos]))
        idx_list.append(int(idx_values[end_pos]))

    if not X_list:
        raise ValueError("No sequence samples generated. Try a smaller sequence length or longer date range.")

    return SequenceDataset(
        X=np.stack(X_list).astype(np.float32),
        y=np.asarray(y_list, dtype=np.int32),
        idx=np.asarray(idx_list, dtype=np.int32),
        feature_names=feature_names,
    )


def chronological_split(dataset: SequenceDataset, test_size: float = 0.2) -> SequenceSplit:
    split_idx = int(len(dataset.X) * (1.0 - float(test_size)))
    split_idx = max(1, min(split_idx, len(dataset.X) - 1))

    return SequenceSplit(
        X_train=dataset.X[:split_idx],
        X_test=dataset.X[split_idx:],
        y_train=dataset.y[:split_idx],
        y_test=dataset.y[split_idx:],
        idx_train=dataset.idx[:split_idx],
        idx_test=dataset.idx[split_idx:],
    )


def fit_standardization(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def compute_class_weight_dict(y_train: np.ndarray) -> dict[int, float]:
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    total = positives + negatives
    if positives == 0 or negatives == 0:
        return {0: 1.0, 1: 1.0}
    return {
        0: total / (2.0 * negatives),
        1: total / (2.0 * positives),
    }


def build_lstm_model(input_shape: tuple[int, int], learning_rate: float = 1e-3):
    try:
        keras = importlib.import_module("tensorflow").keras
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for LSTM training. Install tensorflow and retry."
        ) from exc

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    return model


def save_lstm_artifacts(
    model,
    *,
    out_dir: str | Path,
    feature_names: list[str],
    sequence_length: int,
    mean: np.ndarray,
    std: np.ndarray,
    label_cfg: LabelConfig,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "lstm_15m_model.keras"
    model.save(model_path)

    np.savez_compressed(out / "preprocess.npz", mean=mean, std=std)

    metadata: dict[str, Any] = {
        "sequence_length": int(sequence_length),
        "feature_names": feature_names,
        "threshold_pct": float(label_cfg.threshold_pct),
        "max_adverse_low_pct": float(label_cfg.max_adverse_low_pct),
        "forward_bars": int(label_cfg.forward_bars),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_lstm_artifacts(model_dir: str | Path):
    try:
        keras = importlib.import_module("tensorflow").keras
    except Exception as exc:
        raise RuntimeError("TensorFlow is required for LSTM inference. Install tensorflow and retry.") from exc

    base = Path(model_dir)
    model = keras.models.load_model(base / "lstm_15m_model.keras")
    prep = np.load(base / "preprocess.npz")
    metadata = json.loads((base / "metadata.json").read_text(encoding="utf-8"))

    return {
        "model": model,
        "mean": prep["mean"],
        "std": prep["std"],
        "feature_names": list(metadata["feature_names"]),
        "sequence_length": int(metadata["sequence_length"]),
        "metadata": metadata,
    }

