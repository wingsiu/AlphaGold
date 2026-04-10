#!/usr/bin/env python3
"""Train an LSTM on 15-minute features for asymmetric multi-bar target detection."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from data import DataLoader
try:
    from training.ml_alpha_model_15m_nextbar import prepare_gold_data_15m
    from training.ml_lstm_15m import (
        DEFAULT_FORWARD_BARS,
        DEFAULT_MAX_ADVERSE_LOW_PCT,
        DEFAULT_SEQUENCE_LENGTH,
        DEFAULT_THRESHOLD_PCT,
        LabelConfig,
        apply_standardization,
        build_labeled_feature_frame,
        build_lstm_model,
        build_sequence_dataset,
        chronological_split,
        compute_class_weight_dict,
        fit_standardization,
        save_lstm_artifacts,
    )
except ModuleNotFoundError:
    from ml_alpha_model_15m_nextbar import prepare_gold_data_15m
    from ml_lstm_15m import (
        DEFAULT_FORWARD_BARS,
        DEFAULT_MAX_ADVERSE_LOW_PCT,
        DEFAULT_SEQUENCE_LENGTH,
        DEFAULT_THRESHOLD_PCT,
        LabelConfig,
        apply_standardization,
        build_labeled_feature_frame,
        build_lstm_model,
        build_sequence_dataset,
        chronological_split,
        compute_class_weight_dict,
        fit_standardization,
        save_lstm_artifacts,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a 15-minute LSTM model.")
    parser.add_argument("--table", default="gold_prices")
    parser.add_argument("--start-date", default="2025-05-20")
    parser.add_argument("--end-date", default="2026-02-04")
    parser.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT)
    parser.add_argument("--max-adverse-low-pct", type=float, default=DEFAULT_MAX_ADVERSE_LOW_PCT)
    parser.add_argument("--forward-bars", type=int, default=DEFAULT_FORWARD_BARS)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument(
        "--feature-selection",
        choices=["target_corr", "all"],
        default="target_corr",
        help="Feature selection mode before sequence building.",
    )
    parser.add_argument(
        "--feature-top-n",
        type=int,
        default=32,
        help="Number of features to keep when --feature-selection=target_corr.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--class-weight-pos",
        type=float,
        default=None,
        help="Override positive-class weight (label=1).",
    )
    parser.add_argument(
        "--class-weight-neg",
        type=float,
        default=None,
        help="Override negative-class weight (label=0).",
    )
    parser.add_argument("--model-dir", default="training/ml_models_lstm_15m")
    parser.add_argument("--test-cache-out", default="training/test_period_signals_lstm_15m.csv")
    parser.add_argument("--full-cache-out", default="training/full_period_signals_lstm_15m.csv")
    return parser


def _ensure_tensorflow_available() -> None:
    try:
        importlib.import_module("tensorflow")
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for LSTM training. Install it first, for example: "
            "python3 -m pip install 'tensorflow>=2.16,<3.0'"
        ) from exc


def _select_features(
    merged: pd.DataFrame,
    feature_names: list[str],
    *,
    mode: str,
    top_n: int,
) -> list[str]:
    if mode == "all":
        print(f"Using all features: {len(feature_names)}")
        return feature_names

    k = max(1, min(int(top_n), len(feature_names)))
    feat = merged[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target = pd.to_numeric(merged["is_uptrend"], errors="coerce").fillna(0.0)
    corr = feat.corrwith(target).abs().fillna(0.0)
    selected = corr.sort_values(ascending=False).head(k).index.tolist()

    print(f"Using top {len(selected)} features by |corr(target)| out of {len(feature_names)}")
    print("Top selected features:")
    for name in selected[:12]:
        print(f"  - {name}: {corr[name]:.4f}")
    return selected


def _save_signal_cache(
    *,
    df_15m: pd.DataFrame,
    idx: np.ndarray,
    y_true: np.ndarray,
    probability: np.ndarray,
    out_path: Path,
) -> None:
    out = pd.DataFrame(
        {
            "idx": idx.astype(int),
            "timestamp": df_15m.index[idx.astype(int)].astype(str),
            "actual": y_true.astype(int),
            "lstm_probability": probability.astype(float),
            "lstm_signal": (probability >= 0.5).astype(int),
        }
    ).sort_values("idx").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved signal cache: {out_path} ({len(out):,} rows)")


def _resolve_output_path(path_text: str) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _resolve_class_weight(args: argparse.Namespace, y_train: np.ndarray) -> dict[int, float]:
    auto = compute_class_weight_dict(y_train)
    if args.class_weight_pos is None and args.class_weight_neg is None:
        print(
            "Using auto class weights: "
            f"neg={auto[0]:.4f}, pos={auto[1]:.4f}"
        )
        return auto

    neg = float(args.class_weight_neg) if args.class_weight_neg is not None else float(auto[0])
    pos = float(args.class_weight_pos) if args.class_weight_pos is not None else float(auto[1])
    weights = {0: neg, 1: pos}
    print(
        "Using custom class weights: "
        f"neg={weights[0]:.4f}, pos={weights[1]:.4f}"
    )
    return weights


def main() -> int:
    args = build_parser().parse_args()
    _ensure_tensorflow_available()

    label_cfg = LabelConfig(
        threshold_pct=args.threshold_pct,
        max_adverse_low_pct=args.max_adverse_low_pct,
        forward_bars=args.forward_bars,
    )

    print("Loading and aggregating to 15-minute candles...")
    raw = DataLoader().load_data(args.table, start_date=args.start_date, end_date=args.end_date)
    df_15m = prepare_gold_data_15m(raw)
    print(f"15m candles: {len(df_15m):,}")

    print("Building labeled feature set...")
    merged, feature_names = build_labeled_feature_frame(df_15m, label_cfg=label_cfg)
    feature_names = _select_features(
        merged,
        feature_names,
        mode=args.feature_selection,
        top_n=args.feature_top_n,
    )
    print(f"Labeled rows: {len(merged):,}")
    print(f"Positive rate: {merged['is_uptrend'].mean() * 100.0:.2f}%")

    dataset = build_sequence_dataset(merged, feature_names, sequence_length=args.sequence_length)
    split = chronological_split(dataset, test_size=args.test_size)

    mean, std = fit_standardization(split.X_train)
    X_train = apply_standardization(split.X_train, mean, std)
    X_test = apply_standardization(split.X_test, mean, std)
    X_all = apply_standardization(dataset.X, mean, std)

    class_weight = _resolve_class_weight(args, split.y_train)

    print("Training LSTM...")
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        learning_rate=args.learning_rate,
    )

    callbacks = []
    try:
        keras = importlib.import_module("tensorflow").keras

        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=4,
                restore_best_weights=True,
            )
        )
    except Exception:
        pass

    model.fit(
        X_train,
        split.y_train,
        validation_data=(X_test, split.y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        verbose=1,
        callbacks=callbacks,
    )

    prob_test = model.predict(X_test, verbose=0).reshape(-1)
    pred_test = (prob_test >= 0.5).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    print("\n" + "=" * 80)
    print("LSTM EVALUATION")
    print("=" * 80)
    print(f"Test samples: {len(split.y_test):,}")
    print(f"Actual uptrends: {int(split.y_test.sum())} ({split.y_test.mean() * 100.0:.2f}%)")
    print(f"Predicted uptrends: {int(pred_test.sum())} ({pred_test.mean() * 100.0:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(split.y_test, pred_test, target_names=["No Uptrend", "Uptrend"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(split.y_test, pred_test))
    try:
        print(f"ROC-AUC: {roc_auc_score(split.y_test, prob_test):.4f}")
    except Exception:
        print("ROC-AUC: unavailable")

    prob_all = model.predict(X_all, verbose=0).reshape(-1)

    model_dir = _resolve_output_path(args.model_dir)
    save_lstm_artifacts(
        model,
        out_dir=model_dir,
        feature_names=feature_names,
        sequence_length=args.sequence_length,
        mean=mean,
        std=std,
        label_cfg=label_cfg,
    )
    print(f"Saved model artifacts to: {model_dir}")

    _save_signal_cache(
        df_15m=df_15m,
        idx=split.idx_test,
        y_true=split.y_test,
        probability=prob_test,
        out_path=_resolve_output_path(args.test_cache_out),
    )
    _save_signal_cache(
        df_15m=df_15m,
        idx=dataset.idx,
        y_true=dataset.y,
        probability=prob_all,
        out_path=_resolve_output_path(args.full_cache_out),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

