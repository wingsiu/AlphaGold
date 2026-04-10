#!/usr/bin/env python3
"""Smoke test for the 15-minute LSTM data pipeline."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import DataLoader
try:
    from training.ml_alpha_model_15m_nextbar import prepare_gold_data_15m
    from training.ml_lstm_15m import (
        DEFAULT_FORWARD_BARS,
        DEFAULT_MAX_ADVERSE_LOW_PCT,
        DEFAULT_SEQUENCE_LENGTH,
        DEFAULT_THRESHOLD_PCT,
        LabelConfig,
        build_labeled_feature_frame,
        build_sequence_dataset,
    )
except ModuleNotFoundError:
    from ml_alpha_model_15m_nextbar import prepare_gold_data_15m
    from ml_lstm_15m import (
        DEFAULT_FORWARD_BARS,
        DEFAULT_MAX_ADVERSE_LOW_PCT,
        DEFAULT_SEQUENCE_LENGTH,
        DEFAULT_THRESHOLD_PCT,
        LabelConfig,
        build_labeled_feature_frame,
        build_sequence_dataset,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test LSTM sequence dataset creation.")
    parser.add_argument("--table", default="gold_prices")
    parser.add_argument("--start-date", default="2025-05-20")
    parser.add_argument("--end-date", default="2025-05-23")
    parser.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT)
    parser.add_argument("--max-adverse-low-pct", type=float, default=DEFAULT_MAX_ADVERSE_LOW_PCT)
    parser.add_argument("--forward-bars", type=int, default=DEFAULT_FORWARD_BARS)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    args = parser.parse_args()

    cfg = LabelConfig(
        threshold_pct=args.threshold_pct,
        max_adverse_low_pct=args.max_adverse_low_pct,
        forward_bars=args.forward_bars,
    )

    raw = DataLoader().load_data(args.table, start_date=args.start_date, end_date=args.end_date)
    df_15m = prepare_gold_data_15m(raw)

    merged, feature_names = build_labeled_feature_frame(df_15m, label_cfg=cfg)
    dataset = build_sequence_dataset(merged, feature_names, sequence_length=args.sequence_length)

    print(f"15m candles: {len(df_15m):,}")
    print(f"Labeled rows: {len(merged):,}")
    print(f"Positive rate: {merged['is_uptrend'].mean() * 100.0:.2f}%")
    print(f"Sequence tensor: X={dataset.X.shape}, y={dataset.y.shape}")
    print(f"Sequence positive rate: {dataset.y.mean() * 100.0:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

