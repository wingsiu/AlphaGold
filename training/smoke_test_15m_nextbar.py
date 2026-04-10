#!/usr/bin/env python3
"""Quick validation harness for the 15-minute next-bar model pipeline."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import DataLoader
try:
    from training.ml_alpha_model_15m_nextbar import (
        DEFAULT_FORWARD_BARS,
        DEFAULT_MAX_ADVERSE_LOW_PCT,
        DEFAULT_THRESHOLD_PCT,
        UptrendRecognitionSystem15mNextBar,
        prepare_gold_data_15m,
    )
except ModuleNotFoundError:
    from ml_alpha_model_15m_nextbar import (
        DEFAULT_FORWARD_BARS,
        DEFAULT_MAX_ADVERSE_LOW_PCT,
        DEFAULT_THRESHOLD_PCT,
        UptrendRecognitionSystem15mNextBar,
        prepare_gold_data_15m,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test 15-minute next-bar data prep and labeling.")
    parser.add_argument("--table", default="gold_prices")
    parser.add_argument("--start-date", default="2025-05-20")
    parser.add_argument("--end-date", default="2025-05-23")
    parser.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT)
    parser.add_argument("--max-adverse-low-pct", type=float, default=DEFAULT_MAX_ADVERSE_LOW_PCT)
    parser.add_argument("--forward-bars", type=int, default=DEFAULT_FORWARD_BARS)
    args = parser.parse_args()

    print("Loading raw data...")
    raw = DataLoader().load_data(args.table, start_date=args.start_date, end_date=args.end_date)
    df = prepare_gold_data_15m(raw)
    print(f"15-minute candles: {len(df):,}")
    if df.empty:
        raise ValueError("15-minute aggregation returned no rows; check timestamp parsing and source data coverage")
    print(f"Range: {df.index[0]} -> {df.index[-1]}")

    system = UptrendRecognitionSystem15mNextBar(
        threshold_pct=args.threshold_pct,
        max_adverse_low_pct=args.max_adverse_low_pct,
        forward_bars=args.forward_bars,
    )
    feature_df = system.extractor.create_feature_dataframe(df)
    label_df = system.labeler.create_labels(df, feature_df)
    merged = feature_df.merge(label_df, on="idx", how="inner")

    print(f"Feature rows: {len(feature_df):,}")
    print(f"Label rows: {len(label_df):,}")
    print(f"Merged rows: {len(merged):,}")
    if not merged.empty:
        pos_rate = float(merged['is_uptrend'].mean()) * 100.0
        print(f"Positive rate: {pos_rate:.2f}%")
        print(merged[["idx", "is_uptrend", "future_gain_pct", "future_drawdown_pct"]].head(5).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

