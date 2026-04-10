#!/usr/bin/env python3
"""Generate model signal cache for a full date range."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import DataLoader
try:
    from training.training import prepare_gold_data
except ModuleNotFoundError:
    from training import prepare_gold_data
try:
    from training.ml_alpha_model import UptrendRecognitionSystem
except ModuleNotFoundError:
    from ml_alpha_model import UptrendRecognitionSystem


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RF/GB probabilities for a date range.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--table", default="gold_prices")
    parser.add_argument("--model-dir", default="ml_models")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("Loading price data...")
    raw = DataLoader().load_data(args.table, start_date=args.start_date, end_date=args.end_date)
    df = prepare_gold_data(raw)
    print(f"Candles: {len(df):,}")

    print("Loading models...")
    system = UptrendRecognitionSystem()
    system.load_models(args.model_dir)

    print("Extracting features...")
    feature_df = system.extractor.create_feature_dataframe(df)
    X = feature_df[system.extractor.feature_names].fillna(0)

    rf_model = system.models["random_forest"]
    gb_model = system.models["gradient_boosting"]

    rows = feature_df[["idx"]].copy()
    rows["timestamp"] = df.index[rows["idx"].astype(int)].astype(str).values
    rows["rf_probability"] = rf_model.predict_proba(X)
    rows["gb_probability"] = gb_model.predict_proba(X)
    rows["rf_signal"] = rf_model.predict(X).astype(int)
    rows["gb_signal"] = gb_model.predict(X).astype(int)
    rows = rows.sort_values("idx").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows.to_csv(args.out, index=False)
    print(f"Saved signals: {args.out} ({len(rows):,} rows)")
    print(f"Signal range: {rows['timestamp'].iloc[0]} -> {rows['timestamp'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

