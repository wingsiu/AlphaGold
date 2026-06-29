#!/usr/bin/env python3
"""
Train dip_short_rip XGB filter for v16 (not v15 pattern_models).

Saves to runtime/v16_models/dip_short_rip/filter_prod.joblib

Usage:
  PYTHONPATH=. python3 v16/tools/train_dip_short_rip.py 2024-01-01 2026-06-25
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features
from v16.backtest.ml import _fit_binary
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.dip_short_rip import build_labeled_set, feature_columns


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    start = args[0] if args else "2024-01-01"
    end = args[1] if len(args) > 1 else "2026-06-25"
    cfg = v16_config.DIP_SHORT_RIP

    print(f"Training dip_short_rip | {start} → {end}")
    df = load_gold_1m(start, end)
    feats = build_features(df)
    labeled = build_labeled_set(df, feats, cfg=cfg)
    if labeled.empty:
        print("No labeled rows.")
        return

    feat_cols = feature_columns(feats)
    X = feats.loc[labeled.index, feat_cols]
    y = labeled["short_win"]
    model = _fit_binary(X, y)
    if model is None:
        print("Fit failed (insufficient data).")
        return

    out_dir = PROJECT_ROOT / cfg["model_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "filter_prod.joblib"
    joblib.dump({"model": model, "feature_columns": feat_cols, "pattern": cfg}, out_path)

    pos = int(y.sum())
    print(f"Labeled rows: {len(labeled)}  short_win: {pos} ({pos/len(labeled)*100:.1f}%)")
    print(f"Features: {len(feat_cols)}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
