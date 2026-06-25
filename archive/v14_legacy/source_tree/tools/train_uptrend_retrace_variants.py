#!/usr/bin/env python3
"""
Retrain uptrend_retrace for H=60, TP=30, SL=15/20/25 (fixed labels = execution).

Usage:
  python3 train_uptrend_retrace_variants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from _paths import PROJECT_ROOT

import joblib
import pandas as pd

from config.hybrid_config import WF_CONFIG
from config.pattern_registry import PATTERN_MODEL_DIR, PATTERN_REGISTRY
from xgboost_filter_model.pattern_router import pattern_mask
from xgboost_filter_model.pattern_training import (
    add_pattern_entry_target,
    cycle_model_path,
    feature_columns,
    fit_pattern_model,
    pattern_model_dir,
    pattern_variant_tag,
    prod_model_path,
    prod_train_slice,
    wf_timestamps,
)
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14

PATTERN_NAME = "uptrend_retrace"
HORIZON = 60
TP = 30.0
STOPS = [15, 20, 25]


def train_variant(df: pd.DataFrame, sl: float) -> None:
    spec = PATTERN_REGISTRY[PATTERN_NAME]
    bias = spec["direction_bias"]
    variant = pattern_variant_tag(HORIZON, TP, sl)
    out_dir = pattern_model_dir(PATTERN_MODEL_DIR, PATTERN_NAME, variant=variant)

    mask = pattern_mask(df, PATTERN_NAME)
    df_pat = add_pattern_entry_target(df.loc[mask].copy(), bias)
    feats = feature_columns(df_pat)

    print(f"\n{'='*60}")
    print(f"  {PATTERN_NAME}  H={HORIZON} TP={TP} SL={sl}  →  {variant}")
    print(f"  Samples: {len(df_pat)}  |  Target+: {int(df_pat['target_pattern'].sum())} "
          f"({df_pat['target_pattern'].mean()*100:.1f}%)")
    print(f"{'='*60}")

    if len(df_pat) < 50:
        print("  SKIP: too few samples")
        return

    wf_start, retrain_days = wf_timestamps()
    data_start = df_pat.index.min()
    df_pre = prod_train_slice(df_pat, wf_start)
    prod = fit_pattern_model(df_pre[feats], df_pre["target_pattern"], min_samples=20)
    if prod is None:
        for frac in (0.6, 0.8, 1.0):
            cut = data_start + (df_pat.index.max() - data_start) * frac
            trial = df_pat[df_pat.index < cut]
            prod = fit_pattern_model(trial[feats], trial["target_pattern"], min_samples=20)
            if prod is not None:
                df_pre = trial
                break
    if prod is None:
        print("  SKIP: could not train prod model")
        return

    joblib.dump(prod, prod_model_path(out_dir))
    print(f"  Saved {prod_model_path(out_dir)} (prod n={len(df_pre)})")

    current_start = wf_start
    end_dt = df_pat.index.max()
    cycle = 1
    while current_start < end_dt:
        train_chunk = df_pat[df_pat.index < current_start]
        path = cycle_model_path(out_dir, cycle, current_start.date())
        model = fit_pattern_model(
            train_chunk[feats], train_chunk["target_pattern"], min_samples=20
        )
        joblib.dump(model if model is not None else prod, path)
        print(f"  Cycle {cycle} ({current_start.date()}): n={len(train_chunk)}")
        current_start += pd.Timedelta(days=retrain_days)
        cycle += 1


def main() -> None:
    print(f"Loading data with label H={HORIZON} TP={TP} SL in {STOPS}…")
    for sl in STOPS:
        print(f"\n--- Preparing labels for SL={sl} ---")
        df = prepare_data_v14(
            start_date=WF_CONFIG["full_start"],
            end_date=WF_CONFIG["wf_end"],
            energetic_filter=False,
            label_horizon=HORIZON,
            label_tp=TP,
            label_sl=float(sl),
            fixed_label_tp_sl=True,
        )
        df = prepare_directional_data_v14(df)
        train_variant(df, float(sl))

    print("\nDone. Models under:")
    for sl in STOPS:
        v = pattern_variant_tag(HORIZON, TP, sl)
        print(f"  runtime/bot_assets/wf_models_v14_patterns/{PATTERN_NAME}/{v}/")


if __name__ == "__main__":
    main()
