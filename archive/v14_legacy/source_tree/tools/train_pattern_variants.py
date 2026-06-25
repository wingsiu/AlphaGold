#!/usr/bin/env python3
"""Train pattern variants for any registered pattern (single-stage, fixed labels)."""
from __future__ import annotations

import sys
from pathlib import Path

from _paths import PROJECT_ROOT

import joblib
import pandas as pd

from config.hybrid_config import WF_CONFIG
from config.pattern_registry import PATTERN_MODEL_DIR, PATTERN_REGISTRY, collect_pa_groups
from xgboost_filter_model.pattern_router import pattern_mask
from xgboost_filter_model.pattern_training import (
    add_pattern_entry_target,
    apply_exec_labels,
    backup_variant_models,
    cycle_model_path,
    feature_columns,
    fit_pattern_model,
    pattern_model_dir,
    pattern_variant_tag,
    precompute_future_moves,
    prod_model_path,
    prod_train_slice,
    wf_timestamps,
)
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14


def train_variant(
    pattern_name: str,
    df: pd.DataFrame,
    *,
    horizon: int,
    tp: float,
    sl: float,
    quiet: bool = False,
) -> Path | None:
    spec = PATTERN_REGISTRY[pattern_name]
    bias = spec["direction_bias"]
    variant = pattern_variant_tag(horizon, tp, sl)
    out_dir = pattern_model_dir(PATTERN_MODEL_DIR, pattern_name, variant=variant)

    mask = pattern_mask(df, pattern_name, training=True)
    df_pat = add_pattern_entry_target(df.loc[mask].copy(), bias)
    feats = feature_columns(df_pat)

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  {pattern_name} ({bias})  H={horizon} TP={tp} SL={sl}  →  {variant}")
        print(f"  Samples: {len(df_pat)}  |  Target+: {int(df_pat['target_pattern'].sum())} "
              f"({df_pat['target_pattern'].mean()*100:.1f}%)")
        print(f"{'='*60}")

    if len(df_pat) < 50:
        if not quiet:
            print("  SKIP: too few samples")
        return None

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
        if not quiet:
            print("  SKIP: could not train prod model")
        return None

    backup_variant_models(out_dir, tag=f"h{horizon}_tp{int(tp)}_sl{int(sl)}")
    joblib.dump(prod, prod_model_path(out_dir))
    if not quiet:
        print(f"  Saved {prod_model_path(out_dir)} (n={len(df_pre)})")

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
        if not quiet:
            print(f"  Cycle {cycle} ({current_start.date()}): n={len(train_chunk)}")
        current_start += pd.Timedelta(days=retrain_days)
        cycle += 1
    return out_dir


def main() -> None:
    # Usage: train_pattern_variants.py uptrend_retrace 60 30 15 [20 25]
    if len(sys.argv) < 5:
        print("Usage: python3 train_pattern_variants.py <pattern> <horizon> <tp> <sl> [sl2 ...]")
        sys.exit(1)
    pattern_name = sys.argv[1]
    horizon = int(sys.argv[2])
    tp = float(sys.argv[3])
    stops = [float(x) for x in sys.argv[4:]]

    if pattern_name not in PATTERN_REGISTRY:
        print(f"Unknown pattern: {pattern_name}")
        sys.exit(1)

    print("Loading feature matrix (labels applied per variant H/TP/SL)…")
    df_feat = prepare_data_v14(
        start_date=WF_CONFIG["full_start"],
        end_date=WF_CONFIG["wf_end"],
        energetic_filter=False,
        pa_groups=collect_pa_groups([pattern_name]),
        label_horizon=horizon,
        label_tp=tp,
        label_sl=stops[0],
        fixed_label_tp_sl=True,
    )
    df_feat = prepare_directional_data_v14(df_feat)
    future_by_h = precompute_future_moves(df_feat, [horizon])

    for sl in stops:
        print(f"\n--- Labels H={horizon} TP={tp} SL={sl} ---")
        df = apply_exec_labels(df_feat, horizon, tp, sl, future_moves=future_by_h[horizon])
        train_variant(pattern_name, df, horizon=horizon, tp=tp, sl=sl)


if __name__ == "__main__":
    main()
