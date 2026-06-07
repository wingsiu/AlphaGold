#!/usr/bin/env python3
"""
Train pattern-specialist single-stage models.

Default: all 1-min bars (no HMM / bar_move / volume filter).
With PATTERN_GATE_CONFIG or V14_PATTERN_ENERGETIC_GATE=1: train only on energetic bars.

One XGB model per pattern: P(directional TP hit). Direction fixed by pattern bias.
Labels use each pattern's execution H/TP/SL from PATTERN_REGISTRY (not global 30/25/30).

Each pattern uses its registry feature_set (v2398=93 feats, current=96 feats).
See docs/v14_2398_baseline.md.

Usage:
  python3 train_patterns_v14.py
  python3 train_patterns_v14.py uptrend_retrace
  V14_PATTERN_FEATURE_SET=v2398 python3 train_patterns_v14.py uptrend_retrace  # override all
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from v14._paths import PROJECT_ROOT

import joblib
import pandas as pd

from config.v14_config import WF_CONFIG
from config.v14_patterns import PATTERN_MODEL_DIR, PATTERN_REGISTRY, collect_pa_groups, pattern_feature_set_for
from xgboost_filter_model.energetic_gate import energetic_bar_mask, pattern_gate_config
from xgboost_filter_model.pattern_router import count_pattern_samples, pattern_mask
from xgboost_filter_model.pattern_training import (
    add_pattern_entry_target,
    backup_variant_models,
    cycle_model_path,
    feature_columns,
    fit_pattern_model,
    iter_wf_train_targets,
    label_df_for_pattern,
    pattern_execution,
    pattern_horizons,
    pattern_model_dir,
    pattern_variant_tag,
    precompute_future_moves,
    prod_model_path,
    prod_train_slice,
    wf_train_mode,
    wf_timestamps,
)
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14


def train_one_pattern(pattern_name: str, df: pd.DataFrame, *, feature_set: str) -> None:
    spec = PATTERN_REGISTRY[pattern_name]
    bias = spec["direction_bias"]
    ex = pattern_execution(pattern_name)
    variant = pattern_variant_tag(ex["horizon"], ex["tp"], ex["sl"])
    out_dir = pattern_model_dir(PATTERN_MODEL_DIR, pattern_name, variant=variant)

    mask = pattern_mask(df, pattern_name, training=True)
    df_pat = add_pattern_entry_target(df.loc[mask].copy(), bias)
    feats = feature_columns(df_pat)

    if df_pat.empty:
        print(f"\n  SKIP {pattern_name}: no samples")
        return

    print(f"\n{'='*60}")
    print(f"  Pattern: {pattern_name} ({bias}, single-stage)  →  {variant}")
    print(f"  Features: {feature_set} ({len(feats)} cols)")
    print(f"  Labels  : H={ex['horizon']} TP={ex['tp']} SL={ex['sl']}")
    print(f"  Samples : {len(df_pat)} / {len(df)} total bars")
    print(f"  Range   : {df_pat.index.min()} → {df_pat.index.max()}")
    print(
        f"  Target+ : {int(df_pat['target_pattern'].sum())} "
        f"({df_pat['target_pattern'].mean()*100:.1f}%)"
    )
    print(f"{'='*60}")

    if len(df_pat) < 50:
        print("  SKIP: fewer than 50 pattern samples")
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

    targets = iter_wf_train_targets(
        lambda c, d: cycle_model_path(out_dir, c, d),
    )
    if not targets and wf_train_mode() == "incremental":
        print(f"  No new cycle to train for {pattern_name}")
        return

    if wf_train_mode() == "full":
        backup_variant_models(out_dir, tag=variant)
        joblib.dump(prod, prod_model_path(out_dir))
        print(f"  Saved prod model -> {prod_model_path(out_dir)} (n={len(df_pre)})")
    else:
        print(f"  incremental: prod model unchanged ({prod_model_path(out_dir).name})")

    for cycle, current_start in targets:
        train_chunk = df_pat[df_pat.index < current_start]
        path = cycle_model_path(out_dir, cycle, current_start.date())
        model = fit_pattern_model(
            train_chunk[feats], train_chunk["target_pattern"], min_samples=20
        )
        joblib.dump(model if model is not None else prod, path)
        print(
            f"  Cycle {cycle} ({current_start.date()}): "
            f"train n={len(train_chunk)} -> {path.name}"
        )

    print(f"  Walk-forward complete for {pattern_name} ({wf_train_mode()})")


def main() -> None:
    only = sys.argv[1:] if len(sys.argv) > 1 else list(PATTERN_REGISTRY.keys())
    only = [n for n in only if n in PATTERN_REGISTRY]

    by_feature_set: dict[str, list[str]] = defaultdict(list)
    for name in only:
        by_feature_set[pattern_feature_set_for(name)].append(name)

    print("Training patterns (per registry feature_set — see docs/v14_2398_baseline.md)")
    for pfs, names in sorted(by_feature_set.items()):
        print(f"  {pfs}: {', '.join(names)}")

    for pfs, names in sorted(by_feature_set.items()):
        print(f"\nLoading feature matrix set={pfs}…")
        ex0 = pattern_execution(names[0])
        df_feat = prepare_data_v14(
            start_date=WF_CONFIG["full_start"],
            end_date=WF_CONFIG["wf_end"],
            energetic_filter=False,
            pa_groups=collect_pa_groups(names),
            label_horizon=int(ex0["horizon"]),
            label_tp=float(ex0["tp"]),
            label_sl=float(ex0["sl"]),
            fixed_label_tp_sl=True,
            pattern_feature_set=pfs,
        )
        df_feat = prepare_directional_data_v14(df_feat)
        gate = pattern_gate_config()
        if gate["energetic_filter"]:
            e_mask = energetic_bar_mask(df_feat)
            print(
                f"  Energetic training filter: {int(e_mask.sum())} / {len(df_feat)} bars"
            )
            df_feat = df_feat.loc[e_mask]
        future_by_h = precompute_future_moves(df_feat, pattern_horizons(names))
        print(f"  Matrix: {len(df_feat)} bars | {len(feature_columns(df_feat))} model features")

        if pfs == list(by_feature_set.keys())[0]:
            counts = count_pattern_samples(df_feat)
            print("\nPattern sample counts (full history):")
            for pname, n in counts.items():
                print(f"  {pname:20s}: {n:6d}")

        for name in names:
            df = label_df_for_pattern(df_feat, name, future_by_h)
            train_one_pattern(name, df, feature_set=pfs)


if __name__ == "__main__":
    main()
