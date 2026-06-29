#!/usr/bin/env python3
"""
v15 Pattern Training : ATR-Based Labels
========================================
Trains pattern-specialist single-stage models using ATR-scaled TP/SL targets.
Uses the v15 deterministic feature pipeline (no HMM).

Usage:
  python3 v15/tools/train_patterns_v15.py uptrend_retrace_v15
  python3 v15/tools/train_patterns_v15.py  # all v15 patterns
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import joblib
import pandas as pd

from v15._paths import PROJECT_ROOT
from v15.config.v15_patterns import (
    V15_PATTERN_REGISTRY,
    PATTERN_MODEL_DIR,
    collect_v15_pa_groups,
)
from config.hybrid_config import WF_CONFIG
from xgboost_filter_model.pattern_training import (
    add_pattern_entry_target,
    apply_exec_labels,
    backup_variant_models,
    cycle_model_path,
    feature_columns,
    execution_target_mode,
    execution_tp_sl,
    fit_pattern_model,
    pattern_model_dir,
    pattern_variant_tag,
    precompute_future_moves,
    prod_model_path,
    prod_train_slice,
    wf_timestamps,
    iter_wf_train_targets,
    wf_train_mode,
)
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14
from v15.features import add_v15_energetic_features


# ── v15 pattern router helpers (uses V15_PATTERN_REGISTRY, not v14) ──────

_OPS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _v15_mask_single(df: pd.DataFrame, rules: list[dict]) -> pd.Series:
    """True where all rules pass (per-column comparison)."""
    if not rules:
        return pd.Series(True, index=df.index)
    mask = pd.Series(True, index=df.index)
    for rule in rules:
        feat = rule["feat"]
        if feat not in df.columns:
            return pd.Series(False, index=df.index)
        mask &= _OPS[rule["op"]](df[feat], rule["val"])
    return mask


def v15_pattern_mask(df: pd.DataFrame, pattern_name: str) -> pd.Series:
    """Boolean mask for one v15 pattern (context + pattern rules, minus excludes)."""
    spec = V15_PATTERN_REGISTRY[pattern_name]
    # training uses "pattern" rules (not "router" rules)
    rules = spec.get("pattern", [])
    mask = _v15_mask_single(df, spec.get("context", []))
    mask &= _v15_mask_single(df, rules)
    exclude = spec.get("exclude", [])
    if exclude:
        mask &= ~_v15_mask_single(df, exclude)
    return mask


# ── Training ─────────────────────────────────────────────────────────────

def train_one_v15_pattern(pattern_name: str, df: pd.DataFrame, *, feature_set: str) -> None:
    spec = V15_PATTERN_REGISTRY[pattern_name]
    bias = spec["direction_bias"]
    ex = spec["execution"]
    tp, sl = execution_tp_sl(ex)
    mode = execution_target_mode(ex)
    variant = pattern_variant_tag(ex["horizon"], tp, sl, target_mode=mode)
    out_dir = pattern_model_dir(PATTERN_MODEL_DIR, pattern_name, variant=variant)

    mask = v15_pattern_mask(df, pattern_name)
    df_pat = add_pattern_entry_target(df.loc[mask].copy(), bias)
    feats = feature_columns(df_pat)

    if df_pat.empty:
        print(f"  SKIP {pattern_name}: no samples")
        return

    print(f"\n{'='*70}")
    print(f"  v15 Pattern: {pattern_name} ({bias}, single-stage, {mode})  ->  {variant}")
    print(f"  Features: {feature_set} ({len(feats)} cols)")
    print(f"  Labels  : H={ex['horizon']} TP={tp}xATR SL={sl}xATR")
    print(f"  Samples : {len(df_pat)} / {len(df)} total bars")
    print(f"  Range   : {df_pat.index.min()} -> {df_pat.index.max()}")
    print(f"  Target+ : {int(df_pat['target_pattern'].sum())} "
          f"({df_pat['target_pattern'].mean()*100:.1f}%)")
    print(f"{'='*70}")

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
    only = sys.argv[1:] if len(sys.argv) > 1 else list(V15_PATTERN_REGISTRY.keys())
    only = [n for n in only if n in V15_PATTERN_REGISTRY]

    if not only:
        available = list(V15_PATTERN_REGISTRY.keys())
        print(f"No valid v15 patterns found in args. Available: {available}")
        sys.exit(1)

    by_feature_set: dict[str, list[str]] = defaultdict(list)
    for name in only:
        pfs = V15_PATTERN_REGISTRY[name].get("feature_set", "current")
        by_feature_set[str(pfs).lower()].append(name)

    print("Training v15 patterns with ATR-based TP/SL labels")
    for pfs, names in sorted(by_feature_set.items()):
        print(f"  {pfs}: {', '.join(names)}")

    for pfs, names in sorted(by_feature_set.items()):
        print(f"\nLoading v15 feature matrix set={pfs}...")
        ex0 = V15_PATTERN_REGISTRY[names[0]]["execution"]
        df_feat = prepare_data_v14(
            start_date=WF_CONFIG["full_start"],
            end_date=WF_CONFIG["wf_end"],
            energetic_filter=False,
            pa_groups=collect_v15_pa_groups(names),
            label_horizon=int(ex0["horizon"]),
            label_tp=float(ex0.get("tp_atr", ex0["tp"])),
            label_sl=float(ex0.get("sl_atr", ex0["sl"])),
            fixed_label_tp_sl=True,
            pattern_feature_set=pfs,
        )
        df_feat = prepare_directional_data_v14(df_feat)
        df_feat = add_v15_energetic_features(df_feat)

        future_by_h = precompute_future_moves(df_feat, [int(ex0["horizon"])])
        print(f"  Matrix: {len(df_feat)} bars | {len(feature_columns(df_feat))} model features")

        for name in names:
            ex = V15_PATTERN_REGISTRY[name]["execution"]
            df = apply_exec_labels(
                df_feat,
                int(ex["horizon"]),
                float(ex.get("tp_atr", ex["tp"])),
                float(ex.get("sl_atr", ex["sl"])),
                future_moves=future_by_h[int(ex["horizon"])],
                target_mode=execution_target_mode(ex),
            )
            train_one_v15_pattern(name, df, feature_set=pfs)

    print("\nDone.")


if __name__ == "__main__":
    main()
