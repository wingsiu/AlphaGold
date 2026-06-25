#!/usr/bin/env python3
"""
Train one oil pattern specialist (prod-only by default — no WF cycle joblibs).

Mirrors gold test calendar: train on bars before TEST_START, backtest TEST_START→TEST_END.

Usage (from repo root):
  PYTHONPATH=. .venv/bin/python3 oil/tools/train.py
  PYTHONPATH=. .venv/bin/python3 oil/tools/train.py oil_downtrend_retrace
  PYTHONPATH=. .venv/bin/python3 oil/tools/train.py oil_downtrend_retrace --wf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from oil.bootstrap import apply_oil_registry
from oil.config import FULL_START, PRICE_TABLE, PROD_TRAIN_END, WF_END, WF_START
from oil.patterns import PATTERN_REGISTRY, collect_pa_groups, enrich_pattern_features, pattern_feature_set_for

apply_oil_registry()

from config.pattern_registry import PATTERN_MODEL_DIR  # patched → oil dir
from xgboost_filter_model.pattern_router import pattern_mask
from xgboost_filter_model.pattern_training import (
    add_pattern_entry_target,
    cycle_model_path,
    execution_target_mode,
    execution_tp_sl,
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
    wf_timestamps,
)
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14


def train_one(
    pattern_name: str,
    df_feat: pd.DataFrame,
    *,
    feature_set: str,
    prod_only: bool,
) -> None:
    spec = PATTERN_REGISTRY[pattern_name]
    bias = spec["direction_bias"]
    ex = pattern_execution(pattern_name)
    tp, sl = execution_tp_sl(ex)
    mode = execution_target_mode(ex)
    variant = pattern_variant_tag(ex["horizon"], tp, sl, target_mode=mode)
    out_dir = pattern_model_dir(PATTERN_MODEL_DIR, pattern_name, variant=variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = pattern_mask(df_feat, pattern_name, training=True)
    df_pat = add_pattern_entry_target(df_feat.loc[mask].copy(), bias)
    feats = feature_columns(df_pat)
    if len(df_pat) < 50:
        print(f"SKIP {pattern_name}: fewer than 50 samples ({len(df_pat)})")
        return

    cut = pd.Timestamp(PROD_TRAIN_END)
    if cut.tzinfo is None:
        cut = cut.tz_localize("UTC")
    else:
        cut = cut.tz_convert("UTC")
    df_pre = df_pat[df_pat.index < cut]
    prod = fit_pattern_model(df_pre[feats], df_pre["target_pattern"], min_samples=20)
    if prod is None:
        print(f"SKIP {pattern_name}: could not fit prod model (n={len(df_pre)})")
        return

    joblib.dump(prod, prod_model_path(out_dir))
    pos = int(df_pre["target_pattern"].sum())
    pct = 100.0 * df_pre["target_pattern"].mean()
    print(f"Saved prod -> {prod_model_path(out_dir)} (train n={len(df_pre)}, < {PROD_TRAIN_END})")
    tp_med = float(df_pre["dynamic_tp"].median())
    sl_med = float(df_pre["dynamic_sl"].median())
    label = (
        f"ATR target TP={tp}×ATR SL={sl}×ATR (med {tp_med:.1f}/{sl_med:.1f} DB)"
        if mode == "atr"
        else f"fixed TP={tp} SL={sl} DB"
    )
    print(f"  Target ratio (train): {pos}/{len(df_pre)} = {pct:.2f}% positive — {label}")

    if prod_only:
        for p in out_dir.glob("filter_cycle_*.joblib"):
            p.unlink()
            print(f"Removed {p.name} (prod-only mode)")
        return

    wf_start, _ = wf_timestamps()
    for cycle, current_start in iter_wf_train_targets(
        lambda c, d: cycle_model_path(out_dir, c, d),
    ):
        train_chunk = df_pat[df_pat.index < current_start]
        path = cycle_model_path(out_dir, cycle, current_start.date())
        model = fit_pattern_model(
            train_chunk[feats], train_chunk["target_pattern"], min_samples=20
        )
        joblib.dump(model if model is not None else prod, path)
        print(f"  cycle {cycle} -> {path.name} (n={len(train_chunk)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train oil pattern model(s)")
    parser.add_argument("patterns", nargs="*", help="Pattern name(s); default: all in registry")
    parser.add_argument(
        "--wf",
        action="store_true",
        help="Also write WF cycle_* joblibs (default: prod-only for testing)",
    )
    args = parser.parse_args()
    names = args.patterns or list(PATTERN_REGISTRY.keys())
    names = [n for n in names if n in PATTERN_REGISTRY]
    if not names:
        print("No valid pattern names. Registry:", list(PATTERN_REGISTRY.keys()))
        sys.exit(1)

    prod_only = not args.wf
    print(f"Oil train | table={PRICE_TABLE} | prod_only={prod_only}")
    print(f"  data: {FULL_START} → {WF_END} | prod train < {PROD_TRAIN_END}")

    by_fs: dict[str, list[str]] = {}
    for n in names:
        by_fs.setdefault(pattern_feature_set_for(n), []).append(n)

    for pfs, group in sorted(by_fs.items()):
        ex0 = pattern_execution(group[0])
        print(f"\nLoading features (set={pfs})…")
        df_feat = prepare_data_v14(
            start_date=FULL_START,
            end_date=WF_END,
            energetic_filter=False,
            pa_groups=collect_pa_groups(group),
            label_horizon=int(ex0["horizon"]),
            label_tp=float(ex0["tp"]),
            label_sl=float(ex0["sl"]),
            fixed_label_tp_sl=True,
            pattern_feature_set=pfs,
            price_table=PRICE_TABLE,
        )
        df_feat = prepare_directional_data_v14(df_feat)
        df_feat = enrich_pattern_features(df_feat, group)
        if pfs == "short_impulse":
            n = int(
                (
                    (df_feat["bar_change"] < -14.0)
                    & (df_feat["volume"] > 1000)
                    & (df_feat["oil_session"] == 1)
                ).sum()
            )
            print(f"  short_impulse coarse bars (session+vol+drop): {n}")
        future_by_h = precompute_future_moves(df_feat, pattern_horizons(group))
        print(f"  matrix: {len(df_feat)} bars")

        for name in group:
            df = label_df_for_pattern(df_feat, name, future_by_h)
            train_one(name, df, feature_set=pfs, prod_only=prod_only)


if __name__ == "__main__":
    main()
