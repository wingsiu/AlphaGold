#!/usr/bin/env python3
"""v30 — Deterministic Energetic Gate Sweep — Full Feature Grid
================================================================
Compares production (HMM-based) energetic_bar_mask against the v15
fully-deterministic gate (prev_move + rolling_bm + vol_ratio + range_exp).

Goal: find a deterministic feature set that closely tracks HMM's gate.

Each config sets env vars, runs `energetic_bar_mask_v15()`, and reports:
  - Energetic bar count
  - Overlap % with HMM baseline
  - New bars added / HMM bars missed

Usage:
  # Quick sweep on last 30 days (no backtest)
  python3 v15/research/v30_energetic_deterministic.py

  # Full backtest sweep on WF period
  V30_FULL=1 python3 v15/research/v30_energetic_deterministic.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("V14_HYBRID", "1")
os.environ.setdefault("V14_FVG_MIN_GAP", "0")

from config.v14_config import WF_CONFIG
from config.v14_patterns import PRODUCTION_PATTERNS, backtest_feature_set, collect_pa_groups
from xgboost_filter_model.energetic_gate import energetic_bar_mask as hmm_mask
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14

from v15.energetic_gate import energetic_bar_mask_v15


def clear_env():
    """Reset all V15_EN_* env vars."""
    for k in list(os.environ.keys()):
        if k.startswith("V15_EN_"):
            del os.environ[k]


def run_config(
    df: pd.DataFrame,
    baseline_mask: pd.Series,
    label: str,
    prev_move: float = 0,
    rolling_bm_min: float = 0,
    vol_ratio_min: float = 0,
    range_exp_min: float = 0,
) -> dict:
    """Run one config and return metrics."""
    clear_env()
    os.environ["V15_EN_PREV_MOVE"] = str(prev_move)
    os.environ["V15_EN_ROLLING_BM_WIN"] = "5"
    os.environ["V15_EN_ROLLING_BM_MIN"] = str(rolling_bm_min)
    os.environ["V15_EN_VOL_RATIO_WIN"] = "20"
    os.environ["V15_EN_VOL_RATIO_MIN"] = str(vol_ratio_min)
    os.environ["V15_EN_RANGE_EXP_WIN"] = "5"
    os.environ["V15_EN_RANGE_EXP_MIN"] = str(range_exp_min)

    mask = energetic_bar_mask_v15(df)
    n = int(mask.sum())
    overlap = int((mask & baseline_mask).sum())
    overlap_pct = overlap / max(int(baseline_mask.sum()), 1) * 100
    new_only = int((mask & ~baseline_mask).sum())
    missed = int((~mask & baseline_mask).sum())

    return {
        "label": label,
        "bars": n,
        "overlap_pct": round(overlap_pct, 1),
        "new_only": new_only,
        "missed": missed,
        "prev_move": prev_move,
        "rolling_bm": rolling_bm_min,
        "vol_ratio": vol_ratio_min,
        "range_exp": range_exp_min,
    }


def main():
    print("=" * 70)
    print("  V30 — DETERMINISTIC ENERGETIC GATE — FULL FEATURE SWEEP")
    print("  vs HMM baseline (no HMM in v15 gate)")
    print("=" * 70)

    # --- Load feature matrix ---
    do_full = os.environ.get("V30_FULL", "").strip() in ("1", "yes", "true")
    today = date.today()
    if do_full:
        start = WF_CONFIG["wf_start"]
        period_label = f"full WF: {start} → {today}"
    else:
        start = (today - timedelta(days=60)).strftime("%Y-%m-%d")
        period_label = f"last 60 days: {start} → {today}"
    end = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\nPeriod: {period_label}")
    warmup_days = int(WF_CONFIG.get("feature_warmup_days", 120))
    load_start = (
        pd.Timestamp(start).tz_localize("UTC") - pd.Timedelta(days=warmup_days)
    ).strftime("%Y-%m-%d")

    df = prepare_data_v14(
        start_date=load_start, end_date=end,
        energetic_filter=False, for_live_inference=True,
        pa_groups=collect_pa_groups(list(PRODUCTION_PATTERNS)),
        pattern_feature_set=backtest_feature_set(),
    )
    df = prepare_directional_data_v14(df)
    df_test = df[df.index >= pd.Timestamp(start).tz_localize("UTC")].copy()

    # --- HMM baseline ---
    print("\nComputing HMM baseline mask...")
    baseline_mask = hmm_mask(df_test)
    bl_n = int(baseline_mask.sum())
    print(f"  HMM: {bl_n} / {len(df_test)} bars ({bl_n/len(df_test)*100:.1f}%)")

    # --- Sweep grid ---
    print(f"\n{'='*70}")
    print(f"  SWEEP: PREV_MOVE × ROLLING_BM × VOL_RATIO × RANGE_EXP")
    print(f"{'='*70}")

    configs = []

    # ---- Sweep 1: prev_move alone (0=off, 0.5, 1.0, ..., 6.0) ----
    print("\n[1] prev_move sweep (rolling_bm=off, vol_ratio=off, range_exp=off)")
    header = f"  {'prev_move':>10s} {'bars':>8s} {'HMM%':>8s} {'new':>6s} {'miss':>6s}"
    print(header)
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for pv in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
        r = run_config(df_test, baseline_mask, f"prev={pv}", prev_move=pv)
        configs.append(r)
        print(f"  {pv:10.1f} {r['bars']:8d} {r['overlap_pct']:7.1f}% {r['new_only']:6d} {r['missed']:6d}")

    # ---- Sweep 2: rolling_bm alone (off, 1.0, 1.5, ..., 3.5) ----
    print("\n[2] rolling_bar_move sweep (prev_move=off, vol_ratio=off, range_exp=off)")
    print(header.replace("prev_move", "rolling_bm"))
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for rb in [0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        r = run_config(df_test, baseline_mask, f"roll_bm={rb}", rolling_bm_min=rb)
        configs.append(r)
        print(f"  {rb:10.1f} {r['bars']:8d} {r['overlap_pct']:7.1f}% {r['new_only']:6d} {r['missed']:6d}")

    # ---- Sweep 3: vol_ratio alone ----
    print("\n[3] volume_ratio sweep (prev_move=off, rolling_bm=off, range_exp=off)")
    print(header.replace("prev_move", "vol_ratio"))
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for vr in [0, 0.8, 1.0, 1.2, 1.5, 2.0]:
        r = run_config(df_test, baseline_mask, f"vol_r={vr}", vol_ratio_min=vr)
        configs.append(r)
        print(f"  {vr:10.1f} {r['bars']:8d} {r['overlap_pct']:7.1f}% {r['new_only']:6d} {r['missed']:6d}")

    # ---- Sweep 4: range_exp alone ----
    print("\n[4] range_expansion sweep (prev_move=off, rolling_bm=off, vol_ratio=off)")
    print(header.replace("prev_move", "range_exp"))
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for re in [0, 0.8, 1.0, 1.2, 1.5]:
        r = run_config(df_test, baseline_mask, f"range_e={re}", range_exp_min=re)
        configs.append(r)
        print(f"  {re:10.1f} {r['bars']:8d} {r['overlap_pct']:7.1f}% {r['new_only']:6d} {r['missed']:6d}")

    # ---- Sweep 5: best combos — prev_move + rolling_bm + vol_ratio + range_exp ----
    print(f"\n[5] COMBINATIONS — prev_move + rolling_bm + vol_ratio + range_exp")
    print(f"  {'config':<45s} {'bars':>8s} {'HMM%':>8s} {'new':>6s} {'miss':>6s}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")

    combos = [
        # prev + rolling_bm
        ("prev=1.5 + roll_bm=2.0", 1.5, 2.0, 0, 0),
        ("prev=2.0 + roll_bm=2.0", 2.0, 2.0, 0, 0),
        ("prev=1.5 + roll_bm=2.5", 1.5, 2.5, 0, 0),
        ("prev=1.0 + roll_bm=2.0", 1.0, 2.0, 0, 0),
        # prev + vol_ratio
        ("prev=1.5 + vol_r=1.0", 1.5, 0, 1.0, 0),
        ("prev=2.0 + vol_r=1.0", 2.0, 0, 1.0, 0),
        ("prev=1.5 + vol_r=1.2", 1.5, 0, 1.2, 0),
        # prev + range_exp
        ("prev=1.5 + range_e=1.0", 1.5, 0, 0, 1.0),
        ("prev=2.0 + range_e=1.0", 2.0, 0, 0, 1.0),
        # rolling_bm + vol_ratio
        ("roll_bm=2.0 + vol_r=1.0", 0, 2.0, 1.0, 0),
        ("roll_bm=2.0 + vol_r=1.2", 0, 2.0, 1.2, 0),
        # all three
        ("prev=1.5 + roll_bm=2.0 + vol_r=1.0", 1.5, 2.0, 1.0, 0),
        ("prev=1.5 + roll_bm=2.0 + range_e=1.0", 1.5, 2.0, 0, 1.0),
        ("prev=1.0 + roll_bm=2.0 + vol_r=1.0", 1.0, 2.0, 1.0, 0),
        # all four
        ("prev=1.5 + roll_bm=2.0 + vol_r=1.0 + range_e=1.0", 1.5, 2.0, 1.0, 1.0),
        ("prev=1.0 + roll_bm=2.0 + vol_r=1.0 + range_e=1.0", 1.0, 2.0, 1.0, 1.0),
        # lighter
        ("prev=0.5 + roll_bm=1.5 + vol_r=0.8 + range_e=0.8", 0.5, 1.5, 0.8, 0.8),
        ("prev=0.5 + roll_bm=1.5 + vol_r=1.0", 0.5, 1.5, 1.0, 0),
    ]

    for label, pv, rb, vr, re in combos:
        r = run_config(df_test, baseline_mask, label, prev_move=pv, rolling_bm_min=rb, vol_ratio_min=vr, range_exp_min=re)
        configs.append(r)
        print(f"  {label:<45s} {r['bars']:8d} {r['overlap_pct']:7.1f}% {r['new_only']:6d} {r['missed']:6d}")

    # --- Top configs by overlap ---
    print(f"\n{'='*70}")
    print("  TOP 15 CONFIGS BY HMM OVERLAP %")
    print(f"{'='*70}")
    print(f"  {'rank':>4s} {'label':<45s} {'bars':>8s} {'HMM%':>8s} {'new':>6s} {'miss':>6s}")
    print(f"  {'-'*4} {'-'*45} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")

    sorted_cfgs = sorted(configs, key=lambda x: (-x["overlap_pct"], x["missed"]))
    for i, r in enumerate(sorted_cfgs[:15], 1):
        print(f"  {i:4d} {r['label']:<45s} {r['bars']:8d} {r['overlap_pct']:7.1f}% {r['new_only']:6d} {r['missed']:6d}")

    # --- Day-by-day for top candidates ---
    print(f"\n{'='*70}")
    print("  DAY-BY-DAY — TOP 4 CONFIGS")
    print(f"{'='*70}")

    df_daily = df_test.copy()
    df_daily["_date"] = df_daily.index.floor("D")
    baseline_daily = baseline_mask.groupby(df_daily["_date"]).sum().astype(int)

    for r in sorted_cfgs[:4]:
        label = r["label"]
        clear_env()
        os.environ["V15_EN_PREV_MOVE"] = str(r["prev_move"])
        os.environ["V15_EN_ROLLING_BM_WIN"] = "5"
        os.environ["V15_EN_ROLLING_BM_MIN"] = str(r["rolling_bm"])
        os.environ["V15_EN_VOL_RATIO_WIN"] = "20"
        os.environ["V15_EN_VOL_RATIO_MIN"] = str(r["vol_ratio"])
        os.environ["V15_EN_RANGE_EXP_WIN"] = "5"
        os.environ["V15_EN_RANGE_EXP_MIN"] = str(r["range_exp"])

        mask = energetic_bar_mask_v15(df_test)
        v15_daily = mask.groupby(df_daily["_date"]).sum().astype(int)

        print(f"\n  Config: {label}")
        print(f"  {'Day':>12s} {'HMM':>6s} {'v15':>6s} {'overlap%':>10s}")
        common_dates = baseline_daily.index.intersection(v15_daily.index)
        for d in common_dates[-10:]:
            if baseline_daily[d] == 0 and v15_daily[d] == 0:
                continue
            ovl = min(baseline_daily[d], v15_daily[d])
            pct = ovl / max(baseline_daily[d], 1) * 100
            print(f"  {d.strftime('%Y-%m-%d'):>12s} {baseline_daily[d]:6d} {v15_daily[d]:6d} {pct:10.1f}%")

    print("\nDone. Use V15_EN_* env vars to set the best config in v15 backtest.")

if __name__ == "__main__":
    main()
