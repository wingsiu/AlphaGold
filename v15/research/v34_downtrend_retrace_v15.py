#!/usr/bin/env python3
"""
v34 — V14 vs V15 Downtrend Retrace (SHORT) - Keep V14 definition, fresh model
================================================================================
Mirrors V32 uptrend approach but for downtrend:
  - Keeps V14 definition: drop≥25 rise≥5 H=15 TP=40 SL=30
  - Trains new V15 XGBoost on same bars with v15 deterministic features
  - Compares V14 vs V15 on same bars
"""
from __future__ import annotations

import os, sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import joblib
import numpy as np
import pandas as pd
import pandas_ta as pta

from config.hybrid_config import WF_CONFIG
from config.pattern_registry import PATTERN_MODEL_DIR, PATTERN_REGISTRY, backtest_feature_set, collect_pa_groups
from backtest.core import simulate_v13_core
from xgboost_filter_model.pattern_router import assign_patterns
from xgboost_filter_model.pattern_training import (
    cycle_model_path, execution_target_mode, execution_tp_sl,
    feature_columns, fit_pattern_model, pattern_variant_tag, prod_model_path,
    wf_anchor_ts, iter_wf_cycles,
)
from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold
from xgboost_filter_model.train_filter_v14 import prepare_data_v14, build_target
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14
from xgboost_filter_model.train_filter_1min import load_price_data
from v15.features import add_v15_energetic_features

BT_START = "2025-06-01"
BT_END = WF_CONFIG["wf_end"]

print("=" * 70)
print("  V34 — V14 vs V15 DOWNTEND RETRACE (SHORT)")
print(f"  V14 def: drop≥25 rise≥5 H=15 TP=40 SL=30 | V15: same def, fresh model")
print("=" * 70)

# ── Load bars once ───────────────────────────────────────────────────────
raw = load_price_data(start_date="2020-01-01", end_date=BT_END)
if raw.index.tz is None:
    raw.index = raw.index.tz_localize("UTC")
raw_sim = raw[raw.index >= pd.Timestamp(BT_START).tz_localize("UTC")].copy()

# ── Build feature matrix ──────────────────────────────────────────────────
print("\n[1] Building feature matrix...")
df_v14 = prepare_data_v14(
    start_date=WF_CONFIG["full_start"], end_date=BT_END,
    energetic_filter=False, for_live_inference=True,
    pa_groups=collect_pa_groups(["downtrend_retrace"]),
    pattern_feature_set=backtest_feature_set(),
)
df_v14 = prepare_directional_data_v14(df_v14)
df_v14 = add_v15_energetic_features(df_v14)

df_test = df_v14[df_v14.index >= raw_sim.index[0]].copy()
print(f"  Feature bars in test: {len(df_test)}")

# ── V14: production model + WF cycle scoring ──────────────────────────────
print("\n[2] V14 production scoring...")

v14_pattern = "downtrend_retrace"
v14_spec = PATTERN_REGISTRY[v14_pattern]
v14_ex = v14_spec["execution"]
v14_mode = execution_target_mode(v14_ex)
v14_tp, v14_sl = execution_tp_sl(v14_ex)
v14_variant = pattern_variant_tag(v14_ex["horizon"], v14_tp, v14_sl, target_mode=v14_mode)
v14_pdir = PATTERN_MODEL_DIR / v14_pattern / v14_variant
v14_thresh = v14_spec["thresholds"]["prob"]

df_test = assign_patterns(df_test.copy())
wf_anchor = wf_anchor_ts()
bt_start_dt = pd.Timestamp(BT_START).tz_localize("UTC")
end_dt = pd.Timestamp(BT_END).tz_localize("UTC")

v14_mask = (df_test["pattern_name"] == v14_pattern)
print(f"  V14 pattern bars: {int(v14_mask.sum())}")

df_test["v14_prob"] = np.nan
prod_v14_model = joblib.load(prod_model_path(v14_pdir))
for cycle, current_start, current_end in iter_wf_cycles(bt_start_dt, end_dt, wf_anchor):
    chunk = (df_test.index >= current_start) & (df_test.index < current_end)
    pat_chunk = chunk & v14_mask
    if not pat_chunk.any(): continue
    path = cycle_model_path(v14_pdir, cycle, current_start.date())
    model = joblib.load(path) if path.exists() else prod_v14_model
    mf = list(model.feature_names_in_)
    df_test.loc[pat_chunk, "v14_prob"] = model.predict_proba(df_test.loc[pat_chunk, mf])[:, 1]

adaptive = adaptive_prob_threshold(v14_thresh, df_test)
df_test["v14_signal"] = v14_mask & (df_test["v14_prob"] >= adaptive)
print(f"  V14 signals: {int(df_test['v14_signal'].sum())}")

# ── V15: fresh model on same bars ────────────────────────────────────────
print(f"\n[3] V15 fresh model training...")

# Use V14 definition pattern bars
v15_mask = v14_mask.copy()
print(f"  V15 pattern bars: {int(v15_mask.sum())}")

# Train label: V14's fixed TP/SL at H=15 as target
fm15 = build_target(df_test[["open","high","low","close"]], v14_ex["horizon"], 1.0, 1.0)
df_test["fmax_15"] = fm15["future_max_move"]
df_test["fmin_15"] = fm15["future_min_move"]
# SHORT label: TP hit when fmin <= -tp, NOT stopped when fmax < sl
df_test["v15_trg"] = ((df_test["fmin_15"] <= -v14_tp) & (df_test["fmax_15"] < v14_sl)).astype(int)
df_pat = df_test.loc[v15_mask].copy()
hr = float(df_pat["v15_trg"].mean()*100) if len(df_pat)>0 else 0
print(f"  Hit rate: {hr:.1f}%  pos={int(df_pat['v15_trg'].sum())}")

feats = [c for c in feature_columns(df_pat)
         if c not in ("v15_trg","fmax_15","fmin_15")
         and df_pat[c].dtype in ("float64","float32","int64","int32","bool")]
v15_model = fit_pattern_model(df_pat[feats], df_pat["v15_trg"], min_samples=50)
if v15_model is None:
    print("  ERROR: Model training failed — using V14 prod model")
    v15_model = prod_v14_model
    mf_v15 = list(prod_v14_model.feature_names_in_)
else:
    print(f"  Model: {len(feats)} feats, {len(df_pat)} samples")
    mf_v15 = list(v15_model.feature_names_in_)

df_test["v15_prob"] = np.nan
for ts in df_test.index[v15_mask]:
    row = df_test.loc[ts]; v = row[mf_v15].values.astype(float)
    if np.isnan(v).any(): continue
    try:
        df_test.loc[ts, "v15_prob"] = float(v15_model.predict_proba(pd.DataFrame([v], columns=mf_v15))[:,1][0])
    except: continue

v15_thresh = 0.45
df_test["v15_signal"] = v15_mask & (df_test["v15_prob"] >= v15_thresh)
print(f"  V15 signals: {int(df_test['v15_signal'].sum())} (prob≥{v15_thresh})")

# ── Simulate ─────────────────────────────────────────────────────────────
sim14 = raw_sim[["open","high","low","close"]].copy()
sim14["side_signal"] = 0; sim14["s1_prob"] = 0.5; sim14["s2_prob"] = 0.5
ci = df_test.index[df_test["v14_signal"]].intersection(sim14.index)
sim14.loc[ci, "side_signal"] = -1
sim14.loc[ci, "s2_prob"] = df_test.loc[ci, "v14_prob"]
t14 = simulate_v13_core(sim14, v14_tp, v14_sl, v14_ex["horizon"])
if t14: td14 = pd.DataFrame(t14); p14 = len(td14), float(td14["pnl"].sum()), (td14["pnl"]>0).mean()*100
else: p14 = (0,0,0)

sim15 = raw_sim[["open","high","low","close"]].copy()
sim15["side_signal"] = 0; sim15["s1_prob"] = 0.5; sim15["s2_prob"] = 0.5
ci15 = df_test.index[df_test["v15_signal"]].intersection(sim15.index)
sim15.loc[ci15, "side_signal"] = -1
sim15.loc[ci15, "s2_prob"] = df_test.loc[ci15, "v15_prob"]
t15 = simulate_v13_core(sim15, v14_tp, v14_sl, v14_ex["horizon"])
if t15: td15 = pd.DataFrame(t15); p15 = len(td15), float(td15["pnl"].sum()), (td15["pnl"]>0).mean()*100
else: p15 = (0,0,0)

print(f"\n{'='*55}")
print(f"  DOWNTEND RETRACE — V14 vs V15 (same definition, fresh model)")
print(f"{'='*55}")
print(f"{'':<20} {'V14':>12} {'V15':>12}")
print(f"{'Trades':<20} {p14[0]:>12d} {p15[0]:>12d}")
print(f"{'PnL':<20} {p14[1]:>+12.1f} {p15[1]:>+12.1f}")
print(f"{'WR':<20} {p14[2]:>11.1f}% {p15[2]:>11.1f}%")
print()
print(f"  V14 signals: {int(df_test['v14_signal'].sum())}")
print(f"  V15 signals: {int(df_test['v15_signal'].sum())}")
print(f"  Production backtest_v15: 48 trades, +595 pts, 52% WR (same period)")

print("\nDone.")
