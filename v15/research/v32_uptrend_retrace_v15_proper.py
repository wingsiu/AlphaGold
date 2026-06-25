#!/usr/bin/env python3
"""
v32 — Proper v14 vs v15 Uptrend Retrace Comparison
====================================================
Loads the ACTUAL production v14 model via WF cycle scoring (same as backtest_patterns_v14.py),
trains a v15 model with daily ATR-scaled definition + TP/SL, then runs both through
the same trade simulation core.

This fixes the bug in v15_compare_fixed.py which:
  1. Set side_signal=1 for ALL pattern bars (no model filter) → v14 PnL was wrong
  2. Used a different data preparation path than production

Production baseline (from actual hybrid backtest):
  uptrend_retrace: 221 trades, +1,739 pts, 53% WR (2025-06-01 → 2026-06-12)
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

# V15 config from sweep
RISE_X, DROP_X = 0.30, 0.10
TP_X, SL_X = 0.15, 0.112
HORIZON_V15 = 120
PROB_THRESH_V15 = 0.45

BT_START = "2025-06-01"
BT_END = WF_CONFIG["wf_end"]

print("=" * 70)
print("  V32 — PROPER V14 vs V15 UPTREND RETRACE COMPARISON")
print(f"  V15: dailyATR rise≥{RISE_X}x drop≥{DROP_X}x TP={TP_X}x SL={SL_X}x H={HORIZON_V15}")
print("=" * 70)

# ── Load bars once ───────────────────────────────────────────────────────
raw = load_price_data(start_date="2020-01-01", end_date=BT_END)
if raw.index.tz is None:
    raw.index = raw.index.tz_localize("UTC")
raw_sim = raw[raw.index >= pd.Timestamp(BT_START).tz_localize("UTC")].copy()

# ── Build feature matrix (v14 production path) ────────────────────────────
print("\n[1] Building feature matrix (production path)...")
df_v14 = prepare_data_v14(
    start_date=WF_CONFIG["full_start"], end_date=BT_END,
    energetic_filter=False, for_live_inference=True,
    pa_groups=collect_pa_groups(["uptrend_retrace"]),
    pattern_feature_set=backtest_feature_set(),
)
df_v14 = prepare_directional_data_v14(df_v14)
df_v14 = add_v15_energetic_features(df_v14)

# Daily ATR for v15
daily_atr = pta.atr(
    df_v14["high"].resample("D").max(),
    df_v14["low"].resample("D").min(),
    df_v14["close"].resample("D").last(), length=14
)
df_v14["daily_atr"] = df_v14.index.floor("D").map(daily_atr)

# Future moves
fm120 = build_target(df_v14[["open","high","low","close"]], HORIZON_V15, 1.0, 1.0)
df_v14["fmax_120"] = fm120["future_max_move"]
df_v14["fmin_120"] = fm120["future_min_move"]

df_test = df_v14[df_v14.index >= raw_sim.index[0]].copy()
print(f"  Feature bars in test: {len(df_test)}")

# ── V14: production model + WF cycle scoring ──────────────────────────────
print("\n[2] V14 production scoring (WF cycles, actual models)...")

v14_pattern = "uptrend_retrace"
v14_spec = PATTERN_REGISTRY[v14_pattern]
v14_ex = v14_spec["execution"]
v14_mode = execution_target_mode(v14_ex)
v14_tp, v14_sl = execution_tp_sl(v14_ex)
v14_variant = pattern_variant_tag(v14_ex["horizon"], v14_tp, v14_sl, target_mode=v14_mode)
v14_pdir = PATTERN_MODEL_DIR / v14_pattern / v14_variant
v14_thresh = v14_spec["thresholds"]["prob"]

# Route patterns
df_test = assign_patterns(df_test.copy())

# Score bars through WF cycles (same as backtest_patterns_v14.py)
wf_anchor = wf_anchor_ts()
bt_start_dt = pd.Timestamp(BT_START).tz_localize("UTC")
end_dt = pd.Timestamp(BT_END).tz_localize("UTC")

v14_mask = (df_test["pattern_name"] == v14_pattern)
print(f"  Pattern bars: {int(v14_mask.sum())}")

df_test["v14_prob"] = np.nan

prod_v14_model = joblib.load(prod_model_path(v14_pdir))
for cycle, current_start, current_end in iter_wf_cycles(bt_start_dt, end_dt, wf_anchor):
    chunk = (df_test.index >= current_start) & (df_test.index < current_end)
    pat_chunk = chunk & v14_mask
    if not pat_chunk.any():
        continue
    path = cycle_model_path(v14_pdir, cycle, current_start.date())
    model = joblib.load(path) if path.exists() else prod_v14_model
    mf = list(model.feature_names_in_)
    rows = df_test.loc[pat_chunk]
    p = model.predict_proba(rows[mf])[:, 1]
    df_test.loc[pat_chunk, "v14_prob"] = p

adaptive = adaptive_prob_threshold(v14_thresh, df_test)
df_test["v14_signal"] = v14_mask & (df_test["v14_prob"] >= adaptive)
print(f"  V14 signals: {int(df_test['v14_signal'].sum())} "
      f"(prob≥{v14_thresh}, adaptive)")

# ── Simulate V14 ──────────────────────────────────────────────────────────
sim14 = raw_sim[["open","high","low","close"]].copy()
sim14["side_signal"] = 0
sim14["s1_prob"] = 0.5
sim14["s2_prob"] = 0.5
ci = df_test.index[df_test["v14_signal"]].intersection(sim14.index)
sim14.loc[ci, "side_signal"] = 1
sim14.loc[ci, "s2_prob"] = df_test.loc[ci, "v14_prob"]

t14 = simulate_v13_core(sim14, v14_tp, v14_sl, v14_ex["horizon"])
if t14:
    td14 = pd.DataFrame(t14)
    p14 = {"trades": len(td14), "pnl": float(td14["pnl"].sum()),
           "wr": (td14["pnl"]>0).mean()*100, "avg": float(td14["pnl"].mean()),
           "dd": float((td14["pnl"].cumsum()-td14["pnl"].cumsum().cummax()).min())}
else:
    p14 = {"trades":0,"pnl":0,"wr":0,"avg":0,"dd":0}
print(f"  V14 Sim: {p14['trades']} trades, {p14['pnl']:+.1f} pts, WR={p14['wr']:.1f}%")

# ── V15: daily ATR definition + model ─────────────────────────────────────
print(f"\n[3] V15 daily ATR scoring...")

v15_mask = (df_test["rise_from_low_240"] >= df_test["daily_atr"] * RISE_X) & \
           (df_test["drop_from_high_240"] >= df_test["daily_atr"] * DROP_X)
if "near_high_zone" in df_test.columns:
    v15_mask &= df_test["near_high_zone"] != 1.0
print(f"  V15 pattern bars: {int(v15_mask.sum())}")

# Train V15 model on pattern bars
tp_abs = df_test["daily_atr"] * TP_X
sl_abs = df_test["daily_atr"] * SL_X
df_test["v15_trg"] = ((df_test["fmax_120"] >= tp_abs) & (df_test["fmin_120"] <= sl_abs)).astype(int)
df_pat = df_test.loc[v15_mask].copy()
hr = float(df_pat["v15_trg"].mean()*100) if len(df_pat)>0 else 0

feats = [c for c in feature_columns(df_pat)
         if c not in ("v15_trg","daily_atr","fmax_120","fmin_120")
         and df_pat[c].dtype in ("float64","float32","int64","int32","bool")]
v15_model = fit_pattern_model(df_pat[feats], df_pat["v15_trg"], min_samples=50)
print(f"  Model: {len(feats)} feats, {len(df_pat)} samples, HR={hr:.1f}%")

# Score with v15 model
mf_v15 = list(v15_model.feature_names_in_)
df_test["v15_prob"] = np.nan
for cycle, current_start, current_end in iter_wf_cycles(bt_start_dt, end_dt, wf_anchor):
    chunk = (df_test.index >= current_start) & (df_test.index < current_end)
    pat_chunk = chunk & v15_mask
    if not pat_chunk.any():
        continue
    rows = df_test.loc[pat_chunk]
    try:
        p = v15_model.predict_proba(rows[mf_v15])[:, 1]
        df_test.loc[pat_chunk, "v15_prob"] = p
    except Exception:
        continue

df_test["v15_signal"] = v15_mask & (df_test["v15_prob"] >= PROB_THRESH_V15)
print(f"  V15 signals: {int(df_test['v15_signal'].sum())} (prob≥{PROB_THRESH_V15})")

# ── Simulate V15 ──────────────────────────────────────────────────────────
sim15 = raw_sim[["open","high","low","close"]].copy()
sim15["side_signal"] = 0
sim15["s1_prob"] = 0.5
sim15["s2_prob"] = 0.5
ci15 = df_test.index[df_test["v15_signal"]].intersection(sim15.index)
sim15.loc[ci15, "side_signal"] = 1
sim15.loc[ci15, "s2_prob"] = df_test.loc[ci15, "v15_prob"]

avg_tp = float(tp_abs[v15_mask].mean()) if v15_mask.any() else 30.0
avg_sl = float(sl_abs[v15_mask].mean()) if v15_mask.any() else 25.0

t15 = simulate_v13_core(sim15, avg_tp, avg_sl, HORIZON_V15)
if t15:
    td15 = pd.DataFrame(t15)
    p15 = {"trades": len(td15), "pnl": float(td15["pnl"].sum()),
           "wr": (td15["pnl"]>0).mean()*100, "avg": float(td15["pnl"].mean()),
           "dd": float((td15["pnl"].cumsum()-td15["pnl"].cumsum().cummax()).min())}
else:
    p15 = {"trades":0,"pnl":0,"wr":0,"avg":0,"dd":0}
print(f"  V15 Sim: {p15['trades']} trades, {p15['pnl']:+.1f} pts, WR={p15['wr']:.1f}%, "
      f"TP={avg_tp:.1f} SL={avg_sl:.1f} H={HORIZON_V15}")

# ── Comparison ────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  COMPARISON: V14 (production) vs V15 (daily ATR)")
print(f"  Period: {BT_START} → {BT_END}")
print(f"{'='*55}")
print(f"{'':<22} {'v14 (production)':>15} {'v15 (daily ATR)':>15}")
print(f"{'':-<22} {'':->15} {'':->15}")
v14_h_label = f'{v14_ex["horizon"]}min'
v15_h_label = f'{HORIZON_V15}min'
print(f"{'Horizon':<22} {v14_h_label:>15} {v15_h_label:>15}")
print(f"{'TP / SL':<22} {f'{v14_tp}/{v14_sl}':>15} {f'{avg_tp:.1f}/{avg_sl:.1f}':>15}")
print(f"{'Pattern Bars':<22} {int(v14_mask.sum()):>15d} {int(v15_mask.sum()):>15d}")
print(f"{'Signals':<22} {int(df_test['v14_signal'].sum()):>15d} {int(df_test['v15_signal'].sum()):>15d}")
print(f"{'Trades':<22} {p14['trades']:>15d} {p15['trades']:>15d}")
print(f"{'PnL':<22} {p14['pnl']:>+15.1f} {p15['pnl']:>+15.1f}")
print(f"{'WR':<22} {p14['wr']:>14.1f}% {p15['wr']:>14.1f}%")
print(f"{'Avg/Trade':<22} {p14['avg']:>+15.2f} {p15['avg']:>+15.2f}")
print(f"{'Max DD':<22} {p14['dd']:>+15.1f} {p15['dd']:>+15.1f}")

print("\nDone.")
