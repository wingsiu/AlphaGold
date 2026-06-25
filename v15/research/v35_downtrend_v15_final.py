#!/usr/bin/env python3
"""
v35 — V15 Downtrend Retrace (New Definition + Fresh Model)
=============================================================
New definition: drop_from_high_240 > 10 & rise_from_low_240 > 0.6*drop & wr_90 > -70
Uses V14 fixed TP=$40 SL=$30 at H=15, trains fresh XGBoost with v15 deterministic features
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
print("  V35 — V15 DOWNTEND RETRACE (NEW DEF: drop>10 & bounce>0.6*drop & wr_90>-70)")
print("  TP=$40 SL=$30 H=15 (V14 execution)")
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

# V14 execution params
v14_pattern = "downtrend_retrace"
v14_spec = PATTERN_REGISTRY[v14_pattern]
v14_ex = v14_spec["execution"]
v14_tp, v14_sl = execution_tp_sl(v14_ex)
v14_h = v14_ex["horizon"]
v14_thresh = v14_spec["thresholds"]["prob"]

wf_anchor = wf_anchor_ts()
bt_start_dt = pd.Timestamp(BT_START).tz_localize("UTC")
end_dt = pd.Timestamp(BT_END).tz_localize("UTC")

# ── V14 Mask ──────────────────────────────────────────────────────────────
v14_mask = (df_test["drop_from_high_240"] >= 25) & (df_test["rise_from_low_240"] >= 5)
if "near_low_zone" in df_test.columns:
    v14_mask &= df_test["near_low_zone"] != 1.0
print(f"\n[2] V14 pattern bars: {int(v14_mask.sum())}")

# ── V15 New Mask ──────────────────────────────────────────────────────────
v15_mask = (
    (df_test["drop_from_high_240"] > 10) &
    (df_test["rise_from_low_240"] > 0.6 * df_test["drop_from_high_240"]) &
    (df_test["wr_90"] > -70)
)
if "near_low_zone" in df_test.columns:
    v15_mask &= df_test["near_low_zone"] != 1.0
print(f"  V15 pattern bars (new def): {int(v15_mask.sum())}")
print(f"    drop mean: {df_test.loc[v15_mask,'drop_from_high_240'].mean():.1f}")
print(f"    rise mean: {df_test.loc[v15_mask,'rise_from_low_240'].mean():.1f}")
print(f"    wr_90 mean: {df_test.loc[v15_mask,'wr_90'].mean():.1f}")

# ── Target labels (V14 fixed TP/SL at H=15) ──────────────────────────────
# Compute future moves from raw sim bars (not feature matrix)
raw_ohlc = raw_sim[["open", "high", "low", "close"]].copy()
fm15 = build_target(raw_ohlc, v14_h, 1.0, 1.0)
raw_ohlc["fmax_15"] = fm15["future_max_move"]
raw_ohlc["fmin_15"] = fm15["future_min_move"]

# Align to df_test index for mask selection
df_test["fmax_15"] = raw_ohlc["fmax_15"]
df_test["fmin_15"] = raw_ohlc["fmin_15"]

# SHORT label: TP hit when fmin <= -tp, NOT stopped when fmax < sl
df_test["v15_trg"] = ((df_test["fmin_15"] <= -v14_tp) & (df_test["fmax_15"] < v14_sl)).astype(int)
df_pat = df_test.loc[v15_mask].copy()
n_pos = int(df_pat["v15_trg"].sum())
hr = float(df_pat["v15_trg"].mean() * 100) if len(df_pat) > 0 else 0

# Also check fmin/fmax distributions
print(f"\n[3] V15 training labels:")
print(f"  Samples: {len(df_pat)}  TP hit: {n_pos}  HR: {hr:.1f}%")
print(f"  fmin_15 (on pattern bars): mean={df_pat['fmin_15'].mean():.1f} median={df_pat['fmin_15'].median():.1f} mRN={df_pat['fmin_15'].min():.1f} max={df_pat['fmax_15'].max():.1f}")
print(f"  fmin_15 <= -{v14_tp}: {(df_pat['fmin_15']<=-v14_tp).sum()} bars")
print(f"  fmax_15 < {v14_sl}: {(df_pat['fmax_15']<v14_sl).sum()} bars")

# ── Train V15 model ──────────────────────────────────────────────────────
feats = [c for c in feature_columns(df_pat)
         if c not in ("v15_trg", "fmax_15", "fmin_15")
         and df_pat[c].dtype in ("float64", "float32", "int64", "int32", "bool")]
v15_model = fit_pattern_model(df_pat[feats], df_pat["v15_trg"], min_samples=50)
if v15_model is None:
    print("  ERROR: Model training failed")
    sys.exit(1)
print(f"  Model: {len(feats)} feats, {len(df_pat)} samples")

# ── Score V15 bars ───────────────────────────────────────────────────────
mf_v15 = list(v15_model.feature_names_in_)
df_test["v15_prob"] = np.nan
for ts in df_test.index[v15_mask]:
    row = df_test.loc[ts]
    v = row[mf_v15].values.astype(float)
    if np.isnan(v).any():
        continue
    try:
        df_test.loc[ts, "v15_prob"] = float(
            v15_model.predict_proba(pd.DataFrame([v], columns=mf_v15))[:, 1][0]
        )
    except Exception:
        continue

v15_thresh = 0.45
df_test["v15_signal"] = v15_mask & (df_test["v15_prob"] >= v15_thresh)
print(f"  V15 signals (prob≥{v15_thresh}): {int(df_test['v15_signal'].sum())}")

# ── Simulate V15 ─────────────────────────────────────────────────────────
sim15 = raw_sim[["open","high","low","close"]].copy()
sim15["side_signal"] = 0
sim15["s1_prob"] = 0.5
sim15["s2_prob"] = 0.5
ci15 = df_test.index[df_test["v15_signal"]].intersection(sim15.index)
sim15.loc[ci15, "side_signal"] = -1  # SHORT
sim15.loc[ci15, "s2_prob"] = df_test.loc[ci15, "v15_prob"]

t15 = simulate_v13_core(sim15, v14_tp, v14_sl, v14_h)
if t15:
    td15 = pd.DataFrame(t15)
    p15_trades = len(td15)
    p15_pnl = float(td15["pnl"].sum())
    p15_wr = (td15["pnl"] > 0).mean() * 100
    p15_dd = float((td15["pnl"].cumsum() - td15["pnl"].cumsum().cummax()).min())
    print(f"  V15 Sim: {p15_trades} trades, PnL={p15_pnl:+.1f}, WR={p15_wr:.1f}%, MaxDD={p15_dd:+.1f}")
else:
    p15_trades, p15_pnl, p15_wr, p15_dd = 0, 0.0, 0.0, 0.0
    print("  V15 Sim: no trades")

# ── V14 baseline (production backtest reference) ─────────────────────────
print(f"\n{'='*55}")
print(f"  DOWNTEND RETRACE — V14 vs V15")
print(f"{'='*55}")
print(f"  V14 production (from backtest_v15.py): 48 trades, +595 pts, 52% WR")
print(f"  V15 new def (drop>10 bounce>0.6*drop wr_90>-70):")
print(f"      Bars: {int(v15_mask.sum())} (+{int(v15_mask.sum())-int(v14_mask.sum()):+d} vs V14)")
print(f"      Signals: {int(df_test['v15_signal'].sum())}")
print(f"      Trades: {p15_trades}, PnL={p15_pnl:+.1f}, WR={p15_wr:.1f}%, MaxDD={p15_dd:+.1f}")
print(f"      Config: H={v14_h} TP={v14_tp} SL={v14_sl} prob≥{v15_thresh}")

print("\nDone.")
