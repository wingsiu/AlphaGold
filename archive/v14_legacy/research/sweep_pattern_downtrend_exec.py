#!/usr/bin/env python3
"""
Sweep TP / SL / horizon for downtrend_retrace (execution-only).

Reads fall/bounce from config pattern rules. Uses variant model per SL retrain
or single model path via env PATTERN_MODEL_VARIANT.

Usage:
  python3 sweep_pattern_downtrend_exec.py 2025-06-01 2026-05-23
"""
from __future__ import annotations

import os
import sys
from datetime import date
from itertools import product
from pathlib import Path

from v14._paths import PROJECT_ROOT

import joblib
import numpy as np
import pandas as pd

from v14.backtest.backtest_core import simulate_v13_core
from config.v14_config import EXECUTION_CONFIG, WF_CONFIG
from config.v14_patterns import PATTERN_MODEL_DIR, PATTERN_REGISTRY
from xgboost_filter_model.pattern_training import (
    cycle_model_path,
    feature_columns,
    iter_wf_cycles,
    pattern_variant_tag,
    prod_model_path,
    wf_anchor_ts,
)
from xgboost_filter_model.train_filter_1min import load_price_data
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14

PATTERN = "downtrend_retrace"
HORIZONS = [15, 30, 45, 60]
TARGETS = [15, 20, 30, 40, 45]
STOPS = [10, 15, 20, 25, 30]

args = sys.argv[1:]
bt_start, bt_end = "2025-06-01", date.today().strftime("%Y-%m-%d")
if len(args) >= 1:
    bt_start = args[0]
if len(args) >= 2:
    bt_end = args[1]

MODEL_VARIANT = os.environ.get("PATTERN_MODEL_VARIANT", pattern_variant_tag(60, 30, 15))
PROB_THRESH = PATTERN_REGISTRY[PATTERN]["thresholds"]["prob"]


def _rule_val(rules, feat, default):
    for r in rules:
        if r["feat"] == feat:
            return float(r["val"])
    return default


def pattern_mask_df(df: pd.DataFrame) -> pd.Series:
    spec = PATTERN_REGISTRY[PATTERN]
    fall = _rule_val(spec["pattern"], "drop_from_high_240", 35)
    bounce = _rule_val(spec["pattern"], "rise_from_low_240", 10)
    m = (
        (df["drop_from_high_240"] >= fall)
        & (df["rise_from_low_240"] >= bounce)
        & (df["near_low_zone"] != 1.0)
    )
    return m


def build_scored(bt_start, bt_end, label_sl=15.0):
    warmup = int(WF_CONFIG.get("feature_warmup_days", 120))
    load_start = max(
        pd.to_datetime(WF_CONFIG["full_start"]),
        pd.to_datetime(bt_start) - pd.Timedelta(days=warmup),
    ).strftime("%Y-%m-%d")
    load_end = (pd.to_datetime(bt_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bt_start_dt = pd.to_datetime(bt_start).tz_localize("UTC")

    df = prepare_data_v14(load_start, load_end, energetic_filter=False,
        label_horizon=60, label_tp=30, label_sl=label_sl, fixed_label_tp_sl=True)
    df = prepare_directional_data_v14(df)
    feats = feature_columns(df)
    full = df[df.index >= bt_start_dt].copy()
    pat = pattern_mask_df(full)

    pdir = PATTERN_MODEL_DIR / PATTERN / MODEL_VARIANT
    prod = joblib.load(prod_model_path(pdir))
    wf = wf_anchor_ts()
    end_dt = pd.to_datetime(bt_end).tz_localize("UTC") + pd.Timedelta(days=1)

    prob = pd.Series(np.nan, index=full.index)
    side = pd.Series(0, index=full.index)
    s2 = pd.Series(0.0, index=full.index)
    for cycle, cur, ce in iter_wf_cycles(bt_start_dt, end_dt, wf):
        chunk = (full.index >= cur) & (full.index < ce) & pat
        if chunk.any():
            cp = cycle_model_path(pdir, cycle, cur.date())
            model = joblib.load(cp) if cp.exists() else prod
            p = model.predict_proba(full.loc[chunk, feats])[:, 1]
            prob.loc[chunk] = p
            sig = chunk & (p >= PROB_THRESH)
            side.loc[sig] = -1
            s2.loc[sig] = 1.0 - p

    raw = load_price_data(bt_start, load_end)
    raw = raw[raw.index >= bt_start_dt]
    sim = raw[["open", "high", "low", "close"]].copy()
    sim["side_signal"] = side.reindex(sim.index).fillna(0).astype(int)
    sim["s2_prob"] = s2.reindex(sim.index).fillna(0.0)
    sim["s1_prob"] = prob.reindex(sim.index).fillna(0.0)
    return sim


print(f"\nDowntrend EXEC sweep  {bt_start} → {bt_end}  model={MODEL_VARIANT}\n")
sim_df = build_scored(bt_start, bt_end)
fall = _rule_val(PATTERN_REGISTRY[PATTERN]["pattern"], "drop_from_high_240", 35)
bounce = _rule_val(PATTERN_REGISTRY[PATTERN]["pattern"], "rise_from_low_240", 10)
print(f"Definition: fall>={fall} bounce>={bounce}")
print(f"Signals: {(sim_df['side_signal']!=0).sum()}\n")

rows = []
combos = [(h, tp, sl) for h, tp, sl in product(HORIZONS, TARGETS, STOPS) if sl < tp]
print(f"{'H':>4} {'TP':>4} {'SL':>4} | {'Trades':>6} {'WR%':>6} {'NetPnL':>9} {'Avg':>7}")
print("-" * 52)
cfg = EXECUTION_CONFIG.copy()
for h, tp, sl in combos:
    trades = simulate_v13_core(sim_df, tp, sl, h, config=cfg)
    if not trades:
        rows.append({"horizon": h, "tp": tp, "sl": sl, "trades": 0, "win_rate": 0,
                     "net_pnl": 0, "avg_pnl": 0})
        continue
    tdf = pd.DataFrame(trades)
    net = float(tdf["pnl"].sum())
    n = len(tdf)
    wr = (tdf["pnl"] > 0).mean() * 100
    rows.append({"horizon": h, "tp": tp, "sl": sl, "trades": n,
                 "win_rate": round(wr, 1), "net_pnl": round(net, 2), "avg_pnl": round(net / n, 2)})
    print(f"{h:4d} {tp:4.0f} {sl:4.0f} | {n:6d} {wr:5.1f}% {net:+9.2f} {net/n:+7.2f}")

res = pd.DataFrame(rows).sort_values(["net_pnl", "win_rate"], ascending=False)
out = PROJECT_ROOT / "runtime" / "sweep_downtrend_retrace_exec.csv"
res.to_csv(out, index=False)
print(f"\nSaved -> {out}")
top = res[res["trades"] >= 15].head(10)
print("\nTop (≥15 trades):")
print(top.to_string(index=False))
best = res[res["trades"] >= 15].iloc[0] if (res["trades"] >= 15).any() else res.iloc[0]
print(f"\nBest: H={int(best['horizon'])} TP={int(best['tp'])} SL={int(best['sl'])} "
      f"→ {int(best['trades'])} trades WR={best['win_rate']}% PnL={best['net_pnl']:+.1f}")
