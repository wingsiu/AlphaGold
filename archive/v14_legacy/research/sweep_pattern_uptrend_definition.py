#!/usr/bin/env python3
"""
Sweep uptrend_retrace pattern definition: rise_from_low_240 / drop_from_high_240.

Uses retrained model h60_tp30_sl15; execution H=60 TP=30 SL=15.
Does NOT retrain per combo (fast definition sweep).

Usage:
  python3 sweep_pattern_uptrend_definition.py
  python3 sweep_pattern_uptrend_definition.py 2025-06-01 2026-05-23
"""
from __future__ import annotations

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
    prod_model_path,
    pattern_variant_tag,
    wf_anchor_ts,
)
from xgboost_filter_model.train_filter_1min import load_price_data
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14

PATTERN = "uptrend_retrace"
MODEL_VARIANT = pattern_variant_tag(60, 30, 15)
EXEC_H, EXEC_TP, EXEC_SL = 60, 30.0, 15.0
PROB_THRESH = PATTERN_REGISTRY[PATTERN]["thresholds"]["prob"]

# Grid around current defaults (30 / 10)
RISE_VALUES = [15, 20, 25, 30, 35, 40, 45, 50]
DROP_VALUES = [5, 8, 10, 12, 15, 20, 25, 30]

args = sys.argv[1:]
bt_start = "2025-06-01"
bt_end = date.today().strftime("%Y-%m-%d")
if len(args) >= 1:
    bt_start = args[0]
if len(args) >= 2:
    bt_end = args[1]


def score_probs(full: pd.DataFrame, feats: list[str]) -> pd.Series:
    """Walk-forward model scores on all rows in full (NaN where not scored)."""
    pdir = PATTERN_MODEL_DIR / PATTERN / MODEL_VARIANT
    prod = joblib.load(prod_model_path(pdir))

    bt_start_dt = full.index.min()
    wf = wf_anchor_ts()
    end_dt = full.index.max() + pd.Timedelta(minutes=1)

    prob = pd.Series(np.nan, index=full.index)
    # Score bars that pass loosest grid filter only (speed)
    loose = (
        (full["rise_from_low_240"] >= min(RISE_VALUES))
        & (full["drop_from_high_240"] >= min(DROP_VALUES))
        & (full.get("near_high_zone", 0) != 1.0)
    )

    for cycle, cur, ce in iter_wf_cycles(bt_start_dt, end_dt, wf):
        chunk = (full.index >= cur) & (full.index < ce) & loose
        if chunk.any():
            cp = cycle_model_path(pdir, cycle, cur.date())
            model = joblib.load(cp) if cp.exists() else prod
            prob.loc[chunk] = model.predict_proba(full.loc[chunk, feats])[:, 1]
    return prob


def simulate_combo(
    raw: pd.DataFrame,
    full: pd.DataFrame,
    prob: pd.Series,
    rise: float,
    drop: float,
) -> dict:
    pat = (
        (full["rise_from_low_240"] >= rise)
        & (full["drop_from_high_240"] >= drop)
        & (full["near_high_zone"] != 1.0)
        & (prob >= PROB_THRESH)
    )
    sim = raw[["open", "high", "low", "close"]].copy()
    side = pd.Series(0, index=sim.index, dtype=int)
    s2 = pd.Series(0.0, index=sim.index)
    fired = pat.reindex(sim.index).fillna(False)
    side.loc[fired] = 1
    s2.loc[fired] = prob.reindex(sim.index).loc[fired].astype(float)
    sim["side_signal"] = side
    sim["s2_prob"] = s2
    sim["s1_prob"] = prob.reindex(sim.index).fillna(0.0)

    n_pat = int(
        (
            (full["rise_from_low_240"] >= rise)
            & (full["drop_from_high_240"] >= drop)
            & (full["near_high_zone"] != 1.0)
        ).sum()
    )
    n_sig = int(fired.sum())
    trades = simulate_v13_core(sim, EXEC_TP, EXEC_SL, EXEC_H, config=EXECUTION_CONFIG.copy())
    if not trades:
        return {
            "rise": rise,
            "drop": drop,
            "pattern_bars": n_pat,
            "signals": n_sig,
            "trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "avg_pnl": 0.0,
        }
    tdf = pd.DataFrame(trades)
    net = float(tdf["pnl"].sum())
    n = len(tdf)
    return {
        "rise": rise,
        "drop": drop,
        "pattern_bars": n_pat,
        "signals": n_sig,
        "trades": n,
        "win_rate": round((tdf["pnl"] > 0).mean() * 100, 1),
        "net_pnl": round(net, 2),
        "avg_pnl": round(net / n, 2),
    }


print(f"\n{'='*60}")
print("  Uptrend retrace DEFINITION sweep")
print(f"  Rise grid : {RISE_VALUES}")
print(f"  Drop grid : {DROP_VALUES}")
print(f"  Model     : {MODEL_VARIANT}  |  Exec H={EXEC_H} TP={EXEC_TP} SL={EXEC_SL}")
print(f"  Period    : {bt_start} → {bt_end}")
print(f"{'='*60}\n")

warmup = int(WF_CONFIG.get("feature_warmup_days", 120))
load_start = max(
    pd.to_datetime(WF_CONFIG["full_start"]),
    pd.to_datetime(bt_start) - pd.Timedelta(days=warmup),
).strftime("%Y-%m-%d")
load_end = (pd.to_datetime(bt_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
bt_start_dt = pd.to_datetime(bt_start).tz_localize("UTC")

print(f"Loading data {load_start} → {bt_end}…")
df = prepare_data_v14(
    load_start,
    load_end,
    energetic_filter=False,
    label_horizon=EXEC_H,
    label_tp=EXEC_TP,
    label_sl=EXEC_SL,
    fixed_label_tp_sl=True,
)
df = prepare_directional_data_v14(df)
feats = feature_columns(df)
full = df[df.index >= bt_start_dt].copy()
raw = load_price_data(bt_start, load_end)
raw = raw[raw.index >= bt_start_dt]

print("Scoring model (walk-forward)…")
prob = score_probs(full, feats)

combos = list(product(RISE_VALUES, DROP_VALUES))
rows = []
print(f"\n{'Rise':>5} {'Drop':>5} | {'PatBars':>7} {'Sigs':>6} {'Trades':>6} {'WR%':>6} {'NetPnL':>9} {'Avg':>7}")
print("-" * 62)

for rise, drop in combos:
    r = simulate_combo(raw, full, prob, rise, drop)
    rows.append(r)
    print(
        f"{rise:5.0f} {drop:5.0f} | {r['pattern_bars']:7d} {r['signals']:6d} "
        f"{r['trades']:6d} {r['win_rate']:5.1f}% {r['net_pnl']:+9.2f} {r['avg_pnl']:+7.2f}"
    )

res = pd.DataFrame(rows).sort_values(["net_pnl", "win_rate"], ascending=False)
out = PROJECT_ROOT / "runtime" / "sweep_uptrend_retrace_definition.csv"
res.to_csv(out, index=False)
print(f"\nSaved -> {out}")

top = res[res["trades"] >= 20].head(12)
print("\nTop combos (≥20 trades):")
print(top.to_string(index=False))

best = res[res["trades"] >= 20].iloc[0] if (res["trades"] >= 20).any() else res.iloc[0]
print(
    f"\nBest (≥20 trades): rise>={int(best['rise'])} drop>={int(best['drop'])} → "
    f"{int(best['trades'])} trades, WR={best['win_rate']}%, PnL={best['net_pnl']:+.1f}"
)
cur = res[(res["rise"] == 30) & (res["drop"] == 10)]
if not cur.empty:
    c = cur.iloc[0]
    print(
        f"Current (30/10): {int(c['trades'])} trades, WR={c['win_rate']}%, PnL={c['net_pnl']:+.1f}, "
        f"pattern_bars={int(c['pattern_bars'])}"
    )
