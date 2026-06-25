#!/usr/bin/env python3
"""
Sweep breakthrough_short ret_3m drop threshold (WR90 < -70 fixed).

Uses trained model h30_tp20_sl15; varies only the ret_3m gate.
Exec: H=30 TP=20 SL=15 from config.

Usage:
  python3 sweep_pattern_breakthrough_ret3.py 2025-06-01 2026-05-23
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from v14._paths import PROJECT_ROOT

import joblib
import numpy as np
import pandas as pd

from v14.backtest.backtest_core import simulate_v13_core
from config.v14_config import EXECUTION_CONFIG, WF_CONFIG
from config.v14_patterns import PATTERN_MODEL_DIR, PATTERN_REGISTRY
from xgboost_filter_model.pattern_training import (
    add_pattern_entry_target,
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

PATTERN = "breakthrough_short"
SPEC = PATTERN_REGISTRY[PATTERN]
EXEC = SPEC["execution"]
EXEC_H, EXEC_TP, EXEC_SL = EXEC["horizon"], EXEC["tp"], EXEC["sl"]
MODEL_VARIANT = pattern_variant_tag(EXEC_H, EXEC_TP, EXEC_SL)
PROB_THRESH = SPEC["thresholds"]["prob"]
WR90_MAX = -70.0

RET_VALUES = [-6, -8, -10, -12, -15, -18, -20, -25, -30]

args = sys.argv[1:]
bt_start, bt_end = "2025-06-01", date.today().strftime("%Y-%m-%d")
if len(args) >= 1:
    bt_start = args[0]
if len(args) >= 2:
    bt_end = args[1]


def pattern_mask(full: pd.DataFrame, ret_max: float) -> pd.Series:
    ret_3m = full["ret_3m"] if "ret_3m" in full.columns else full["close"] - full["close"].shift(3)
    return (full["wr_90"] < WR90_MAX) & (ret_3m < ret_max)


def score_probs(full: pd.DataFrame, feats: list[str], loose: pd.Series) -> pd.Series:
    pdir = PATTERN_MODEL_DIR / PATTERN / MODEL_VARIANT
    prod = joblib.load(prod_model_path(pdir))
    bt_start_dt = full.index.min()
    wf = wf_anchor_ts()
    end_dt = full.index.max() + pd.Timedelta(minutes=1)
    prob = pd.Series(np.nan, index=full.index)
    for cycle, cur, ce in iter_wf_cycles(bt_start_dt, end_dt, wf):
        chunk = (full.index >= cur) & (full.index < ce) & loose
        if not chunk.any():
            continue
        cp = cycle_model_path(pdir, cycle, cur.date())
        model = joblib.load(cp) if cp.exists() else prod
        prob.loc[chunk] = model.predict_proba(full.loc[chunk, feats])[:, 1]
    return prob


def simulate_combo(raw, full, prob, ret_max: float) -> dict:
    pat = pattern_mask(full, ret_max)
    sig = pat & (prob >= PROB_THRESH)
    sim = raw[["open", "high", "low", "close"]].copy()
    side = pd.Series(0, index=sim.index, dtype=int)
    s2 = pd.Series(0.0, index=sim.index)
    fired = sig.reindex(sim.index).fillna(False)
    side.loc[fired] = -1
    s2.loc[fired] = 1.0 - prob.reindex(sim.index).loc[fired].astype(float)
    sim["side_signal"] = side
    sim["s2_prob"] = s2
    sim["s1_prob"] = prob.reindex(sim.index).fillna(0.0)
    sim["exec_tp"] = EXEC_TP
    sim["exec_sl"] = EXEC_SL
    sim["exec_horizon"] = EXEC_H

    cfg = EXECUTION_CONFIG.copy()
    cfg["close_on_reverse"] = False
    trades = simulate_v13_core(sim, EXEC_TP, EXEC_SL, EXEC_H, config=cfg)
    n_pat = int(pat.sum())
    n_sig = int(fired.sum())
    if not trades:
        return {
            "ret_3m_max": ret_max,
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
        "ret_3m_max": ret_max,
        "pattern_bars": n_pat,
        "signals": n_sig,
        "trades": n,
        "win_rate": round((tdf["pnl"] > 0).mean() * 100, 1),
        "net_pnl": round(net, 2),
        "avg_pnl": round(net / n, 2),
    }


print(f"\n{'='*60}")
print("  Breakthrough SHORT — ret_3m drop sweep")
print(f"  WR(90) < {WR90_MAX}  |  ret_3m thresholds: {RET_VALUES}")
print(f"  Model: {MODEL_VARIANT}  |  Exec H={EXEC_H} TP={EXEC_TP} SL={EXEC_SL}")
print(f"  Period: {bt_start} → {bt_end}")
print(f"{'='*60}\n")

warmup = int(WF_CONFIG.get("feature_warmup_days", 120))
load_start = (
    pd.to_datetime(bt_start) - pd.Timedelta(days=warmup)
).strftime("%Y-%m-%d")
load_end = (pd.to_datetime(bt_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

print("Loading data…")
full = prepare_data_v14(
    start_date=load_start,
    end_date=load_end,
    energetic_filter=False,
    label_horizon=EXEC_H,
    label_tp=EXEC_TP,
    label_sl=EXEC_SL,
    fixed_label_tp_sl=True,
)
full = prepare_directional_data_v14(full)
bt_start_dt = pd.to_datetime(bt_start).tz_localize("UTC")
full = full[full.index >= bt_start_dt].copy()
feats = feature_columns(full)

loose = pattern_mask(full, max(RET_VALUES))
print(f"Scoring with model (loose ret < {max(RET_VALUES)})…")
prob = score_probs(full, feats, loose)

raw = load_price_data(start_date=bt_start, end_date=load_end)
raw = raw[raw.index >= bt_start_dt].copy()

rows = []
print(f"\n{'ret':>6s} {'bars':>6s} {'tgt+':>6s} {'tgt%':>6s} {'sig':>5s} {'trd':>5s} {'WR':>6s} {'PnL':>8s} {'avg':>6s}")
print("-" * 62)
for ret_max in RET_VALUES:
    pat = pattern_mask(full, ret_max)
    tgt = add_pattern_entry_target(full.loc[pat].copy(), "short")
    tgt_n = int(tgt["target_pattern"].sum())
    tgt_pct = tgt["target_pattern"].mean() * 100 if len(tgt) else 0.0
    r = simulate_combo(raw, full, prob, ret_max)
    r["target_plus"] = tgt_n
    r["target_pct"] = round(tgt_pct, 1)
    rows.append(r)
    print(
        f"{ret_max:6.0f} {r['pattern_bars']:6d} {tgt_n:6d} {tgt_pct:5.1f}% "
        f"{r['signals']:5d} {r['trades']:5d} {r['win_rate']:5.1f}% "
        f"{r['net_pnl']:+8.1f} {r['avg_pnl']:+6.2f}"
    )

out = PROJECT_ROOT / "runtime" / "sweep_breakthrough_ret3.csv"
pd.DataFrame(rows).to_csv(out, index=False)
best = max(rows, key=lambda x: x["net_pnl"])
print(f"\nBest PnL: ret_3m < {best['ret_3m_max']:.0f} → {best['trades']} trades, PnL={best['net_pnl']:+.1f}")
print(f"Saved -> {out}")
