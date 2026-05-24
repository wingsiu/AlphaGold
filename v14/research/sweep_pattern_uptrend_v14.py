#!/usr/bin/env python3
"""
Sweep TP / SL / horizon for uptrend_retrace pattern (execution-only).

Scores the pattern model once, then re-simulates exits for each combo.
Does NOT retrain models (fast execution sweep).

Usage:
  python3 sweep_pattern_uptrend_v14.py
  python3 sweep_pattern_uptrend_v14.py 2025-06-01 2026-05-23
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from itertools import product
from pathlib import Path

from v14._paths import PROJECT_ROOT

import joblib
import numpy as np
import pandas as pd

from v14.backtest.backtest_core import simulate_v13_core
from config.v14_config import EXECUTION_CONFIG, WF_CONFIG
from config.v14_patterns import PATTERN_MODEL_DIR, PATTERN_REGISTRY
from xgboost_filter_model.pattern_router import assign_patterns
from xgboost_filter_model.pattern_training import (
    cycle_model_path,
    feature_columns,
    iter_wf_cycles,
    prod_model_path,
    wf_anchor_ts,
)
from xgboost_filter_model.train_filter_1min import load_price_data
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14

PATTERN_NAME = "uptrend_retrace"

HORIZONS = [15, 30, 45, 60]
TARGETS = [15, 20, 30, 40, 45]
STOPS = [10, 15, 20, 25, 30]

args = sys.argv[1:]
bt_start = "2025-06-01"
bt_end = date.today().strftime("%Y-%m-%d")
if len(args) >= 1:
    bt_start = args[0]
if len(args) >= 2:
    bt_end = args[1]


def build_scored_sim_df(bt_start: str, bt_end: str) -> pd.DataFrame:
    spec = PATTERN_REGISTRY[PATTERN_NAME]
    warmup_days = int(WF_CONFIG.get("feature_warmup_days", 120))
    load_start_dt = max(
        pd.to_datetime(WF_CONFIG["full_start"]),
        pd.to_datetime(bt_start) - pd.Timedelta(days=warmup_days),
    )
    load_start = load_start_dt.strftime("%Y-%m-%d")
    bt_end_date = bt_end.split("T")[0] if "T" in bt_end else bt_end
    load_end = (pd.to_datetime(bt_end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bt_start_date = bt_start.split("T")[0] if "T" in bt_start else bt_start

    print(f"Loading data {load_start} → {bt_end}…")
    df = prepare_data_v14(start_date=load_start, end_date=load_end, energetic_filter=False)
    df = prepare_directional_data_v14(df)
    feats = feature_columns(df)

    bt_start_dt = pd.to_datetime(bt_start)
    if bt_start_dt.tzinfo is None:
        bt_start_dt = bt_start_dt.tz_localize("UTC")
    else:
        bt_start_dt = bt_start_dt.tz_convert("UTC")

    df_test = df[df.index >= bt_start_dt].copy()
    df_test = assign_patterns(df_test)

    pdir = PATTERN_MODEL_DIR / PATTERN_NAME
    prod = joblib.load(prod_model_path(pdir))

    wf_anchor = wf_anchor_ts()
    end_dt = pd.to_datetime(bt_end).tz_localize("UTC") + pd.Timedelta(days=1)

    # Score on full 1m timeline (signals only on pattern bars)
    full_df = df_test.copy()
    full_df["prob"] = np.nan
    full_df["s1_prob"] = np.nan
    full_df["s2_prob"] = np.nan
    full_df["side_signal"] = 0

    prob_thresh = spec["thresholds"]["prob"]
    pat_mask = full_df["pattern_name"] == PATTERN_NAME

    for cycle, current_start, current_end in iter_wf_cycles(bt_start_dt, end_dt, wf_anchor):
        chunk = (full_df.index >= current_start) & (full_df.index < current_end) & pat_mask
        if chunk.any():
            path = cycle_model_path(pdir, cycle, current_start.date())
            model = joblib.load(path) if path.exists() else prod
            p = model.predict_proba(full_df.loc[chunk, feats])[:, 1]
            full_df.loc[chunk, "prob"] = p
            full_df.loc[chunk, "s1_prob"] = p
            sig = chunk & (full_df["prob"] >= prob_thresh)
            full_df.loc[sig, "side_signal"] = 1
            full_df.loc[sig, "s2_prob"] = full_df.loc[sig, "prob"]

    raw_df = load_price_data(start_date=bt_start_date, end_date=load_end)
    raw_df = raw_df[raw_df.index >= bt_start_dt].copy()
    sim_df = raw_df[["open", "high", "low", "close"]].copy()
    for col in ("side_signal", "s1_prob", "s2_prob"):
        sim_df[col] = full_df[col]
    sim_df["side_signal"] = sim_df["side_signal"].fillna(0).astype(int)

    n_sig = (sim_df["side_signal"] != 0).sum()
    print(f"Pattern bars: {pat_mask.sum()}  |  Entry signals: {n_sig}")
    return sim_df


def run_sweep(sim_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cfg = EXECUTION_CONFIG.copy()
    combos = [(h, tp, sl) for h, tp, sl in product(HORIZONS, TARGETS, STOPS) if sl < tp]
    print(f"\nSweeping {len(combos)} combos (SL must be < TP)…\n")
    print(f"{'H':>4} {'TP':>4} {'SL':>4} | {'Trades':>6} {'Win%':>6} {'NetPnL':>9} {'Avg':>7}")
    print("-" * 52)

    for horizon, tp, sl in combos:
        trades = simulate_v13_core(sim_df, tp, sl, horizon, config=cfg)
        if not trades:
            rows.append(
                {
                    "horizon": horizon,
                    "tp": tp,
                    "sl": sl,
                    "trades": 0,
                    "win_rate": 0.0,
                    "net_pnl": 0.0,
                    "avg_pnl": 0.0,
                }
            )
            continue
        tdf = pd.DataFrame(trades)
        net = float(tdf["pnl"].sum())
        wr = float((tdf["pnl"] > 0).mean() * 100)
        n = len(tdf)
        avg = net / n
        rows.append(
            {
                "horizon": horizon,
                "tp": tp,
                "sl": sl,
                "trades": n,
                "win_rate": round(wr, 1),
                "net_pnl": round(net, 2),
                "avg_pnl": round(avg, 2),
            }
        )
        print(f"{horizon:4d} {tp:4.0f} {sl:4.0f} | {n:6d} {wr:5.1f}% {net:+9.2f} {avg:+7.2f}")

    return pd.DataFrame(rows).sort_values(["net_pnl", "win_rate"], ascending=False)


def main() -> None:
    if PATTERN_NAME not in PATTERN_REGISTRY:
        print(f"Unknown pattern: {PATTERN_NAME}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Uptrend retrace param sweep  |  {bt_start} → {bt_end}")
    print(f"  Horizons: {HORIZONS}")
    print(f"  Targets : {TARGETS}")
    print(f"  Stops   : {STOPS}")
    print(f"{'='*60}\n")

    sim_df = build_scored_sim_df(bt_start, bt_end)
    results = run_sweep(sim_df)

    out = PROJECT_ROOT / "runtime" / "sweep_uptrend_retrace_params.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved -> {out}")

    top = results[results["trades"] >= 10].head(15)
    print(f"\nTop combos (≥10 trades):")
    print(top.to_string(index=False))

    best = results[results["trades"] >= 10].iloc[0] if (results["trades"] >= 10).any() else results.iloc[0]
    print(
        f"\nBest (≥10 trades): H={int(best.horizon)} TP={int(best.tp)} SL={int(best.sl)} "
        f"→ {int(best.trades)} trades, WR={best.win_rate}%, PnL={best.net_pnl:+.1f}"
    )


if __name__ == "__main__":
    main()
