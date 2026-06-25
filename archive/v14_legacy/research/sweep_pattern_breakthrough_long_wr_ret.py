#!/usr/bin/env python3
"""
Sweep WR(90) and ret_3m gates for breakthrough_long at fixed exec H45/TP30/SL15.

Trains one walk-forward model for H45 TP30 SL15, then backtests each gate combo.

Usage:
  python3 sweep_pattern_breakthrough_long_wr_ret.py 2025-06-01 2026-05-23
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
from config.v14_patterns import PATTERN_REGISTRY
from train_pattern_variants import train_variant
from xgboost_filter_model.pattern_training import (
    add_pattern_entry_target,
    apply_exec_labels,
    cycle_model_path,
    feature_columns,
    iter_wf_cycles,
    precompute_future_moves,
    prod_model_path,
    wf_anchor_ts,
)
from xgboost_filter_model.train_filter_1min import load_price_data
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14

PATTERN = "breakthrough_long"
EXEC_H, EXEC_TP, EXEC_SL = 45, 30.0, 15.0
PROB_THRESH = PATTERN_REGISTRY[PATTERN]["thresholds"]["prob"]

WR90_VALUES = [-70, -60, -50, -40, -30, -20, -10]
RET_VALUES = [4, 6, 8, 10, 12, 15, 18, 20]

args = sys.argv[1:]
bt_start, bt_end = "2025-06-01", date.today().strftime("%Y-%m-%d")
if len(args) >= 1:
    bt_start = args[0]
if len(args) >= 2:
    bt_end = args[1]

OUT_CSV = PROJECT_ROOT / "runtime" / "sweep_breakthrough_long_wr_ret.csv"


def gate_mask(full: pd.DataFrame, wr_min: float, ret_min: float) -> pd.Series:
    ret_3m = full["ret_3m"] if "ret_3m" in full.columns else full["close"] - full["close"].shift(3)
    return (full["wr_90"] > wr_min) & (ret_3m > ret_min)


def score_probs(
    full: pd.DataFrame,
    feats: list[str],
    out_dir: Path,
    loose: pd.Series,
    bt_start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.Series:
    prod = joblib.load(prod_model_path(out_dir))
    wf = wf_anchor_ts()
    prob = pd.Series(np.nan, index=full.index)
    for cycle, cur, ce in iter_wf_cycles(bt_start_dt, end_dt, wf):
        chunk = (full.index >= cur) & (full.index < ce) & loose
        if not chunk.any():
            continue
        cp = cycle_model_path(out_dir, cycle, cur.date())
        model = joblib.load(cp) if cp.exists() else prod
        prob.loc[chunk] = model.predict_proba(full.loc[chunk, feats])[:, 1]
    return prob


def backtest_combo(
    raw: pd.DataFrame,
    full: pd.DataFrame,
    prob: pd.Series,
    wr_min: float,
    ret_min: float,
) -> dict:
    pat = gate_mask(full, wr_min, ret_min)
    sig = (pat & (prob >= PROB_THRESH)).fillna(False)
    sim = raw[["open", "high", "low", "close"]].copy()
    side = pd.Series(0, index=sim.index, dtype=int)
    s2 = pd.Series(0.0, index=sim.index)
    fired = sig.reindex(sim.index, fill_value=False)
    side.loc[fired] = 1
    s2.loc[fired] = prob.reindex(sim.index).loc[fired].astype(float)
    sim["side_signal"] = side
    sim["s2_prob"] = s2
    sim["s1_prob"] = prob.reindex(sim.index).fillna(0.0)

    cfg = EXECUTION_CONFIG.copy()
    cfg["close_on_reverse"] = False
    trades = simulate_v13_core(sim, EXEC_TP, EXEC_SL, EXEC_H, config=cfg)
    n_pat = int(pat.sum())
    n_sig = int(fired.sum())
    if not trades:
        return {
            "wr90_min": wr_min,
            "ret_3m_min": ret_min,
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
        "wr90_min": wr_min,
        "ret_3m_min": ret_min,
        "pattern_bars": n_pat,
        "signals": n_sig,
        "trades": n,
        "win_rate": round((tdf["pnl"] > 0).mean() * 100, 1),
        "net_pnl": round(net, 2),
        "avg_pnl": round(net / n, 2),
    }


def main() -> None:
    combos = list(product(WR90_VALUES, RET_VALUES))
    print(f"\n{'='*70}")
    print(f"  Breakthrough LONG — WR(90) × ret_3m sweep")
    print(f"  Exec: H={EXEC_H} TP={EXEC_TP} SL={EXEC_SL}  |  prob≥{PROB_THRESH}")
    print(f"  WR(90) > {WR90_VALUES}")
    print(f"  ret_3m > {RET_VALUES}")
    print(f"  Combos: {len(combos)}  |  {bt_start} → {bt_end}")
    print(f"  Output: {OUT_CSV}")
    print(f"{'='*70}\n")

    warmup = int(WF_CONFIG.get("feature_warmup_days", 120))
    load_start = max(
        pd.to_datetime(WF_CONFIG["full_start"]),
        pd.to_datetime(bt_start) - pd.Timedelta(days=warmup),
    ).strftime("%Y-%m-%d")
    load_end = (pd.to_datetime(bt_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bt_start_dt = pd.to_datetime(bt_start).tz_localize("UTC")
    end_dt = pd.to_datetime(bt_end).tz_localize("UTC") + pd.Timedelta(days=1)

    print("Loading features…", flush=True)
    df_feat = prepare_data_v14(
        load_start,
        load_end,
        energetic_filter=False,
        label_horizon=EXEC_H,
        label_tp=EXEC_TP,
        label_sl=EXEC_SL,
        fixed_label_tp_sl=True,
    )
    df_feat = prepare_directional_data_v14(df_feat)
    future = precompute_future_moves(df_feat, [EXEC_H])[EXEC_H]
    df = apply_exec_labels(df_feat, EXEC_H, EXEC_TP, EXEC_SL, future_moves=future)

    print(f"Training H={EXEC_H} TP={EXEC_TP} SL={EXEC_SL}…", flush=True)
    out_dir = train_variant(PATTERN, df, horizon=EXEC_H, tp=EXEC_TP, sl=EXEC_SL, quiet=True)
    if out_dir is None:
        print("Training failed — too few samples.")
        sys.exit(1)

    full = df[df.index >= bt_start_dt].copy()
    feats = feature_columns(full)
    loose_wr = min(WR90_VALUES)
    loose_ret = min(RET_VALUES)
    loose = gate_mask(full, loose_wr, loose_ret)
    print(f"Scoring model on loose gate WR>{loose_wr}, ret>{loose_ret} ({int(loose.sum())} bars)…", flush=True)
    prob = score_probs(full, feats, out_dir, loose, bt_start_dt, end_dt)

    raw = load_price_data(bt_start, load_end)
    raw = raw[raw.index >= bt_start_dt]

    rows: list[dict] = []
    print(f"\n{'WR>':>5s} {'ret>':>5s} {'bars':>6s} {'tgt+':>6s} {'sig':>5s} {'trd':>5s} {'WR%':>6s} {'PnL':>8s} {'avg':>6s}")
    print("-" * 58)
    for i, (wr_min, ret_min) in enumerate(combos, 1):
        pat = gate_mask(full, wr_min, ret_min)
        tgt = add_pattern_entry_target(full.loc[pat].copy(), "long")
        r = backtest_combo(raw, full, prob, wr_min, ret_min)
        r["target_plus"] = int(tgt["target_pattern"].sum())
        r["target_pct"] = round(float(tgt["target_pattern"].mean()) * 100, 1) if len(tgt) else 0.0
        rows.append(r)
        pd.DataFrame(rows).sort_values(["net_pnl", "win_rate"], ascending=False).to_csv(OUT_CSV, index=False)
        print(
            f"{wr_min:5.0f} {ret_min:5.0f} {r['pattern_bars']:6d} {r['target_plus']:6d} "
            f"{r['signals']:5d} {r['trades']:5d} {r['win_rate']:5.1f}% "
            f"{r['net_pnl']:+8.1f} {r['avg_pnl']:+6.2f}",
            flush=True,
        )

    res = pd.DataFrame(rows).sort_values(["net_pnl", "win_rate"], ascending=False)
    res.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")

    viable = res[res["trades"] >= 10]
    if not viable.empty:
        best = viable.iloc[0]
        print(
            f"\nBest (≥10 trades): WR(90)>{best.wr90_min:.0f} ret_3m>{best.ret_3m_min:.0f} "
            f"→ {int(best.trades)} tr WR={best.win_rate}% PnL={best.net_pnl:+.1f}"
        )
    best_any = res.iloc[0]
    print(
        f"Best any: WR(90)>{best_any.wr90_min:.0f} ret_3m>{best_any.ret_3m_min:.0f} "
        f"→ {int(best_any.trades)} tr PnL={best_any.net_pnl:+.1f}"
    )

    default = res[(res["wr90_min"] == -30) & (res["ret_3m_min"] == 8)]
    if not default.empty:
        d = default.iloc[0]
        print(
            f"Default WR>-30 ret>8: {int(d.trades)} tr WR={d.win_rate}% PnL={d.net_pnl:+.1f}"
        )


if __name__ == "__main__":
    main()
