#!/usr/bin/env python3
"""
Sweep H / TP / SL for downtrend_retrace with retrain per combo (aligned labels + exec).

Uses pattern rules from config (fall/bounce). Trains walk-forward model for each
combo, then backtests SHORT on the Friday WF grid.

Usage:
  python3 sweep_pattern_downtrend_retrain.py
  python3 sweep_pattern_downtrend_retrain.py 2025-06-01 2026-05-23
"""
from __future__ import annotations

import sys
from datetime import date
from itertools import product
from pathlib import Path

from v14._paths import PROJECT_ROOT

import joblib
import pandas as pd

from v14.backtest.backtest_core import simulate_v13_core
from config.v14_config import EXECUTION_CONFIG, WF_CONFIG
from config.v14_patterns import PATTERN_MODEL_DIR, PATTERN_REGISTRY
from train_pattern_variants import train_variant
from xgboost_filter_model.pattern_router import pattern_mask
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

PATTERN = "downtrend_retrace"
HORIZONS = [15, 30, 45, 60]
TARGETS = [15, 20, 30, 40, 45]
STOPS = [10, 15, 20, 25, 30]
PROB_THRESH = PATTERN_REGISTRY[PATTERN]["thresholds"]["prob"]

args = sys.argv[1:]
bt_start = "2025-06-01"
bt_end = date.today().strftime("%Y-%m-%d")
if len(args) >= 1:
    bt_start = args[0]
if len(args) >= 2:
    bt_end = args[1]

OUT_CSV = PROJECT_ROOT / "runtime" / "sweep_downtrend_retrace_retrain.csv"


def _rule_val(spec: dict, feat: str) -> float:
    for r in spec["pattern"]:
        if r["feat"] == feat:
            return float(r["val"])
    return 0.0


def backtest_variant(
    out_dir: Path,
    df: pd.DataFrame,
    raw: pd.DataFrame,
    bt_start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    horizon: int,
    tp: float,
    sl: float,
) -> dict:
    feats = feature_columns(df)
    full = df[df.index >= bt_start_dt].copy()
    pat = pattern_mask(full, PATTERN)
    prod = joblib.load(prod_model_path(out_dir))
    wf = wf_anchor_ts()

    prob = pd.Series(float("nan"), index=full.index)
    side = pd.Series(0, index=full.index, dtype=int)
    s2 = pd.Series(0.0, index=full.index)

    for cycle, cur, ce in iter_wf_cycles(bt_start_dt, end_dt, wf):
        chunk = (full.index >= cur) & (full.index < ce) & pat
        if not chunk.any():
            continue
        cp = cycle_model_path(out_dir, cycle, cur.date())
        model = joblib.load(cp) if cp.exists() else prod
        idx = full.index[chunk]
        p = model.predict_proba(full.loc[idx, feats])[:, 1]
        prob.loc[idx] = p
        sig_idx = idx[p >= PROB_THRESH]
        side.loc[sig_idx] = -1
        s2.loc[sig_idx] = 1.0 - prob.loc[sig_idx]

    sim = raw[["open", "high", "low", "close"]].copy()
    sim["side_signal"] = side.reindex(sim.index).fillna(0).astype(int)
    sim["s2_prob"] = s2.reindex(sim.index).fillna(0.0)
    sim["s1_prob"] = prob.reindex(sim.index).fillna(0.0)

    trades = simulate_v13_core(sim, tp, sl, horizon, config=EXECUTION_CONFIG.copy())
    n_pat = int(pat.sum())
    n_sig = int((sim["side_signal"] != 0).sum())
    if not trades:
        return {
            "horizon": horizon,
            "tp": tp,
            "sl": sl,
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
        "horizon": horizon,
        "tp": tp,
        "sl": sl,
        "pattern_bars": n_pat,
        "signals": n_sig,
        "trades": n,
        "win_rate": round((tdf["pnl"] > 0).mean() * 100, 1),
        "net_pnl": round(net, 2),
        "avg_pnl": round(net / n, 2),
    }


def main() -> None:
    spec = PATTERN_REGISTRY[PATTERN]
    fall = _rule_val(spec, "drop_from_high_240")
    bounce = _rule_val(spec, "rise_from_low_240")
    combos = [(h, tp, sl) for h, tp, sl in product(HORIZONS, TARGETS, STOPS) if sl < tp]

    print(f"\n{'='*70}")
    print(f"  Downtrend retrace RETRAIN sweep  |  {bt_start} → {bt_end}")
    print(f"  Definition: fall≥{fall:.0f}  bounce≥{bounce:.0f}  |  prob≥{PROB_THRESH}")
    print(f"  Combos: {len(combos)} (retrain + backtest each)")
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
    raw = load_price_data(bt_start, load_end)
    raw = raw[raw.index >= bt_start_dt]

    rows: list[dict] = []
    print(f"{'#':>4} {'H':>4} {'TP':>4} {'SL':>4} | {'Trades':>6} {'WR%':>6} {'NetPnL':>9} {'Avg':>7}")
    print("-" * 52)

    for i, (h, tp, sl) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] Train H={h} TP={tp} SL={sl}…", flush=True)
        df = prepare_data_v14(
            load_start,
            load_end,
            energetic_filter=False,
            label_horizon=h,
            label_tp=tp,
            label_sl=sl,
            fixed_label_tp_sl=True,
        )
        df = prepare_directional_data_v14(df)
        out_dir = train_variant(PATTERN, df, horizon=h, tp=tp, sl=sl, quiet=True)
        if out_dir is None:
            row = {
                "horizon": h, "tp": tp, "sl": sl,
                "pattern_bars": 0, "signals": 0, "trades": 0,
                "win_rate": 0.0, "net_pnl": 0.0, "avg_pnl": 0.0,
            }
        else:
            row = backtest_variant(out_dir, df, raw, bt_start_dt, end_dt, h, tp, sl)
        rows.append(row)
        pd.DataFrame(rows).sort_values(["net_pnl", "win_rate"], ascending=False).to_csv(
            OUT_CSV, index=False
        )
        print(
            f"{i:4d} {h:4d} {tp:4.0f} {sl:4.0f} | {row['trades']:6d} "
            f"{row['win_rate']:5.1f}% {row['net_pnl']:+9.2f} {row['avg_pnl']:+7.2f}",
            flush=True,
        )

    res = pd.DataFrame(rows).sort_values(["net_pnl", "win_rate"], ascending=False)
    res.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")

    top = res[res["trades"] >= 10].head(15)
    print("\nTop combos (≥10 trades):")
    print(top.to_string(index=False))

    if (res["trades"] >= 10).any():
        best = res[res["trades"] >= 10].iloc[0]
        print(
            f"\nBest: H={int(best.horizon)} TP={int(best.tp)} SL={int(best.sl)} "
            f"→ {int(best.trades)} trades WR={best.win_rate}% PnL={best.net_pnl:+.1f}"
        )
    baseline = res[(res["horizon"] == 60) & (res["tp"] == 30) & (res["sl"] == 25)]
    if not baseline.empty:
        b = baseline.iloc[0]
        print(
            f"Baseline H=60 TP=30 SL=25: {int(b.trades)} trades "
            f"WR={b.win_rate}% PnL={b.net_pnl:+.1f}"
        )


if __name__ == "__main__":
    main()
