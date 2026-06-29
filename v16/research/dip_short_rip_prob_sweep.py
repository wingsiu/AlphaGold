#!/usr/bin/env python3
"""
Sweep ML prob threshold — single WF train pass, filter only (no retrain).

Usage:
  PYTHONPATH=. python3 v16/research/dip_short_rip_prob_sweep.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/dip_short_rip_prob_sweep.py 2025-06-01 2026-06-25 --quick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features
from v16.backtest.ml import walk_forward_short_probs
from v16.backtest.position_sim import simulate_single_position
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.dip_short_rip import (
    build_labeled_set,
    feature_columns,
    resolve_execution,
    router_mask,
)

PROBS_FULL = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85, 0.90]
PROBS_QUICK = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = v16_config.DIP_SHORT_RIP
    ex = resolve_execution(cfg, mechanical=False)
    tp, sl, h = float(ex["tp"]), float(ex["sl"]), int(ex["horizon"])
    probs = PROBS_QUICK if args.quick else PROBS_FULL

    print("=" * 80)
    print(f"  dip_short_rip PROB sweep (no retrain)  |  {args.start} → {args.end}")
    print(f"  exit: TP{tp:.0f}/SL{ex['sl']:.0f}/H{h}  |  single position + same_dir_refresh")
    print("=" * 80)

    df = load_gold_1m(args.start, args.end)
    feats = build_features(df)
    router = router_mask(feats, df.index, cfg=cfg)
    print(f"\nRouter pool: {int(router.sum())}")

    labeled = build_labeled_set(df, feats, cfg=cfg)
    print("Walk-forward scoring (one pass)...", flush=True)
    scores = walk_forward_short_probs(labeled, feats, feature_columns(feats))
    scores["signal_ts"] = pd.to_datetime(scores["signal_ts"], utc=True)
    print(f"OOS scored rows: {len(scores)}")

    # map signal_ts -> p_short for router hits
    router_ts = df.index[router]
    score_ix = pd.Index(scores["signal_ts"])
    on_router = scores[scores["signal_ts"].isin(router_ts)].copy()

    rows = []
    refresh = cfg.get("same_dir_refresh", "entry")
    upgrade = bool(cfg.get("upgrade_stop", False))

    for p in probs:
        sub = on_router[on_router["p_short"] >= p]
        ml_ok = pd.Series(False, index=df.index)
        if not sub.empty:
            ml_ok.loc[df.index.intersection(sub["signal_ts"])] = True
        signals = router & ml_ok
        tdf = simulate_single_position(
            df, signals, side=-1, tp=tp, sl=sl, horizon=h,
            same_dir_refresh=refresh, upgrade_stop=upgrade,
        )
        if tdf.empty:
            rows.append({"prob": p, "trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0, "max_dd": 0.0})
            continue
        eq = tdf["pnl"].cumsum()
        max_dd = float((eq - eq.cummax()).min())
        rows.append({
            "prob": p,
            "trades": len(tdf),
            "wr": round(float((tdf["pnl"] > 0).mean() * 100), 1),
            "net": round(float(tdf["pnl"].sum()), 1),
            "avg": round(float(tdf["pnl"].mean()), 2),
            "max_dd": round(max_dd, 1),
            "tp_hit": int((tdf["exit_reason"] == "target_hit").sum()),
            "sl_hit": int((tdf["exit_reason"] == "stop_loss").sum()),
        })

    out_df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "runtime" / "v16_dip_short_rip_prob_sweep.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\n{'prob':>6}  {'trades':>6}  {'WR%':>6}  {'net':>8}  {'avg':>6}  {'maxDD':>8}")
    print("-" * 52)
    for _, r in out_df.sort_values("prob").iterrows():
        print(f"{r['prob']:>6.2f}  {int(r['trades']):>6}  {r['wr']:>6.1f}  {r['net']:>+8.1f}  {r['avg']:>+6.2f}  {r['max_dd']:>+8.1f}")

    best = out_df.sort_values("net", ascending=False).iloc[0]
    best_avg = out_df.sort_values("avg", ascending=False).iloc[0]
    print(f"\nBest net : p>={best['prob']:.2f}  {int(best['trades'])} tr  net={best['net']:+.1f}")
    print(f"Best avg : p>={best_avg['prob']:.2f}  {int(best_avg['trades'])} tr  avg={best_avg['avg']:+.2f}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
