#!/usr/bin/env python3
"""
Compare ML filters on impulse_1m_15m sample pool.

Mechanical filter: first 1m |body|>=5pt in 15m slot, enter after slot close.
Models: xgb, rf, hgb (sklearn), logreg, lstm

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_ml_compare.py
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_ml_compare.py 2025-06-01 2026-06-25 --models xgb lstm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features
from v16.backtest.impulse_ml import (
    MODEL_NAMES,
    evaluate_threshold_sweep,
    walk_forward_model_scores,
)
from v16.backtest.position_sim import simulate_position_sided_scaleout
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_side_signals, count_signals

THRESHOLDS = (0.50, 0.52, 0.55, 0.58, 0.60, 0.65)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oos_start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    args = parser.parse_args()

    models = [m.lower() for m in args.models if m.lower() in MODEL_NAMES]
    if not models:
        models = list(MODEL_NAMES)

    cfg = v16_config.MOMENTUM_15M_HOLD
    so = cfg.get("scaleout", v16_config.EXIT_CONFIG)

    print("=" * 88)
    print(f"  impulse_1m_15m ML compare  |  train {args.train_start}  OOS {args.oos_start} → {args.end}")
    print(f"  filter: first 1m |body|>={cfg['min_move_pts']} in 15m slot")
    print(f"  models: {', '.join(models)}  |  label: scale-out +{so['first_scale_pnl']:.0f}/+{so['final_scale_pnl']:.0f}")
    print("=" * 88)

    t0 = time.time()
    df = load_gold_1m(args.train_start, args.end)
    feats = build_features(df)
    labeled = build_labeled_set(df, cfg=cfg)

    oos_start = pd.Timestamp(args.oos_start, tz="UTC")
    df_oos = df[df.index >= oos_start]
    base_sides = build_side_signals(df_oos, cfg=cfg)
    n = count_signals(base_sides)
    print(f"\nLabeled (full): {len(labeled)}  WR={labeled['win'].mean()*100:.1f}%")
    print(f"OOS mechanical: {n['total']} signals (L{n['long']} S{n['short']})")

    mech = simulate_position_sided_scaleout(
        df_oos,
        base_sides,
        scaleout_kw=dict(so),
        same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
    )
    print(
        f"Mechanical OOS: {len(mech)} tr  WR={mech['win'].mean()*100:.1f}%  "
        f"net={mech['pnl'].sum():+.1f}  avg={mech['pnl'].mean():+.2f}"
    )

    all_sweeps: list[pd.DataFrame] = []
    best_rows: list[dict] = []

    for model in models:
        print(f"\n--- {model.upper()} walk-forward ---")
        t1 = time.time()
        scores = walk_forward_model_scores(df, feats, labeled, model, prob_threshold=0.0)
        if scores.empty:
            print(f"  No scores for {model}")
            continue

        scores_oos = scores[pd.to_datetime(scores["signal_ts"], utc=True) >= oos_start]
        print(f"  OOS scored: {len(scores_oos)}  ({time.time() - t1:.1f}s)")

        sweep = evaluate_threshold_sweep(df_oos, base_sides, scores_oos, cfg, THRESHOLDS)
        all_sweeps.append(sweep)

        print(f"  {'prob':>5} {'sig':>5} {'tr':>5} {'WR%':>6} {'net':>9} {'avg':>6}")
        for _, r in sweep.iterrows():
            print(
                f"  {r['prob']:5.2f} {int(r['signals']):5d} {int(r['trades']):5d} "
                f"{r['wr']:6.1f} {r['net']:+9.1f} {r['avg']:+6.2f}"
            )

        best = sweep.sort_values("net", ascending=False).iloc[0]
        best_rows.append(
            {
                "model": model,
                "best_prob": best["prob"],
                "signals": int(best["signals"]),
                "trades": int(best["trades"]),
                "wr": best["wr"],
                "net": best["net"],
                "avg": best["avg"],
            }
        )

    if not all_sweeps:
        return

    sweep_df = pd.concat(all_sweeps, ignore_index=True)
    best_df = pd.DataFrame(best_rows).sort_values("net", ascending=False)

    out_sweep = PROJECT_ROOT / "runtime" / "v16_momentum_15m_hold_ml_sweep.csv"
    out_best = PROJECT_ROOT / "runtime" / "v16_momentum_15m_hold_ml_best.csv"
    sweep_df.to_csv(out_sweep, index=False)
    best_df.to_csv(out_best, index=False)

    print("\n" + "=" * 88)
    print("Best threshold per model (position sim OOS):")
    print(f"{'model':8s} {'prob':>5} {'trades':>6} {'WR%':>6} {'net':>9} {'avg':>6}")
    print("-" * 48)
    for _, r in best_df.iterrows():
        print(
            f"{r['model']:8s} {r['best_prob']:5.2f} {int(r['trades']):6d} "
            f"{r['wr']:6.1f} {r['net']:+9.1f} {r['avg']:+6.2f}"
        )
    print(f"\nSaved {out_sweep}")
    print(f"Saved {out_best}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
