#!/usr/bin/env python3
"""
Sweep entry minute (0–4) in the post-impulse 15m slot — mechanical + ML predict.

Impulse in prior slot; predict/enter at minute N of next 15m slot.
Exit: TP25/SL35/H60 (sweep winner). ML labels match execution.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_entry_min_sweep.py
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_entry_min_sweep.py 2025-06-01 2026-06-25 --models rf lstm
"""
from __future__ import annotations

import argparse
import copy
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
    apply_ml_filter_to_sides,
    evaluate_threshold_sweep,
    walk_forward_model_scores,
)
from v16.backtest.position_sim import simulate_position_sided
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_side_signals, count_signals

EXECUTION = {"tp": 25.0, "sl": 35.0, "horizon": 60}
ENTRY_MINS = (0, 1, 2, 3, 4)
THRESHOLDS = (0.50, 0.55, 0.58, 0.60, 0.65, 0.70)
DEFAULT_MODELS = ("rf", "lstm")


def _cfg_for_entry(entry_min: int) -> dict:
    c = copy.deepcopy(v16_config.MOMENTUM_15M_HOLD)
    c["entry_minute_in_slot"] = entry_min
    c["ml_label_mode"] = "execution"
    c["execution"] = dict(EXECUTION)
    return c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oos_start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    args = parser.parse_args()

    models = [m.lower() for m in args.models]
    oos_start = pd.Timestamp(args.oos_start, tz="UTC")

    print("=" * 92)
    print(f"  impulse entry-minute sweep  |  OOS {args.oos_start} → {args.end}")
    print(f"  exit TP{EXECUTION['tp']:.0f}/SL{EXECUTION['sl']:.0f}/H{EXECUTION['horizon']}")
    print(f"  entry minutes: {list(ENTRY_MINS)}  |  models: {', '.join(models)}")
    print("=" * 92)

    t0 = time.time()
    df = load_gold_1m(args.train_start, args.end)
    feats = build_features(df)
    df_oos = df[df.index >= oos_start]

    mech_rows: list[dict] = []
    ml_rows: list[dict] = []

    for entry_min in ENTRY_MINS:
        cfg = _cfg_for_entry(entry_min)
        print(f"\n{'='*40} entry minute {entry_min} {'='*40}")

        labeled = build_labeled_set(df, cfg=cfg)
        sides_oos = build_side_signals(df_oos, cfg=cfg)
        n = count_signals(sides_oos)
        print(f"  OOS signals: {n['total']}  labeled WR: {labeled['win'].mean()*100:.1f}%")

        mech = simulate_position_sided(
            df_oos,
            sides_oos,
            tp=EXECUTION["tp"],
            sl=EXECUTION["sl"],
            horizon=EXECUTION["horizon"],
            same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
        )
        mech_net = float(mech["pnl"].sum()) if not mech.empty else 0.0
        mech_wr = float(mech["win"].mean() * 100) if not mech.empty else 0.0
        mech_rows.append(
            {
                "entry_min": entry_min,
                "mode": "mechanical",
                "model": "mech",
                "prob": None,
                "signals": n["total"],
                "trades": len(mech),
                "wr": round(mech_wr, 1),
                "net": round(mech_net, 1),
                "avg": round(float(mech["pnl"].mean()), 2) if not mech.empty else 0.0,
            }
        )
        print(f"  Mechanical: {len(mech)} tr  WR={mech_wr:.1f}%  net={mech_net:+.1f}")

        for model in models:
            t1 = time.time()
            scores = walk_forward_model_scores(df, feats, labeled, model, prob_threshold=0.0)
            if scores.empty:
                print(f"  {model}: no scores")
                continue
            scores_oos = scores[pd.to_datetime(scores["signal_ts"], utc=True) >= oos_start]
            sweep = evaluate_threshold_sweep(
                df_oos,
                sides_oos,
                scores_oos,
                cfg,
                THRESHOLDS,
                exit_mode="execution",
                execution=EXECUTION,
            )
            best = sweep.sort_values("net", ascending=False).iloc[0]
            ml_rows.append(
                {
                    "entry_min": entry_min,
                    "mode": "ml",
                    "model": model,
                    "prob": best["prob"],
                    "signals": int(best["signals"]),
                    "trades": int(best["trades"]),
                    "wr": best["wr"],
                    "net": best["net"],
                    "avg": best["avg"],
                }
            )
            print(
                f"  {model:5s} ({time.time()-t1:.1f}s) best p>={best['prob']:.2f}: "
                f"{int(best['trades'])} tr  WR={best['wr']:.1f}%  net={best['net']:+.1f}"
            )

    mech_df = pd.DataFrame(mech_rows)
    ml_df = pd.DataFrame(ml_rows)
    out = pd.concat([mech_df, ml_df], ignore_index=True)
    out_path = PROJECT_ROOT / "runtime" / "v16_momentum_15m_hold_entry_min_sweep.csv"
    out.to_csv(out_path, index=False)

    print("\n" + "=" * 92)
    print("Summary — mechanical by entry minute:")
    for _, r in mech_df.iterrows():
        print(
            f"  min {int(r['entry_min'])}: {int(r['trades'])} tr  WR={r['wr']:.1f}%  "
            f"net={r['net']:+.1f}  avg={r['avg']:+.2f}"
        )

    if not ml_df.empty:
        print("\nBest ML per entry minute:")
        for em in ENTRY_MINS:
            sub = ml_df[ml_df["entry_min"] == em].sort_values("net", ascending=False)
            if sub.empty:
                continue
            r = sub.iloc[0]
            print(
                f"  min {int(r['entry_min'])} {r['model']} p>={r['prob']:.2f}: "
                f"{int(r['trades'])} tr  WR={r['wr']:.1f}%  net={r['net']:+.1f}"
            )

    print(f"\nSaved -> {out_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
