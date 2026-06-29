#!/usr/bin/env python3
"""
Exit sweep: structure-change vs opposite-signal (v15 close_on_reverse).

Mechanical + ET ML (v15 14d retrain) on winner pre-close config.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_exit_sweep.py 2025-06-01 2026-06-25
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
from v16.backtest.impulse_features import attach_structure_features, structure_kwargs
from v16.backtest.impulse_ml import filter_signal_table, walk_forward_model_scores
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_signal_table

EXIT_MODES = {
    "struct_always": {
        "exit_on_structure_change": True,
        "exit_on_structure_change_min_pnl": -1e9,
        "exit_on_reverse_signal": False,
    },
    "reverse_always": {
        "exit_on_structure_change": False,
        "exit_on_reverse_signal": True,
        "exit_on_reverse_signal_min_pnl": -1e9,
    },
    "struct_and_reverse": {
        "exit_on_structure_change": True,
        "exit_on_structure_change_min_pnl": -1e9,
        "exit_on_reverse_signal": True,
        "exit_on_reverse_signal_min_pnl": -1e9,
    },
    "reverse_profit_only": {
        "exit_on_structure_change": False,
        "exit_on_reverse_signal": True,
        "exit_on_reverse_signal_min_pnl": 0.0,
    },
    "none": {
        "exit_on_structure_change": False,
        "exit_on_reverse_signal": False,
    },
}


def _stats(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {"trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0, "struct": 0, "reverse": 0}
    reasons = tdf["exit_reason"].value_counts()
    return {
        "trades": len(tdf),
        "wr": round(float(tdf["win"].mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
        "struct": int(reasons.get("structure_change", 0)),
        "reverse": int(reasons.get("reverse_signal", 0)),
        "tp": int(reasons.get("target_hit", 0)),
        "sl": int(reasons.get("stop_loss", 0)),
        "to": int(reasons.get("timeout", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--ml", action="store_true", help="also run ET ML p>=0.50 (14d retrain)")
    args = parser.parse_args()

    base = copy.deepcopy(v16_config.MOMENTUM_V16_WINNER_PRECLOSE)
    oos = pd.Timestamp(args.start, tz="UTC")

    print("=" * 88)
    print(f"  Exit sweep — winner pre-close R=3 H=120  |  {args.start} → {args.end}")
    print("  struct-change | reverse-signal (v15) | both | none")
    if args.ml:
        print("  ML: ET p>=0.50, v15 14d retrain")
    print("=" * 88)

    t0 = time.time()
    df = load_gold_1m(args.train_start, args.end)
    df_oos = df[df.index >= oos]
    signals = build_signal_table(df_oos, cfg=base)

    ml_scores = None
    if args.ml:
        labeled = build_labeled_set(df, cfg=base)
        feats = build_features(df)
        skw = structure_kwargs(base)
        if skw:
            feats = attach_structure_features(df, feats, **skw)
        print("Training ET walk-forward (14d)…")
        ml_scores = walk_forward_model_scores(
            df, feats, labeled, "et", prob_threshold=0.0, retrain_freq="14D", cfg=base
        )
        ml_scores = ml_scores[pd.to_datetime(ml_scores["signal_ts"], utc=True) >= oos]
        ml_scores = ml_scores[ml_scores["p_win"] >= 0.50]

    rows: list[dict] = []
    for mode, exit_kw in EXIT_MODES.items():
        for lane, filt in [("mech", signals), ("ml", filter_signal_table(signals, ml_scores) if ml_scores is not None else None)]:
            if lane == "ml" and filt is None:
                continue
            cfg = copy.deepcopy(base)
            cfg["impulse_stop"] = {**cfg.get("impulse_stop", {}), **exit_kw}
            tdf = simulate_position_impulse_stop(df_oos, filt, cfg=cfg)
            rows.append({"mode": mode, "lane": lane, **_stats(tdf)})

    out = pd.DataFrame(rows)
    path = PROJECT_ROOT / "runtime" / f"v16_winner_exit_sweep{'_ml' if args.ml else ''}.csv"
    out.to_csv(path, index=False)

    print(f"\n{'mode':18s} {'lane':5s} {'tr':>5} {'WR%':>6} {'net':>9} {'avg':>7} {'SC':>4} {'RV':>4} {'TP':>4} {'SL':>4}")
    print("-" * 80)
    for _, r in out.sort_values("net", ascending=False).iterrows():
        print(
            f"{r['mode']:18s} {r['lane']:5s} {int(r['trades']):5d} {r['wr']:6.1f} "
            f"{r['net']:+9.1f} {r['avg']:+7.2f} {int(r['struct']):4d} {int(r['reverse']):4d} "
            f"{int(r['tp']):4d} {int(r['sl']):4d}"
        )

    best = out.sort_values("net", ascending=False).iloc[0]
    print(f"\nBest: {best['mode']} ({best['lane']}) → net={best['net']:+.1f} ({int(best['trades'])} tr)")
    print(f"Saved {path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
