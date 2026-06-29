#!/usr/bin/env python3
"""
ML walk-forward on breakout + pre-break close ≤10pt + with-trend + next_open.

Mechanical OOS ~+843 (765 tr). Labels match pre-close filtered fills.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_breakout_preclose_ml.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_breakout_preclose_ml.py 2025-06-01 2026-06-25 --models logreg lgb gbc
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_breakout_preclose_ml.py 2025-06-01 2026-06-25 --pre-close 8
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
from v16.backtest.impulse_entry import build_resolved_entry_table
from v16.backtest.impulse_features import (
    attach_structure_features,
    impulse_ml_feature_columns,
    structure_kwargs,
)
from v16.backtest.impulse_ml import MODEL_NAMES, evaluate_threshold_sweep, walk_forward_model_scores
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_signal_table
from v16.structure.filter import apply_structure_gate

THRESHOLDS = (0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70)
DEFAULT_MODELS = ("logreg", "gbc", "lgb", "hgb", "mlp", "rf", "xgb", "ens", "et")
ALL_MODELS = tuple(m for m in MODEL_NAMES if m != "lstm")


def _cfg(pre_close_pts: float) -> dict:
    c = copy.deepcopy(v16_config.MOMENTUM_BREAKOUT_PRECLOSE)
    c["entry_breakout"] = {
        **c.get("entry_breakout", {}),
        "max_pre_break_close_dist_pts": float(pre_close_pts),
    }
    return c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oos_start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--pre-close", type=float, default=10.0, help="max pt bar-before-close to trigger")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    cfg = _cfg(args.pre_close)
    is_cfg = cfg.get("impulse_stop", {})
    tp_r = float(is_cfg.get("tp_multiple", 3.0))
    horizon = int(is_cfg.get("horizon", 90))

    models = list(ALL_MODELS if args.all_models else (args.models or DEFAULT_MODELS))
    oos_start = pd.Timestamp(args.oos_start, tz="UTC")

    print("=" * 96)
    print(f"  breakout pre-close + ML  |  train {args.train_start}  OOS {args.oos_start} → {args.end}")
    print(f"  pre_break close ≤{args.pre_close:.0f}pt  |  next_open  |  with-trend  |  TP={tp_r}R  H={horizon}")
    print(f"  models ({len(models)}): {', '.join(models)}")
    print("=" * 96)

    t0 = time.time()
    df = load_gold_1m(args.train_start, args.end)
    feats = build_features(df)
    skw = structure_kwargs(cfg)
    if skw:
        feats = attach_structure_features(df, feats, **skw)

    labeled = build_labeled_set(df, cfg=cfg)
    print(
        f"\nLabeled: {len(labeled)}  WR={labeled['win'].mean()*100:.1f}%  "
        f"SL med={labeled['sl_pts'].median():.1f}pt"
    )
    if "entry_style" in labeled.columns:
        print(f"  entry styles: {labeled['entry_style'].value_counts().to_dict()}")

    feat_n = len(impulse_ml_feature_columns(feats, labeled))
    print(f"Features: {feat_n}")

    df_oos = df[df.index >= oos_start]
    signals_oos = build_signal_table(df_oos, cfg=cfg)
    gated_oos = apply_structure_gate(df_oos, signals_oos, cfg=cfg)
    fills_oos = build_resolved_entry_table(df_oos, gated_oos, cfg=cfg)

    mech = simulate_position_impulse_stop(
        df_oos,
        signals_oos,
        tp_multiple=tp_r,
        horizon=horizon,
        cfg=cfg,
    )
    mech_net = float(mech["pnl"].sum()) if not mech.empty else 0.0
    print(
        f"\nMechanical OOS: gate {len(gated_oos)}  fills {len(fills_oos)}  "
        f"tr {len(mech)}  WR={mech['win'].mean()*100:.1f}%  "
        f"net={mech_net:+.1f}  avg={mech['pnl'].mean():+.2f}"
    )

    best_rows: list[dict] = []
    all_sweeps: list[pd.DataFrame] = []

    for model in models:
        print(f"\n--- {model.upper()} ---")
        t1 = time.time()
        scores = walk_forward_model_scores(
            df,
            feats,
            labeled,
            model,
            seq_len=args.seq_len,
            prob_threshold=0.0,
            cfg=cfg,
        )
        if scores.empty:
            print("  no scores")
            continue
        scores_oos = scores[pd.to_datetime(scores["signal_ts"], utc=True) >= oos_start]
        sweep = evaluate_threshold_sweep(
            df_oos,
            pd.Series(0, index=df_oos.index),
            scores_oos,
            cfg,
            THRESHOLDS,
            exit_mode="impulse_stop",
            signal_table=signals_oos,
        )
        sweep["model"] = model
        all_sweeps.append(sweep)

        print(f"  {'prob':>5} {'sig':>5} {'tr':>5} {'WR%':>6} {'net':>10} {'avg':>7}")
        for _, r in sweep.iterrows():
            print(
                f"  {r['prob']:5.2f} {int(r['signals']):5d} {int(r['trades']):5d} "
                f"{r['wr']:6.1f} {r['net']:+10.1f} {r['avg']:+7.2f}"
            )
        best = sweep.sort_values("net", ascending=False).iloc[0]
        best_rows.append(
            {
                "pre_close_pts": args.pre_close,
                "model": model,
                **{k: best[k] for k in best.index},
                "train_secs": round(time.time() - t1, 1),
            }
        )

    if not all_sweeps:
        return

    tag = f"preclose{args.pre_close:.0f}"
    sweep_df = pd.concat(all_sweeps, ignore_index=True)
    best_df = pd.DataFrame(best_rows).sort_values("net", ascending=False)
    out_s = PROJECT_ROOT / f"runtime/v16_breakout_preclose_ml_{tag}_sweep.csv"
    out_b = PROJECT_ROOT / f"runtime/v16_breakout_preclose_ml_{tag}_best.csv"
    sweep_df.to_csv(out_s, index=False)
    best_df.to_csv(out_b, index=False)

    print("\n" + "=" * 96)
    print(f"Mechanical: net={mech_net:+.1f}  ({len(mech)} tr)")
    print("Best ML per model:")
    for _, r in best_df.iterrows():
        beat = "✓" if r["net"] > mech_net else " "
        print(
            f"  {beat} {r['model']:6s} p>={r['prob']:.2f}: {int(r['trades'])} tr  "
            f"WR={r['wr']:.1f}%  net={r['net']:+.1f}  avg={r['avg']:+.2f}"
        )
    top = best_df.iloc[0]
    print(
        f"\nTop: {top['model']} p>={top['prob']:.2f} → net={top['net']:+.1f}  "
        f"({'beats' if top['net'] > mech_net else 'below'} mechanical {mech_net:+.1f})"
    )
    print(f"Saved {out_s}")
    print(f"Saved {out_b}")
    print(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
