#!/usr/bin/env python3
"""
ML filter on impulse-bar stop (SL @ impulse H/L, TP = ratio × SL).

Default: min_move=3pt body (change sweep winner), R=3, H=120.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_impulse_stop_ml.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_impulse_stop_ml.py 2025-06-01 2026-06-25 --min-move 5
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_impulse_stop_ml.py 2025-06-01 2026-06-25 --all-models
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
from v16.backtest.impulse_features import (
    attach_structure_features,
    impulse_ml_feature_columns,
    structure_kwargs,
)
from v16.backtest.impulse_ml import (
    MODEL_NAMES,
    evaluate_threshold_sweep,
    walk_forward_model_scores,
)
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_signal_table

THRESHOLDS = (0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70)
# Best performers from prior impulse-stop ML research
DEFAULT_MODELS = ("logreg", "gbc", "lgb", "mlp", "hgb", "rf", "xgb", "ens", "et", "lstm")
ALL_MODELS = tuple(MODEL_NAMES)


def _cfg_ml(
    *,
    min_move: float,
    change_mode: str,
    tp_r: float,
    horizon: int,
) -> dict:
    c = copy.deepcopy(v16_config.MOMENTUM_15M_HOLD)
    c["min_move_pts"] = min_move
    c["change_mode"] = change_mode
    c["ml_label_mode"] = "impulse_stop"
    c["impulse_stop"] = {
        "tp_multiple": tp_r,
        "horizon": horizon,
        "min_sl_pts": 1.0,
        "max_sl_pts": 80.0,
    }
    return c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oos_start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--min-move", type=float, default=None)
    parser.add_argument("--change-mode", default=None, choices=["body", "range", "either"])
    parser.add_argument("--tp-r", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=30, help="sequence window for tabular agg + default LSTM")
    parser.add_argument("--lstm-seq-len", type=int, default=None, help="LSTM window (default: --seq-len)")
    args = parser.parse_args()

    base = v16_config.MOMENTUM_15M_HOLD
    is_cfg = base.get("impulse_stop", {})
    min_move = float(args.min_move if args.min_move is not None else base.get("min_move_pts", 3.0))
    change_mode = args.change_mode or base.get("change_mode", "body")
    tp_r = float(args.tp_r if args.tp_r is not None else is_cfg.get("tp_multiple", 3.0))
    horizon = int(args.horizon if args.horizon is not None else is_cfg.get("horizon", 120))

    if args.all_models:
        models = list(ALL_MODELS)
    elif args.models:
        models = [m.lower() for m in args.models if m.lower() in MODEL_NAMES]
    else:
        models = list(DEFAULT_MODELS)
    if not models:
        models = list(DEFAULT_MODELS)

    cfg = _cfg_ml(min_move=min_move, change_mode=change_mode, tp_r=tp_r, horizon=horizon)
    oos_start = pd.Timestamp(args.oos_start, tz="UTC")

    print("=" * 92)
    print(f"  impulse stop + ML  |  train {args.train_start}  OOS {args.oos_start} → {args.end}")
    print(f"  filter: 1m {change_mode} >={min_move:.1f}pt  |  TP={tp_r:.1f}×SL  H={horizon}")
    print(f"  models ({len(models)}): {', '.join(models)}")
    print(f"  tabular seq={args.seq_len}  lstm seq={args.lstm_seq_len or args.seq_len}")
    print("=" * 92)

    t0 = time.time()
    df = load_gold_1m(args.train_start, args.end)
    feats = build_features(df)
    skw = structure_kwargs(cfg)
    if skw:
        feats = attach_structure_features(df, feats, **skw)
    labeled = build_labeled_set(df, cfg=cfg)
    print(f"\nLabeled: {len(labeled)}  WR={labeled['win'].mean()*100:.1f}%  "
          f"SL med={labeled['sl_pts'].median():.1f}pt")

    feat_names = impulse_ml_feature_columns(feats, labeled)
    struct_n = sum(1 for c in feat_names if c.startswith("struct_"))
    print(
        f"Features: {len(feat_names)} "
        f"(v16 + v15 dip + impulse + {struct_n} structure)"
    )

    df_oos = df[df.index >= oos_start]
    signals_oos = build_signal_table(df_oos, cfg=cfg)
    base_sides = pd.Series(0, index=df_oos.index, dtype=int)
    base_sides.loc[signals_oos.index] = signals_oos["side"].astype(int)

    mech = simulate_position_impulse_stop(
        df_oos,
        signals_oos,
        tp_multiple=tp_r,
        horizon=horizon,
        min_sl_pts=1.0,
        max_sl_pts=80.0,
        same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
        cfg=cfg,
    )
    mech_net = float(mech["pnl"].sum()) if not mech.empty else 0.0
    print(
        f"\nMechanical OOS: {len(signals_oos)} sig → {len(mech)} tr  "
        f"WR={mech['win'].mean()*100:.1f}%  net={mech_net:+.1f}  avg={mech['pnl'].mean():+.2f}"
    )

    all_sweeps: list[pd.DataFrame] = []
    best_rows: list[dict] = []

    for model in models:
        print(f"\n--- {model.upper()} ---")
        t1 = time.time()
        scores = walk_forward_model_scores(
            df,
            feats,
            labeled,
            model,
            seq_len=args.seq_len,
            lstm_seq_len=args.lstm_seq_len,
            prob_threshold=0.0,
            cfg=cfg,
        )
        if scores.empty:
            print("  no scores")
            continue
        scores_oos = scores[pd.to_datetime(scores["signal_ts"], utc=True) >= oos_start]
        sweep = evaluate_threshold_sweep(
            df_oos,
            base_sides,
            scores_oos,
            cfg,
            THRESHOLDS,
            exit_mode="impulse_stop",
            signal_table=signals_oos,
        )
        sweep["model"] = model
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
                "min_move_pts": min_move,
                "change_mode": change_mode,
                "seq_len": args.seq_len,
                "lstm_seq_len": args.lstm_seq_len or args.seq_len,
                "model": model,
                **{k: best[k] for k in best.index},
                "train_secs": round(time.time() - t1, 1),
            }
        )

    if not all_sweeps:
        return

    sweep_df = pd.concat(all_sweeps, ignore_index=True)
    best_df = pd.DataFrame(best_rows).sort_values("net", ascending=False)
    tag = f"mv{min_move:.0f}_{change_mode}"
    out_sweep = PROJECT_ROOT / f"runtime/v16_momentum_impulse_stop_ml_{tag}_sweep.csv"
    out_best = PROJECT_ROOT / f"runtime/v16_momentum_impulse_stop_ml_{tag}_best.csv"
    sweep_df.to_csv(out_sweep, index=False)
    best_df.to_csv(out_best, index=False)

    print("\n" + "=" * 92)
    print(f"Mechanical: net={mech_net:+.1f}  ({len(mech)} tr)")
    print("Best ML per model:")
    for _, r in best_df.iterrows():
        beat = "✓" if r["net"] > mech_net else " "
        print(
            f"  {beat} {r['model']:6s} p>={r['prob']:.2f}: {int(r['trades'])} tr  "
            f"WR={r['wr']:.1f}%  net={r['net']:+.1f}  avg={r['avg']:+.2f}"
        )
    top = best_df.iloc[0]
    if top["net"] > mech_net:
        print(f"\nML beats mechanical: {top['model']} p>={top['prob']:.2f} → net={top['net']:+.1f}")
    else:
        print(f"\nNo ML combo beat mechanical (+{mech_net:.1f})")
    print(f"\nSaved {out_sweep}")
    print(f"Saved {out_best}")
    print(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
