#!/usr/bin/env python3
"""
LSTM walk-forward filter on impulse_1m_15m (1m >=5pt bar in 15m slot).

Mechanical rule = pattern filter. LSTM sees last 30 x 1m bars at signal time.
Labels = scale-out outcome (+5/+10, SL20, H10).

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_lstm.py
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_lstm.py 2025-06-01 2026-06-25
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
from v16.backtest.impulse_ml import apply_ml_filter_to_sides
from v16.backtest.lstm_filter import SEQ_LEN, walk_forward_lstm_scores
from v16.backtest.position_sim import simulate_position_sided_scaleout
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_side_signals, count_signals


def _run_backtest(
    df: pd.DataFrame,
    sides: pd.Series,
    cfg: dict,
    *,
    label: str,
) -> pd.DataFrame:
    so = dict(cfg.get("scaleout", v16_config.EXIT_CONFIG))
    tdf = simulate_position_sided_scaleout(
        df,
        sides,
        scaleout_kw=so,
        same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
    )
    if tdf.empty:
        print(f"\n{label}: no trades")
        return tdf
    print(
        f"\n{label}: {len(tdf)} tr  WR={tdf['win'].mean()*100:.1f}%  "
        f"net={tdf['pnl'].sum():+.1f}  avg={tdf['pnl'].mean():+.2f}"
    )
    if "scaled_half" in tdf.columns:
        print(f"  +5 scale: {tdf['scaled_half'].mean()*100:.1f}%")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"    {reason:12s}: {len(g):4d}  PnL={g['pnl'].sum():+.1f}")
    return tdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oos_start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--prob", type=float, default=None, help="single threshold (else sweep)")
    args = parser.parse_args()

    cfg = v16_config.MOMENTUM_15M_HOLD
    so = cfg.get("scaleout", v16_config.EXIT_CONFIG)

    print("=" * 80)
    print(f"  impulse_1m_15m LSTM  |  train from {args.train_start}")
    print(f"  OOS eval: {args.oos_start} → {args.end}")
    print(f"  filter: first 1m |body|>={cfg['min_move_pts']} in 15m slot, enter after slot close")
    print(f"  LSTM seq: {SEQ_LEN} x 1m bars  |  label: scale-out +{so['first_scale_pnl']:.0f}/+{so['final_scale_pnl']:.0f}")
    print("=" * 80)

    t0 = time.time()
    df = load_gold_1m(args.train_start, args.end)
    feats = build_features(df)
    labeled = build_labeled_set(df, cfg=cfg)
    print(f"\nLabeled signals (full history): {len(labeled)}  "
          f"win rate {labeled['win'].mean()*100:.1f}%")

    oos_start = pd.Timestamp(args.oos_start, tz="UTC")
    df_oos = df[df.index >= oos_start]
    feats_oos = feats.loc[df_oos.index]
    base_sides = build_side_signals(df_oos, cfg=cfg)
    n = count_signals(base_sides)
    print(f"OOS mechanical signals: {n['total']} (L{n['long']} S{n['short']})")

    _run_backtest(df_oos, base_sides, cfg, label="Mechanical (OOS)")

    print("\nTraining LSTM walk-forward (monthly retrain)…")
    scores_all = walk_forward_lstm_scores(
        df, feats, labeled, prob_threshold=0.0
    )
    if scores_all.empty:
        print("No LSTM scores.")
        return

    scores_oos = scores_all[pd.to_datetime(scores_all["signal_ts"], utc=True) >= oos_start]
    print(f"LSTM OOS scored signals: {len(scores_oos)}")

    # Calibration: bucket probs vs actual win rate on OOS labeled rows
    cal = scores_oos.merge(
        labeled[["win", "pnl"]],
        left_on="signal_ts",
        right_index=True,
        how="left",
    )
    if not cal.empty:
        cal["bucket"] = pd.cut(cal["p_win"], bins=[0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0])
        print("\nOOS calibration (independent labels, not position sim):")
        for b, g in cal.groupby("bucket", observed=True):
            if len(g) < 5:
                continue
            print(f"  {b}: n={len(g):4d}  actual WR={g['win'].mean()*100:.1f}%  avg pnl={g['pnl'].mean():+.2f}")

    thresholds = [args.prob] if args.prob is not None else [0.50, 0.55, 0.58, 0.60, 0.65, 0.70]
    sweep_rows = []
    best_net = -1e9
    best_p = None
    best_tdf = None

    for p in thresholds:
        sub = scores_oos[scores_oos["p_win"] >= p]
        filtered = apply_ml_filter_to_sides(df_oos, base_sides, sub)
        kept = int((filtered != 0).sum())
        tdf = simulate_position_sided_scaleout(
            df_oos,
            filtered,
            scaleout_kw=dict(so),
            same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
        )
        net = float(tdf["pnl"].sum()) if not tdf.empty else 0.0
        wr = float(tdf["win"].mean() * 100) if not tdf.empty else 0.0
        avg = float(tdf["pnl"].mean()) if not tdf.empty else 0.0
        sweep_rows.append({"prob": p, "signals": kept, "trades": len(tdf), "wr": wr, "net": net, "avg": avg})
        print(f"\nLSTM p>={p:.2f}: {kept} signals -> {len(tdf)} trades  WR={wr:.1f}%  net={net:+.1f}  avg={avg:+.2f}")
        if net > best_net:
            best_net = net
            best_p = p
            best_tdf = tdf

    sweep = pd.DataFrame(sweep_rows)
    out_sweep = PROJECT_ROOT / "runtime" / "v16_momentum_15m_hold_lstm_sweep.csv"
    sweep.to_csv(out_sweep, index=False)

    if best_tdf is not None and not best_tdf.empty:
        out_tr = PROJECT_ROOT / "runtime" / "v16_momentum_15m_hold_lstm_trades.csv"
        best_tdf.to_csv(out_tr, index=False)
        print(f"\nBest threshold p>={best_p:.2f}: net={best_net:+.1f}")
        print(f"Sweep -> {out_sweep}")
        print(f"Trades -> {out_tr}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
