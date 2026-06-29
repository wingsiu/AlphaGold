#!/usr/bin/env python3
"""
Full walk-forward backtest report — breakout entry at impulse H/L (no jargon).

Entry (plain):
  After a 3pt+ 1m impulse in a 15m slot, wait for price to break the impulse bar
  high (long) or low (short) in the next slot. Fill price = that break level.

Exit:
  Stop = impulse bar low (long) / high (short).  TP = 3 × stop distance.  Max hold 120m.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_breakout_wf_report.py
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_breakout_wf_report.py 2025-06-01 2026-06-25
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features
from v16.backtest.impulse_features import attach_structure_features, structure_kwargs
from v16.backtest.impulse_ml import walk_forward_model_scores
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_signal_table
from v16.structure.filter import apply_structure_gate

TRAIN_START = "2024-01-01"
ML_MODEL = "lgb"
ML_PROB = 0.50


def _stats(tdf: pd.DataFrame, *, label: str) -> dict:
    if tdf.empty:
        print(f"\n{label}: no trades")
        return {}
    tdf = tdf.copy()
    tdf["month"] = pd.to_datetime(tdf["entry_time"], utc=True).dt.to_period("M").astype(str)
    net = float(tdf["pnl"].sum())
    wins = int((tdf["pnl"] > 0).sum())
    losses = int((tdf["pnl"] <= 0).sum())
    cum = tdf["pnl"].cumsum()
    dd = float((cum - cum.cummax()).min())

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  Trades:      {len(tdf)}")
    print(f"  Win / Loss:  {wins} / {losses}  ({wins/len(tdf)*100:.1f}% WR)")
    print(f"  Net PnL:     {net:+.1f} pt")
    print(f"  Avg / trade: {tdf['pnl'].mean():+.2f} pt")
    print(f"  Median:      {tdf['pnl'].median():+.2f} pt")
    print(f"  Best / Worst:{tdf['pnl'].max():+.1f} / {tdf['pnl'].min():+.1f} pt")
    print(f"  Max DD:      {dd:+.1f} pt (cumsum)")
    print(f"  SL median:   {tdf['sl'].median():.1f} pt")

    for reason, g in tdf.groupby("exit_reason"):
        print(f"    {reason:12s}: {len(g):4d}  net={g['pnl'].sum():+.1f}  WR={(g['win'].mean()*100):.1f}%")

    for side, name in [(1, "LONG"), (-1, "SHORT")]:
        g = tdf[tdf["side"] == side]
        if len(g):
            print(f"  {name}: {len(g)} tr  net={g['pnl'].sum():+.1f}  WR={g['win'].mean()*100:.1f}%")

    monthly = tdf.groupby("month").agg(
        trades=("pnl", "count"),
        net=("pnl", "sum"),
        wr=("win", "mean"),
    )
    monthly["wr"] = (monthly["wr"] * 100).round(1)
    monthly["net"] = monthly["net"].round(1)
    neg = int((monthly["net"] < 0).sum())
    print(f"\n  Monthly ({len(monthly)} months, {neg} losing):")
    print(monthly.to_string())
    return {"label": label, "trades": len(tdf), "wr": wins / len(tdf) * 100, "net": net, "avg": float(tdf["pnl"].mean()), "max_dd": dd}


def _ml_oos_trades(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    labeled: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: dict,
    *,
    oos_start: pd.Timestamp,
    prob: float,
) -> pd.DataFrame:
    scores = walk_forward_model_scores(df, feats, labeled, ML_MODEL, prob_threshold=0.0, cfg=cfg)
    if scores.empty:
        return pd.DataFrame()
    scores = scores[pd.to_datetime(scores["signal_ts"], utc=True) >= oos_start]
    scores = scores[scores["p_win"] >= prob]
    keep = signals.index.intersection(pd.to_datetime(scores["signal_ts"], utc=True))
    filt = signals.loc[keep]
    is_cfg = cfg.get("impulse_stop", {})
    return simulate_position_impulse_stop(
        df[df.index >= oos_start],
        filt,
        tp_multiple=float(is_cfg.get("tp_multiple", 3.0)),
        horizon=int(is_cfg.get("horizon", 120)),
        cfg=cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oos_start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--with-trend-only", action="store_true", help="15m structure gate")
    parser.add_argument("--trigger-fill", action="store_true", help="fill at break level (optimistic)")
    args = parser.parse_args()

    oos_start = pd.Timestamp(args.oos_start, tz="UTC")
    if args.trigger_fill and args.with_trend_only:
        cfg = copy.deepcopy(v16_config.MOMENTUM_BREAKOUT_IDEAL_ML)
    elif args.trigger_fill:
        cfg = copy.deepcopy(v16_config.MOMENTUM_BREAKOUT_ALL_IDEAL)
    elif args.with_trend_only:
        cfg = copy.deepcopy(v16_config.MOMENTUM_BREAKOUT_ML)
    else:
        cfg = copy.deepcopy(v16_config.MOMENTUM_BREAKOUT_NEXT_OPEN)

    fill_type = cfg.get("entry_breakout", {}).get("fill", "next_open")

    print("=" * 72)
    print("  BREAKOUT IMPULSE BACKTEST — full stats")
    print("=" * 72)
    print("""
  SETUP
  -----
  Signal:  first 1m bar |body| >= 3pt in a 15m slot
  Entry:   price breaks impulse bar HIGH (long) or LOW (short)
           in the next 15m slot → enter """
          + ("at break level (optimistic)" if fill_type == "trigger" else "at NEXT 1m bar OPEN after break")
          + """
  Exit:    stop = impulse bar low/high  |  TP = 3× risk  |  timeout 120m
  Filter:  """
          + ("15m trend must match trade direction" if args.with_trend_only else "none (all impulses)")
          + """
  Data:    """
          + f"{TRAIN_START} → {args.end}  |  OOS from {args.oos_start}"
    )

    df = load_gold_1m(TRAIN_START, args.end)
    signals = build_signal_table(df, cfg=cfg)
    gated = apply_structure_gate(df, signals, cfg=cfg)

    is_cfg = cfg.get("impulse_stop", {})
    kw = dict(
        tp_multiple=float(is_cfg.get("tp_multiple", 3.0)),
        horizon=int(is_cfg.get("horizon", 120)),
        cfg=cfg,
    )

    # Full sample mechanical
    mech_all = simulate_position_impulse_stop(df, signals, **kw)
    _stats(mech_all, label=f"MECHANICAL — full sample {TRAIN_START} → {args.end}")

    # OOS mechanical
    df_oos = df[df.index >= oos_start]
    sig_oos = build_signal_table(df_oos, cfg=cfg)
    mech_oos = simulate_position_impulse_stop(df_oos, sig_oos, **kw)
    _stats(mech_oos, label=f"MECHANICAL — OOS {args.oos_start} → {args.end}")

    # WF ML
    print(f"\n  Loading features + labels for walk-forward ML ({ML_MODEL})…")
    feats = build_features(df)
    skw = structure_kwargs(cfg)
    if skw:
        feats = attach_structure_features(df, feats, **skw)
    labeled = build_labeled_set(df, cfg=cfg)
    print(f"  Train labels: {len(labeled)} trades  WR={labeled['win'].mean()*100:.1f}%")

    ml_all = _ml_oos_trades(df, feats, labeled, signals, cfg, oos_start=pd.Timestamp(TRAIN_START, tz="UTC"), prob=ML_PROB)
    _stats(ml_all, label=f"ML {ML_MODEL.upper()} p>={ML_PROB} — full WF from {TRAIN_START}")

    ml_oos = _ml_oos_trades(df, feats, labeled, signals, cfg, oos_start=oos_start, prob=ML_PROB)
    _stats(ml_oos, label=f"ML {ML_MODEL.upper()} p>={ML_PROB} — OOS {args.oos_start} → {args.end}")

    # Save trades
    tag = "with_trend" if args.with_trend_only else "all"
    out = PROJECT_ROOT / f"runtime/v16_breakout_wf_{tag}_mech_oos.csv"
    mech_oos.to_csv(out, index=False)
    out_ml = PROJECT_ROOT / f"runtime/v16_breakout_wf_{tag}_ml_oos.csv"
    ml_oos.to_csv(out_ml, index=False)
    print(f"\nSaved mechanical OOS trades -> {out}")
    print(f"Saved ML OOS trades         -> {out_ml}")


if __name__ == "__main__":
    main()
