#!/usr/bin/env python3
"""Full stat report for V16 winner best config (ET ML p>=0.50, R=3, struct-exit)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features
from v16.backtest.impulse_entry import build_resolved_entry_table
from v16.backtest.impulse_features import attach_structure_features, structure_kwargs
from v16.backtest.impulse_ml import filter_signal_table, walk_forward_model_scores
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_signal_table
from v16.structure.filter import apply_structure_gate

OOS_START = "2025-06-01"
END = "2026-06-25"
TRAIN_START = "2024-01-01"
ML_MODEL = "et"
ML_PROB = 0.50


def _risk_stats(tdf: pd.DataFrame) -> dict:
    eq = tdf["pnl"].cumsum()
    dd = eq - eq.cummax()
    wins = tdf[tdf["pnl"] > 0]["pnl"]
    losses = tdf[tdf["pnl"] <= 0]["pnl"]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    return {
        "max_dd": float(dd.min()),
        "profit_factor": gross_win / gross_loss if gross_loss else float("inf"),
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "payoff": abs(float(wins.mean()) / float(losses.mean())) if len(losses) and losses.mean() != 0 else 0.0,
        "expectancy": float(tdf["pnl"].mean()),
    }


def _hold_minutes(tdf: pd.DataFrame) -> pd.Series:
    return (pd.to_datetime(tdf["exit_time"]) - pd.to_datetime(tdf["entry_time"])).dt.total_seconds() / 60.0


def _monthly(tdf: pd.DataFrame) -> pd.DataFrame:
    t = tdf.copy()
    t["month"] = pd.to_datetime(t["entry_time"]).dt.to_period("M").astype(str)
    return (
        t.groupby("month")
        .agg(trades=("pnl", "count"), wr=("win", "mean"), net=("pnl", "sum"), avg=("pnl", "mean"))
        .assign(wr=lambda x: (x["wr"] * 100).round(1), net=lambda x: x["net"].round(1), avg=lambda x: x["avg"].round(2))
    )


def _streaks(tdf: pd.DataFrame) -> tuple[int, int]:
    w = (tdf["pnl"] > 0).astype(int).tolist()
    max_w = max_l = cw = cl = 0
    for x in w:
        if x:
            cw += 1
            cl = 0
            max_w = max(max_w, cw)
        else:
            cl += 1
            cw = 0
            max_l = max(max_l, cl)
    return max_w, max_l


def main() -> None:
    cfg = copy.deepcopy(v16_config.MOMENTUM_V16_WINNER_PRECLOSE)
    is_cfg = cfg.get("impulse_stop", {})
    oos_start = pd.Timestamp(OOS_START, tz="UTC")

    print("=" * 72)
    print("  V16 WINNER — FULL STAT REPORT")
    print("=" * 72)
    print("\nCONFIG")
    print(f"  Pattern:      impulse_1m_15m / pre-close breakout ≤10pt")
    print(f"  Entry:        next_open after breakout, with-trend structure gate")
    print(f"  SL:           impulse 1m bar H/L")
    print(f"  TP:           R={is_cfg.get('tp_multiple', 3.0)}  H={is_cfg.get('horizon', 120)}")
    print(f"  Struct exit:  always on 15m trend flip")
    print(f"  ML:           {ML_MODEL.upper()}  p>={ML_PROB:.2f}")
    print(f"  OOS window:   {OOS_START} → {END}")
    print(f"  Train from:   {TRAIN_START}")

    df = load_gold_1m(TRAIN_START, END)
    df_oos = df[df.index >= oos_start]
    signals = build_signal_table(df_oos, cfg=cfg)
    gated = apply_structure_gate(df_oos, signals, cfg=cfg)
    fills = build_resolved_entry_table(df_oos, gated, cfg=cfg)

    labeled = build_labeled_set(df, cfg=cfg)
    feats = build_features(df)
    skw = structure_kwargs(cfg)
    if skw:
        feats = attach_structure_features(df, feats, **skw)

    scores = walk_forward_model_scores(df, feats, labeled, ML_MODEL, seq_len=30, prob_threshold=0.0, cfg=cfg)
    scores_oos = scores[pd.to_datetime(scores["signal_ts"], utc=True) >= oos_start]
    passed = scores_oos[scores_oos["p_win"] >= ML_PROB]
    filt = filter_signal_table(signals, passed)
    tdf = simulate_position_impulse_stop(df_oos, filt, cfg=cfg)
    tdf = tdf.copy()
    tdf["hold_min"] = _hold_minutes(tdf)

    out_trades = PROJECT_ROOT / "runtime" / "v16_winner_preclose_et_trades.csv"
    tdf.to_csv(out_trades, index=False)

    print("\nPIPELINE (OOS)")
    print(f"  Raw signals:          {len(signals):5d}")
    print(f"  Structure gate:       {len(gated):5d}")
    print(f"  Resolved fills:       {len(fills):5d}")
    print(f"  ML scored (OOS):      {len(scores_oos):5d}")
    print(f"  ML passed p>={ML_PROB:.2f}:     {len(passed):5d}")
    print(f"  Traded:               {len(tdf):5d}")

    r = _risk_stats(tdf)
    max_w, max_l = _streaks(tdf)
    days = (pd.Timestamp(END) - pd.Timestamp(OOS_START)).days
    months = days / 30.44

    print("\nPERFORMANCE")
    print(f"  Trades:               {len(tdf)}")
    print(f"  Win rate:             {tdf['win'].mean()*100:.1f}%  ({int(tdf['win'].sum())}W / {int((~tdf['win']).sum())}L)")
    print(f"  Net PnL:              {tdf['pnl'].sum():+.1f} pt")
    print(f"  Gross win / loss:     {r['gross_win']:+.1f} / -{r['gross_loss']:.1f}")
    print(f"  Avg / median trade:   {tdf['pnl'].mean():+.2f} / {tdf['pnl'].median():+.2f} pt")
    print(f"  Avg win / avg loss:   {r['avg_win']:+.2f} / {r['avg_loss']:+.2f} pt")
    print(f"  Payoff ratio:         {r['payoff']:.2f}")
    print(f"  Profit factor:        {r['profit_factor']:.2f}")
    print(f"  Max drawdown:         {r['max_dd']:+.1f} pt")
    print(f"  Best / worst trade:   {tdf['pnl'].max():+.1f} / {tdf['pnl'].min():+.1f} pt")
    print(f"  Max win / loss streak:{max_w} / {max_l}")
    print(f"  Trades / month:       {len(tdf)/months:.1f}")
    print(f"  PnL / month:          {tdf['pnl'].sum()/months:+.1f} pt")

    print("\nHOLD TIME (minutes)")
    print(f"  Mean / median:        {tdf['hold_min'].mean():.1f} / {tdf['hold_min'].median():.1f}")
    print(f"  P25 / P75:            {tdf['hold_min'].quantile(0.25):.1f} / {tdf['hold_min'].quantile(0.75):.1f}")

    if "sl" in tdf.columns:
        print("\nSL / TP (pts at entry)")
        print(f"  SL mean / median:     {tdf['sl'].mean():.1f} / {tdf['sl'].median():.1f}")
        print(f"  TP mean / median:     {tdf['tp'].mean():.1f} / {tdf['tp'].median():.1f}")
        print(f"  R at entry (TP/SL):   {(tdf['tp']/tdf['sl']).mean():.2f}")

    print("\nEXIT REASONS")
    print(f"  {'reason':18s} {'n':>5} {'%':>6} {'net':>9} {'avg':>8} {'WR%':>6}")
    for reason, g in tdf.groupby("exit_reason"):
        print(
            f"  {reason:18s} {len(g):5d} {len(g)/len(tdf)*100:5.1f}% "
            f"{g['pnl'].sum():+9.1f} {g['pnl'].mean():+8.2f} {g['win'].mean()*100:5.1f}%"
        )

    print("\nBY SIDE")
    print(f"  {'side':6s} {'n':>5} {'WR%':>6} {'net':>9} {'avg':>8}")
    for side, name in [(1, "LONG"), (-1, "SHORT")]:
        sub = tdf[tdf["side"] == side]
        if sub.empty:
            continue
        print(
            f"  {name:6s} {len(sub):5d} {sub['win'].mean()*100:5.1f}% "
            f"{sub['pnl'].sum():+9.1f} {sub['pnl'].mean():+8.2f}"
        )

    print("\nBY ENTRY STYLE")
    if "entry_style" in tdf.columns:
        for style, g in tdf.groupby("entry_style"):
            print(f"  {style:12s} {len(g):4d}  WR={g['win'].mean()*100:.1f}%  net={g['pnl'].sum():+.1f}")

    print("\nMONTHLY")
    print(f"  {'month':8s} {'tr':>4} {'WR%':>6} {'net':>9} {'avg':>7}")
    for month, row in _monthly(tdf).iterrows():
        print(f"  {month:8s} {int(row['trades']):4d} {row['wr']:6.1f} {row['net']:+9.1f} {row['avg']:+7.2f}")

    print("\nTOP 5 WINS")
    for _, row in tdf.nlargest(5, "pnl").iterrows():
        print(
            f"  {row['entry_time']}  {('LONG' if row['side']==1 else 'SHORT'):5s}  "
            f"{row['pnl']:+.1f}pt  {row['exit_reason']}  hold={row['hold_min']:.0f}m"
        )

    print("\nTOP 5 LOSSES")
    for _, row in tdf.nsmallest(5, "pnl").iterrows():
        print(
            f"  {row['entry_time']}  {('LONG' if row['side']==1 else 'SHORT'):5s}  "
            f"{row['pnl']:+.1f}pt  {row['exit_reason']}  hold={row['hold_min']:.0f}m"
        )

    print(f"\nTrades saved → {out_trades}")


if __name__ == "__main__":
    main()
