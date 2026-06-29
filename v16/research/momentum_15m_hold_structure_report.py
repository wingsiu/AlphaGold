#!/usr/bin/env python3
"""
15m zigzag structure × impulse_1m_15m (impulse-bar stop).

Reports OOS PnL buckets by:
  - trend alignment (signal side vs struct_trend)
  - pullback % of prior leg
  - leg age (15m bars since last swing)

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_structure_report.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_structure_report.py 2025-06-01 2026-06-25 --sweep-atr
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
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_signal_table
from v16.structure.swing_zigzag import build_structure_context, build_swing_table, build_15m_ohlc

TP_R = 3.0
HORIZON = 120
ATR_MULTS = (2.0, 2.5, 3.0, 3.5, 4.0)


def _attach_structure(
    tdf: pd.DataFrame,
    struct: pd.DataFrame,
) -> pd.DataFrame:
    out = tdf.copy()
    if out.empty or struct.empty:
        return out
    idx = pd.DatetimeIndex(out["signal_ts"] if "signal_ts" in out.columns else out.index)
    for c in struct.columns:
        out[c] = struct.reindex(idx)[c].values
    out["struct_aligned"] = out["side"] * out["struct_trend"]
    out["with_trend"] = out["struct_aligned"] > 0
    out["counter_trend"] = out["struct_aligned"] < 0
    out["range_regime"] = out["struct_trend"] == 0
    return out


def _bucket_report(tdf: pd.DataFrame, col: str, *, bins, labels: list[str], title: str) -> None:
    if tdf.empty or col not in tdf.columns:
        return
    sub = tdf.copy()
    sub["_bin"] = pd.cut(sub[col], bins=bins, labels=labels, include_lowest=True)
    print(f"\n{title}")
    rows = []
    for lab, g in sub.groupby("_bin", observed=True):
        if g.empty:
            continue
        rows.append(
            {
                "bucket": str(lab),
                "trades": len(g),
                "wr": round(g["win"].mean() * 100, 1),
                "net": round(g["pnl"].sum(), 1),
                "avg": round(g["pnl"].mean(), 2),
            }
        )
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


def _alignment_report(tdf: pd.DataFrame, *, label: str) -> None:
    if tdf.empty:
        print(f"\n{label}: no trades")
        return
    print(f"\n{label}: {len(tdf)} tr  net={tdf['pnl'].sum():+.1f}  WR={tdf['win'].mean()*100:.1f}%")
    for name, mask in [
        ("WITH trend (HH/HL or LH/LL)", tdf["with_trend"]),
        ("COUNTER trend", tdf["counter_trend"]),
        ("RANGE / mixed", tdf["range_regime"]),
    ]:
        g = tdf[mask]
        if g.empty:
            continue
        print(
            f"  {name:28s}: {len(g):4d} tr  WR={g['win'].mean()*100:.1f}%  "
            f"net={g['pnl'].sum():+.1f}  avg={g['pnl'].mean():+.2f}"
        )
    for side, sname in [(1, "LONG"), (-1, "SHORT")]:
        sub = tdf[tdf["side"] == side]
        if sub.empty:
            continue
        wt = sub[sub["with_trend"]]
        ct = sub[sub["counter_trend"]]
        print(f"  {sname} with-trend: {len(wt)} tr net={wt['pnl'].sum():+.1f}" if len(wt) else f"  {sname} with-trend: 0")
        print(f"  {sname} counter:    {len(ct)} tr net={ct['pnl'].sum():+.1f}" if len(ct) else f"  {sname} counter: 0")


def run_one(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: dict,
    *,
    atr_mult: float,
    horizon: int,
    tp_r: float,
) -> pd.DataFrame:
    struct = build_structure_context(df, atr_mult=atr_mult)
    tdf = simulate_position_impulse_stop(
        df,
        signals,
        tp_multiple=tp_r,
        horizon=horizon,
        min_sl_pts=1.0,
        max_sl_pts=80.0,
        same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
    )
    return _attach_structure(tdf, struct)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--atr-mult", type=float, default=3.0)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--tp-r", type=float, default=TP_R)
    parser.add_argument("--sweep-atr", action="store_true")
    args = parser.parse_args()

    cfg = v16_config.MOMENTUM_15M_HOLD
    print("=" * 80)
    print(f"  impulse × 15m structure  |  {args.start} → {args.end}")
    print(f"  zigzag ATR mult={args.atr_mult}  TP={args.tp_r}R  H={args.horizon}")
    print("=" * 80)

    df = load_gold_1m(args.start, args.end)
    signals = build_signal_table(df, cfg=cfg)
    print(f"\nImpulse signals: {len(signals)}")

    ohlc = build_15m_ohlc(df)
    swings = build_swing_table(ohlc, atr_mult=args.atr_mult)
    print(f"15m swings (ATR×{args.atr_mult}): {len(swings)}")
    if not swings.empty:
        trend_counts = swings["trend"].value_counts().sort_index()
        print(f"  trend at swings: up={trend_counts.get(1,0)} down={trend_counts.get(-1,0)} range={trend_counts.get(0,0)}")

    if args.sweep_atr:
        rows = []
        for m in ATR_MULTS:
            tdf = run_one(df, signals, cfg, atr_mult=m, horizon=args.horizon, tp_r=args.tp_r)
            wt = tdf[tdf["with_trend"]]
            ct = tdf[tdf["counter_trend"]]
            rows.append(
                {
                    "atr_mult": m,
                    "trades": len(tdf),
                    "net_all": round(tdf["pnl"].sum(), 1),
                    "net_with": round(wt["pnl"].sum(), 1) if len(wt) else 0.0,
                    "net_counter": round(ct["pnl"].sum(), 1) if len(ct) else 0.0,
                    "wr_with": round(wt["win"].mean() * 100, 1) if len(wt) else 0.0,
                }
            )
        sweep = pd.DataFrame(rows)
        out = PROJECT_ROOT / "runtime" / "v16_momentum_structure_atr_sweep.csv"
        sweep.to_csv(out, index=False)
        print("\nATR mult sweep (with-trend filter potential):")
        print(sweep.to_string(index=False))
        print(f"\nSaved -> {out}")
        best = sweep.sort_values("net_with", ascending=False).iloc[0]
        args.atr_mult = float(best["atr_mult"])
        print(f"\nBest net_with at atr_mult={args.atr_mult}")

    tdf = run_one(df, signals, cfg, atr_mult=args.atr_mult, horizon=args.horizon, tp_r=args.tp_r)
    _alignment_report(tdf, label="All trades")

    _bucket_report(
        tdf,
        "struct_pullback_pct",
        bins=[-0.001, 0.25, 0.5, 0.75, 1.01, 999],
        labels=["0-25%", "25-50%", "50-75%", "75-100%", ">100%"],
        title="Pullback vs prior leg (15m structure)",
    )
    _bucket_report(
        tdf,
        "struct_leg_age_15m",
        bins=[-1, 2, 5, 10, 20, 999],
        labels=["0-2", "3-5", "6-10", "11-20", "20+"],
        title="Leg age (15m bars since last swing)",
    )

    # Filter: with-trend only
    wt = tdf[tdf["with_trend"]]
    _alignment_report(wt, label="WITH-trend only (filter)")

    out_tr = PROJECT_ROOT / "runtime" / "v16_momentum_structure_trades.csv"
    tdf.to_csv(out_tr, index=False)
    print(f"\nSaved trades -> {out_tr}")

    out_sw = PROJECT_ROOT / "runtime" / "v16_momentum_15m_swings.csv"
    swings.to_csv(out_sw)
    print(f"Saved swings -> {out_sw}")


if __name__ == "__main__":
    main()
