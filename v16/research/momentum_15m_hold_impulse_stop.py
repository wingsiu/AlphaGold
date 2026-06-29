#!/usr/bin/env python3
"""
impulse_1m_15m — SL at impulse bar low/high, TP = 3 × stop distance.

Long:  SL dist = entry - impulse_low_bid,  stop @ impulse low
Short: SL dist = impulse_high_ask - entry, stop @ impulse high

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_impulse_stop.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_impulse_stop.py 2025-06-01 2026-06-25 --sweep-h
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

TP_R = 3.0
HORIZONS = (30, 45, 60, 90, 120)


def _report(tdf: pd.DataFrame, *, label: str) -> None:
    if tdf.empty:
        print(f"\n{label}: no trades")
        return
    print(
        f"\n{label}: {len(tdf)} tr  WR={tdf['win'].mean()*100:.1f}%  "
        f"net={tdf['pnl'].sum():+.1f}  avg={tdf['pnl'].mean():+.2f}"
    )
    print(f"  SL dist median={tdf['sl'].median():.1f}pt  TP dist median={tdf['tp'].median():.1f}pt")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"    {reason:12s}: {len(g):4d}  PnL={g['pnl'].sum():+.1f}")
    for side, name in [(1, "LONG"), (-1, "SHORT")]:
        sub = tdf[tdf["side"] == side]
        if not sub.empty:
            print(
                f"  {name}: {len(sub)} tr  WR={sub['win'].mean()*100:.1f}%  "
                f"net={sub['pnl'].sum():+.1f}  avg SL={sub['sl'].mean():.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--tp-r", type=float, default=TP_R)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--sweep-h", action="store_true")
    parser.add_argument("--min-sl", type=float, default=1.0)
    parser.add_argument("--max-sl", type=float, default=80.0)
    args = parser.parse_args()

    cfg = v16_config.MOMENTUM_15M_HOLD
    print("=" * 80)
    print(f"  impulse stop @ bar H/L  |  {args.start} → {args.end}")
    print(f"  SL = entry − impulse low (long) / impulse high − entry (short)")
    print(f"  TP = {args.tp_r:.0f} × SL distance")
    print("=" * 80)

    df = load_gold_1m(args.start, args.end)
    signals = build_signal_table(df, cfg=cfg)
    print(f"\nImpulse signals: {len(signals)}")

    if args.sweep_h:
        rows = []
        for h in HORIZONS:
            tdf = simulate_position_impulse_stop(
                df,
                signals,
                tp_multiple=args.tp_r,
                horizon=h,
                min_sl_pts=args.min_sl,
                max_sl_pts=args.max_sl,
                same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
            )
            net = float(tdf["pnl"].sum()) if not tdf.empty else 0.0
            rows.append(
                {
                    "horizon": h,
                    "trades": len(tdf),
                    "wr": round(float(tdf["win"].mean() * 100), 1) if not tdf.empty else 0.0,
                    "net": round(net, 1),
                    "avg": round(float(tdf["pnl"].mean()), 2) if not tdf.empty else 0.0,
                    "sl_med": round(float(tdf["sl"].median()), 1) if not tdf.empty else 0.0,
                }
            )
        sweep = pd.DataFrame(rows)
        out = PROJECT_ROOT / "runtime" / "v16_momentum_impulse_stop_h_sweep.csv"
        sweep.to_csv(out, index=False)
        print("\nHorizon sweep:")
        print(sweep.to_string(index=False))
        print(f"\nSaved -> {out}")
        best = sweep.sort_values("net", ascending=False).iloc[0]
        tdf = simulate_position_impulse_stop(
            df, signals, tp_multiple=args.tp_r, horizon=int(best["horizon"]),
            min_sl_pts=args.min_sl, max_sl_pts=args.max_sl,
            same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
        )
        _report(tdf, label=f"Best H={int(best['horizon'])}")
    else:
        tdf = simulate_position_impulse_stop(
            df,
            signals,
            tp_multiple=args.tp_r,
            horizon=args.horizon,
            min_sl_pts=args.min_sl,
            max_sl_pts=args.max_sl,
            same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
        )
        _report(tdf, label=f"H={args.horizon}")

    if not args.sweep_h:
        out = PROJECT_ROOT / "runtime" / "v16_momentum_impulse_stop_trades.csv"
        if not tdf.empty:
            tdf.to_csv(out, index=False)
            print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
