#!/usr/bin/env python3
"""
Sweep TP/SL ratio (tp_multiple) × horizon for impulse-bar stop exit.

SL distance = entry to impulse bar low (long) / high (short).
TP distance = tp_multiple × SL distance.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_impulse_ratio_sweep.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_impulse_ratio_sweep.py 2025-06-01 2026-06-25 --quick
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
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

GRIDS = {
    "quick": {
        "tp_r": [2.0, 2.5, 3.0, 4.0, 5.0],
        "horizon": [45, 60, 90, 120],
    },
    "full": {
        "tp_r": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
        "horizon": [30, 45, 60, 90, 120, 150],
    },
}


def _stats(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {
            "trades": 0,
            "wr": 0.0,
            "net": 0.0,
            "avg": 0.0,
            "max_dd": 0.0,
            "sl_med": 0.0,
            "tp_med": 0.0,
            "tp_hit": 0,
            "sl_hit": 0,
            "timeout": 0,
        }
    eq = tdf["pnl"].cumsum()
    reasons = tdf["exit_reason"].value_counts()
    return {
        "trades": len(tdf),
        "wr": round(float((tdf["pnl"] > 0).mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
        "max_dd": round(float((eq - eq.cummax()).min()), 1),
        "sl_med": round(float(tdf["sl"].median()), 1),
        "tp_med": round(float(tdf["tp"].median()), 1),
        "tp_hit": int(reasons.get("target_hit", 0)),
        "sl_hit": int(reasons.get("stop_loss", 0)),
        "timeout": int(reasons.get("timeout", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--min-sl", type=float, default=1.0)
    parser.add_argument("--max-sl", type=float, default=80.0)
    args = parser.parse_args()

    grid = GRIDS["quick" if args.quick else "full"]
    cfg = v16_config.MOMENTUM_15M_HOLD
    combos = list(itertools.product(grid["tp_r"], grid["horizon"]))

    print("=" * 96)
    print(f"  impulse stop TP/SL ratio × horizon  |  {args.start} → {args.end}")
    print(f"  SL @ impulse H/L  |  TP = ratio × SL  |  combos={len(combos)}")
    print("=" * 96)

    df = load_gold_1m(args.start, args.end)
    signals = build_signal_table(df, cfg=cfg)
    print(f"Signals: {len(signals)}")

    rows = []
    t0 = time.time()
    refresh = cfg.get("same_dir_refresh", "entry")
    for tp_r, h in combos:
        tdf = simulate_position_impulse_stop(
            df,
            signals,
            tp_multiple=float(tp_r),
            horizon=int(h),
            min_sl_pts=args.min_sl,
            max_sl_pts=args.max_sl,
            same_dir_refresh=refresh,
        )
        rows.append({"tp_r": tp_r, "horizon": h, **_stats(tdf)})

    elapsed = time.time() - t0
    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "runtime" / "v16_momentum_impulse_stop_ratio_h_sweep.csv"
    out.to_csv(out_path, index=False)
    print(f"\nDone in {elapsed:.1f}s  ->  {out_path}")
    print(f"Profitable: {int((out['net'] > 0).sum())} / {len(out)}")

    print("\nTop 20 by net:")
    print(
        f"{'R':>4} {'H':>4} {'trades':>6} {'WR%':>6} {'net':>9} {'avg':>6} "
        f"{'maxDD':>8} {'SL':>5} {'TP':>5} {'TP#':>5} {'SL#':>5} {'TO':>5}"
    )
    print("-" * 78)
    for _, r in out.sort_values("net", ascending=False).head(20).iterrows():
        print(
            f"{r['tp_r']:4.1f} {int(r['horizon']):4d} {int(r['trades']):6d} "
            f"{r['wr']:6.1f} {r['net']:+9.1f} {r['avg']:+6.2f} {r['max_dd']:+8.1f} "
            f"{r['sl_med']:5.1f} {r['tp_med']:5.1f} {int(r['tp_hit']):5d} "
            f"{int(r['sl_hit']):5d} {int(r['timeout']):5d}"
        )

    print("\nBest net per TP ratio:")
    for r_val in sorted(out["tp_r"].unique()):
        r = out[out["tp_r"] == r_val].sort_values("net", ascending=False).iloc[0]
        print(
            f"  R={r['tp_r']:.1f} H={int(r['horizon']):3d}: "
            f"net={r['net']:+.1f}  WR={r['wr']:.1f}%  tr={int(r['trades'])}"
        )

    print("\nBest net per horizon:")
    for h in sorted(out["horizon"].unique()):
        r = out[out["horizon"] == h].sort_values("net", ascending=False).iloc[0]
        print(
            f"  H={int(r['horizon']):3d} R={r['tp_r']:.1f}: "
            f"net={r['net']:+.1f}  WR={r['wr']:.1f}%  tr={int(r['trades'])}"
        )


if __name__ == "__main__":
    main()
