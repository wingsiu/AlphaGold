#!/usr/bin/env python3
"""
Sweep retrace-% SL × day-level target for pre-close breakout + with-trend.

Exit:
  SL distance = retrace_sl_pct × impulse bar range (high - low)
  Long TP  = day_low_rolling + target_offset  (default 80pt)
  Short TP = day_high_rolling - target_offset

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_retrace_day_target_sweep.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_retrace_day_target_sweep.py 2025-06-01 2026-06-25 --quick
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_retrace_day_target_sweep.py 2025-06-01 2026-06-25 --target-offset 60 80 100 --horizons 90 120
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.day_levels import attach_day_levels
from v16.backtest.impulse_entry import build_resolved_entry_table
from v16.backtest.position_sim import (
    simulate_position_impulse_stop,
    simulate_position_retrace_day_target,
)
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_signal_table
from v16.structure.filter import apply_structure_gate

GRIDS = {
    "quick": {
        "retrace_sl_pct": [0.5, 0.75, 1.0, 1.25, 1.5],
        "target_offset": [60.0, 80.0, 100.0, 120.0],
        "horizon": [90, 120],
    },
    "full": {
        "retrace_sl_pct": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "target_offset": [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0],
        "horizon": [60, 90, 120, 150],
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
            "invalid_skip": 0,
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
        "invalid_skip": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--target-offset", nargs="+", type=float, default=None)
    parser.add_argument("--retrace-sl-pct", nargs="+", type=float, default=None)
    args = parser.parse_args()

    grid = GRIDS["quick" if args.quick else "full"]
    retrace_pcts = args.retrace_sl_pct or grid["retrace_sl_pct"]
    target_offsets = args.target_offset or grid["target_offset"]
    horizons = args.horizons or grid["horizon"]
    combos = list(itertools.product(retrace_pcts, target_offsets, horizons))

    cfg = copy.deepcopy(v16_config.MOMENTUM_BREAKOUT_PRECLOSE)

    print("=" * 104)
    print(f"  retrace-% SL × day target × horizon  |  {args.start} → {args.end}")
    print(f"  pre-close ≤10  |  next_open  |  with-trend")
    print(f"  SL = pct × impulse range  |  long TP = day_low + N  |  short TP = day_high - N")
    print(f"  combos={len(combos)}  (SL% × target ± × H)")
    print("=" * 104)

    t0 = time.time()
    df = attach_day_levels(load_gold_1m(args.start, args.end))
    signals = build_signal_table(df, cfg=cfg)
    gated = apply_structure_gate(df, signals, cfg=cfg)
    fills = build_resolved_entry_table(df, gated, cfg=cfg)
    print(f"Signals: {len(signals)}  gated: {len(gated)}  fills: {len(fills)}")

    print("\nBaseline impulse-stop R=3:")
    for h in sorted(horizons):
        baseline = simulate_position_impulse_stop(
            df, signals, tp_multiple=3.0, horizon=int(h), cfg=cfg,
        )
        if baseline.empty:
            print(f"  H={h:3d}: no trades")
            continue
        b_net = float(baseline["pnl"].sum())
        print(
            f"  H={h:3d}: tr={len(baseline)} WR={baseline['win'].mean()*100:.1f}% "
            f"net={b_net:+.1f} avg={baseline['pnl'].mean():+.2f}"
        )

    rows: list[dict] = []
    for pct, tgt_off, horizon in combos:
        tdf = simulate_position_retrace_day_target(
            df,
            signals,
            retrace_sl_pct=float(pct),
            target_offset=float(tgt_off),
            horizon=int(horizon),
            cfg=cfg,
        )
        skipped = max(len(fills) - len(tdf), 0)
        row = {
            "retrace_sl_pct": pct,
            "target_offset": tgt_off,
            "horizon": int(horizon),
            **_stats(tdf),
        }
        row["invalid_skip"] = skipped
        rows.append(row)

    elapsed = time.time() - t0
    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "runtime" / "v16_retrace_day_target_sweep.csv"
    out.to_csv(out_path, index=False)

    profitable = int((out["net"] > 0).sum())
    print(f"\nDone in {elapsed:.1f}s  ->  {out_path}")
    print(f"Profitable: {profitable} / {len(out)}")

    print("\nTop 20 by net:")
    print(
        f"{'SL%':>5} {'tgt':>4} {'H':>4} {'trades':>6} {'skip':>5} {'WR%':>6} {'net':>9} {'avg':>6} "
        f"{'maxDD':>8} {'TP#':>5} {'SL#':>5} {'TO':>5}"
    )
    print("-" * 88)
    for _, r in out.sort_values("net", ascending=False).head(20).iterrows():
        print(
            f"{r['retrace_sl_pct']:5.2f} {int(r['target_offset']):4d} {int(r['horizon']):4d} "
            f"{int(r['trades']):6d} {int(r['invalid_skip']):5d} {r['wr']:6.1f} {r['net']:+9.1f} "
            f"{r['avg']:+6.2f} {r['max_dd']:+8.1f} {int(r['tp_hit']):5d} "
            f"{int(r['sl_hit']):5d} {int(r['timeout']):5d}"
        )

    print("\nBest net per horizon:")
    for h in sorted(out["horizon"].unique()):
        r = out[out["horizon"] == h].sort_values("net", ascending=False).iloc[0]
        print(
            f"  H={int(r['horizon']):3d} SL={r['retrace_sl_pct']:.0%} tgt±{int(r['target_offset']):3d}: "
            f"net={r['net']:+.1f}  WR={r['wr']:.1f}%  tr={int(r['trades'])}"
        )

    print("\nBest net per target offset:")
    for t in sorted(out["target_offset"].unique()):
        r = out[out["target_offset"] == t].sort_values("net", ascending=False).iloc[0]
        print(
            f"  tgt±{int(r['target_offset']):3d} SL={r['retrace_sl_pct']:.0%} H={int(r['horizon']):3d}: "
            f"net={r['net']:+.1f}  WR={r['wr']:.1f}%  tr={int(r['trades'])}"
        )

    if not out.empty:
        best = out.sort_values("net", ascending=False).iloc[0]
        print(
            f"\nBest overall: SL={best['retrace_sl_pct']:.0%} tgt±{int(best['target_offset'])} "
            f"H={int(best['horizon'])} → tr={int(best['trades'])} WR={best['wr']}% "
            f"net={best['net']:+.1f} avg={best['avg']:+.2f}"
        )


if __name__ == "__main__":
    main()
