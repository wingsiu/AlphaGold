#!/usr/bin/env python3
"""
Day-level entry filter + impulse-stop exit (pre-close breakout + with-trend).

Entry filter (room to day target):
  Long:  entry < day_low_rolling + offset
  Short: entry > day_high_rolling - offset

Exit: impulse bar SL, TP = 3R (default), horizon swept.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_day_level_entry_sweep.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_day_level_entry_sweep.py 2025-06-01 2026-06-25 --quick
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
from v16.backtest.day_levels import attach_day_levels, filter_entries_by_day_level
from v16.backtest.impulse_entry import build_resolved_entry_table
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_signal_table
from v16.structure.filter import apply_structure_gate

GRIDS = {
    "quick": {
        "offset": [None, 60.0, 80.0, 100.0, 120.0],
        "horizon": [90, 120],
        "tp_r": [3.0],
    },
    "full": {
        "offset": [None, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0],
        "horizon": [60, 90, 120, 150],
        "tp_r": [2.5, 3.0, 3.5],
    },
}


def _cfg(offset: float | None, tp_r: float, horizon: int) -> dict:
    c = copy.deepcopy(v16_config.MOMENTUM_BREAKOUT_PRECLOSE)
    c["impulse_stop"] = {
        **c.get("impulse_stop", {}),
        "tp_multiple": float(tp_r),
        "horizon": int(horizon),
    }
    if offset is None:
        c["day_level_entry_filter"] = {"enabled": False}
    else:
        c["day_level_entry_filter"] = {"enabled": True, "offset": float(offset)}
    return c


def _stats(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {"trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0, "max_dd": 0.0}
    eq = tdf["pnl"].cumsum()
    return {
        "trades": len(tdf),
        "wr": round(float((tdf["pnl"] > 0).mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
        "max_dd": round(float((eq - eq.cummax()).min()), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--offsets", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--tp-r", nargs="+", type=float, default=None)
    args = parser.parse_args()

    grid = GRIDS["quick" if args.quick else "full"]
    offsets: list[float | None] = (
        [None if str(x).lower() in ("none", "off", "any") else float(x) for x in args.offsets]
        if args.offsets is not None
        else grid["offset"]
    )
    horizons = args.horizons or grid["horizon"]
    tp_rs = args.tp_r or grid["tp_r"]
    combos = list(itertools.product(offsets, horizons, tp_rs))

    print("=" * 104)
    print(f"  day-level entry filter × impulse-stop  |  {args.start} → {args.end}")
    print(f"  pre-close ≤10  |  next_open  |  with-trend")
    print(f"  long if entry < day_low+N  |  short if entry > day_high-N  |  exit impulse 3R")
    print(f"  combos={len(combos)}")
    print("=" * 104)

    t0 = time.time()
    df = attach_day_levels(load_gold_1m(args.start, args.end))
    base_cfg = copy.deepcopy(v16_config.MOMENTUM_BREAKOUT_PRECLOSE)
    signals = build_signal_table(df, cfg=base_cfg)
    gated = apply_structure_gate(df, signals, cfg=base_cfg)
    all_fills = build_resolved_entry_table(df, gated, cfg=base_cfg)
    print(f"Signals: {len(signals)}  gated: {len(gated)}  fills: {len(all_fills)}")

    rows: list[dict] = []
    for offset, horizon, tp_r in combos:
        cfg = _cfg(offset, tp_r, horizon)
        filtered_fills = (
            all_fills
            if offset is None
            else filter_entries_by_day_level(df, all_fills, float(offset))
        )
        tdf = simulate_position_impulse_stop(
            df, signals, tp_multiple=float(tp_r), horizon=int(horizon), cfg=cfg,
        )
        rows.append(
            {
                "day_offset": "any" if offset is None else offset,
                "horizon": int(horizon),
                "tp_r": float(tp_r),
                "fills_in": len(filtered_fills),
                "filtered_out": len(all_fills) - len(filtered_fills),
                **_stats(tdf),
            }
        )

    elapsed = time.time() - t0
    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "runtime" / "v16_day_level_entry_filter_sweep.csv"
    out.to_csv(out_path, index=False)

    print(f"\nDone in {elapsed:.1f}s  ->  {out_path}")
    print(f"Profitable: {int((out['net'] > 0).sum())} / {len(out)}")

    print("\nTop 20 by net:")
    print(f"{'offset':>6} {'H':>4} {'R':>4} {'fills':>6} {'out':>5} {'trades':>6} {'WR%':>6} {'net':>9} {'avg':>6} {'maxDD':>8}")
    print("-" * 72)
    for _, r in out.sort_values("net", ascending=False).head(20).iterrows():
        off = r["day_offset"]
        off_s = "any" if off == "any" else f"{float(off):.0f}"
        print(
            f"{off_s:>6} {int(r['horizon']):4d} {r['tp_r']:4.1f} {int(r['fills_in']):6d} "
            f"{int(r['filtered_out']):5d} {int(r['trades']):6d} {r['wr']:6.1f} "
            f"{r['net']:+9.1f} {r['avg']:+6.2f} {r['max_dd']:+8.1f}"
        )

    no_filter = out[out["day_offset"] == "any"]
    if not no_filter.empty:
        print("\nNo filter baseline (best per H, R=3):")
        nf = no_filter[no_filter["tp_r"] == 3.0] if (no_filter["tp_r"] == 3.0).any() else no_filter
        for h in sorted(nf["horizon"].unique()):
            r = nf[nf["horizon"] == h].iloc[0]
            print(f"  H={int(r['horizon']):3d}: net={r['net']:+.1f}  tr={int(r['trades'])}")

    print("\nBest per day offset (any H/R):")
    for off in sorted(out["day_offset"].unique(), key=lambda x: (x == "any", x)):
        sub = out[out["day_offset"] == off]
        r = sub.sort_values("net", ascending=False).iloc[0]
        off_s = "any" if off == "any" else f"±{float(off):.0f}"
        print(
            f"  {off_s:>5} H={int(r['horizon']):3d} R={r['tp_r']:.1f}: "
            f"net={r['net']:+.1f}  WR={r['wr']:.1f}%  tr={int(r['trades'])}  "
            f"fills={int(r['fills_in'])}"
        )

    best = out.sort_values("net", ascending=False).iloc[0]
    off_s = "any" if best["day_offset"] == "any" else f"±{float(best['day_offset']):.0f}"
    print(
        f"\nBest overall: offset={off_s} H={int(best['horizon'])} R={best['tp_r']:.1f} "
        f"→ tr={int(best['trades'])} WR={best['wr']}% net={best['net']:+.1f} avg={best['avg']:+.2f}"
    )


if __name__ == "__main__":
    main()
