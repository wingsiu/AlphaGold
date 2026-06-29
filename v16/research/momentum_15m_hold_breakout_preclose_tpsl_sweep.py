#!/usr/bin/env python3
"""
Sweep TP/SL ratio (tp_multiple) × horizon for breakout pre-close ≤10pt + with-trend.

SL @ impulse bar H/L; TP = tp_multiple × SL distance; fill = next_open.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_breakout_preclose_tpsl_sweep.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_breakout_preclose_tpsl_sweep.py 2025-06-01 2026-06-25 --quick
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_breakout_preclose_tpsl_sweep.py 2025-06-01 2026-06-25 --pre-close 8
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
from v16.backtest.impulse_entry import build_resolved_entry_table
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_signal_table
from v16.structure.filter import apply_structure_gate

GRIDS = {
    "quick": {
        "tp_r": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        "horizon": [45, 60, 75, 90, 105, 120],
    },
    "full": {
        "tp_r": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
        "horizon": [30, 45, 60, 75, 90, 105, 120, 150],
    },
}


def _cfg(pre_close_pts: float, tp_r: float, horizon: int) -> dict:
    c = copy.deepcopy(v16_config.MOMENTUM_BREAKOUT_PRECLOSE)
    c["entry_breakout"] = {
        **c.get("entry_breakout", {}),
        "max_pre_break_close_dist_pts": float(pre_close_pts),
    }
    c["impulse_stop"] = {
        **c.get("impulse_stop", {}),
        "tp_multiple": float(tp_r),
        "horizon": int(horizon),
    }
    return c


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


def _run(df: pd.DataFrame, signals: pd.DataFrame, cfg: dict) -> dict:
    is_cfg = cfg["impulse_stop"]
    gated = apply_structure_gate(df, signals, cfg=cfg)
    fills = build_resolved_entry_table(df, gated, cfg=cfg)
    tdf = simulate_position_impulse_stop(
        df,
        signals,
        tp_multiple=float(is_cfg["tp_multiple"]),
        horizon=int(is_cfg["horizon"]),
        cfg=cfg,
    )
    return {"fills": len(fills), **_stats(tdf)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--pre-close", type=float, default=10.0)
    args = parser.parse_args()

    grid = GRIDS["quick" if args.quick else "full"]
    combos = list(itertools.product(grid["tp_r"], grid["horizon"]))

    print("=" * 96)
    print(f"  pre-close breakout TP×R × horizon  |  {args.start} → {args.end}")
    print(f"  pre_break close ≤{args.pre_close:.0f}pt  |  next_open  |  with-trend  |  combos={len(combos)}")
    print("=" * 96)

    t0 = time.time()
    df = load_gold_1m(args.start, args.end)
    signals = build_signal_table(df, cfg=v16_config.MOMENTUM_15M_HOLD)
    print(f"Signals: {len(signals)}")

    baselines: list[dict] = []
    for name, base_cfg in [
        ("open+trend H90 R3", copy.deepcopy(v16_config.MOMENTUM_OPEN_STRUCTURE_ML)),
        (
            f"preclose≤{args.pre_close:.0f} current R3 H90",
            _cfg(args.pre_close, 3.0, 90),
        ),
    ]:
        row = _run(df, signals, base_cfg)
        baselines.append({"scenario": name, **row})

    rows: list[dict] = []
    for tp_r, h in combos:
        cfg = _cfg(args.pre_close, tp_r, h)
        row = _run(df, signals, cfg)
        rows.append({"tp_r": tp_r, "horizon": h, **row})

    elapsed = time.time() - t0
    out = pd.DataFrame(rows)
    suffix = f"_preclose{int(args.pre_close)}"
    out_path = PROJECT_ROOT / "runtime" / f"v16_breakout_preclose_tpsl_sweep{suffix}.csv"
    out.to_csv(out_path, index=False)

    print(f"\nBaselines:")
    for b in baselines:
        print(
            f"  {b['scenario']:<28} fills={b['fills']:4d} tr={b['trades']:4d} "
            f"WR={b['wr']:5.1f}% net={b['net']:+8.1f} avg={b['avg']:+5.2f}"
        )

    profitable = int((out["net"] > 0).sum())
    print(f"\nDone in {elapsed:.1f}s  ->  {out_path}")
    print(f"Profitable: {profitable} / {len(out)}")

    print("\nTop 20 by net:")
    print(
        f"{'R':>4} {'H':>4} {'fills':>6} {'trades':>6} {'WR%':>6} {'net':>9} {'avg':>6} "
        f"{'maxDD':>8} {'SL':>5} {'TP':>5} {'TP#':>5} {'SL#':>5} {'TO':>5}"
    )
    print("-" * 88)
    for _, r in out.sort_values("net", ascending=False).head(20).iterrows():
        print(
            f"{r['tp_r']:4.1f} {int(r['horizon']):4d} {int(r['fills']):6d} {int(r['trades']):6d} "
            f"{r['wr']:6.1f} {r['net']:+9.1f} {r['avg']:+6.2f} {r['max_dd']:+8.1f} "
            f"{r['sl_med']:5.1f} {r['tp_med']:5.1f} {int(r['tp_hit']):5d} "
            f"{int(r['sl_hit']):5d} {int(r['timeout']):5d}"
        )

    print("\nBest net per TP ratio:")
    for r_val in sorted(out["tp_r"].unique()):
        r = out[out["tp_r"] == r_val].sort_values("net", ascending=False).iloc[0]
        print(
            f"  R={r['tp_r']:.1f} H={int(r['horizon']):3d}: "
            f"net={r['net']:+.1f}  WR={r['wr']:.1f}%  tr={int(r['trades'])}  avg={r['avg']:+.2f}"
        )

    print("\nBest net per horizon:")
    for h in sorted(out["horizon"].unique()):
        r = out[out["horizon"] == h].sort_values("net", ascending=False).iloc[0]
        print(
            f"  H={int(r['horizon']):3d} R={r['tp_r']:.1f}: "
            f"net={r['net']:+.1f}  WR={r['wr']:.1f}%  tr={int(r['trades'])}  avg={r['avg']:+.2f}"
        )

    best = out.sort_values("net", ascending=False).iloc[0]
    print(
        f"\nBest: R={best['tp_r']:.1f} H={int(best['horizon'])} "
        f"→ tr={int(best['trades'])} WR={best['wr']}% net={best['net']:+.1f} avg={best['avg']:+.2f}"
    )


if __name__ == "__main__":
    main()
