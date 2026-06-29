#!/usr/bin/env python3
"""
Sweep TP / SL / horizon for impulse_1m_15m (fixed exit, single position).

Mechanical filter: first 1m |body|>=5pt in 15m slot, enter after slot close.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_tpsl_sweep.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_tpsl_sweep.py 2025-06-01 2026-06-25 --quick
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
from v16.backtest.position_sim import simulate_position_sided
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_side_signals

GRIDS = {
    "quick": {
        "tp": [15.0, 20.0, 25.0, 30.0, 35.0],
        "sl": [15.0, 20.0, 25.0, 30.0],
        "horizon": [10, 15, 20, 30, 45],
    },
    "full": {
        "tp": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0],
        "sl": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
        "horizon": [8, 10, 12, 15, 20, 25, 30, 45, 60],
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
            "tp_hit": 0,
            "sl_hit": 0,
            "timeout": 0,
        }
    eq = tdf["pnl"].cumsum()
    max_dd = float((eq - eq.cummax()).min())
    reasons = tdf["exit_reason"].value_counts()
    return {
        "trades": len(tdf),
        "wr": round(float((tdf["pnl"] > 0).mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
        "max_dd": round(max_dd, 1),
        "tp_hit": int(reasons.get("target_hit", 0)),
        "sl_hit": int(reasons.get("stop_loss", 0)),
        "timeout": int(reasons.get("timeout", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    grid = GRIDS["quick" if args.quick else "full"]
    cfg = v16_config.MOMENTUM_15M_HOLD
    combos = list(itertools.product(grid["tp"], grid["sl"], grid["horizon"]))

    print("=" * 92)
    print(f"  impulse_1m_15m TP/SL/H sweep  |  {args.start} → {args.end}")
    print(f"  filter: first 1m |body|>={cfg['min_move_pts']} in 15m slot  |  combos={len(combos)}")
    print("=" * 92)

    df = load_gold_1m(args.start, args.end)
    sides = build_side_signals(df, cfg=cfg)
    print(f"Signals: {int((sides != 0).sum())}")

    rows = []
    t0 = time.time()
    refresh = cfg.get("same_dir_refresh", "entry")
    for tp, sl, h in combos:
        tdf = simulate_position_sided(
            df,
            sides,
            tp=float(tp),
            sl=float(sl),
            horizon=int(h),
            same_dir_refresh=refresh,
            upgrade_stop=bool(cfg.get("upgrade_stop", False)),
        )
        rows.append({"tp": tp, "sl": sl, "horizon": h, **_stats(tdf)})

    elapsed = time.time() - t0
    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "runtime" / "v16_momentum_15m_hold_tpsl_sweep.csv"
    out.to_csv(out_path, index=False)
    print(f"\nDone in {elapsed:.1f}s  ->  {out_path}")

    profitable = int((out["net"] > 0).sum())
    print(f"Profitable combos: {profitable} / {len(out)}")

    print("\nTop 20 by net PnL:")
    print(f"{'TP':>4} {'SL':>4} {'H':>4} {'trades':>6} {'WR%':>6} {'net':>9} {'avg':>6} {'maxDD':>8} {'TP':>5} {'SL':>5} {'TO':>5}")
    print("-" * 72)
    for _, r in out.sort_values("net", ascending=False).head(20).iterrows():
        print(
            f"{r['tp']:4.0f} {r['sl']:4.0f} {int(r['horizon']):4d} {int(r['trades']):6d} "
            f"{r['wr']:6.1f} {r['net']:+9.1f} {r['avg']:+6.2f} {r['max_dd']:+8.1f} "
            f"{int(r['tp_hit']):5d} {int(r['sl_hit']):5d} {int(r['timeout']):5d}"
        )

    # Best per horizon
    print("\nBest net per horizon:")
    for h in sorted(out["horizon"].unique()):
        r = out[out["horizon"] == h].sort_values("net", ascending=False).iloc[0]
        print(
            f"  H{int(r['horizon']):2d} TP{int(r['tp']):2.0f}/SL{int(r['sl']):2.0f}: "
            f"net={r['net']:+.1f}  WR={r['wr']:.1f}%  tr={int(r['trades'])}"
        )


if __name__ == "__main__":
    main()
