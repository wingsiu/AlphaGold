#!/usr/bin/env python3
"""
Sweep 1m impulse change threshold (min_move_pts) × change_mode.

Filter: first 1m bar in 15m slot with |body| or range >= threshold.
Exit: impulse-bar SL, TP = 3× SL, H=120 (sweep winner).

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_impulse_change_sweep.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_impulse_change_sweep.py 2025-06-01 2026-06-25 --quick
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
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_signal_table

GRIDS = {
    "quick": {
        "min_move": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        "change_mode": ["body"],
    },
    "full": {
        "min_move": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0],
        "change_mode": ["body", "range", "either"],
    },
}


def _stats(tdf: pd.DataFrame, n_signals: int) -> dict:
    if tdf.empty:
        return {
            "signals": n_signals,
            "trades": 0,
            "wr": 0.0,
            "net": 0.0,
            "avg": 0.0,
            "max_dd": 0.0,
            "sl_med": 0.0,
            "tp_hit": 0,
            "sl_hit": 0,
            "timeout": 0,
        }
    eq = tdf["pnl"].cumsum()
    reasons = tdf["exit_reason"].value_counts()
    return {
        "signals": n_signals,
        "trades": len(tdf),
        "wr": round(float((tdf["pnl"] > 0).mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
        "max_dd": round(float((eq - eq.cummax()).min()), 1),
        "sl_med": round(float(tdf["sl"].median()), 1),
        "tp_hit": int(reasons.get("target_hit", 0)),
        "sl_hit": int(reasons.get("stop_loss", 0)),
        "timeout": int(reasons.get("timeout", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tp-r", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args()

    grid = GRIDS["quick" if args.quick else "full"]
    base_cfg = v16_config.MOMENTUM_15M_HOLD
    is_cfg = dict(base_cfg.get("impulse_stop", {}))
    tp_r = float(args.tp_r if args.tp_r is not None else is_cfg.get("tp_multiple", 3.0))
    horizon = int(args.horizon if args.horizon is not None else is_cfg.get("horizon", 120))
    combos = list(itertools.product(grid["min_move"], grid["change_mode"]))

    print("=" * 96)
    print(f"  impulse change sweep  |  {args.start} → {args.end}")
    print(f"  exit: impulse SL  TP={tp_r:.1f}×SL  H={horizon}  |  combos={len(combos)}")
    print("=" * 96)

    df = load_gold_1m(args.start, args.end)
    rows = []
    t0 = time.time()

    for min_move, mode in combos:
        cfg = copy.deepcopy(base_cfg)
        cfg["min_move_pts"] = float(min_move)
        cfg["change_mode"] = mode
        signals = build_signal_table(df, cfg=cfg)
        tdf = simulate_position_impulse_stop(
            df,
            signals,
            tp_multiple=tp_r,
            horizon=horizon,
            min_sl_pts=float(is_cfg.get("min_sl_pts", 1.0)),
            max_sl_pts=float(is_cfg.get("max_sl_pts", 80.0)),
            same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
        )
        rows.append(
            {
                "min_move_pts": min_move,
                "change_mode": mode,
                **_stats(tdf, len(signals)),
            }
        )

    elapsed = time.time() - t0
    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "runtime" / "v16_momentum_impulse_change_sweep.csv"
    out.to_csv(out_path, index=False)
    print(f"\nDone in {elapsed:.1f}s  ->  {out_path}")
    print(f"Profitable: {int((out['net'] > 0).sum())} / {len(out)}")

    print("\nTop 20 by net PnL:")
    print(
        f"{'min':>5} {'mode':>6} {'signals':>7} {'trades':>6} {'WR%':>6} "
        f"{'net':>9} {'avg':>6} {'SL':>5} {'TP#':>5} {'SL#':>5} {'TO':>5}"
    )
    print("-" * 72)
    for _, r in out.sort_values("net", ascending=False).head(20).iterrows():
        print(
            f"{r['min_move_pts']:5.1f} {r['change_mode']:>6} {int(r['signals']):7d} "
            f"{int(r['trades']):6d} {r['wr']:6.1f} {r['net']:+9.1f} {r['avg']:+6.2f} "
            f"{r['sl_med']:5.1f} {int(r['tp_hit']):5d} {int(r['sl_hit']):5d} {int(r['timeout']):5d}"
        )

    print("\nBest per change_mode (body baseline):")
    for mode in sorted(out["change_mode"].unique()):
        sub = out[out["change_mode"] == mode].sort_values("net", ascending=False)
        r = sub.iloc[0]
        print(
            f"  {mode:6s} min={r['min_move_pts']:.1f}: "
            f"sig={int(r['signals'])}  net={r['net']:+.1f}  WR={r['wr']:.1f}%"
        )

    body = out[out["change_mode"] == "body"].sort_values("min_move_pts")
    if not body.empty:
        print("\nBody mode by threshold:")
        for _, r in body.iterrows():
            print(
                f"  >={r['min_move_pts']:.0f}pt: {int(r['signals']):5d} sig  "
                f"{int(r['trades']):4d} tr  WR={r['wr']:.1f}%  net={r['net']:+.1f}"
            )


if __name__ == "__main__":
    main()
