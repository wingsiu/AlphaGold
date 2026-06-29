#!/usr/bin/env python3
"""
TP×R sweep on V16 winner configs + exit when 15m structure trend changes.

struct_exit modes:
  off     — no structure exit
  profit  — exit if trend flips and PnL > 0
  always  — exit on any trend flip

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_winner_tpsl_struct_exit_sweep.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_winner_tpsl_struct_exit_sweep.py 2025-06-01 2026-06-25 --preset open
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

PRESETS = {
    "preclose": v16_config.MOMENTUM_V16_WINNER_PRECLOSE,
    "open": v16_config.MOMENTUM_V16_WINNER,
}
TP_GRID = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)
STRUCT_MODES = {
    "off": (False, None),
    "profit": (True, 0.0),
    "always": (True, -1e9),
}


def _stats(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {"trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0, "max_dd": 0.0, "struct_ex": 0}
    eq = tdf["pnl"].cumsum()
    reasons = tdf["exit_reason"].value_counts()
    return {
        "trades": len(tdf),
        "wr": round(float((tdf["pnl"] > 0).mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
        "max_dd": round(float((eq - eq.cummax()).min()), 1),
        "struct_ex": int(reasons.get("structure_change", 0)),
        "tp_hit": int(reasons.get("target_hit", 0)),
        "sl_hit": int(reasons.get("stop_loss", 0)),
        "timeout": int(reasons.get("timeout", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--preset", choices=list(PRESETS), default="preclose")
    args = parser.parse_args()

    base = PRESETS[args.preset]
    is0 = base.get("impulse_stop", {})
    horizon = int(is0.get("horizon", 120))
    combos = list(itertools.product(TP_GRID, STRUCT_MODES.keys()))

    print("=" * 100)
    print(f"  V16 winner TP sweep + structure-change exit  |  {args.start} → {args.end}")
    print(f"  preset={args.preset}  H={horizon}  SL=impulse H/L")
    print(f"  struct_exit: off | profit (pnl>0) | always (on trend flip)")
    print(f"  combos={len(combos)}")
    print("=" * 100)

    t0 = time.time()
    df = load_gold_1m(args.start, args.end)
    signals = build_signal_table(df, cfg=base)
    print(f"Signals (raw): {len(signals)}")

    rows: list[dict] = []
    for tp_r, mode in combos:
        struct_exit, min_pnl = STRUCT_MODES[mode]
        cfg = copy.deepcopy(base)
        cfg["impulse_stop"] = {
            **cfg.get("impulse_stop", {}),
            "tp_multiple": float(tp_r),
            "horizon": horizon,
            "stop_mode": "impulse_bar",
            "exit_on_structure_change": struct_exit,
            "exit_on_structure_change_min_pnl": 0.0 if min_pnl is None else float(min_pnl),
        }
        tdf = simulate_position_impulse_stop(df, signals, cfg=cfg)
        rows.append({"tp_r": tp_r, "struct_mode": mode, "horizon": horizon, **_stats(tdf)})

    out = pd.DataFrame(rows)
    path = PROJECT_ROOT / "runtime" / f"v16_winner_{args.preset}_tpsl_struct_exit_sweep.csv"
    out.to_csv(path, index=False)
    print(f"\nDone {time.time()-t0:.1f}s -> {path}")

    print("\nTop 20 by net:")
    print(f"{'R':>4} {'mode':>7} {'trades':>6} {'WR%':>6} {'net':>9} {'avg':>6} {'SC':>4} {'TP':>4} {'SL':>4} {'TO':>4}")
    print("-" * 76)
    for _, r in out.sort_values("net", ascending=False).head(20).iterrows():
        print(
            f"{r['tp_r']:4.1f} {r['struct_mode']:>7} {int(r['trades']):6d} "
            f"{r['wr']:6.1f} {r['net']:+9.1f} {r['avg']:+6.2f} {int(r['struct_ex']):4d} "
            f"{int(r['tp_hit']):4d} {int(r['sl_hit']):4d} {int(r['timeout']):4d}"
        )

    print("\nBest per TP:")
    for tp in TP_GRID:
        sub = out[out["tp_r"] == tp].sort_values("net", ascending=False)
        r = sub.iloc[0]
        print(
            f"  R={tp:.1f} best={r['struct_mode']:7s}: net={r['net']:+.1f}  "
            f"tr={int(r['trades'])}  TP={int(r['tp_hit'])}  SC={int(r['struct_ex'])}"
        )

    best = out.sort_values("net", ascending=False).iloc[0]
    print(
        f"\nBest overall: R={best['tp_r']:.1f} mode={best['struct_mode']} "
        f"→ net={best['net']:+.1f} ({int(best['trades'])} tr) WR={best['wr']}%"
    )


if __name__ == "__main__":
    main()
