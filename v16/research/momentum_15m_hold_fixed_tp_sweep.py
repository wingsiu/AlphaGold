#!/usr/bin/env python3
"""
Fixed TP (points) sweep on V16 winner configs + structure-change exit.

TP = fixed pts from entry; SL = impulse 1m bar H/L (variable).

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_fixed_tp_sweep.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_fixed_tp_sweep.py 2025-06-01 2026-06-25 --preset open
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
TP_PTS_GRID = (10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120)
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
    combos = list(itertools.product(TP_PTS_GRID, STRUCT_MODES.keys()))

    print("=" * 100)
    print(f"  V16 winner fixed-TP sweep + structure-change exit  |  {args.start} → {args.end}")
    print(f"  preset={args.preset}  H={horizon}  SL=impulse H/L")
    print(f"  struct_exit: off | profit (pnl>0) | always (on trend flip)")
    print(f"  combos={len(combos)}")
    print("=" * 100)

    t0 = time.time()
    df = load_gold_1m(args.start, args.end)
    signals = build_signal_table(df, cfg=base)
    print(f"Signals (raw): {len(signals)}")

    rows: list[dict] = []
    for tp_pts, mode in combos:
        struct_exit, min_pnl = STRUCT_MODES[mode]
        cfg = copy.deepcopy(base)
        cfg["impulse_stop"] = {
            **cfg.get("impulse_stop", {}),
            "tp_mode": "fixed_pts",
            "tp_pts": float(tp_pts),
            "horizon": horizon,
            "stop_mode": "impulse_bar",
            "exit_on_structure_change": struct_exit,
            "exit_on_structure_change_min_pnl": 0.0 if min_pnl is None else float(min_pnl),
        }
        tdf = simulate_position_impulse_stop(df, signals, cfg=cfg)
        rows.append({"tp_pts": tp_pts, "struct_mode": mode, "horizon": horizon, **_stats(tdf)})

    # Baseline: R=3 multiple TP for comparison
    for mode in STRUCT_MODES:
        struct_exit, min_pnl = STRUCT_MODES[mode]
        cfg = copy.deepcopy(base)
        cfg["impulse_stop"] = {
            **cfg.get("impulse_stop", {}),
            "tp_mode": "multiple",
            "tp_multiple": 3.0,
            "horizon": horizon,
            "stop_mode": "impulse_bar",
            "exit_on_structure_change": struct_exit,
            "exit_on_structure_change_min_pnl": 0.0 if min_pnl is None else float(min_pnl),
        }
        tdf = simulate_position_impulse_stop(df, signals, cfg=cfg)
        rows.append(
            {
                "tp_pts": "R3",
                "struct_mode": mode,
                "horizon": horizon,
                **_stats(tdf),
            }
        )

    out = pd.DataFrame(rows)
    path = PROJECT_ROOT / "runtime" / f"v16_winner_{args.preset}_fixed_tp_sweep.csv"
    out.to_csv(path, index=False)
    print(f"\nDone {time.time()-t0:.1f}s -> {path}")

    fixed = out[out["tp_pts"] != "R3"].copy()
    fixed["tp_pts"] = fixed["tp_pts"].astype(float)

    print("\nTop 20 by net:")
    print(f"{'TP':>5} {'mode':>7} {'trades':>6} {'WR%':>6} {'net':>9} {'avg':>6} {'SC':>4} {'TP':>4} {'SL':>4} {'TO':>4}")
    print("-" * 78)
    for _, r in out.sort_values("net", ascending=False).head(20).iterrows():
        tp_lbl = f"{int(r['tp_pts']):3d}" if r["tp_pts"] != "R3" else " R3"
        print(
            f"{tp_lbl:>5} {r['struct_mode']:>7} {int(r['trades']):6d} "
            f"{r['wr']:6.1f} {r['net']:+9.1f} {r['avg']:+6.2f} {int(r['struct_ex']):4d} "
            f"{int(r['tp_hit']):4d} {int(r['sl_hit']):4d} {int(r['timeout']):4d}"
        )

    print("\nBest per fixed TP (always struct exit):")
    sub_always = fixed[fixed["struct_mode"] == "always"]
    for tp in TP_PTS_GRID:
        sub = sub_always[sub_always["tp_pts"] == tp]
        if sub.empty:
            continue
        r = sub.iloc[0]
        print(
            f"  TP={int(tp):3d}pt: net={r['net']:+.1f}  tr={int(r['trades'])}  "
            f"TP_hit={int(r['tp_hit'])}  SC={int(r['struct_ex'])}  WR={r['wr']}%"
        )

    best = out.sort_values("net", ascending=False).iloc[0]
    tp_lbl = best["tp_pts"] if best["tp_pts"] == "R3" else f"{int(best['tp_pts'])}pt"
    print(
        f"\nBest overall: TP={tp_lbl} mode={best['struct_mode']} "
        f"→ net={best['net']:+.1f} ({int(best['trades'])} tr) WR={best['wr']}%"
    )


if __name__ == "__main__":
    main()
