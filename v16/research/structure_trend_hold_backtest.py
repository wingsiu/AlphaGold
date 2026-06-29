#!/usr/bin/env python3
"""
Structure trend hold — with-trend retrace entry, hold until structure breaks.

Usage:
  PYTHONPATH=. python3 v16/research/structure_trend_hold_backtest.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/structure_trend_hold_backtest.py 2025-06-01 2026-06-25 --sweep-pb
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.structure_hold_sim import simulate_structure_trend_hold
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.structure_trend_hold import build_structure_retrace_signals


def _report(tdf: pd.DataFrame, label: str) -> dict:
    if tdf.empty:
        print(f"\n{label}: no trades")
        return {"label": label, "trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0, "hold_med": 0.0}
    reasons = tdf["exit_reason"].value_counts()
    hold_med = float(tdf["hold_min"].median()) if "hold_min" in tdf.columns else 0.0
    print(f"\n{label}: {len(tdf)} tr  WR={tdf['win'].mean()*100:.1f}%  "
          f"net={tdf['pnl'].sum():+.1f}  avg={tdf['pnl'].mean():+.2f}  hold_med={hold_med:.0f}m")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"  {reason:18s} {len(g):4d}  ({len(g)/len(tdf)*100:4.1f}%)  net={g['pnl'].sum():+.1f}")
    return {
        "label": label,
        "trades": len(tdf),
        "wr": round(float(tdf["win"].mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
        "hold_med": round(hold_med, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--sweep-pb", action="store_true", help="sweep pullback band")
    args = parser.parse_args()

    print("=" * 72)
    print(f"  structure_trend_hold  |  {args.start} → {args.end}")
    print("  enter: with-trend retrace 30–90m | exit: swing break + struct flip")
    print("=" * 72)

    df = load_gold_1m(args.start, args.end)
    rows: list[dict] = []

    if args.sweep_pb:
        bands = ((0.10, 0.40), (0.15, 0.55), (0.15, 0.65), (0.20, 0.60), (0.25, 0.70))
        for min_pb, max_pb in bands:
            cfg = copy.deepcopy(v16_config.STRUCTURE_TREND_HOLD)
            cfg["entry"]["min_pullback_pct"] = min_pb
            cfg["entry"]["max_pullback_pct"] = max_pb
            sig = build_structure_retrace_signals(df, cfg=cfg)
            tdf = simulate_structure_trend_hold(df, sig, cfg=cfg)
            rows.append(_report(tdf, f"pb {min_pb:.0%}–{max_pb:.0%}"))
    else:
        cfg = v16_config.STRUCTURE_TREND_HOLD
        sig = build_structure_retrace_signals(df, cfg=cfg)
        print(f"\nSignals: {len(sig)} (long {(sig['side']==1).sum()}, short {(sig['side']==-1).sum()})")
        tdf = simulate_structure_trend_hold(df, sig, cfg=cfg)
        rows.append(_report(tdf, "default"))

        for variant, exit_kw in [
            ("struct_only", {"on_swing_break": False, "on_structure_change": True}),
            ("swing_only", {"on_swing_break": True, "on_structure_change": False}),
        ]:
            c2 = copy.deepcopy(cfg)
            c2["exit"] = {**c2["exit"], **exit_kw}
            tdf2 = simulate_structure_trend_hold(df, sig, cfg=c2)
            rows.append(_report(tdf2, variant))

    out = pd.DataFrame(rows)
    path = PROJECT_ROOT / "runtime" / "v16_structure_trend_hold_backtest.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
