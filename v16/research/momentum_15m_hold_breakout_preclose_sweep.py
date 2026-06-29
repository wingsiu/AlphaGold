#!/usr/bin/env python3
"""
Breakout + next_open when bar BEFORE break closed near trigger level.

Rule:
  bar j-1: close within N pt of impulse H/L (compression at level)
  bar j:   breaks trigger
  bar j+1: enter at open

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_breakout_preclose_sweep.py 2025-06-01 2026-06-25
"""
from __future__ import annotations

import argparse
import copy
import sys
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

PRE_CLOSE_SWEEP = (None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0)


def _cfg(*, gate: bool, pre_close_dist: float | None, horizon: int) -> dict:
    c = copy.deepcopy(v16_config.MOMENTUM_15M_HOLD)
    c["entry_mode"] = "breakout"
    c["entry_breakout"] = {
        **c.get("entry_breakout", {}),
        "fill": "next_open",
        "max_close_dist_pts": None,
        "max_pre_break_close_dist_pts": pre_close_dist,
        "max_entry_gap_pts": None,
    }
    c["impulse_stop"] = {**c["impulse_stop"], "tp_multiple": 3.0, "horizon": horizon}
    sc = dict(c.get("structure", {}))
    sc["gate"] = {"enabled": gate, "require_with_trend": True, "max_leg_age_15m": None}
    c["structure"] = sc
    return c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    args = parser.parse_args()

    print("=" * 88)
    print(f"  pre-break close near trigger  |  {args.start} → {args.end}")
    print("  bar j-1 close within N pt of break level → break on j → enter j+1 open")
    print("=" * 88)

    df = load_gold_1m(args.start, args.end)
    signals = build_signal_table(df, cfg=v16_config.MOMENTUM_15M_HOLD)
    rows: list[dict] = []

    # baselines
    for name, cfg in [
        ("open+trend H90", copy.deepcopy(v16_config.MOMENTUM_OPEN_STRUCTURE_ML)),
        ("breakout no pre-close filter", _cfg(gate=True, pre_close_dist=None, horizon=90)),
    ]:
        gated = apply_structure_gate(df, signals, cfg=cfg)
        fills = build_resolved_entry_table(df, gated, cfg=cfg)
        tdf = simulate_position_impulse_stop(df, signals, cfg=cfg)
        rows.append(
            {
                "scenario": name,
                "gate": True,
                "pre_close_max": "any",
                "horizon": cfg["impulse_stop"]["horizon"],
                "fills": len(fills),
                "trades": len(tdf),
                "wr": round(float(tdf["win"].mean() * 100), 1) if not tdf.empty else 0,
                "net": round(float(tdf["pnl"].sum()), 1) if not tdf.empty else 0,
                "avg": round(float(tdf["pnl"].mean()), 2) if not tdf.empty else 0,
            }
        )

    for gate in (False, True):
        for horizon in (90, 120):
            for pre_d in PRE_CLOSE_SWEEP:
                cfg = _cfg(gate=gate, pre_close_dist=pre_d, horizon=horizon)
                gated = apply_structure_gate(df, signals, cfg=cfg)
                fills = build_resolved_entry_table(df, gated, cfg=cfg)
                tdf = simulate_position_impulse_stop(df, signals, cfg=cfg)
                rows.append(
                    {
                        "scenario": "",
                        "gate": gate,
                        "pre_close_max": pre_d if pre_d is not None else "any",
                        "horizon": horizon,
                        "fills": len(fills),
                        "trades": len(tdf),
                        "wr": round(float(tdf["win"].mean() * 100), 1) if not tdf.empty else 0,
                        "net": round(float(tdf["pnl"].sum()), 1) if not tdf.empty else 0,
                        "avg": round(float(tdf["pnl"].mean()), 2) if not tdf.empty else 0,
                    }
                )

    out = pd.DataFrame(rows)
    sweep = out[out["scenario"] == ""].sort_values("net", ascending=False)
    path = PROJECT_ROOT / "runtime/v16_breakout_preclose_dist_sweep.csv"
    out.to_csv(path, index=False)

    print("\nBaselines:")
    print(out[out["scenario"] != ""].to_string(index=False))
    print("\nTop 15 (pre-break close distance):")
    print(sweep.head(15).to_string(index=False))
    print(f"\nSaved -> {path}")
    b = sweep.iloc[0]
    print(
        f"\nBest: gate={b['gate']} pre_close≤{b['pre_close_max']}pt H={int(b['horizon'])} "
        f"→ tr={int(b['trades'])} WR={b['wr']}% net={b['net']:+.1f} avg={b['avg']:+.2f}"
    )


if __name__ == "__main__":
    main()
