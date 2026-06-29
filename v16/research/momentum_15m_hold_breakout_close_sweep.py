#!/usr/bin/env python3
"""
Breakout + next_open ONLY when break-bar CLOSE is near trigger level.

Sweeps max_close_dist_pts (how close close must be to impulse H/L break).

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_breakout_close_sweep.py 2025-06-01 2026-06-25
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

CLOSE_DIST_SWEEP = (None, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0)
OPEN_GAP_SWEEP = (None, 2.0, 3.0, 5.0)  # optional 2nd filter on next open vs trigger
HORIZONS = (90, 120)


def _cfg(
    *,
    gate: bool,
    max_close_dist: float | None,
    max_open_gap: float | None,
    horizon: int,
) -> dict:
    c = copy.deepcopy(v16_config.MOMENTUM_15M_HOLD)
    c["entry_mode"] = "breakout"
    c["entry_breakout"] = {
        **c.get("entry_breakout", {}),
        "fill": "next_open",
        "max_close_dist_pts": max_close_dist,
        "max_entry_gap_pts": max_open_gap,
    }
    c["impulse_stop"] = {**c["impulse_stop"], "tp_multiple": 3.0, "horizon": horizon}
    sc = dict(c.get("structure", {}))
    sc["gate"] = {
        "enabled": gate,
        "require_with_trend": True,
        "max_leg_age_15m": None,
    }
    c["structure"] = sc
    return c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    args = parser.parse_args()

    print("=" * 88)
    print(f"  breakout close-near-trigger sweep  |  {args.start} → {args.end}")
    print("  Rule: bar breaks impulse H/L → close within N pt → enter next 1m open")
    print("=" * 88)

    df = load_gold_1m(args.start, args.end)
    signals = build_signal_table(df, cfg=v16_config.MOMENTUM_15M_HOLD)

    rows: list[dict] = []

    # baselines
    for label, cfg in [
        ("open+trend H90", copy.deepcopy(v16_config.MOMENTUM_OPEN_STRUCTURE_ML)),
        ("breakout+next_open no filter", _cfg(gate=True, max_close_dist=None, max_open_gap=None, horizon=90)),
    ]:
        tdf = simulate_position_impulse_stop(df, signals, cfg=cfg)
        rows.append(
            {
                "label": label,
                "gate": True,
                "max_close_dist": "",
                "max_open_gap": "",
                "horizon": cfg["impulse_stop"]["horizon"],
                "fills": len(build_resolved_entry_table(df, apply_structure_gate(df, signals, cfg=cfg), cfg=cfg)),
                "trades": len(tdf),
                "wr": round(float(tdf["win"].mean() * 100), 1) if not tdf.empty else 0,
                "net": round(float(tdf["pnl"].sum()), 1) if not tdf.empty else 0,
                "avg": round(float(tdf["pnl"].mean()), 2) if not tdf.empty else 0,
            }
        )

    for gate in (False, True):
        for horizon in HORIZONS:
            for max_cd in CLOSE_DIST_SWEEP:
                for max_gap in (None,):  # primary sweep close only
                    cfg = _cfg(
                        gate=gate,
                        max_close_dist=max_cd,
                        max_open_gap=max_gap,
                        horizon=horizon,
                    )
                    gated = apply_structure_gate(df, signals, cfg=cfg)
                    fills = build_resolved_entry_table(df, gated, cfg=cfg)
                    tdf = simulate_position_impulse_stop(df, signals, cfg=cfg)
                    rows.append(
                        {
                            "label": "",
                            "gate": gate,
                            "max_close_dist": max_cd if max_cd is not None else "any",
                            "max_open_gap": max_gap if max_gap is not None else "any",
                            "horizon": horizon,
                            "fills": len(fills),
                            "trades": len(tdf),
                            "wr": round(float(tdf["win"].mean() * 100), 1) if not tdf.empty else 0,
                            "net": round(float(tdf["pnl"].sum()), 1) if not tdf.empty else 0,
                            "avg": round(float(tdf["pnl"].mean()), 2) if not tdf.empty else 0,
                        }
                    )

    # fine sweep best region with open gap filter
    for max_cd in (1.0, 1.5, 2.0, 3.0):
        for max_gap in OPEN_GAP_SWEEP:
            if max_gap is None:
                continue
            cfg = _cfg(gate=True, max_close_dist=max_cd, max_open_gap=max_gap, horizon=90)
            gated = apply_structure_gate(df, signals, cfg=cfg)
            fills = build_resolved_entry_table(df, gated, cfg=cfg)
            tdf = simulate_position_impulse_stop(df, signals, cfg=cfg)
            rows.append(
                {
                    "label": "close+open_gap",
                    "gate": True,
                    "max_close_dist": max_cd,
                    "max_open_gap": max_gap,
                    "horizon": 90,
                    "fills": len(fills),
                    "trades": len(tdf),
                    "wr": round(float(tdf["win"].mean() * 100), 1) if not tdf.empty else 0,
                    "net": round(float(tdf["pnl"].sum()), 1) if not tdf.empty else 0,
                    "avg": round(float(tdf["pnl"].mean()), 2) if not tdf.empty else 0,
                }
            )

    out = pd.DataFrame(rows)
    sweep = out[out["label"] == ""].sort_values("net", ascending=False)
    path = PROJECT_ROOT / "runtime/v16_breakout_close_dist_sweep.csv"
    out.to_csv(path, index=False)

    print("\nBaselines:")
    print(out[out["label"] != ""].to_string(index=False))
    print("\nTop 12 breakout (close-near sweep):")
    print(sweep.head(12).to_string(index=False))
    print(f"\nSaved -> {path}")

    best = sweep.iloc[0]
    print(
        f"\nBest breakout: gate={best['gate']} close≤{best['max_close_dist']}pt "
        f"H={int(best['horizon'])} → fills={int(best['fills'])} tr={int(best['trades'])} "
        f"WR={best['wr']}% net={best['net']:+.1f} avg={best['avg']:+.2f}"
    )


if __name__ == "__main__":
    main()
