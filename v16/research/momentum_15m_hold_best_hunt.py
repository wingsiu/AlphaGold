#!/usr/bin/env python3
"""
Hunt for best impulse setup under realistic next_open fills (OOS).

Sweeps entry mode, structure gate, leg age, breakout gap cap, TP×R, horizon.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_best_hunt.py 2025-06-01 2026-06-25
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
from v16.backtest.impulse_entry import build_resolved_entry_table
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_signal_table
from v16.structure.filter import apply_structure_gate

BASE = v16_config.MOMENTUM_15M_HOLD


def _cfg(
    *,
    entry_mode: str,
    gate: bool,
    leg_age: int | None,
    max_gap: float | None,
    tp_r: float,
    horizon: int,
) -> dict:
    c = copy.deepcopy(BASE)
    c["entry_mode"] = entry_mode
    c["entry_breakout"] = {
        **c.get("entry_breakout", {}),
        "fill": "next_open",
        "max_entry_gap_pts": max_gap,
    }
    c["impulse_stop"] = {
        **c.get("impulse_stop", {}),
        "tp_multiple": tp_r,
        "horizon": horizon,
    }
    sc = dict(c.get("structure", {}))
    sc["gate"] = {
        "enabled": gate,
        "require_with_trend": True,
        "max_leg_age_15m": leg_age,
    }
    c["structure"] = sc
    return c


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
    if tdf.empty:
        return {"trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0, "fills": len(fills)}
    return {
        "fills": len(fills),
        "trades": len(tdf),
        "wr": round(float(tdf["win"].mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    args = parser.parse_args()

    print("=" * 88)
    print(f"  BEST HUNT (next_open fills)  |  {args.start} → {args.end}")
    print("=" * 88)

    df = load_gold_1m(args.start, args.end)
    signals = build_signal_table(df, cfg=BASE)

    rows: list[dict] = []
    grid = list(
        itertools.product(
            ("open", "breakout"),
            (False, True),  # gate
            (None, 2, 5),  # leg_age
            (None, 1.0, 2.0, 3.0),  # max_gap (breakout only)
            (2.0, 2.5, 3.0),  # tp_r
            (60, 90, 120),  # horizon
        )
    )

    for entry_mode, gate, leg_age, max_gap, tp_r, horizon in grid:
        if entry_mode == "open" and max_gap is not None:
            continue
        if not gate and leg_age is not None:
            continue
        cfg = _cfg(
            entry_mode=entry_mode,
            gate=gate,
            leg_age=leg_age if gate else None,
            max_gap=max_gap if entry_mode == "breakout" else None,
            tp_r=tp_r,
            horizon=horizon,
        )
        r = _run(df, signals, cfg)
        rows.append(
            {
                "entry": entry_mode,
                "gate": gate,
                "leg_age": leg_age if gate else "",
                "max_gap": max_gap if entry_mode == "breakout" else "",
                "tp_r": tp_r,
                "horizon": horizon,
                **r,
            }
        )

    out = pd.DataFrame(rows).sort_values("net", ascending=False)
    path = PROJECT_ROOT / "runtime/v16_momentum_best_hunt.csv"
    out.to_csv(path, index=False)

    print("\nTop 15 by net:")
    print(out.head(15).to_string(index=False))
    print(f"\nSaved -> {path}")

    top = out.iloc[0]
    print(
        f"\nBest: {top['entry']} gate={top['gate']} leg={top['leg_age']} "
        f"gap={top['max_gap']} R={top['tp_r']} H={int(top['horizon'])}  "
        f"→ {int(top['trades'])} tr  WR={top['wr']}%  net={top['net']:+.1f}  avg={top['avg']:+.2f}"
    )


if __name__ == "__main__":
    main()
