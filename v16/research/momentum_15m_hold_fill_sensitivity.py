#!/usr/bin/env python3
"""
Fill sensitivity: open vs breakout+with-trend under ideal / conservative / pessimistic fills.

Fill modes (entry_fill.mode):
  ideal         — trigger/open at model price
  conservative  — breakout: max(trigger, open); +0.125pt slip
  pessimistic   — conservative + 0.25pt slip; cancel if SL touched while waiting;
                  skip fill bar if SL touched intrabar

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_fill_sensitivity.py 2025-06-01 2026-06-25
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

FILL_MODES = ("ideal", "conservative", "pessimistic")
HORIZON = 120
TP_R = 3.0


def _cfg(scenario: str, fill_mode: str) -> dict:
    if scenario == "open":
        c = copy.deepcopy(v16_config.MOMENTUM_15M_HOLD)
    else:
        c = copy.deepcopy(v16_config.MOMENTUM_BREAKOUT_STRUCTURE)
    c["entry_fill"] = {
        **c.get("entry_fill", {}),
        "mode": fill_mode,
        "slippage_pts": 0.25,
        "intrabar_stop_first": True,
        "cancel_on_stop_during_wait": fill_mode == "pessimistic",
    }
    return c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    args = parser.parse_args()

    print("=" * 88)
    print(f"  fill sensitivity  |  {args.start} → {args.end}")
    print(f"  exit: impulse SL  TP={TP_R}R  H={HORIZON}")
    print("=" * 88)

    df = load_gold_1m(args.start, args.end)
    base_signals = build_signal_table(df, cfg=v16_config.MOMENTUM_15M_HOLD)
    print(f"\nImpulse signals: {len(base_signals)}")

    rows: list[dict] = []
    for scenario in ("open", "breakout+with_trend"):
        for fill_mode in FILL_MODES:
            cfg = _cfg(scenario, fill_mode)
            gated = apply_structure_gate(df, base_signals, cfg=cfg)
            entries = build_resolved_entry_table(df, gated, cfg=cfg)
            is_cfg = cfg.get("impulse_stop", {})
            tdf = simulate_position_impulse_stop(
                df,
                base_signals,
                tp_multiple=float(is_cfg.get("tp_multiple", TP_R)),
                horizon=int(is_cfg.get("horizon", HORIZON)),
                min_sl_pts=float(is_cfg.get("min_sl_pts", 1.0)),
                max_sl_pts=float(is_cfg.get("max_sl_pts", 80.0)),
                same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
                cfg=cfg,
            )
            net = float(tdf["pnl"].sum()) if not tdf.empty else 0.0
            wr = float(tdf["win"].mean() * 100) if not tdf.empty else 0.0
            avg = float(tdf["pnl"].mean()) if not tdf.empty else 0.0
            row = {
                "scenario": scenario,
                "fill_mode": fill_mode,
                "after_gate": len(gated),
                "fills": len(entries),
                "trades": len(tdf),
                "wr": round(wr, 1),
                "net": round(net, 1),
                "avg": round(avg, 2),
            }
            rows.append(row)
            print(
                f"  {scenario:22s} {fill_mode:14s}  "
                f"fills={row['fills']:4d}  tr={row['trades']:4d}  "
                f"WR={row['wr']:5.1f}%  net={row['net']:+9.1f}  avg={row['avg']:+.2f}"
            )

    out = pd.DataFrame(rows)
    path = PROJECT_ROOT / "runtime/v16_momentum_fill_sensitivity.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved -> {path}")

    # Headline: does breakout+struct beat open under pessimistic?
    pivot = out.pivot(index="fill_mode", columns="scenario", values="net")
    print("\nNet PnL matrix:")
    print(pivot.to_string())
    if "pessimistic" in pivot.index:
        pess = pivot.loc["pessimistic"]
        if "breakout+with_trend" in pess.index and "open" in pess.index:
            diff = pess["breakout+with_trend"] - pess["open"]
            winner = "breakout+with_trend" if diff > 0 else "open"
            print(f"\nPessimistic: {winner} wins by {abs(diff):.1f} pt")


if __name__ == "__main__":
    main()
