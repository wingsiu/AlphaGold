#!/usr/bin/env python3
"""
Breakout entry × 15m structure gate (with-trend + optional fresh leg).

Compares:
  open              — baseline slot open
  breakout          — stop at impulse H/L
  open + struct     — with-trend gate on open
  breakout + struct — combined (research target)
  breakout + struct + leg_age≤2

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_breakout_structure.py 2025-06-01 2026-06-25
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

HORIZON = 120
TP_R = 3.0

SCENARIOS = (
    ("open", False, None),
    ("breakout", False, None),
    ("open+with_trend", True, None),
    ("breakout+with_trend", True, None),
    ("breakout+with_trend+leg≤2", True, 2),
)


def _cfg(
    entry_mode: str,
    *,
    gate: bool,
    max_leg_age: int | None,
) -> dict:
    c = copy.deepcopy(v16_config.MOMENTUM_15M_HOLD)
    c["entry_mode"] = entry_mode
    sc = dict(c.get("structure", {}))
    sc["gate"] = {
        "enabled": gate,
        "require_with_trend": True,
        "max_leg_age_15m": max_leg_age,
    }
    c["structure"] = sc
    return c


def _run(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: dict,
    *,
    label: str,
) -> dict:
    gated = apply_structure_gate(df, signals, cfg=cfg)
    entries = build_resolved_entry_table(df, gated, cfg=cfg)
    is_cfg = cfg.get("impulse_stop", {})
    tdf = simulate_position_impulse_stop(
        df,
        signals,
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
        "scenario": label,
        "entry_mode": cfg.get("entry_mode"),
        "struct_gate": bool(cfg.get("structure", {}).get("gate", {}).get("enabled")),
        "max_leg_age": cfg.get("structure", {}).get("gate", {}).get("max_leg_age_15m"),
        "signals": len(signals),
        "after_gate": len(gated),
        "fills": len(entries),
        "trades": len(tdf),
        "wr": round(wr, 1),
        "net": round(net, 1),
        "avg": round(avg, 2),
    }
    print(
        f"\n{label}: gate {len(gated)}/{len(signals)} sig  "
        f"fills={len(entries)}  tr={row['trades']}  WR={row['wr']:.1f}%  "
        f"net={row['net']:+.1f}  avg={row['avg']:+.2f}"
    )
    if not tdf.empty:
        for reason, g in tdf.groupby("exit_reason"):
            print(f"    {reason:12s}: {len(g):4d}  PnL={g['pnl'].sum():+.1f}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    args = parser.parse_args()

    print("=" * 80)
    print(f"  breakout × structure  |  {args.start} → {args.end}")
    print(f"  exit: impulse SL  TP={TP_R}R  H={HORIZON}")
    print("=" * 80)

    df = load_gold_1m(args.start, args.end)
    signals = build_signal_table(df, cfg=v16_config.MOMENTUM_15M_HOLD)
    print(f"\nImpulse signals: {len(signals)}")

    rows: list[dict] = []
    for label, gate, max_age in SCENARIOS:
        mode = "breakout" if label.startswith("breakout") else "open"
        cfg = _cfg(mode, gate=gate, max_leg_age=max_age)
        rows.append(_run(df, signals, cfg, label=label))

    summary = pd.DataFrame(rows)
    out = PROJECT_ROOT / "runtime/v16_momentum_breakout_structure.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved -> {out}")

    best = summary.sort_values("net", ascending=False).iloc[0]
    print(
        f"\nBest: {best['scenario']}  net={best['net']:+.1f}  "
        f"({int(best['trades'])} tr  WR={best['wr']:.1f}%)"
    )


if __name__ == "__main__":
    main()
