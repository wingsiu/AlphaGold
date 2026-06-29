#!/usr/bin/env python3
"""Backtest slot-15m SL + breakout entry (no slot-open, no structure gate)."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v16.backtest.impulse_entry import build_resolved_entry_table
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_signal_table
from v16.structure.filter import apply_structure_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--open", action="store_true", help="use slot-open entry (legacy)")
    parser.add_argument("--impulse-sl", action="store_true", help="SL at 1m impulse bar H/L")
    parser.add_argument("--reverse", action="store_true", help="fade impulse (uptrend+down bar→long)")
    args = parser.parse_args()

    if args.reverse:
        cfg = v16_config.MOMENTUM_SLOT_BREAKOUT_REVERSE
        label = "SLOT_BREAKOUT_REVERSE"
    elif args.open:
        cfg = copy.deepcopy(v16_config.MOMENTUM_SLOT_BREAKOUT)
        cfg["entry_mode"] = "open"
        cfg["entry_minute_in_slot"] = 0
        label = "SLOT_OPEN"
    else:
        cfg = v16_config.MOMENTUM_SLOT_BREAKOUT
        label = "SLOT_BREAKOUT"

    if args.impulse_sl:
        cfg["impulse_stop"] = {**cfg["impulse_stop"], "stop_mode": "impulse_bar", "max_sl_pts": 80.0}

    is_cfg = cfg["impulse_stop"]
    df = load_gold_1m(args.start, args.end)
    signals = build_signal_table(df, cfg=cfg)
    gated = apply_structure_gate(df, signals, cfg=cfg)
    fills = build_resolved_entry_table(df, gated, cfg=cfg)
    tdf = simulate_position_impulse_stop(df, signals, cfg=cfg)
    print(f"{label}  {args.start} → {args.end}  SL={is_cfg.get('stop_mode', 'impulse_bar')}")
    print(
        f"  filters: |body|>={cfg.get('min_move_pts')}  vol>={cfg.get('min_impulse_volume', 'off')}  "
        f"bars_after>={cfg.get('min_bars_after_impulse', 'off')}  "
        f"structure_gate={cfg.get('structure', {}).get('gate', {}).get('enabled', False)}  "
        f"reverse={cfg.get('reverse_impulse', False)}"
    )
    print(f"  signals={len(signals)}  gated={len(gated)}  fills={len(fills)}  tr={len(tdf)}")
    if not tdf.empty:
        print(
            f"  WR={tdf['win'].mean()*100:.1f}%  net={tdf['pnl'].sum():+.1f}  "
            f"avg={tdf['pnl'].mean():+.2f}  SL med={tdf['sl'].median():.1f}  TP med={tdf['tp'].median():.1f}"
        )
        print(f"  exits: {tdf['exit_reason'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
