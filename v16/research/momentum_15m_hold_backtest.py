#!/usr/bin/env python3
"""
Backtest impulse_1m_15m — 1m >=5pt bar in 15m slot; follow if held at slot close.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_backtest.py 2025-06-01 2026-06-25
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v16._paths import PROJECT_ROOT
from v16.backtest.position_sim import simulate_position_sided, simulate_position_sided_scaleout
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_side_signals, count_signals


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    start = args[0] if args else "2025-06-01"
    end = args[1] if len(args) > 1 else "2026-06-25"
    cfg = v16_config.MOMENTUM_15M_HOLD

    print("=" * 72)
    print(f"  impulse_1m_15m  |  {start} → {end}")
    print(f"  1m |body|>={cfg['min_move_pts']} first in slot → enter next slot open")
    print(f"  entry: minute {cfg['entry_minute_in_slot']} of new slot | single position")

    exit_mode = cfg.get("exit_mode", "fixed")
    if exit_mode == "scaleout":
        so = cfg.get("scaleout", v16_config.EXIT_CONFIG)
        print(
            f"  exit: scale-out +{so['first_scale_pnl']:.0f}/+{so['final_scale_pnl']:.0f} "
            f"SL{so['initial_sl']:.0f} H{so['horizon_minutes']}"
        )
    else:
        ex = cfg["execution"]
        print(f"  exit: TP{ex['tp']:.0f}/SL{ex['sl']:.0f}/H{ex['horizon']}")
    print("=" * 72)

    df = load_gold_1m(start, end)
    sides = build_side_signals(df, cfg=cfg)
    n = count_signals(sides)
    print(f"\nSignals: {n['total']} (long {n['long']}, short {n['short']})")

    if exit_mode == "scaleout":
        so = dict(cfg.get("scaleout", v16_config.EXIT_CONFIG))
        tdf = simulate_position_sided_scaleout(
            df,
            sides,
            scaleout_kw=so,
            same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
        )
    else:
        ex = cfg["execution"]
        tdf = simulate_position_sided(
            df,
            sides,
            tp=float(ex["tp"]),
            sl=float(ex["sl"]),
            horizon=int(ex["horizon"]),
            same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
            upgrade_stop=bool(cfg.get("upgrade_stop", False)),
        )

    if tdf.empty:
        print("No trades.")
        return

    for label, sub in [("ALL", tdf), ("LONG", tdf[tdf["side"] == 1]), ("SHORT", tdf[tdf["side"] == -1])]:
        if sub.empty:
            continue
        print(f"\n{label}: {len(sub)} trades  WR={sub['win'].mean()*100:.1f}%  "
              f"net={sub['pnl'].sum():+.1f}  avg={sub['pnl'].mean():+.2f}")

    if "target_updates" in tdf.columns:
        print(f"\nHorizon refreshes: {int(tdf['target_updates'].sum())}")
    if "scaled_half" in tdf.columns:
        print(f"Scaled half at +5: {int(tdf['scaled_half'].sum())} ({tdf['scaled_half'].mean()*100:.1f}%)")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"  {reason:12s}: {len(g):4d}  PnL={g['pnl'].sum():+.1f}")

    out = PROJECT_ROOT / "runtime" / "v16_momentum_15m_hold_trades.csv"
    tdf.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
