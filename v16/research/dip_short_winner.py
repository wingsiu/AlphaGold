#!/usr/bin/env python3
"""
Best v16 dip-short lane (profit hunt winner).

Rule: 2 consecutive 15m UP + slot rip + prev body>=8
Exit: v15-style TP30 / SL25 / H30

Jun 2025 → Jun 2026: ~425 trades, +1543 pts

Usage:
  PYTHONPATH=. python3 v16/research/dip_short_winner.py 2025-06-01 2026-06-25
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features
from v16.backtest.fixed_tpsl_sim import simulate_fixed_tpsl
from v16.backtest.signals import build_labeled_set
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.research.profit_hunt import signal_cfg


WINNER_PATCH = {
    "mode": "dip_short_15m",
    "dip_require_two_prev_up": True,
    "dip_min_prev_body_pts": 8.0,
    "dip_short_min_above_open_pts": 5.0,
    "dip_short_min_slot_high_pts": 10.0,
    "dip_max_minute_in_slot": 10,
}
TP, SL, H = 30.0, 25.0, 30


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    bt_start = args[0] if args else "2025-06-01"
    bt_end = args[1] if len(args) > 1 else "2026-06-25"

    print("=" * 70)
    print(f"  DIP SHORT WINNER  |  {bt_start} → {bt_end}")
    print("  2x15m UP + prev body>=8 | short slot rip | TP30 SL25 H30")
    print("=" * 70)

    df = load_gold_1m(bt_start, bt_end)
    feats = build_features(df)

    with signal_cfg(WINNER_PATCH):
        labeled = build_labeled_set(df, feats)

    rows = []
    for ts, row in labeled.iterrows():
        i = int(row["entry_idx"])
        ep = float(df.iloc[i]["open_bid"])
        r = simulate_fixed_tpsl(df, i, -1, ep, tp=TP, sl=SL, horizon=H)
        rows.append(
            {
                "signal_ts": ts,
                "pnl": r.pnl,
                "exit_reason": r.exit_reason,
                "win": r.pnl > 0,
                "prev_15m_body": float(feats.loc[ts, "prev_15m_body"]),
            }
        )

    tdf = pd.DataFrame(rows)
    if tdf.empty:
        print("No trades.")
        return

    print(f"\nTrades     : {len(tdf)}")
    print(f"Win rate   : {tdf['win'].mean()*100:.1f}%")
    print(f"Net PnL    : {tdf['pnl'].sum():+.1f}")
    print(f"Avg/trade  : {tdf['pnl'].mean():+.2f}")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"  {reason:12s}: {len(g):4d}  WR={g['win'].mean()*100:.0f}%  PnL={g['pnl'].sum():+.1f}")

    out = PROJECT_ROOT / "runtime" / "v16_dip_short_winner_trades.csv"
    tdf.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
