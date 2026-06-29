#!/usr/bin/env python3
"""
15m double-down dip long backtest.

Rule (long only):
  - Previous completed 15m bar closed DOWN
  - Current 15m slot is DOWN (price below slot open)
  - Price is >= 5 pts below slot open
  - minute_in_15m < 10
  - Slot running low is >= 10 pts below slot open

Usage:
  PYTHONPATH=. python3 v16/research/dip_long_15m_backtest.py 2025-06-01 2026-06-25
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
from v16.backtest.scaleout_sim import simulate_scaleout_trade
from v16.backtest.signals import build_labeled_set, candidate_mask, dip_long_15m_mask, _exit_kwargs
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m


def _run_long_only(df: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    kw = _exit_kwargs()
    for ts, row in labeled.iterrows():
        entry_idx = int(row["entry_idx"])
        nxt = df.iloc[entry_idx]
        ep = float(nxt["open_ask"])
        res = simulate_scaleout_trade(df, entry_idx, 1, ep, **kw)
        rows.append(
            {
                "signal_ts": ts,
                "side": 1,
                "minute_in_15m": int(row["minute_in_15m"]),
                "dip_from_slot_open": float(row["dip_from_slot_open"]),
                "slot_low_dip": float(row["slot_low_dip"]),
                "pnl": res.pnl,
                "exit_reason": res.exit_reason,
                "scaled_half": res.scaled_half,
                "win": res.pnl > 0,
            }
        )
    return pd.DataFrame(rows)


def _print_stats(name: str, tdf: pd.DataFrame) -> None:
    if tdf.empty:
        print(f"\n{name}: no trades")
        return
    print(f"\n{name}")
    print(f"  Trades     : {len(tdf)}")
    print(f"  Win rate   : {tdf['win'].mean()*100:.1f}%")
    print(f"  Scaled 50% : {tdf['scaled_half'].mean()*100:.1f}%")
    print(f"  Net PnL    : {tdf['pnl'].sum():+.1f}")
    print(f"  Avg/trade  : {tdf['pnl'].mean():+.2f}")
    print(f"  Avg dip    : {tdf['dip_from_slot_open'].mean():+.2f} pts below open")
    for reason, grp in tdf.groupby("exit_reason"):
        print(
            f"    {reason:12s}: {len(grp):4d}  "
            f"WR={grp['win'].mean()*100:.0f}%  PnL={grp['pnl'].sum():+.1f}"
        )


def _count_mask(df: pd.DataFrame, feats: pd.DataFrame, mask_fn) -> int:
    return int(mask_fn(feats, df.index).sum())


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    bt_start = args[0] if args else "2025-06-01"
    bt_end = args[1] if len(args) > 1 else pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    cfg = v16_config.SIGNAL_CONFIG

    print("=" * 70)
    print(f"  15m DOUBLE-DOWN DIP LONG  |  {bt_start} → {bt_end}")
    print(
        f"  prev 15m DOWN | slot DOWN | dip>={cfg['dip_min_below_open_pts']} pts | "
        f"slot low dip>={cfg['dip_min_slot_low_pts']} | minute<{cfg['dip_max_minute_in_slot']}"
    )
    print(
        f"  Exit: +{v16_config.EXIT_CONFIG['first_scale_pnl']:.0f} half | "
        f"+{v16_config.EXIT_CONFIG['final_scale_pnl']:.0f} all | "
        f"SL={v16_config.EXIT_CONFIG['initial_sl']:.0f}"
    )
    print("=" * 70)

    df = load_gold_1m(bt_start, bt_end)
    feats = build_features(df)

    old_mode = cfg["mode"]
    v16_config.SIGNAL_CONFIG["mode"] = "dip_long_15m"
    try:
        labeled = build_labeled_set(df, feats)
        n = _count_mask(df, feats, dip_long_15m_mask)
        print(f"\nSignal bars: {n}  (labeled entries: {len(labeled)})")

        if labeled.empty:
            print("No signals — loosen dip thresholds or widen date range.")
            return

        trades = _run_long_only(df, labeled)
        _print_stats("Dip long (mechanical)", trades)

        # Variant: minute-only (drop slot-low >= 10 requirement)
        saved = v16_config.SIGNAL_CONFIG["dip_require_slot_low"]
        v16_config.SIGNAL_CONFIG["dip_require_slot_low"] = False
        labeled2 = build_labeled_set(df, feats)
        _print_stats("Dip long (minute<10 only, no slot-low filter)", _run_long_only(df, labeled2))
        v16_config.SIGNAL_CONFIG["dip_require_slot_low"] = saved

        out = PROJECT_ROOT / "runtime" / "v16_dip_long_15m_trades.csv"
        trades.to_csv(out, index=False)
        print(f"\nSaved -> {out}")
    finally:
        v16_config.SIGNAL_CONFIG["mode"] = old_mode


if __name__ == "__main__":
    main()
