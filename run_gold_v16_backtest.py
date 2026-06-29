#!/usr/bin/env python3
"""Gold v16 combined backtest — full statistics (hybrid-style report).

One command (backtest + full stats + all trades + saved report):
  python3 run_gold_v16_backtest.py
  python3 run_hybrid_backtest.py

Optional:
  python3 run_hybrid_backtest.py 2025-06-01 2026-06-25
  python3 run_hybrid_backtest.py --last30   # stats but only last 30 trades
  python3 run_hybrid_backtest.py --quick    # skip full stats
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("V14_HYBRID", "1")

from v16.backtest.report import print_full_stats, tee_stdout, trades_to_dataframe
from v16.config.gold_config import BACKTEST, GOLD_TRAIN_START
from v16.gold.combined_run import run_gold_v16_combined, save_combined_trades

STATS_OUT = ROOT / "runtime" / "gold_v16_full_statistics.txt"


def _print_premerge(run_stats: dict) -> None:
    print("\n--- Pre-merge legs ---")
    hybrid = run_stats.get("hybrid", {})
    mom = run_stats.get("v16_momentum", {})
    dip = run_stats.get("v16_dip_short", {})
    h_n = sum(v["trades"] for v in hybrid.values()) if hybrid else 0
    h_pnl = sum(v["pnl"] for v in hybrid.values()) if hybrid else 0.0
    print(f"  {'hybrid patterns+energetic':28s} {h_n:4d} trades  PnL={h_pnl:+8.1f}")
    print(f"  {'v16 momentum':28s} {mom.get('trades', 0):4d} trades  PnL={mom.get('pnl', 0):+.1f}")
    print(f"  {'v16 dip short':28s} {dip.get('trades', 0):4d} trades  PnL={dip.get('pnl', 0):+.1f}")
    raw_n = run_stats.get("raw_n", 0)
    raw_pnl = h_pnl + mom.get("pnl", 0) + dip.get("pnl", 0)
    print(f"  {'raw total':28s} {raw_n:4d} trades  PnL={raw_pnl:+.1f}")
    print(f"\n  Dropped by merge: {raw_n - run_stats.get('merged_n', 0)}")


def _run_backtest() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    start = args[0] if args else BACKTEST["default_start"]
    end = args[1] if len(args) > 1 else BACKTEST["default_end"]
    show_all = "--last30" not in flags and "--tail30" not in flags
    quick = "--quick" in flags

    print("=" * 72)
    print("  GOLD v16 COMBINED BACKTEST")
    print(f"  Stack: hybrid + momentum + dip short (single slot)")
    print("=" * 72)

    merged, run_stats = run_gold_v16_combined(
        GOLD_TRAIN_START, end, oos_start=start, verbose=True
    )
    _print_premerge(run_stats)

    out = save_combined_trades(merged)
    fmt_path = out.with_name("gold_v16_combined_trades_hybrid_fmt.csv")
    trades_to_dataframe(merged).to_csv(fmt_path, index=False)

    if not quick:
        print_full_stats(
            merged,
            title="GOLD v16 COMBINED (merged portfolio)",
            start=start,
            end=end,
            csv_path=str(fmt_path),
            show_all_trades=show_all,
        )

    print(f"\n  CSV: {out}")
    print("DONE.")


def main() -> None:
    with tee_stdout(STATS_OUT):
        _run_backtest()
    print(f"\n  Full stats saved: {STATS_OUT}")


if __name__ == "__main__":
    main()
