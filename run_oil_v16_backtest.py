#!/usr/bin/env python3
"""Oil v16 combined backtest — full statistics (same report as gold).

One command (backtest + full stats + all trades + saved report):
  python3 run_oil_v16_backtest.py
  python3 run_oil_backtest.py

Optional:
  python3 run_oil_backtest.py 2024-07-01 2026-06-30 --struct-hold
  python3 run_oil_backtest.py --last30
  python3 run_oil_backtest.py --quick
  python3 run_oil_backtest.py --rip
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from v16.backtest.report import print_full_stats, tee_stdout, trades_to_dataframe
from v16.config.oil_config import BACKTEST, OIL_LEG_MODELS
from v16.oil.combined_run import leg_stats_table, run_oil_v16_combined

STATS_OUT = ROOT / "runtime" / "oil_v16_full_statistics.txt"


def _run_backtest() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    start = args[0] if args else BACKTEST["default_start"]
    end = args[1] if len(args) > 1 else BACKTEST["default_end"]
    wr90_exit = "fixed_tpsl" if "--fixed-tpsl" in flags else "struct_hold"
    include_rip = BACKTEST.get("include_rip_short", False) and "--no-rip" not in flags
    if "--rip" in flags:
        include_rip = True
    show_all = "--last30" not in flags and "--tail30" not in flags
    quick = "--quick" in flags

    print("=" * 72)
    print("  OIL v16 COMBINED BACKTEST")
    print(f"  Period: {start} → {end}")
    print(f"  WR90 exit: {wr90_exit}  |  Models: {OIL_LEG_MODELS}")
    print(f"  Legs: WR90 + ret + ret_short + long_ret + SI" + (" + rip" if include_rip else ""))
    print("=" * 72)

    merged, stats = run_oil_v16_combined(
        start, end, wr90_exit=wr90_exit, include_rip=include_rip
    )
    print(leg_stats_table(stats))

    raw_n = stats.get("_raw", 0)
    merged_n = len(merged)
    if raw_n and raw_n > merged_n:
        print(f"\n  Dropped by merge: {raw_n - merged_n}")

    csv_path = ROOT / BACKTEST["trades_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fmt_path = csv_path.with_name("oil_v16_combined_trades_hybrid_fmt.csv")
    if merged:
        pd.DataFrame(merged).to_csv(csv_path, index=False)
        trades_to_dataframe(merged, asset="oil").to_csv(fmt_path, index=False)

    if not quick:
        print_full_stats(
            merged,
            title="OIL v16 COMBINED (merged portfolio)",
            start=start,
            end=end,
            csv_path=str(fmt_path) if merged else None,
            show_all_trades=show_all,
            asset="oil",
        )

    print(f"\n  CSV: {csv_path}")
    print("DONE.")


def main() -> None:
    with tee_stdout(STATS_OUT):
        _run_backtest()
    print(f"\n  Full stats saved: {STATS_OUT}")


if __name__ == "__main__":
    main()
