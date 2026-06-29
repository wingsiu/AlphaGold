#!/usr/bin/env python3
"""Display full extended stats for gold v16 combined backtest CSV."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from v16.backtest.report import print_full_stats, trades_to_dataframe

CSV = ROOT / "runtime" / "gold_v16_combined_trades.csv"
HYBRID_FMT = ROOT / "runtime" / "gold_v16_combined_trades_hybrid_fmt.csv"


def main() -> None:
    if not CSV.exists():
        print(f"Missing {CSV} — run: PYTHONPATH=. V14_HYBRID=1 python3 run_gold_v16_backtest.py")
        sys.exit(1)
    tdf = pd.read_csv(CSV)
    tdf["entry"] = pd.to_datetime(tdf["entry"], utc=True)
    tdf["exit"] = pd.to_datetime(tdf["exit"], utc=True)
    start = tdf["entry"].min().strftime("%Y-%m-%d")
    end = tdf["entry"].max().strftime("%Y-%m-%d")
    rows = tdf.to_dict("records")
    fmt = trades_to_dataframe(rows)
    fmt.to_csv(HYBRID_FMT, index=False)
    print_full_stats(
        fmt,
        title="GOLD v16 COMBINED — Full Statistics",
        start=start,
        end=end,
        csv_path=str(HYBRID_FMT),
        show_all_trades="--all" in sys.argv,
    )
    print("Done.")


if __name__ == "__main__":
    main()
