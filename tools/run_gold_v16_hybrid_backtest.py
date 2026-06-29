#!/usr/bin/env python3
"""Run v16 gold hybrid backtest for a window — mobile-API compatible CSV output."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from v16.gold.hybrid_legs import run_hybrid_legs

OUT = ROOT / "runtime" / "gold_v16_hybrid_backtest_trades.csv"


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: run_gold_v16_hybrid_backtest.py START END", file=sys.stderr)
        sys.exit(1)
    start, end = sys.argv[1], sys.argv[2]
    start_d = start.split("T")[0]
    end_d = end.split("T")[0]
    trades = run_hybrid_legs(start_d, end_d, verbose=True)
    rows = []
    for t in trades:
        typ = str(t.get("type", t.get("_leg", "pattern")))
        src = typ if typ == "energetic" else "pattern"
        rows.append(
            {
                "entry_time": pd.Timestamp(t["entry"]).tz_convert("UTC"),
                "exit_time": pd.Timestamp(t["exit"]).tz_convert("UTC"),
                "pnl": float(t["pnl"]),
                "side": int(t["side"]),
                "source": src,
                "matched_pattern": typ if src == "pattern" else pd.NA,
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {len(rows)} trades -> {OUT}")


if __name__ == "__main__":
    main()
