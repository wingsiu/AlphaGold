#!/usr/bin/env python3
"""
Backtest oil pattern(s) on the gold-style holdout window (default 2025-06-01 → 2026-05-23).

Uses filter_prod.joblib only when cycle files are absent (prod-only testing).

Usage:
  PYTHONPATH=. V14_NO_TIME_FILTER=1 .venv/bin/python3 oil/tools/backtest.py
  PYTHONPATH=. V14_NO_TIME_FILTER=1 .venv/bin/python3 oil/tools/backtest.py oil_downtrend_retrace
  PYTHONPATH=. .venv/bin/python3 oil/tools/backtest.py 2025-06-01 2026-05-23 oil_downtrend_retrace
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oil.bootstrap import apply_oil_registry
from oil.config import PRICE_TABLE, TEST_END, TEST_START, TRADES_CSV
from oil.patterns import PATTERN_REGISTRY

apply_oil_registry()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    bt_start = TEST_START
    bt_end = TEST_END
    patterns: list[str] = []
    if len(args) >= 2 and args[0][:4] == "2025":
        bt_start, bt_end = args[0], args[1]
        patterns = [a for a in args[2:] if a in PATTERN_REGISTRY]
    else:
        patterns = [a for a in args if a in PATTERN_REGISTRY]
    if not patterns:
        patterns = list(PATTERN_REGISTRY.keys())

    env = os.environ.copy()
    env.setdefault("V14_NO_TIME_FILTER", "1")
    env["V14_HYBRID"] = "0"
    env["V14_PRICE_TABLE"] = PRICE_TABLE
    env["V14_BT_TRADES_OUT"] = str(TRADES_CSV.relative_to(ROOT))

    cmd = [
        sys.executable,
        str(ROOT / "oil" / "backtest" / "pattern_backtest.py"),
        bt_start,
        bt_end,
        *patterns,
    ]
    print("Oil backtest:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)

    gold_csv = TRADES_CSV
    if gold_csv.exists():
        import pandas as pd

        tdf = pd.read_csv(gold_csv)
        if not tdf.empty and "pnl" in tdf.columns:
            wins = int((tdf["pnl"] > 0).sum())
            net = float(tdf["pnl"].sum())
            wr = 100.0 * wins / len(tdf)
            print(f"\n{'='*60}")
            print(f"  OIL BACKTEST  {bt_start} → {bt_end}  patterns={','.join(patterns)}")
            print(f"{'='*60}")
            print(f"  Trades   : {len(tdf)}")
            print(f"  Win rate : {wr:.1f}%")
            print(f"  Net PnL  : {net:+.2f}  (DB price units; ÷100 ≈ spot $ per contract)")
            print(f"  Avg/trade: {net/len(tdf):+.2f}")
            if "pattern" in tdf.columns:
                print("\n  By pattern:")
                for name, g in tdf.groupby("pattern"):
                    print(
                        f"    {name}: {len(g)} trades  PnL={g['pnl'].sum():+.1f}  "
                        f"WR={(g['pnl']>0).mean()*100:.0f}%"
                    )

    if not TRADES_CSV.exists():
        print(f"Warning: expected {TRADES_CSV}")


if __name__ == "__main__":
    main()
