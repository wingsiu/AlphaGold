#!/usr/bin/env python3
"""Gold v16 — replay parity vs combined backtest, or live via trading_bot_gold_v16.py.

Replay (validate bot logic matches backtest):
  python3 gold_live_bot_v16.py --replay 2025-06-01 2026-06-25
  python3 gold_live_bot_v16.py --replay --quick
  python3 _check_gold_v16_parity.py 2025-06-01 2026-06-25

Live (IG):
  python3 trading_bot_gold_v16.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v16.config.gold_config import BACKTEST
from v16.gold.parity_check import run_parity


def main() -> None:
    p = argparse.ArgumentParser(description="Gold v16 replay / parity")
    p.add_argument("--replay", action="store_true", help="Run replay vs backtest parity")
    p.add_argument("--quick", action="store_true", help="Replay last 7 days only")
    p.add_argument("start", nargs="?", default=None)
    p.add_argument("end", nargs="?", default=None)
    args = p.parse_args()

    if not args.replay:
        print("Gold v16 live runs via trading_bot_gold_v16.py (hybrid + momentum + dip).")
        print("Use --replay to validate vs combined backtest:")
        print("  python3 gold_live_bot_v16.py --replay 2025-06-01 2026-06-25")
        sys.exit(0)

    if args.quick:
        end_d = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        start_d = end_d - timedelta(days=6)
        start, end = str(start_d), str(end_d)
    else:
        start = args.start or BACKTEST["default_start"]
        end = args.end or (args.start or BACKTEST["default_end"])

    print("=" * 72)
    print("  GOLD v16 REPLAY / PARITY")
    print(f"  Period: {start} → {end}")
    print("  Stack: hybrid + momentum + dip short")
    print("=" * 72)

    report = run_parity(start, end)
    print(report)
    print("Reports: runtime/gold_v16_parity_latest.txt")
    print("         runtime/gold_v16_replay_trades.csv")


if __name__ == "__main__":
    main()
