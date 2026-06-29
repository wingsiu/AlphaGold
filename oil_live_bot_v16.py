#!/usr/bin/env python3
"""Oil v16 bot — replay mode (parity vs backtest). Live IG wiring TBD.

Usage:
  PYTHONPATH=. python3 oil_live_bot_v16.py --replay [start] [end]
  PYTHONPATH=. python3 oil_live_bot_v16.py --replay --quick   # last 7 days
  PYTHONPATH=. python3 _check_oil_v16_parity.py 2024-07-01 2024-07-31
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v16.config.oil_config import BACKTEST, OIL_LEG_MODELS
from v16.oil.parity_check import run_parity


def main() -> None:
    p = argparse.ArgumentParser(description="Oil v16 bot (replay / parity)")
    p.add_argument("--replay", action="store_true", help="Run minute replay vs backtest parity")
    p.add_argument("--quick", action="store_true", help="Replay last 7 HKT days only")
    p.add_argument("--struct-hold", action="store_true", default=True, help="WR90 struct-hold exit (default)")
    p.add_argument("--fixed-tpsl", action="store_true", help="WR90 fixed TP/SL exit")
    p.add_argument("start", nargs="?", default=None)
    p.add_argument("end", nargs="?", default=None)
    args = p.parse_args()

    if not args.replay:
        print("Oil v16 live IG mode not wired yet. Use --replay to validate vs backtest.")
        print("  PYTHONPATH=. python3 oil_live_bot_v16.py --replay 2024-07-01 2024-07-31")
        sys.exit(1)

    if args.quick:
        end_d = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        start_d = end_d - timedelta(days=6)
        start, end = str(start_d), str(end_d)
    else:
        start = args.start or BACKTEST["default_start"]
        end = args.end or (args.start or BACKTEST["default_end"])

    wr90_exit = "fixed_tpsl" if args.fixed_tpsl else "struct_hold"

    print("=" * 72)
    print("  OIL v16 REPLAY BOT")
    print(f"  Period: {start} → {end}")
    print(f"  WR90 exit: {wr90_exit}")
    print(f"  Legs: WR90 + ret + ret_short + long_ret + SI")
    print(f"  Models: {OIL_LEG_MODELS}")
    print("=" * 72)

    report = run_parity(start, end, wr90_exit=wr90_exit)
    print(report)
    print("Reports: runtime/oil_v16_parity_latest.txt")
    print("         runtime/oil_v16_replay_trades.csv")


if __name__ == "__main__":
    main()
