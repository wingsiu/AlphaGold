#!/usr/bin/env python3
"""
Pattern + energetic fallback backtest (two-bot logic).

Bot 1 — Pattern router (6 production patterns):
  - Scores all bars, priority routing, per-pattern TP/SL/H.

Bot 2 — Energetic S1/S2 fallback:
  - Only enters when flat AND no pattern signal / no pattern position.
  - Uses ENERGETIC_EXECUTION_CONFIG (reverse exit, global refresh).

Time filter (v10 weak slots) applies to BOTH legs when enabled in
config/v14_config.py (default ON → runtime/v14_weak_time_slots.json).

Usage:
  .venv/bin/python3 run_hybrid_backtest.py
  .venv/bin/python3 run_hybrid_backtest.py 2025-06-01 2026-05-23

Disable time filter:
  V14_NO_TIME_FILTER=1 .venv/bin/python3 run_hybrid_backtest.py

Rebuild weak slots from hybrid baseline then re-run filtered:
  .venv/bin/python3 run_hybrid_time_filter.py 2025-06-01 2026-05-23

Trades CSV: runtime/v14_pattern_backtest_trades.csv
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

os.environ.setdefault("V14_HYBRID", "1")
os.environ.setdefault("V14_FVG_MIN_GAP", "0")

from run_pattern_backtest import TRADES_CSV, print_full_stats  # noqa: E402
from xgboost_filter_model.time_slot_filter import resolve_v14_time_filter_path  # noqa: E402

import pandas as pd  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    cmd = [sys.executable, str(PROJECT_ROOT / "v14" / "backtest" / "backtest_patterns_v14.py"), *argv]
    filter_path = resolve_v14_time_filter_path(PROJECT_ROOT)
    if filter_path:
        print(f"Hybrid backtest: pattern-first → energetic fallback + time filter\n  ({filter_path})\n")
    else:
        print("Hybrid backtest: pattern-first → energetic fallback (time filter OFF)\n")

    csv_mtime_before = TRADES_CSV.stat().st_mtime if TRADES_CSV.exists() else None
    subprocess.run(cmd, cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}, check=True)
    csv_updated = (
        TRADES_CSV.exists()
        and (csv_mtime_before is None or TRADES_CSV.stat().st_mtime > csv_mtime_before)
    )

    if not csv_updated:
        print("\nNo trades written for this window (no entry signals or early exit).")
        print("Previous CSV was left unchanged — stats below are NOT for this run.\n")
        sys.exit(0)

    tdf = pd.read_csv(TRADES_CSV)
    if tdf.empty:
        print("No trades in CSV.")
        sys.exit(0)

    print_full_stats(tdf)


if __name__ == "__main__":
    main()
