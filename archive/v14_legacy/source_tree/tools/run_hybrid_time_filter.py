#!/usr/bin/env python3
"""
Two-pass v10-style time filter for hybrid (pattern + energetic combined).

Pass 1 — hybrid backtest without time filter → session heatmaps on ALL trades
Pass 2 — save weak cells → re-run hybrid with filter on both legs

Usage:
  .venv/bin/python3 run_hybrid_time_filter.py 2025-06-01 2026-05-23
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from _paths import PROJECT_ROOT
TRADES_CSV = PROJECT_ROOT / "runtime" / "v14_pattern_backtest_trades.csv"
BASELINE_TRADES_CSV = PROJECT_ROOT / "v14" / "runtime" / "results" / "v14_hybrid_trades_baseline.csv"
FILTER_JSON = PROJECT_ROOT / "runtime" / "v14_weak_time_slots.json"

from config.hybrid_config import TIME_FILTER_CONFIG
from xgboost_filter_model.time_slot_filter import (
    find_bad_slots,
    print_blocked_cells,
    save_weak_filter,
)

MIN_TRADES = int(TIME_FILTER_CONFIG.get("min_trades", 3))
MIN_TRADES_EXCLUSIVE = bool(TIME_FILTER_CONFIG.get("min_trades_exclusive", True))
MAX_TOTAL_PNL = float(TIME_FILTER_CONFIG.get("max_total_pnl", 0.0))
MAX_WIN_RATE = float(TIME_FILTER_CONFIG.get("max_win_rate", 40.0))
REQUIRE_WR = bool(TIME_FILTER_CONFIG.get("require_low_win_rate", False))


def parse_summary(stdout: str) -> tuple[int, float, float]:
    trades, wr, pnl = 0, 0.0, 0.0
    for line in stdout.split("\n"):
        if "Trades   :" in line or "Trades       :" in line:
            m = re.search(r"Trades\s*:\s*(\d+)", line)
            if m:
                trades = int(m.group(1))
        elif "Win rate :" in line or "Win Rate     :" in line:
            m = re.search(r"Win [Rr]ate\s*:\s*([0-9.]+)", line)
            if m:
                wr = float(m.group(1))
        elif "Net PnL  :" in line or "Net PnL      :" in line:
            m = re.search(r"Net PnL\s*:\s*([0-9.+-]+)", line)
            if m:
                pnl = float(m.group(1))
    return trades, wr, pnl


def run_hybrid(date_args: list[str], *, use_filter: bool) -> tuple[int, float, float]:
    env = os.environ.copy()
    env["V14_HYBRID"] = "1"
    env.setdefault("V14_FVG_MIN_GAP", "0")
    if use_filter:
        env.pop("V14_NO_TIME_FILTER", None)
        env["V14_TIME_FILTER_JSON"] = str(FILTER_JSON)
    else:
        env["V14_NO_TIME_FILTER"] = "1"
        env.pop("V14_TIME_FILTER_JSON", None)
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "oil" / "backtest" / "pattern_backtest.py"), *date_args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr[-4000:] if r.stderr else r.stdout[-4000:])
        r.check_returncode()
    return parse_summary(r.stdout)


def main() -> None:
    date_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    sweep_wr = "--no-wr" in sys.argv
    require_wr = False if sweep_wr else REQUIRE_WR

    rule = f"trades>{MIN_TRADES}, pnl<{MAX_TOTAL_PNL}"
    if require_wr:
        rule += f", wr<{MAX_WIN_RATE}%"
    print("Hybrid v10-style time filter (combined pattern + energetic heatmaps)")
    print(f"Weak-cell rule: {rule}\n")

    print("Pass 1: hybrid baseline (no time filter)...")
    t1, wr1, pnl1 = run_hybrid(date_args, use_filter=False)
    print(f"  Baseline: trades={t1}  WR={wr1:.1f}%  PnL={pnl1:+.1f}\n")

    if not TRADES_CSV.exists():
        print(f"ERROR: missing {TRADES_CSV}")
        sys.exit(1)
    shutil.copy(TRADES_CSV, BASELINE_TRADES_CSV)

    cells = find_bad_slots(
        BASELINE_TRADES_CSV,
        min_trades=MIN_TRADES,
        min_trades_exclusive=MIN_TRADES_EXCLUSIVE,
        max_total_pnl=MAX_TOTAL_PNL,
        max_win_rate=MAX_WIN_RATE,
        require_low_win_rate=require_wr,
    )
    save_weak_filter(cells, FILTER_JSON)
    print(f"Blocked slots ({len(cells)}):")
    print_blocked_cells(cells)

    print("\nPass 2: hybrid with time filter (both legs)...")
    t2, wr2, pnl2 = run_hybrid(date_args, use_filter=True)
    print(f"  Filtered: trades={t2}  WR={wr2:.1f}%  PnL={pnl2:+.1f}")
    print(f"  Delta PnL: {pnl2 - pnl1:+.1f}")

    out = PROJECT_ROOT / "time_filter_hybrid_results.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("pass,trades,win_rate,net_pnl,blocked_cells\n")
        f.write(f"baseline,{t1},{wr1},{pnl1},0\n")
        f.write(f"filtered,{t2},{wr2},{pnl2},{len(cells)}\n")
    print(f"\nSaved {out} and {FILTER_JSON}")


if __name__ == "__main__":
    main()
