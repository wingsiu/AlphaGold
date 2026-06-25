#!/usr/bin/env python3
"""
Bi-weekly hybrid retrain (intended workflow).

1. Train S1 / S2 / pattern models — incremental only (latest WF cycle).
   Uses all bars strictly BEFORE that cycle start. Older cycle_*.joblib unchanged.
   Training never uses the weak time filter (labels are bar-based).

2. Run hybrid backtest WITHOUT weak filter → trade heatmaps.

3. Rebuild runtime/hybrid_weak_time_slots.json from those trades.

4. Optional: print filtered backtest summary (verification).

Env (defaults):
  V14_WF_TRAIN_MODE=incremental
  V14_WF_FORCE_LATEST=1  — only if you must overwrite an existing cycle file

Usage:
  PYTHONPATH=. .venv/bin/python3 tools/retrain_hybrid_wf.py
  PYTHONPATH=. .venv/bin/python3 tools/retrain_hybrid_wf.py 2025-06-01 2026-05-23
  V14_WF_TRAIN_MODE=full .venv/bin/python3 tools/retrain_hybrid_wf.py  # bootstrap all cycles
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from _paths import PROJECT_ROOT

# Reuse weak-filter builder from run_hybrid_time_filter
from tools import run_hybrid_time_filter as rhtf


def _run_script(label: str, script: Path, extra_env: dict | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("V14_WF_TRAIN_MODE", "incremental")
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    if extra_env:
        env.update(extra_env)
    print(f"\n=== {label} ===")
    res = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
    )
    if res.returncode != 0:
        raise SystemExit(f"{label} failed (exit {res.returncode})")


def main() -> None:
    date_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not date_args:
        wf_start = "2025-06-01"
        end = date.today().strftime("%Y-%m-%d")
        date_args = [wf_start, end]

    from xgboost_filter_model.pattern_training import wf_incremental_train_target

    pending = wf_incremental_train_target()
    if pending:
        c, s = pending
        print(f"WF train target: cycle_{c} start {s.date()} (data before {s.date()})")
    elif os.environ.get("V14_WF_TRAIN_MODE", "incremental") == "incremental":
        print(
            "No WF train target: current 14d cycle still active. "
            "Train only after cycle end + grace (e.g. 2026-06-06 for cycle 38). "
            "Use V14_WF_TRAIN_AS_OF=YYYY-MM-DD or wait."
        )
        if os.environ.get("V14_SKIP_TRAIN_IF_NOT_READY", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            print("Skipping model training.")
            sys.exit(0)

    print("Hybrid WF retrain: incremental models → unfiltered backtest → weak filter")
    print(f"  Backtest window for weak cells: {date_args[0]} → {date_args[-1]}")
    print(f"  WF train mode: {os.environ.get('V14_WF_TRAIN_MODE', 'incremental')}")

    _run_script(
        "Stage 1 (energetic filter)",
        PROJECT_ROOT / "xgboost_filter_model" / "train_filter_v14.py",
    )
    _run_script(
        "Stage 2 (directional)",
        PROJECT_ROOT / "xgboost_filter_model" / "train_stage2_v14_directional.py",
    )
    _run_script(
        "Pattern specialists",
        PROJECT_ROOT / "v14" / "tools" / "train_patterns_v14.py",
    )

    print("\n=== Weak time filter (from unfiltered hybrid backtest) ===")
    print("Pass 1: hybrid baseline (no time filter)...")
    t1, wr1, pnl1 = rhtf.run_hybrid(date_args, use_filter=False)
    print(f"  Baseline: trades={t1}  WR={wr1:.1f}%  PnL={pnl1:+.1f}\n")

    if not rhtf.TRADES_CSV.exists():
        raise SystemExit(f"Missing {rhtf.TRADES_CSV}")

    import shutil

    shutil.copy(rhtf.TRADES_CSV, rhtf.BASELINE_TRADES_CSV)
    cells = rhtf.find_bad_slots(
        rhtf.BASELINE_TRADES_CSV,
        min_trades=rhtf.MIN_TRADES,
        min_trades_exclusive=rhtf.MIN_TRADES_EXCLUSIVE,
        max_total_pnl=rhtf.MAX_TOTAL_PNL,
        max_win_rate=rhtf.MAX_WIN_RATE,
        require_low_win_rate=rhtf.REQUIRE_WR,
    )
    rhtf.save_weak_filter(cells, rhtf.FILTER_JSON)
    print(f"Blocked slots ({len(cells)}):")
    rhtf.print_blocked_cells(cells)

    print("\nPass 2: hybrid with new weak filter (verification)...")
    t2, wr2, pnl2 = rhtf.run_hybrid(date_args, use_filter=True)
    print(f"  Filtered: trades={t2}  WR={wr2:.1f}%  PnL={pnl2:+.1f}")
    print(f"  Delta PnL: {pnl2 - pnl1:+.1f}")
    print(f"\nSaved {rhtf.FILTER_JSON}")


if __name__ == "__main__":
    main()
