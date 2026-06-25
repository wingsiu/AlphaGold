#!/usr/bin/env python3
"""
Bi-weekly hybrid retrain (intended workflow).

1. Train S1 / S2 / pattern models — incremental only (latest WF cycle).
   Uses all bars strictly BEFORE that cycle start. Older cycle_*.joblib unchanged.
   Training never uses the weak time filter (labels are bar-based).

2. Run hybrid backtest WITHOUT weak filter (history before cycle start) → heatmaps.

3. Save runtime/bot_assets/wf_time_filters/weak_time_slots_cycle_{n}_{date}.json
   and symlink runtime/hybrid_weak_time_slots.json → that file.

4. Optional: print filtered backtest summary (verification, per-cycle filters).

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

import pandas as pd

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


def _weak_filter_targets() -> list[tuple[int, pd.Timestamp]]:
    from xgboost_filter_model.pattern_training import (
        iter_wf_cycles,
        wf_anchor_ts,
        wf_incremental_train_target,
        wf_train_as_of,
        wf_train_mode,
    )

    pending = wf_incremental_train_target()
    if pending:
        return [pending]
    if wf_train_mode() == "full":
        anchor = wf_anchor_ts()
        as_of = wf_train_as_of()
        return [(c, s) for c, s, _ in iter_wf_cycles(anchor, as_of, anchor)]
    return []


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

    print("Hybrid WF retrain: incremental models → unfiltered backtest → per-cycle weak filter")
    print(f"  Verification backtest window: {date_args[0]} → {date_args[-1]}")
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
        PROJECT_ROOT / "tools" / "train_patterns.py",
    )

    targets = _weak_filter_targets()
    if not targets:
        raise SystemExit("No WF cycle target for weak-filter rebuild")

    print("\n=== Weak time filter (unfiltered backtest → per-cycle JSON) ===")
    last_cycle_path = None
    for cycle, cycle_start in targets:
        print(f"\nPass 1: cycle_{cycle} baseline (no time filter)...")
        cells = rhtf.build_weak_filter_for_cycle(cycle, cycle_start)
        cycle_path = rhtf.save_weak_filter_cycle(cells, cycle, cycle_start.date())
        rhtf.publish_current_weak_filter(cycle_path)
        rhtf.save_weak_filter(cells, rhtf.FILTER_JSON)
        last_cycle_path = cycle_path
        print(f"  Blocked slots ({len(cells)}):")
        rhtf.print_blocked_cells(cells)
        print(f"  Saved {cycle_path}")

    print("\nPass 2: hybrid with per-cycle weak filters (verification)...")
    t1, wr1, pnl1 = rhtf.run_hybrid(date_args, use_filter=False)
    t2, wr2, pnl2 = rhtf.run_hybrid(date_args, use_filter=True, per_cycle_filter=True)
    print(f"  Baseline: trades={t1}  WR={wr1:.1f}%  PnL={pnl1:+.1f}")
    print(f"  Filtered: trades={t2}  WR={wr2:.1f}%  PnL={pnl2:+.1f}")
    print(f"  Delta PnL: {pnl2 - pnl1:+.1f}")
    if last_cycle_path:
        print(f"\nPublished {rhtf.FILTER_JSON} → {last_cycle_path.name}")


if __name__ == "__main__":
    main()
