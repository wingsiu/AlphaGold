#!/usr/bin/env python3
"""
Train + A/B each new pattern against the 2398 baseline (one at a time).

Usage:
  .venv/bin/python3 try_add_pattern.py 2025-06-01 2026-05-23
  .venv/bin/python3 try_add_pattern.py 2025-06-01 2026-05-23 reversal_wick_long
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from v14._paths import PROJECT_ROOT

import pandas as pd

from config.v14_patterns import (
    BASELINE_PATTERNS,
    PATTERN_REGISTRY,
    REVERSAL_TRIAL_PATTERNS,
)

TRADES_CSV = PROJECT_ROOT / "runtime" / "v14_pattern_backtest_trades.csv"
OUT_CSV = PROJECT_ROOT / "runtime" / "try_add_pattern_results.csv"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def backtest_stats(patterns: list[str], bt_start: str, bt_end: str) -> dict:
    run(
        [
            sys.executable,
            str(PROJECT_ROOT / "v14" / "backtest" / "backtest_patterns_v14.py"),
            bt_start,
            bt_end,
            *patterns,
        ]
    )
    tdf = pd.read_csv(TRADES_CSV)
    if tdf.empty:
        return {"trades": 0, "pnl": 0.0, "wr": 0.0}
    return {
        "trades": len(tdf),
        "pnl": round(float(tdf["pnl"].sum()), 1),
        "wr": round(float((tdf["pnl"] > 0).mean() * 100), 1),
    }


def pattern_only_stats(trial: str) -> dict:
    tdf = pd.read_csv(TRADES_CSV)
    sub = tdf[tdf["pattern"] == trial] if "pattern" in tdf.columns else tdf.iloc[0:0]
    if sub.empty:
        return {"trial_trades": 0, "trial_pnl": 0.0}
    return {
        "trial_trades": len(sub),
        "trial_pnl": round(float(sub["pnl"].sum()), 1),
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    bt_start = "2025-06-01"
    bt_end = "2026-05-23"
    if len(args) >= 2:
        bt_start, bt_end = args[0], args[1]
        trials = [a for a in args[2:] if a in REVERSAL_TRIAL_PATTERNS]
        if not trials:
            trials = list(REVERSAL_TRIAL_PATTERNS)
    elif len(args) == 1:
        if args[0] in PATTERN_REGISTRY:
            trials = [args[0]]
        else:
            bt_start = args[0]
            trials = list(REVERSAL_TRIAL_PATTERNS)
    else:
        trials = list(REVERSAL_TRIAL_PATTERNS)

    trials = [t for t in trials if t in REVERSAL_TRIAL_PATTERNS]
    if not trials:
        print("No trial patterns. Choose from:", ", ".join(REVERSAL_TRIAL_PATTERNS))
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  Add-pattern A/B  |  {bt_start} → {bt_end}")
    print(f"  Baseline: {', '.join(BASELINE_PATTERNS)}")
    print(f"  Trials  : {', '.join(trials)}")
    print(f"{'='*70}\n")

    print("Baseline backtest…")
    base = backtest_stats(list(BASELINE_PATTERNS), bt_start, bt_end)
    print(f"  BASE  {base['trades']:4d} trades  WR={base['wr']:5.1f}%  PnL={base['pnl']:+.1f}\n")

    rows: list[dict] = []
    for trial in trials:
        spec = PATTERN_REGISTRY[trial]
        rule = spec["pattern"][0]
        print(f"--- {trial}: {rule['feat']} {rule['op']} {rule['val']}  pa={spec.get('pa_groups')} ---")
        print("  Training…")
        run([sys.executable, str(PROJECT_ROOT / "train_patterns_v14.py"), trial])
        print("  Backtest baseline + trial…")
        combined = backtest_stats([*BASELINE_PATTERNS, trial], bt_start, bt_end)
        extra = pattern_only_stats(trial)
        row = {
            "trial": trial,
            "rule": f"{rule['feat']}{rule['op']}{rule['val']}",
            "baseline_trades": base["trades"],
            "baseline_pnl": base["pnl"],
            "combined_trades": combined["trades"],
            "combined_pnl": combined["pnl"],
            "delta_trades": combined["trades"] - base["trades"],
            "delta_pnl": round(combined["pnl"] - base["pnl"], 1),
            **extra,
        }
        rows.append(row)
        print(
            f"  COMB  {combined['trades']:4d} trades  PnL={combined['pnl']:+.1f}  "
            f"(Δ{row['delta_trades']:+d} tr, Δ{row['delta_pnl']:+.1f})  "
            f"trial-only: {extra['trial_trades']} tr / {extra['trial_pnl']:+.1f}\n"
        )

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
