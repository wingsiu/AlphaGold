#!/usr/bin/env python3
"""
Sweep reversal_fvg_short: min_gap × time_from_fvg_bear max (combined A/B vs 4-pattern baseline).

Usage:
  PYTHONUNBUFFERED=1 .venv/bin/python3 sweep_reversal_fvg_short.py 2025-06-01 2026-05-23
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from v14._paths import PROJECT_ROOT

import pandas as pd

from config.v14_patterns import BASELINE_PATTERNS

PATTERN = "reversal_fvg_short"
TIME_ENV = "V14_REVERSAL_FVG_SHORT_TIME_FROM_FVG_BEAR_MAX"
GAPS = [0, 1, 3, 5, 10]
TIMES = [15, 20, 30, 45, 60]
TRADES_CSV = PROJECT_ROOT / "runtime" / "v14_pattern_backtest_trades.csv"
OUT_CSV = PROJECT_ROOT / "runtime" / "sweep_reversal_fvg_short.csv"


def backtest_stats(patterns: list[str], bt_start: str, bt_end: str, env: dict) -> dict:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "v14" / "backtest" / "backtest_patterns_v14.py"), bt_start, bt_end, *patterns],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    tdf = pd.read_csv(TRADES_CSV)
    if tdf.empty:
        return {"trades": 0, "pnl": 0.0, "wr": 0.0, "trial_trades": 0, "trial_pnl": 0.0}
    trial = tdf[tdf["pattern"] == PATTERN] if PATTERN in patterns else tdf.iloc[0:0]
    return {
        "trades": len(tdf),
        "pnl": round(float(tdf["pnl"].sum()), 1),
        "wr": round(float((tdf["pnl"] > 0).mean() * 100), 1),
        "trial_trades": len(trial),
        "trial_pnl": round(float(trial["pnl"].sum()), 1) if len(trial) else 0.0,
    }


def train_pattern(env: dict) -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "train_patterns_v14.py"), PATTERN],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    bt_start = "2025-06-01"
    bt_end = "2026-05-23"
    if len(args) >= 2:
        bt_start, bt_end = args[0], args[1]

    print(f"\n{'='*72}")
    print(f"  reversal_fvg_short sweep  |  {bt_start} → {bt_end}")
    print(f"  min_gap (pts): {GAPS}  |  time_max (min): {TIMES}")
    print(f"  Combined: baseline + {PATTERN}")
    print(f"{'='*72}\n")

    base_env = os.environ.copy()
    base_env.pop(TIME_ENV, None)
    base_env.pop("V14_FVG_MIN_GAP", None)

    print("Baseline (4 patterns)…", flush=True)
    base = backtest_stats(list(BASELINE_PATTERNS), bt_start, bt_end, base_env)
    print(f"  BASE  {base['trades']:4d} tr  WR={base['wr']:5.1f}%  PnL={base['pnl']:+.1f}\n", flush=True)

    rows: list[dict] = []
    combined = [*BASELINE_PATTERNS, PATTERN]

    for gap in GAPS:
        gap_env = base_env.copy()
        gap_env["V14_FVG_MIN_GAP"] = str(gap)
        print(f"=== min_gap={gap} — train once ===", flush=True)
        train_pattern(gap_env)

        for tmax in TIMES:
            run_env = gap_env.copy()
            run_env[TIME_ENV] = str(tmax)
            print(f"  time<{tmax} backtest…", flush=True)
            comb = backtest_stats(combined, bt_start, bt_end, run_env)
            row = {
                "min_gap": gap,
                "time_max": tmax,
                "baseline_trades": base["trades"],
                "baseline_pnl": base["pnl"],
                "combined_trades": comb["trades"],
                "combined_pnl": comb["pnl"],
                "delta_trades": comb["trades"] - base["trades"],
                "delta_pnl": round(comb["pnl"] - base["pnl"], 1),
                "trial_trades": comb["trial_trades"],
                "trial_pnl": comb["trial_pnl"],
            }
            rows.append(row)
            pd.DataFrame(rows).sort_values(["delta_pnl", "combined_pnl"], ascending=False).to_csv(
                OUT_CSV, index=False
            )
            print(
                f"    gap={gap:2d} time<{tmax:2d}  "
                f"COMB {comb['trades']:4d} tr PnL={comb['pnl']:+.1f} "
                f"(Δ{row['delta_trades']:+d}, Δ{row['delta_pnl']:+.1f})  "
                f"trial {comb['trial_trades']} tr / {comb['trial_pnl']:+.1f}",
                flush=True,
            )

    res = pd.DataFrame(rows).sort_values(["delta_pnl", "combined_pnl"], ascending=False)
    res.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    print("\nTop 10 by ΔPnL:")
    print(res.head(10).to_string(index=False))
    profitable = res[res["delta_pnl"] > 0]
    if not profitable.empty:
        best = profitable.iloc[0]
        print(
            f"\nBest profitable: min_gap={int(best.min_gap)} time<{int(best.time_max)} "
            f"→ COMB PnL={best.combined_pnl:+.1f} (Δ{best.delta_pnl:+.1f})"
        )
    else:
        print("\nNo combo beat baseline on combined PnL.")


if __name__ == "__main__":
    main()
