#!/usr/bin/env python3
"""Sweep stage2-min-prob-up (long) and stage2-min-prob-down (short) independently.

All other params fixed at Candidate C's best values:
  s1=0.55  ladv=15  ltgt=0.008  mwr=30  (+ C's weak filter from pass1)

Two-pass per combo:
  Pass 1: no weak filter → generate weak filter
  Pass 2: with generated weak filter → record results

Usage:
    python3 training/testing/sweep_stage2_directional.py
    python3 training/testing/sweep_stage2_directional.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.testing.sweep_utils import (
    fmt_sec, generate_weak_filter, parse_report, print_progress, print_top,
)

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
IMAGE_TREND_ML = PROJECT_ROOT / "training" / "image_trend_ml.py"
MODEL_IN       = PROJECT_ROOT / "runtime" / "bot_assets" / "backtest_model_best_base_weak_nostate.joblib"
RESULTS_DIR    = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Fixed at Candidate C's values ────────────────────────────────────────────
FIXED = {
    "stage1_min_prob":        0.55,
    "long_adverse_limit":     15,
    "long_target_threshold":  0.008,
    "min_window_range":       30,
    "short_adverse_limit":    18,
    "short_target_threshold": 0.008,
    "adverse_limit":          15,
    "trend_threshold":        0.008,
    "window":                 150,
    "horizon":                25,
    "min_15m_drop":           15,
    "max_flat_ratio":         2.5,
    "classifier":             "gradient_boosting",
}

# ── Grid: stage2 up vs down ───────────────────────────────────────────────────
# Baseline C used stage2=0.58 for both (same up/down)
PARAM_GRID = {
    "stage2_up":   [0.50, 0.55, 0.58, 0.62, 0.65],
    "stage2_down": [0.50, 0.55, 0.58, 0.62, 0.65],
}


def _build_cmd(stage2_up: float, stage2_down: float, out_base: Path, weak_filter: "Path | None") -> list[str]:
    cmd = [
        sys.executable, str(IMAGE_TREND_ML),
        "--start-date",           "2025-05-20",
        "--end-date",             "2026-04-10",
        "--test-start-date",      "2025-11-25T17:02:00+00:00",
        "--timeframe",            "1min",
        "--eval-mode",            "single_split",
        "--disable-time-filter",
        "--window",                     str(FIXED["window"]),
        "--window-15m",                 "0",
        "--min-window-range",           str(FIXED["min_window_range"]),
        "--min-15m-drop",               str(FIXED["min_15m_drop"]),
        "--min-15m-rise",               "0",
        "--horizon",                    str(FIXED["horizon"]),
        "--trend-threshold",            str(FIXED["trend_threshold"]),
        "--adverse-limit",              str(FIXED["adverse_limit"]),
        "--long-target-threshold",      str(FIXED["long_target_threshold"]),
        "--short-target-threshold",     str(FIXED["short_target_threshold"]),
        "--long-adverse-limit",         str(FIXED["long_adverse_limit"]),
        "--short-adverse-limit",        str(FIXED["short_adverse_limit"]),
        "--classifier",                 FIXED["classifier"],
        "--max-flat-ratio",             str(FIXED["max_flat_ratio"]),
        "--stage1-min-prob",            str(FIXED["stage1_min_prob"]),
        "--stage2-min-prob",            "0.58",   # fallback (overridden by up/down below)
        "--stage2-min-prob-up",         str(stage2_up),
        "--stage2-min-prob-down",       str(stage2_down),
        "--model-in",   str(MODEL_IN),
        "--model-out",  str(out_base) + "_model.joblib",
        "--report-out", str(out_base) + "_report.json",
        "--trades-out", str(out_base) + "_trades.csv",
    ]
    if weak_filter:
        cmd += ["--weak-periods-json", str(weak_filter)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    keys   = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    print(f"Total combinations: {len(combos)}  (2-pass each → {len(combos)*2} runs)")
    print(f"Fixed: s1={FIXED['stage1_min_prob']}  ladv={FIXED['long_adverse_limit']}  "
          f"ltgt={FIXED['long_target_threshold']}  mwr={FIXED['min_window_range']}")

    if args.dry_run:
        for i, c in enumerate(combos, 1):
            print(f"  {i:3d}: up={c['stage2_up']}  down={c['stage2_down']}")
        return

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = RESULTS_DIR / f"sweep_s2dir_{timestamp}.csv"
    fieldnames  = [
        "stage2_up", "stage2_down",
        "trades", "total_pnl", "avg_trade", "win_rate_pct", "profit_factor",
        "avg_trades_per_day", "avg_day", "positive_days_pct",
        "trade_max_drawdown", "daily_max_drawdown",
        "long_trades", "long_wr", "long_pnl",
        "short_trades", "short_wr", "short_pnl", "error",
    ]

    sweep_start = time.time()
    durations: list[float] = []
    all_rows: list[dict]   = []

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, combo in enumerate(combos, 1):
            up, dn = combo["stage2_up"], combo["stage2_down"]
            print_progress(i, len(combos), sweep_start, durations)
            print(f"  stage2_up={up}  stage2_down={dn}")
            t0     = time.time()
            prefix = RESULTS_DIR / f"s2dir_{timestamp}_{i:03d}"

            # Pass 1 — no weak filter (stream output so errors are visible)
            base1 = Path(str(prefix) + "_p1")
            rc1   = subprocess.run(
                _build_cmd(up, dn, base1, None),
                cwd=str(PROJECT_ROOT), check=False,
            ).returncode

            metrics: dict = {}
            if rc1 != 0:
                metrics = {"error": f"pass1 rc={rc1}"}
            else:
                trades1   = Path(str(base1) + "_trades.csv")
                weak_json = Path(str(prefix) + "_wf.json")
                cells     = generate_weak_filter(trades1, weak_json)

                # Pass 2 — with weak filter
                base2 = Path(str(prefix) + "_p2")
                rc2   = subprocess.run(
                    _build_cmd(up, dn, base2, weak_json if cells else None),
                    cwd=str(PROJECT_ROOT), check=False,
                ).returncode

                report2 = Path(str(base2) + "_report.json")
                if rc2 == 0 and report2.exists():
                    metrics = parse_report(report2)
                else:
                    metrics = {"error": f"pass2 rc={rc2}"}

            durations.append(time.time() - t0)
            row = {**combo, **metrics}
            for fn in fieldnames:
                row.setdefault(fn, "")
            all_rows.append(row)
            writer.writerow(row); f.flush()

            if "error" in metrics and metrics["error"]:
                print(f"  ⚠ {metrics['error']}")
            else:
                print(f"  → {fmt_sec(durations[-1])}  "
                      f"trades={metrics.get('trades')}  pnl={metrics.get('total_pnl')}  "
                      f"avg={metrics.get('avg_trade')}  wr={metrics.get('win_rate_pct')}%  "
                      f"pf={metrics.get('profit_factor')}  dd={metrics.get('trade_max_drawdown')}")
                print(f"     long: {metrics.get('long_trades')}t  wr={metrics.get('long_wr')}%  "
                      f"pnl={metrics.get('long_pnl')} | "
                      f"short: {metrics.get('short_trades')}t  wr={metrics.get('short_wr')}%  "
                      f"pnl={metrics.get('short_pnl')}")

    print(f"\n✅ Sweep done in {fmt_sec(time.time()-sweep_start)}. Results: {summary_csv}")
    print_top(all_rows, sort_key="total_pnl", param_keys=["stage2_up", "stage2_down"])


if __name__ == "__main__":
    main()

