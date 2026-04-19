#!/usr/bin/env python3
"""Parameter sweep to find a better model (no retrain).

Runs backtest_no_retrain.py (option C) with different parameter combinations
and saves a summary CSV to training/testing/results/.

Usage:
    python3 training/testing/sweep_params.py
    python3 training/testing/sweep_params.py --dry-run   # just print combos
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_TREND_ML = PROJECT_ROOT / "training" / "image_trend_ml.py"
MODEL_IN = PROJECT_ROOT / "runtime" / "bot_assets" / "backtest_model_best_base_weak_nostate.joblib"
WEAK_FILTER = PROJECT_ROOT / "runtime" / "bot_assets" / "weak-filter.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Parameter grid ────────────────────────────────────────────────────────────
# Baseline (best_base):
#   stage1=0.48  stage2=0.50  long_adverse=12  long_target=0.006  min_range=40

PARAM_GRID = {
    "stage1_min_prob":        [0.48, 0.55, 0.60],
    "stage2_min_prob":        [0.50, 0.55, 0.58],
    "long_adverse_limit":     [12, 15, 18],
    "long_target_threshold":  [0.006, 0.008],
    "min_window_range":       [30, 40, 50],
}

# Fixed params (not swept)
FIXED = {
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


def _build_cmd(combo: dict, run_id: str) -> list[str]:
    """Build command to call image_trend_ml.py directly (no-retrain mode)."""
    out_base = RESULTS_DIR / f"run_{run_id}"
    return [
        sys.executable, str(IMAGE_TREND_ML),
        "--start-date", "2025-05-20",
        "--end-date", "2026-04-10",
        "--test-start-date", "2025-11-25T17:02:00+00:00",
        "--timeframe", "1min",
        "--eval-mode", "single_split",
        "--disable-time-filter",
        "--window", "150",
        "--window-15m", "0",
        "--min-window-range", str(combo["min_window_range"]),
        "--min-15m-drop", str(FIXED["min_15m_drop"]),
        "--min-15m-rise", "0",
        "--horizon", str(FIXED["horizon"]),
        "--trend-threshold", str(FIXED["trend_threshold"]),
        "--adverse-limit", str(FIXED["adverse_limit"]),
        "--long-target-threshold", str(combo["long_target_threshold"]),
        "--short-target-threshold", str(FIXED["short_target_threshold"]),
        "--long-adverse-limit", str(combo["long_adverse_limit"]),
        "--short-adverse-limit", str(FIXED["short_adverse_limit"]),
        "--classifier", FIXED["classifier"],
        "--max-flat-ratio", str(FIXED["max_flat_ratio"]),
        "--stage1-min-prob", str(combo["stage1_min_prob"]),
        "--stage2-min-prob", str(combo["stage2_min_prob"]),
        "--model-in", str(MODEL_IN),
        "--model-out", f"{out_base}_model.joblib",
        "--report-out", f"{out_base}_report.json",
        "--trades-out", f"{out_base}_trades.csv",
        "--weak-periods-json", str(WEAK_FILTER),
    ]


def _parse_report(report_path: Path) -> dict:
    """Extract key metrics from the saved report JSON."""
    try:
        r = json.loads(report_path.read_text())
        dp = r.get("directional_pnl", r)
        a = dp.get("all", dp)
        return {
            "trades":             a.get("trades", 0),
            "total_pnl":          round(a.get("total_pnl", 0.0), 2),
            "avg_trade":          round(a.get("avg_trade", 0.0), 4),
            "win_rate_pct":       round(a.get("win_rate_pct", 0.0), 2),
            "profit_factor":      round(a.get("profit_factor") or 0.0, 3),
            "avg_trades_per_day": round(a.get("avg_trades_per_day", 0.0), 2),
            "avg_day":            round(dp.get("avg_day") or 0.0, 2),
            "positive_days_pct":  round(dp.get("positive_days_pct") or 0.0, 2),
            "trade_max_drawdown": round(a.get("trade_max_drawdown", 0.0), 2),
            "daily_max_drawdown": round(a.get("daily_max_drawdown", 0.0), 2),
        }
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print combos only, don't run")
    args = ap.parse_args()

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total combinations: {len(combos)}")
    if args.dry_run:
        for i, c in enumerate(combos, 1):
            print(f"  {i:3d}: {c}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = RESULTS_DIR / f"sweep_{timestamp}.csv"
    fieldnames = keys + [
        "trades", "total_pnl", "avg_trade", "win_rate_pct", "profit_factor",
        "avg_trades_per_day", "avg_day", "positive_days_pct",
        "trade_max_drawdown", "daily_max_drawdown", "error",
    ]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, combo in enumerate(combos, 1):
            label = " | ".join(f"{k}={v}" for k, v in combo.items())
            run_id = f"{timestamp}_{i:03d}"
            print(f"\n[{i}/{len(combos)}] {label}")

            cmd = _build_cmd(combo, run_id)
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

            # Parse the report JSON written by this run
            report_path = RESULTS_DIR / f"run_{run_id}_report.json"
            if report_path.exists():
                metrics = _parse_report(report_path)
            else:
                metrics = {"error": "no report file"}

            if result.returncode != 0:
                metrics["error"] = result.stderr[-300:] if result.stderr else "non-zero exit"

            row = {**combo, **metrics}
            # Fill missing fieldnames with ""
            for fn in fieldnames:
                row.setdefault(fn, "")
            writer.writerow(row)
            f.flush()

            status = f"  → trades={metrics.get('trades','?')} total_pnl={metrics.get('total_pnl','?')} avg={metrics.get('avg_trade','?')} wr={metrics.get('win_rate_pct','?')}%"
            print(status)

    print(f"\n✅ Sweep done. Results: {summary_csv}")
    _print_top(summary_csv)


def _print_top(csv_path: Path, n: int = 10) -> None:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append(row)
            except Exception:
                pass
    rows = [r for r in rows if not r.get("error")]
    rows.sort(key=lambda r: float(r.get("total_pnl", 0)), reverse=True)
    print(f"\n=== Top {min(n, len(rows))} by Total PnL ===")
    print(f"{'total_pnl':>10} {'avg_trade':>9} {'wr%':>6} {'pf':>5} {'trades':>7} | params")
    print("-" * 80)
    for r in rows[:n]:
        params = " ".join(f"{k}={r[k]}" for k in PARAM_GRID)
        print(f"{float(r['total_pnl']):>10.2f} {float(r['avg_trade']):>9.4f} {float(r['win_rate_pct']):>6.1f} {float(r['profit_factor']):>5.3f} {int(r['trades']):>7} | {params}")


if __name__ == "__main__":
    main()

