#!/usr/bin/env python3
"""Sweep horizon (prediction hold bars) to find optimal hold time.

All other params fixed at Candidate D's best values:
  s1=0.55  s2_up=0.65  s2_down=0.62  ladv=15  ltgt=0.008  mwr=30

Two-pass per combo:
  Pass 1: no weak filter → generate weak filter
  Pass 2: with generated weak filter → record results

We sweep:
  - horizon: bars to hold (1min bars, so horizon=N → N-min hold)
  - max_hold_minutes: optional hard cap (None = off; only tested for horizon values
    where we want to test a cap shorter than the label horizon)

Usage:
    python3 training/testing/sweep_hold_time.py
    python3 training/testing/sweep_hold_time.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
IMAGE_TREND_ML = PROJECT_ROOT / "training" / "image_trend_ml.py"
MODEL_IN       = PROJECT_ROOT / "runtime" / "bot_assets" / "backtest_model_best_base_weak_nostate.joblib"
RESULTS_DIR    = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Fixed at Candidate D's values ────────────────────────────────────────────
FIXED = {
    "stage1_min_prob":        0.55,
    "stage2_min_prob":        0.58,   # fallback
    "stage2_min_prob_up":     0.65,
    "stage2_min_prob_down":   0.62,
    "long_adverse_limit":     15,
    "short_adverse_limit":    18,
    "long_target_threshold":  0.008,
    "short_target_threshold": 0.008,
    "adverse_limit":          15,
    "trend_threshold":        0.008,
    "min_window_range":       30,
    "window":                 150,
    "min_15m_drop":           15,
    "max_flat_ratio":         2.5,
    "classifier":             "gradient_boosting",
}

# ── Grid ─────────────────────────────────────────────────────────────────────
# Candidate D baseline: horizon=25, max_hold_minutes=None
# We test longer horizons (retrains labels) and also a pure cap approach
# (max_hold_minutes with horizon fixed at 25) to separate the two effects.
#
# Each entry: (horizon, max_hold_minutes)
#   max_hold_minutes=None  → no hard cap
COMBOS: list[tuple[int, float | None]] = [
    # ── vary horizon (retrains labels) ──────────────────────────────────────
    (25,  None),   # D baseline
    (30,  None),
    (35,  None),
    (40,  None),
    (45,  None),
    (50,  None),
    (60,  None),
    # ── horizon=25 + hard cap (no label change) ──────────────────────────────
    (25,  30.0),
    (25,  35.0),
    (25,  40.0),
    (25,  45.0),
    (25,  50.0),
    (25,  60.0),
]

# Weak-filter thresholds
WEAK_MIN_TRADES   = 3
WEAK_MAX_WIN_RATE = 40.0


def _build_cmd(
    horizon: int,
    max_hold: float | None,
    out_base: Path,
    weak_filter: Path | None,
) -> list[str]:
    cmd = [
        sys.executable, str(IMAGE_TREND_ML),
        "--start-date",       "2025-05-20",
        "--end-date",         "2026-04-10",
        "--test-start-date",  "2025-11-25T17:02:00+00:00",
        "--timeframe",        "1min",
        "--eval-mode",        "single_split",
        "--disable-time-filter",
        "--window",                     str(FIXED["window"]),
        "--window-15m",                 "0",
        "--min-window-range",           str(FIXED["min_window_range"]),
        "--min-15m-drop",               str(FIXED["min_15m_drop"]),
        "--min-15m-rise",               "0",
        "--horizon",                    str(horizon),
        "--trend-threshold",            str(FIXED["trend_threshold"]),
        "--adverse-limit",              str(FIXED["adverse_limit"]),
        "--long-target-threshold",      str(FIXED["long_target_threshold"]),
        "--short-target-threshold",     str(FIXED["short_target_threshold"]),
        "--long-adverse-limit",         str(FIXED["long_adverse_limit"]),
        "--short-adverse-limit",        str(FIXED["short_adverse_limit"]),
        "--classifier",                 FIXED["classifier"],
        "--max-flat-ratio",             str(FIXED["max_flat_ratio"]),
        "--stage1-min-prob",            str(FIXED["stage1_min_prob"]),
        "--stage2-min-prob",            str(FIXED["stage2_min_prob"]),
        "--stage2-min-prob-up",         str(FIXED["stage2_min_prob_up"]),
        "--stage2-min-prob-down",       str(FIXED["stage2_min_prob_down"]),
        "--model-in",   str(MODEL_IN),
        "--model-out",  str(out_base) + "_model.joblib",
        "--report-out", str(out_base) + "_report.json",
        "--trades-out", str(out_base) + "_trades.csv",
    ]
    if max_hold is not None:
        cmd += ["--max-hold-minutes", str(max_hold)]
    if weak_filter:
        cmd += ["--weak-periods-json", str(weak_filter)]
    return cmd


def _generate_weak_filter(trades_csv: Path, out_json: Path) -> list[dict]:
    pnl = rebuild_directional_pnl(trades_csv)
    try:
        session_heatmaps = pnl["all"]["time_distribution"]["session_heatmaps"]
    except KeyError:
        out_json.write_text(json.dumps({"weak_cells": []}, indent=2))
        return []

    weak_cells: list[dict] = []
    for session in ("hkt", "london", "ny"):
        day_map = session_heatmaps.get(session, {}).get("cell_stats", {})
        for day, hour_map in day_map.items():
            for hour, st in hour_map.items():
                if not st:
                    continue
                t         = int(st.get("trades", 0))
                pnl_total = float(st.get("total_pnl", 0.0))
                wr        = st.get("win_rate_pct", None)
                wr        = float(wr) if wr is not None else None
                if t >= WEAK_MIN_TRADES and pnl_total < 0.0 and wr is not None and wr < WEAK_MAX_WIN_RATE:
                    weak_cells.append({"session": session, "day": str(day), "hour": hour})

    out_json.write_text(json.dumps({"weak_cells": weak_cells}, indent=2))
    return weak_cells


def _parse_report(report_path: Path) -> dict:
    try:
        r   = json.loads(report_path.read_text())
        dp  = r.get("directional_pnl", r)
        a   = dp.get("all", dp)
        long_s  = dp.get("long_up",    {})
        short_s = dp.get("short_down", {})
        return {
            "trades":             int(a.get("trades", 0)),
            "total_pnl":          round(float(a.get("total_pnl", 0)), 2),
            "avg_trade":          round(float(a.get("avg_trade", 0)), 4),
            "win_rate_pct":       round(float(a.get("win_rate_pct", 0)), 2),
            "profit_factor":      round(float(a.get("profit_factor") or 0), 3),
            "avg_trades_per_day": round(float(a.get("avg_trades_per_day", 0)), 2),
            "avg_day":            round(float(dp.get("avg_day") or 0), 2),
            "positive_days_pct":  round(float(dp.get("positive_days_pct") or 0), 2),
            "trade_max_drawdown": round(float(a.get("trade_max_drawdown", 0)), 2),
            "daily_max_drawdown": round(float(a.get("daily_max_drawdown", 0)), 2),
            "long_trades":        int(long_s.get("trades", 0)),
            "long_wr":            round(float(long_s.get("win_rate_pct", 0)), 1),
            "long_pnl":           round(float(long_s.get("total_pnl", 0)), 2),
            "short_trades":       int(short_s.get("trades", 0)),
            "short_wr":           round(float(short_s.get("win_rate_pct", 0)), 1),
            "short_pnl":          round(float(short_s.get("total_pnl", 0)), 2),
        }
    except Exception as e:
        return {"error": str(e)}


def _fmt_sec(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec:02d}s" if m else f"{sec}s"


def _bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total) if total else 0
    pct    = int(100 * done / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:3d}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Total combinations: {len(COMBOS)}  (2-pass each → {len(COMBOS)*2} runs)")
    print(f"Fixed: s1={FIXED['stage1_min_prob']}  s2_up={FIXED['stage2_min_prob_up']}  "
          f"s2_dn={FIXED['stage2_min_prob_down']}  ladv={FIXED['long_adverse_limit']}  "
          f"ltgt={FIXED['long_target_threshold']}  mwr={FIXED['min_window_range']}")

    if args.dry_run:
        for i, (h, mh) in enumerate(COMBOS, 1):
            cap = f"  max_hold={mh}" if mh else ""
            print(f"  {i:3d}: horizon={h}{cap}")
        return

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = RESULTS_DIR / f"sweep_holdtime_{timestamp}.csv"
    fieldnames  = ["horizon", "max_hold_minutes",
                   "trades", "total_pnl", "avg_trade", "win_rate_pct", "profit_factor",
                   "avg_trades_per_day", "avg_day", "positive_days_pct",
                   "trade_max_drawdown", "daily_max_drawdown",
                   "long_trades", "long_wr", "long_pnl",
                   "short_trades", "short_wr", "short_pnl", "error"]

    sweep_start = time.time()
    durations: list[float] = []

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (horizon, max_hold) in enumerate(COMBOS, 1):
            now_str = datetime.now().strftime("%H:%M:%S")
            elapsed = time.time() - sweep_start
            if durations:
                avg_sec = sum(durations) / len(durations)
                eta_str = (datetime.now() + timedelta(seconds=avg_sec * (len(COMBOS) - i + 1))).strftime("%H:%M:%S")
                print(f"\n{_bar(i-1, len(COMBOS))} [{i}/{len(COMBOS)}] {now_str}  elapsed={_fmt_sec(elapsed)}  eta={eta_str}")
            else:
                print(f"\n[{i}/{len(COMBOS)}] {now_str}  elapsed={_fmt_sec(elapsed)}")

            cap_str = f"  max_hold={max_hold}" if max_hold else ""
            print(f"  horizon={horizon}{cap_str}")

            t0     = time.time()
            prefix = RESULTS_DIR / f"holdtime_{timestamp}_{i:03d}"

            # Pass 1 — no weak filter
            base1 = Path(str(prefix) + "_p1")
            rc1   = subprocess.run(
                _build_cmd(horizon, max_hold, base1, None),
                capture_output=True, cwd=str(PROJECT_ROOT)
            ).returncode

            metrics: dict = {}
            if rc1 != 0:
                metrics = {"error": f"pass1 rc={rc1}"}
            else:
                trades1   = Path(str(base1) + "_trades.csv")
                weak_json = Path(str(prefix) + "_wf.json")
                cells = _generate_weak_filter(trades1, weak_json)

                base2 = Path(str(prefix) + "_p2")
                rc2   = subprocess.run(
                    _build_cmd(horizon, max_hold, base2, weak_json if cells else None),
                    capture_output=True, cwd=str(PROJECT_ROOT)
                ).returncode

                report2 = Path(str(base2) + "_report.json")
                metrics = _parse_report(report2) if (rc2 == 0 and report2.exists()) else {"error": f"pass2 rc={rc2}"}

            durations.append(time.time() - t0)

            row = {"horizon": horizon, "max_hold_minutes": max_hold if max_hold else "", **metrics}
            for fn in fieldnames:
                row.setdefault(fn, "")
            writer.writerow(row)
            f.flush()

            if "error" in metrics:
                print(f"  ⚠ {metrics['error']}")
            else:
                print(f"  → {_fmt_sec(durations[-1])}  "
                      f"trades={metrics['trades']}  pnl={metrics['total_pnl']}  "
                      f"avg={metrics['avg_trade']}  wr={metrics['win_rate_pct']}%  "
                      f"pf={metrics['profit_factor']}  dd={metrics['trade_max_drawdown']}")
                print(f"     long: {metrics['long_trades']}t  wr={metrics['long_wr']}%  pnl={metrics['long_pnl']} | "
                      f"short: {metrics['short_trades']}t  wr={metrics['short_wr']}%  pnl={metrics['short_pnl']}")

    print(f"\n✅ Sweep done in {_fmt_sec(time.time()-sweep_start)}. Results: {summary_csv}")
    _print_top(summary_csv)


def _print_top(csv_path: Path, n: int = 10) -> None:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows = [r for r in rows if not r.get("error")]
    rows.sort(key=lambda r: float(r.get("total_pnl", 0)), reverse=True)
    print(f"\n=== Top {min(n, len(rows))} by Total PnL ===")
    print(f"{'hz':>4} {'cap':>6} | {'total_pnl':>10} {'avg':>8} {'wr%':>6} {'pf':>5} "
          f"{'trades':>7} {'avg_day':>8} {'dd':>9} | {'l_wr':>6} {'s_wr':>6}")
    print("-" * 90)
    for r in rows[:n]:
        cap = r.get("max_hold_minutes") or "-"
        print(f"{r['horizon']:>4} {cap:>6} | "
              f"{float(r['total_pnl']):>10.2f} {float(r['avg_trade']):>8.4f} "
              f"{float(r['win_rate_pct']):>6.1f} {float(r['profit_factor']):>5.3f} "
              f"{int(r['trades']):>7} {float(r['avg_day']):>8.2f} "
              f"{float(r['trade_max_drawdown']):>9.2f} | "
              f"{float(r['long_wr']):>6.1f} {float(r['short_wr']):>6.1f}")


if __name__ == "__main__":
    main()

