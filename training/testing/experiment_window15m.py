#!/usr/bin/env python3
"""
experiment_window15m.py
-----------------------
Train + backtest with window_15m=100 (and various min_15m_drop values)
on the predefined C date range, then compare vs Candidate E baseline.

Usage:
    python3 training/testing/experiment_window15m.py
"""
from __future__ import annotations
import subprocess, sys, json, time
from pathlib import Path
from datetime import timezone, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl

SCRIPT      = PROJECT_ROOT / "training/image_trend_ml.py"
OUT_DIR     = PROJECT_ROOT / "runtime/_tmp_experiment_15m"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Date range (same as Candidate E baseline "C" run) ─────────────────────
START_DATE       = "2025-05-20"
END_DATE         = "2026-04-10"
TEST_START_DATE  = "2025-11-25T17:02:00+00:00"

# ── Fixed params (same as Candidate E) ────────────────────────────────────
BASE = {
    "timeframe":            "1min",
    "eval_mode":            "single_split",
    "window":               150,
    "min_window_range":     30,
    "horizon":              25,
    "trend_threshold":      0.008,
    "adverse_limit":        15,
    "long_target_threshold":  0.008,
    "short_target_threshold": 0.008,
    "long_adverse_limit":   15,
    "short_adverse_limit":  18,
    "classifier":           "gradient_boosting",
    "max_flat_ratio":       2.5,
    "stage1_min_prob":      0.55,
    "stage2_min_prob":      0.58,
    "stage2_min_prob_up":   0.65,
    "stage2_min_prob_down": 0.62,
    "max_hold_minutes":     60,
    "min_15m_rise":         0,
}

# ── Sweep: window_15m=100 with different min_15m_drop values ──────────────
COMBOS = [
    {"window_15m": 100, "min_15m_drop": 15},
]

WEAK_FILTER = PROJECT_ROOT / "runtime/bot_assets/weak-filter.json"


def _parse_report(report_path: Path) -> dict:
    try:
        r  = json.loads(report_path.read_text())
        dp = r.get("directional_pnl", r)
        a  = dp.get("all", dp)
        return {
            "trades":       int(a.get("trades", 0)),
            "total_pnl":    round(float(a.get("total_pnl", 0)), 2),
            "win_rate_pct": round(float(a.get("win_rate_pct", 0)), 2),
            "profit_factor":round(float(a.get("profit_factor") or 0), 3),
            "max_dd":       round(float(a.get("trade_max_drawdown") or 0), 2),
            "avg_trade":    round(float(a.get("avg_trade", 0)), 4),
        }
    except Exception as e:
        return {"error": str(e)}


def run_combo(combo: dict, idx: int) -> dict:
    stem    = f"exp15m_{idx:02d}_w15m{combo['window_15m']}_drop{combo['min_15m_drop']}"
    model   = OUT_DIR / f"{stem}_model.joblib"
    report  = OUT_DIR / f"{stem}_report.json"
    trades  = OUT_DIR / f"{stem}_trades.csv"

    params  = {**BASE, **combo}

    cmd = [
        sys.executable, str(SCRIPT),
        "--start-date",             START_DATE,
        "--end-date",               END_DATE,
        "--test-start-date",        TEST_START_DATE,
        "--timeframe",              params["timeframe"],
        "--eval-mode",              params["eval_mode"],
        "--disable-time-filter",
        "--window",                 str(params["window"]),
        "--window-15m",             str(params["window_15m"]),
        "--min-window-range",       str(params["min_window_range"]),
        "--min-15m-drop",           str(params["min_15m_drop"]),
        "--min-15m-rise",           str(params["min_15m_rise"]),
        "--horizon",                str(params["horizon"]),
        "--trend-threshold",        str(params["trend_threshold"]),
        "--adverse-limit",          str(params["adverse_limit"]),
        "--long-target-threshold",  str(params["long_target_threshold"]),
        "--short-target-threshold", str(params["short_target_threshold"]),
        "--long-adverse-limit",     str(params["long_adverse_limit"]),
        "--short-adverse-limit",    str(params["short_adverse_limit"]),
        "--classifier",             params["classifier"],
        "--max-flat-ratio",         str(params["max_flat_ratio"]),
        "--stage1-min-prob",        str(params["stage1_min_prob"]),
        "--stage2-min-prob",        str(params["stage2_min_prob"]),
        "--stage2-min-prob-up",     str(params["stage2_min_prob_up"]),
        "--stage2-min-prob-down",   str(params["stage2_min_prob_down"]),
        "--max-hold-minutes",       str(params["max_hold_minutes"]),
        "--model-out",              str(model.relative_to(PROJECT_ROOT)),
        "--report-out",             str(report.relative_to(PROJECT_ROOT)),
        "--trades-out",             str(trades.relative_to(PROJECT_ROOT)),
        "--weak-periods-json",      str(WEAK_FILTER),
    ]

    label = f"window_15m={combo['window_15m']}  min_15m_drop={combo['min_15m_drop']}"
    print(f"\n[{idx+1}/{len(COMBOS)}] {label}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ✗ FAILED ({elapsed:.0f}s)")
        print(result.stderr[-1000:])
        return {**combo, "error": "non-zero exit", "elapsed": elapsed}

    metrics = _parse_report(report) if report.exists() else {"error": "no report"}
    metrics.update(combo)
    metrics["elapsed"] = round(elapsed)
    print(f"  ✓ {elapsed:.0f}s  trades={metrics.get('trades','?')}  "
          f"pnl={metrics.get('total_pnl','?')}  "
          f"wr={metrics.get('win_rate_pct','?')}%  "
          f"pf={metrics.get('profit_factor','?')}  "
          f"dd={metrics.get('max_dd','?')}")
    return metrics


def main():
    print("=" * 65)
    print("Experiment: window_15m=100 vs Candidate E (window_15m=0)")
    print(f"Date range: {START_DATE} → {END_DATE}  (test from {TEST_START_DATE[:10]})")
    print("=" * 65)

    # Candidate E baseline (for reference)
    print("\n📌 Candidate E baseline: PnL=$5,217  WR=53.4%  PF=1.749  DD=-$171")

    results = []
    for i, combo in enumerate(COMBOS):
        r = run_combo(combo, i)
        results.append(r)

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'window_15m':>10}  {'drop':>4}  {'trades':>6}  {'pnl':>8}  {'wr%':>6}  {'pf':>6}  {'dd':>8}")
    print("-" * 65)
    for r in results:
        if "error" in r:
            print(f"  w15m={r.get('window_15m')}  drop={r.get('min_15m_drop')}  ERROR: {r['error']}")
        else:
            print(f"{r['window_15m']:>10}  {r['min_15m_drop']:>4}  "
                  f"{r['trades']:>6}  {r['total_pnl']:>8.2f}  "
                  f"{r['win_rate_pct']:>6.1f}  {r['profit_factor']:>6.3f}  "
                  f"{r['max_dd']:>8.2f}")
    print("-" * 65)
    print("  Candidate E (baseline):                 5217.00   53.4   1.749   -171.00")

    # Save results JSON
    out_json = OUT_DIR / "experiment_15m_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {out_json}")


if __name__ == "__main__":
    main()

