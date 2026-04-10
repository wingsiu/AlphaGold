#!/usr/bin/env python3
"""Analyse all existing backtest results and find the best configuration to
target $40 profit per trading day from spot gold.

What this script does:
  1. Loads every available backtest trades CSV.
  2. Computes per-day PnL statistics (mean, median, hit-rate of positive days).
  3. Works out how many contracts/units are required to hit $40/day on average.
  4. Prints a ranked strategy comparison table.
  5. Shows a daily PnL distribution for the best strategy so you can see consistency.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / "training"

# ── Strategy registry ────────────────────────────────────────────────────────
STRATEGIES: list[dict] = [
    {
        "name": "GB 15m-nextbar TP0.60 corr",
        "csv":  TRAINING_DIR / "backtest_trades_15m_nextbar_060_corr.csv",
    },
    {
        "name": "GB 15m-nextbar TP0.35",
        "csv":  TRAINING_DIR / "backtest_trades_15m_nextbar_035.csv",
    },
    {
        "name": "SR walk-forward (span060 sl17)",
        "csv":  TRAINING_DIR / "l2" / "backtest_sr_trades_gc_c0_span060_sl17.csv",
    },
    {
        "name": "SR walk-forward (default)",
        "csv":  TRAINING_DIR / "l2" / "backtest_sr_trades_gc_c0.csv",
    },
    {
        "name": "LSTM v3w16 test-period",
        "csv":  TRAINING_DIR / "backtest_trades_lstm_15m_v3w16_testperiod.csv",
    },
    {
        "name": "LSTM v3w10 test-period",
        "csv":  TRAINING_DIR / "backtest_trades_lstm_15m_v3w10_testperiod.csv",
    },
    {
        "name": "LSTM v3w test-period",
        "csv":  TRAINING_DIR / "backtest_trades_lstm_15m_v3w_testperiod.csv",
    },
    {
        "name": "LSTM v3 test-period",
        "csv":  TRAINING_DIR / "backtest_trades_lstm_15m_v3_testperiod.csv",
    },
    {
        "name": "LSTM v2 test-period",
        "csv":  TRAINING_DIR / "backtest_trades_lstm_15m_v2_testperiod.csv",
    },
]

TARGET_DAILY_USD = 40.0
MAX_CONTRACTS_LIMIT = 50          # hard cap for sanity
DAYS_IN_PERIOD_FALLBACK = 259     # ~trading days May-2025 to Feb-2026


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_time_col(df: pd.DataFrame) -> str:
    for col in ("entry_time", "ts_event", "entry_ts", "time", "date"):
        if col in df.columns:
            return col
    return ""


def _resolve_pnl_col(df: pd.DataFrame) -> str:
    for col in ("pnl_usd", "pnl", "profit", "pnl_points", "points"):
        if col in df.columns:
            return col
    return ""


def load_strategy(info: dict) -> dict | None:
    path = info["csv"]
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"  [WARN] Could not read {path.name}: {exc}")
        return None

    pnl_col = _resolve_pnl_col(df)
    if not pnl_col:
        print(f"  [WARN] No PnL column in {path.name}")
        return None

    df[pnl_col] = pd.to_numeric(df[pnl_col], errors="coerce")
    df = df.dropna(subset=[pnl_col])

    time_col = _resolve_time_col(df)
    if time_col:
        df["_date"] = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.normalize()
    else:
        df["_date"] = pd.NaT

    total_pnl = float(df[pnl_col].sum())
    n_trades = len(df)
    win_rate = float((df[pnl_col] > 0).mean()) * 100

    # Per-day stats
    if df["_date"].notna().any():
        daily = df.groupby("_date")[pnl_col].sum()
        n_days = len(daily)
        avg_daily = float(daily.mean())
        median_daily = float(daily.median())
        pct_positive_days = float((daily > 0).mean()) * 100
        worst_day = float(daily.min())
        best_day  = float(daily.max())
    else:
        n_days = DAYS_IN_PERIOD_FALLBACK
        avg_daily = total_pnl / n_days
        median_daily = avg_daily
        pct_positive_days = win_rate
        worst_day = float(df[pnl_col].min())
        best_day  = float(df[pnl_col].max())
        daily = None

    # Contracts needed
    if avg_daily > 0:
        contracts_needed = min(
            int(np.ceil(TARGET_DAILY_USD / avg_daily)),
            MAX_CONTRACTS_LIMIT,
        )
        projected_daily = avg_daily * contracts_needed
    else:
        contracts_needed = None
        projected_daily  = None

    return {
        "name":               info["name"],
        "file":               path.name,
        "n_trades":           n_trades,
        "n_days":             n_days,
        "total_pnl_1unit":    total_pnl,
        "avg_daily_1unit":    avg_daily,
        "median_daily_1unit": median_daily,
        "win_rate_pct":       win_rate,
        "pct_positive_days":  pct_positive_days,
        "worst_day_1unit":    worst_day,
        "best_day_1unit":     best_day,
        "contracts_needed":   contracts_needed,
        "projected_daily_usd": projected_daily,
        "daily_series":       daily,
        "df":                 df,
        "pnl_col":            pnl_col,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("  GOLD STRATEGY ANALYSIS — TARGET $40 / TRADING DAY")
    print("=" * 72)

    results = []
    for info in STRATEGIES:
        r = load_strategy(info)
        if r:
            results.append(r)

    if not results:
        print("No backtest CSVs found. Run the backtests first.")
        return

    # ── Ranked summary table ─────────────────────────────────────────────────
    results_sorted = sorted(results, key=lambda x: x["avg_daily_1unit"], reverse=True)

    print(f"\n{'Rank':<5} {'Strategy':<35} {'Days':>5} {'Trd':>5} {'WR%':>6} "
          f"{'Pos-Day%':>9} {'Avg$/Day':>9} {'Med$/Day':>9} {'Worst Day':>10} "
          f"{'Units→$40':>10} {'Proj $/Day':>11}")
    print("-" * 130)

    for rank, r in enumerate(results_sorted, 1):
        cu = f"{r['contracts_needed']}" if r['contracts_needed'] else "N/A"
        pj = f"${r['projected_daily_usd']:.2f}" if r['projected_daily_usd'] else "N/A"
        print(
            f"{rank:<5} {r['name']:<35} {r['n_days']:>5} {r['n_trades']:>5} "
            f"{r['win_rate_pct']:>5.1f}% {r['pct_positive_days']:>8.1f}% "
            f"${r['avg_daily_1unit']:>8.2f} ${r['median_daily_1unit']:>8.2f} "
            f"${r['worst_day_1unit']:>9.2f} {cu:>10} {pj:>11}"
        )

    # ── Best strategy deep-dive ───────────────────────────────────────────────
    best = results_sorted[0]
    print()
    print("=" * 72)
    print(f"  BEST STRATEGY: {best['name']}")
    print("=" * 72)
    print(f"  File            : {best['file']}")
    print(f"  Trading days    : {best['n_days']}")
    print(f"  Total trades    : {best['n_trades']}")
    print(f"  Win rate        : {best['win_rate_pct']:.1f}%")
    print(f"  Positive days   : {best['pct_positive_days']:.1f}%")
    print()
    print(f"  Per 1 contract/unit:")
    print(f"    Avg daily PnL : ${best['avg_daily_1unit']:.2f}")
    print(f"    Med daily PnL : ${best['median_daily_1unit']:.2f}")
    print(f"    Best day      : ${best['best_day_1unit']:.2f}")
    print(f"    Worst day     : ${best['worst_day_1unit']:.2f}")
    print(f"    Total PnL     : ${best['total_pnl_1unit']:.2f}")
    print()

    if best["contracts_needed"]:
        n = best["contracts_needed"]
        print(f"  To hit ${TARGET_DAILY_USD:.0f}/day average:")
        print(f"    Units required  : {n}")
        print(f"    Projected avg   : ${best['projected_daily_usd']:.2f}/day")
        print(f"    Projected worst : ${best['worst_day_1unit'] * n:.2f} (single worst day × {n} units)")
        print(f"    Projected total : ${best['total_pnl_1unit'] * n:.2f} over {best['n_days']} days")
    else:
        print(f"  Strategy does not have positive average daily PnL — cannot target $40/day.")

    # ── Daily PnL distribution for best strategy ─────────────────────────────
    if best["daily_series"] is not None and len(best["daily_series"]) > 5:
        daily = best["daily_series"]
        n = best["contracts_needed"] or 1
        scaled = daily * n
        print()
        print(f"  Daily PnL distribution ({n} units):")
        buckets = [
            ("< -$100",       scaled < -100),
            ("-$100 to -$50",  (scaled >= -100) & (scaled < -50)),
            ("-$50 to $0",    (scaled >= -50)  & (scaled < 0)),
            ("$0 to $20",     (scaled >= 0)    & (scaled < 20)),
            ("$20 to $40",    (scaled >= 20)   & (scaled < 40)),
            ("$40 to $80",    (scaled >= 40)   & (scaled < 80)),
            ("> $80",         scaled >= 80),
        ]
        for label, mask in buckets:
            count = int(mask.sum())
            pct   = count / len(scaled) * 100
            bar   = "█" * int(pct / 2)
            print(f"    {label:>18}: {count:>4} days ({pct:>5.1f}%)  {bar}")

        print()
        print(f"  Monthly breakdown ({n} units):")
        monthly = (daily * n).resample("ME").sum()
        for month, pnl in monthly.items():
            print(f"    {month.strftime('%Y-%m')}: ${pnl:.2f}")

    # ── Exit reason breakdown ─────────────────────────────────────────────────
    df = best["df"]
    if "exit_reason" in df.columns:
        print()
        print("  Exit reason breakdown:")
        er = df.groupby("exit_reason")[best["pnl_col"]].agg(["count", "mean", "sum"])
        er.columns = ["trades", "avg_pnl", "total_pnl"]
        for reason, row in er.iterrows():
            print(f"    {reason:<15}: {int(row['trades']):>5} trades  avg=${row['avg_pnl']:.2f}  total=${row['total_pnl']:.2f}")

    # ── Recommendation ────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  RECOMMENDATION")
    print("=" * 72)

    feasible = [r for r in results_sorted if r["contracts_needed"] and r["contracts_needed"] <= MAX_CONTRACTS_LIMIT and r["avg_daily_1unit"] > 0]

    if not feasible:
        print("  No strategy achieves positive daily PnL. Retrain models with updated data.")
    else:
        best_f = feasible[0]
        n = best_f["contracts_needed"]
        print(f"  Strategy  : {best_f['name']}")
        print(f"  Units     : {n}")
        print(f"  Avg daily : ${best_f['projected_daily_usd']:.2f}")
        print(f"  Win days  : {best_f['pct_positive_days']:.1f}%")
        print()
        print("  Key risks:")
        print(f"    - Worst single day (×{n} units): ${best_f['worst_day_1unit'] * n:.2f}")
        wdr_pct = abs(best_f['worst_day_1unit'] / best_f['avg_daily_1unit']) if best_f['avg_daily_1unit'] else 0
        print(f"    - Worst day is {wdr_pct:.1f}× the average day — size cautiously")
        print(f"    - Backtested period is 2025-05-20 to 2026-02-04; re-validate on 2026-02-04 onward")
        print()
        print("  Next step: retrain models with the new data (2026-02-04 to 2026-04-11)")
        print("  and re-run this analysis to confirm performance holds.")


if __name__ == "__main__":
    main()

