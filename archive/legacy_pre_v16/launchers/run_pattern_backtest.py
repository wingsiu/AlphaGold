#!/usr/bin/env python3
"""
Run combined 6-pattern backtest + full stats report (v15).

Usage (from project root):
  .venv/bin/python3 run_pattern_backtest.py
  .venv/bin/python3 run_pattern_backtest.py 2025-06-01 2026-05-23
  .venv/bin/python3 run_pattern_backtest.py 90
  .venv/bin/python3 run_pattern_backtest.py 2025-06-01 2026-05-23 breakthrough_short

Runs v15/backtest/backtest_v15.py (deterministic energetic gate — matches live v15 bot).
Legacy v14 backtest archived under archive/v14_legacy/.

Default patterns: PRODUCTION_PATTERNS from config/v14_patterns.py.
Time filter: ON by default (runtime/hybrid_weak_time_slots.json). Disable: V14_NO_TIME_FILTER=1
Hybrid (pattern-first + energetic fallback):
  .venv/bin/python3 run_hybrid_backtest.py
Trades CSV: runtime/v15_backtest_trades.csv
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
TRADES_CSV = PROJECT_ROOT / "runtime" / "v15_backtest_trades.csv"


def run_backtest(argv: list[str]) -> None:
    import os

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "V14_HYBRID": "1"}
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "v15" / "backtest" / "backtest_v15.py"),
        *argv,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def print_full_stats(tdf: pd.DataFrame) -> None:
    from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl

    wins = int((tdf["pnl"] > 0).sum())
    net_pnl = float(tdf["pnl"].sum())
    wr = float(wins / len(tdf) * 100.0) if len(tdf) else 0.0
    cum = tdf["pnl"].cumsum()
    max_dd = float((cum - cum.cummax()).min())

    print(f"\n{'='*60}")
    print("  FULL STATISTICS")
    print(f"{'='*60}")
    print(f"  Trades       : {len(tdf)}  (W:{wins}  L:{len(tdf)-wins})")
    print(f"  Win Rate     : {wr:.1f}%")
    print(f"  Net PnL      : {net_pnl:+.1f}")
    print(f"  Avg Trade    : {net_pnl/len(tdf):+.2f}")
    print(f"  Max Drawdown : {max_dd:+.1f}")

    if "pattern" in tdf.columns:
        print("\n  By pattern:")
        for name, grp in tdf.groupby("pattern", dropna=False):
            label = name if pd.notna(name) else "unknown"
            print(
                f"    {label:20s}: {len(grp):4d} trades  "
                f"PnL={grp['pnl'].sum():+.1f}  WR={(grp['pnl']>0).mean()*100:.0f}%  "
                f"avg={grp['pnl'].mean():+.2f}"
            )

    long_t = tdf[tdf["side"] == 1]
    short_t = tdf[tdf["side"] == -1]
    if len(long_t):
        print(
            f"\n  LONG : {len(long_t):4d} trades  PnL={long_t['pnl'].sum():+.1f}  "
            f"WR={(long_t['pnl']>0).mean()*100:.0f}%  avg={long_t['pnl'].mean():+.2f}"
        )
    if len(short_t):
        print(
            f"  SHORT: {len(short_t):4d} trades  PnL={short_t['pnl'].sum():+.1f}  "
            f"WR={(short_t['pnl']>0).mean()*100:.0f}%  avg={short_t['pnl'].mean():+.2f}"
        )

    print("\n  Exit Breakdown:")
    for reason, grp in tdf.groupby("exit_reason"):
        wr_r = (grp["pnl"] > 0).mean() * 100
        print(f"    {reason:18s}: {len(grp):4d}  WR={wr_r:5.1f}%  avg={grp['pnl'].mean():7.2f}")

    stats = rebuild_directional_pnl(TRADES_CSV)
    all_stats = stats.get("all", {})
    gross_win = float(all_stats.get("gross_profit") or 0.0)
    gross_loss = abs(float(all_stats.get("gross_loss") or 0.0))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    print(f"\n  Profit Factor   : {pf:.2f}")
    print(f"  Daily Drawdown  : {float(all_stats.get('daily_max_drawdown') or 0.0):.1f}")
    print(f"  Avg Day PnL     : {float(all_stats.get('avg_day') or 0.0):.1f}")
    print(f"  Positive Days   : {float(all_stats.get('positive_days_pct') or 0.0):.1f}%")
    print(f"  Trades/Day      : {float(all_stats.get('avg_trades_per_day') or 0.0):.1f}")
    print(f"  Avg Duration    : {float(all_stats.get('avg_duration_min') or 0.0):.1f} min")

    st = stats.get("streaks", {})
    print("\n  Streaks:")
    print(f"    Max Win Streak   : {int(st.get('max_win_streak', 0))}")
    print(f"    Max Loss Streak  : {int(st.get('max_loss_streak', 0))}")

    avg_win = float(all_stats.get("avg_win") or 0.0)
    avg_loss = float(all_stats.get("avg_loss") or 0.0)
    expectancy = (wr / 100.0) * avg_win + (1.0 - wr / 100.0) * avg_loss
    trades_per_day = float(all_stats.get("avg_trades_per_day") or 0.0)
    recovery_factor = (net_pnl / abs(max_dd)) if max_dd < 0 else float("inf")

    tdf_daily = tdf.copy()
    tdf_daily["entry_time"] = pd.to_datetime(tdf_daily["entry_time"], utc=True)
    tdf_daily["trade_day"] = tdf_daily["entry_time"].dt.tz_convert("America/New_York").dt.floor("D")
    daily_pnl = tdf_daily.groupby("trade_day")["pnl"].sum().astype(float)
    mean_day = float(daily_pnl.mean()) if len(daily_pnl) else 0.0
    std_day = float(daily_pnl.std(ddof=1)) if len(daily_pnl) > 1 else 0.0
    downside = daily_pnl[daily_pnl < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sharpe = (mean_day / std_day) * np.sqrt(252.0) if std_day > 0 else 0.0
    sortino = (mean_day / downside_std) * np.sqrt(252.0) if downside_std > 0 else 0.0

    print("\n  Risk-Adjusted:")
    print(f"    Expectancy/Trade   : {expectancy:.2f}")
    print(f"    Expectancy/Day     : {expectancy * trades_per_day:.2f}")
    print(f"    Recovery Factor    : {recovery_factor:.3f}")
    print(f"    Sharpe  (annualized): {sharpe:.2f}")
    print(f"    Sortino (annualized): {sortino:.2f}")

    target_hit_stats = stats.get("target_hit", {})
    reverse_stats = stats.get("reverse_signal", {})
    timeout_stats = stats.get("timeout", {})
    print("\n  Exit Reason Details:")
    print(f"    Target Hit   : {int(target_hit_stats.get('trades', 0))} trades  avg={float(target_hit_stats.get('avg_pnl') or 0.0):.2f}")
    print(
        f"    Reverse Sig  : {int(reverse_stats.get('trades', 0))} trades  "
        f"WR={float(reverse_stats.get('win_rate_pct') or 0.0):.1f}%  "
        f"avg={float(reverse_stats.get('avg_pnl') or 0.0):.2f}"
    )
    print(
        f"    Timeout      : {int(timeout_stats.get('trades', 0))} trades  "
        f"WR={float(timeout_stats.get('win_rate_pct') or 0.0):.1f}%  "
        f"avg={float(timeout_stats.get('avg_pnl') or 0.0):.2f}"
    )

    print("\n  Target Updates:")
    print(
        f"    Mean={float(all_stats.get('target_updates_mean') or 0.0):.2f}  "
        f"Median={float(all_stats.get('target_updates_median') or 0.0):.2f}  "
        f"Max={int(all_stats.get('target_updates_max') or 0)}"
    )

    mdf = tdf.copy()
    mdf["entry_time"] = pd.to_datetime(mdf["entry_time"], utc=True)
    mdf["month"] = mdf["entry_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)
    monthly = mdf.groupby("month")["pnl"].agg(
        trades="size",
        total_pnl="sum",
        avg_trade="mean",
        win_rate=lambda s: (s > 0).mean() * 100.0,
    ).reset_index()
    monthly["win_rate"] = monthly["win_rate"].map(lambda v: f"{v:.1f}%")

    print(f"\n{'='*60}")
    print("  MONTHLY STATISTICS")
    print(f"{'='*60}")
    print(monthly.to_string(index=False))

    ydf = tdf.copy()
    ydf["entry_time"] = pd.to_datetime(ydf["entry_time"], utc=True)
    ydf["year"] = ydf["entry_time"].dt.tz_convert("UTC").dt.year
    yearly = ydf.groupby("year")["pnl"].agg(
        trades="size",
        total_pnl="sum",
        avg_trade="mean",
        win_rate=lambda s: (s > 0).mean() * 100.0,
    ).reset_index()
    yearly["win_rate"] = yearly["win_rate"].map(lambda v: f"{v:.1f}%")
    print(f"\n{'='*60}")
    print("  YEARLY STATISTICS")
    print(f"{'='*60}")
    print(yearly.to_string(index=False))

    from config.hybrid_config import WF_CONFIG
    from xgboost_filter_model.pattern_training import iter_wf_cycles, wf_anchor_ts

    cdf = tdf.copy()
    cdf["entry_time"] = pd.to_datetime(cdf["entry_time"], utc=True)
    wf = wf_anchor_ts()
    bt_start_dt = cdf["entry_time"].min()
    end_dt = cdf["entry_time"].max() + pd.Timedelta(minutes=1)
    cycle_ranges: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for cycle, cur, ce in iter_wf_cycles(bt_start_dt, end_dt, wf):
        cycle_ranges.append((cur, ce, f"cycle_{cycle} ({cur.date()} to {ce.date()})"))

    def assign_cycle(ts: pd.Timestamp) -> str:
        ts = pd.to_datetime(ts, utc=True)
        for cur, ce, label in cycle_ranges:
            if cur <= ts < ce:
                return label
        return cycle_ranges[-1][2] if cycle_ranges else "unknown"

    cdf["retrain_cycle"] = cdf["entry_time"].apply(assign_cycle)
    cycle_stats = cdf.groupby("retrain_cycle")["pnl"].agg(
        trades="size",
        total_pnl="sum",
        avg_trade="mean",
        win_rate=lambda s: (s > 0).mean() * 100.0,
    ).reset_index()
    cycle_stats["win_rate"] = cycle_stats["win_rate"].map(lambda v: f"{v:.1f}%")
    print(f"\n{'='*60}")
    print(f"  RETRAIN CYCLE STATISTICS  (every {WF_CONFIG['retrain_days']}d from WF anchor)")
    print(f"{'='*60}")
    print(cycle_stats.to_string(index=False))

    wdf = tdf.copy()
    wdf["entry_time"] = pd.to_datetime(wdf["entry_time"], utc=True)
    wdf["weekday_utc2"] = (wdf["entry_time"] + pd.Timedelta(hours=2)).dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_utc2 = wdf.groupby("weekday_utc2")["pnl"].agg(
        trades="size",
        total_pnl="sum",
        avg_trade="mean",
        win_rate_pct=lambda s: (s > 0).mean() * 100.0,
    ).reindex([d for d in weekday_order if d in wdf["weekday_utc2"].unique()]).reset_index()
    weekday_utc2["win_rate_pct"] = weekday_utc2["win_rate_pct"].map(lambda v: f"{v:.1f}%")
    print("\n  Weekday (UTC+2):")
    print(weekday_utc2.to_string(index=False))

    td = all_stats.get("time_distribution", {})
    session_rows = td.get("by_session", [])
    if session_rows:
        session_df = pd.DataFrame(session_rows)
        print("\n  Session Breakdown:")
        print(session_df[["session", "trades", "total_pnl", "avg_trade", "win_rate_pct"]].to_string(index=False))

    sh = td.get("session_heatmaps", {})
    if sh:
        print(f"\n{'='*60}")
        print("  SESSION HEATMAPS")
        print(f"{'='*60}")
        for sess in ("hkt", "london", "ny"):
            sess_block = sh.get(sess)
            if not sess_block:
                continue
            rendered = sess_block.get("rendered_tables", {})
            for metric_key in ("total_pnl", "win_rate_pct", "trade_count"):
                table = rendered.get(metric_key)
                if table:
                    print(f"\n{table}")

    from backtest.trade_display import print_trades_table_hkt

    print_trades_table_hkt(tdf, tail=30)

    print(f"\n{'='*60}\n")


def main() -> None:
    argv = sys.argv[1:]
    run_backtest(argv)
    if not TRADES_CSV.exists():
        print(f"No trades file at {TRADES_CSV}")
        sys.exit(1)
    tdf = pd.read_csv(TRADES_CSV)
    if tdf.empty:
        print("No trades in CSV.")
        sys.exit(0)
    print_full_stats(tdf)


if __name__ == "__main__":
    main()
