#!/usr/bin/env python3
"""Display full stats for the v15 backtest CSV (deterministic energetic gate)."""
import pandas as pd
import numpy as np
from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl

tdf = pd.read_csv("runtime/v15_backtest_trades.csv")
wins = int((tdf["pnl"] > 0).sum())
net_pnl = float(tdf["pnl"].sum())
wr = wins / len(tdf) * 100
cum = tdf["pnl"].cumsum()
max_dd = float((cum - cum.cummax()).min())

print("=" * 60)
print("  V15 HYBRID BACKTEST — Deterministic Energetic Gate (no HMM)")
print(f"  {tdf['entry_time'].min()} → {tdf['entry_time'].max()}")
print("=" * 60)
print(f"  Trades       : {len(tdf)}  (W:{wins}  L:{len(tdf)-wins})")
print(f"  Win Rate     : {wr:.1f}%")
print(f"  Net PnL      : {net_pnl:+.2f}")
print(f"  Avg/Trade    : {net_pnl/len(tdf):+.2f}")
print(f"  Max DD       : {max_dd:+.2f}")

# By source
print()
print("  By source:")
for src, grp in tdf.groupby("source", dropna=False):
    src_wr = (grp["pnl"] > 0).mean() * 100
    print(f"    {str(src):10s}: {len(grp):4d} trades  PnL={grp['pnl'].sum():+.1f}  WR={src_wr:.1f}%  avg={grp['pnl'].mean():+.2f}")

# By matched_pattern
print()
print("  By matched_pattern:")
col = "matched_pattern" if "matched_pattern" in tdf.columns else "pattern"
for name, grp in tdf.groupby(col, dropna=False):
    label = name if pd.notna(name) else "unknown"
    print(f"    {label:20s}: {len(grp):4d}  PnL={grp['pnl'].sum():+.1f}  WR={(grp['pnl']>0).mean()*100:.0f}%  avg={grp['pnl'].mean():+.2f}")

# LONG/SHORT
print()
if "side" in tdf.columns:
    long_t = tdf[tdf["side"] == 1]
    short_t = tdf[tdf["side"] == -1]
    print(f"  LONG : {len(long_t):4d} trades  PnL={long_t['pnl'].sum():+.1f}  WR={(long_t['pnl']>0).mean()*100:.0f}%  avg={long_t['pnl'].mean():+.2f}")
    print(f"  SHORT: {len(short_t):4d} trades  PnL={short_t['pnl'].sum():+.1f}  WR={(short_t['pnl']>0).mean()*100:.0f}%  avg={short_t['pnl'].mean():+.2f}")

# Exit breakdown
print()
print("  Exit Breakdown:")
for reason, grp in tdf.groupby("exit_reason"):
    wr_r = (grp["pnl"] > 0).mean() * 100
    print(f"    {reason:18s}: {len(grp):4d}  WR={wr_r:5.1f}%  avg={grp['pnl'].mean():7.2f}")

# Rebuild full stats
stats = rebuild_directional_pnl("runtime/v15_backtest_trades.csv")
all_stats = stats.get("all", {})
gross_win = float(all_stats.get("gross_profit") or 0.0)
gross_loss = abs(float(all_stats.get("gross_loss") or 0.0))
pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

print()
print(f"  Profit Factor   : {pf:.2f}")
print(f"  Daily Drawdown  : {float(all_stats.get('daily_max_drawdown') or 0.0):.1f}")
print(f"  Avg Day PnL     : {float(all_stats.get('avg_day') or 0.0):.1f}")
print(f"  Positive Days   : {float(all_stats.get('positive_days_pct') or 0.0):.1f}%")
print(f"  Trades/Day      : {float(all_stats.get('avg_trades_per_day') or 0.0):.1f}")
print(f"  Avg Duration    : {float(all_stats.get('avg_duration_min') or 0.0):.1f} min")

st = stats.get("streaks", {})
print()
print("  Streaks:")
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

print()
print("  Risk-Adjusted:")
print(f"    Expectancy/Trade   : {expectancy:.2f}")
print(f"    Expectancy/Day     : {expectancy * trades_per_day:.2f}")
print(f"    Recovery Factor    : {recovery_factor:.3f}")
print(f"    Sharpe  (annualized): {sharpe:.2f}")
print(f"    Sortino (annualized): {sortino:.2f}")

print()
print("  Target Updates:")
print(f"    Mean={float(all_stats.get('target_updates_mean') or 0.0):.2f}  Median={float(all_stats.get('target_updates_median') or 0.0):.2f}  Max={int(all_stats.get('target_updates_max') or 0)}")

# Monthly breakdown
print()
print("  Monthly Breakdown:")
tdf["entry_ts"] = pd.to_datetime(tdf["entry_time"])
tdf["month"] = tdf["entry_ts"].dt.to_period("M")
for m, grp in tdf.groupby("month"):
    print(f"    {m}: {len(grp):3d}t  PnL={grp['pnl'].sum():+8.1f}  WR={(grp['pnl']>0).mean()*100:.0f}%  avg={grp['pnl'].mean():+.2f}")

# Top 5 / Bottom 5
print()
print("  Top 5 Best Trades:")
for i, (_, t) in enumerate(tdf.nlargest(5, "pnl").iterrows()):
    dur = str(pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])) if "exit_time" in tdf.columns else "?"
    print(f"    {i+1}. entry={t['entry_ts']}  PnL={t['pnl']:+.1f}  dur={dur}  reason={t['exit_reason']}  pattern={t.get('matched_pattern','?')}")

print()
print("  Bottom 5 Worst Trades:")
for i, (_, t) in enumerate(tdf.nsmallest(5, "pnl").iterrows()):
    dur = str(pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])) if "exit_time" in tdf.columns else "?"
    print(f"    {i+1}. entry={t['entry_ts']}  PnL={t['pnl']:+.1f}  dur={dur}  reason={t['exit_reason']}  pattern={t.get('matched_pattern','?')}")

# Trade distribution
print()
print("  Trade PnL Distribution:")
bins = [-100, -80, -60, -40, -20, -10, 0, 10, 20, 40, 60, 80, 100, 200, 500]
counts, edges = np.histogram(tdf["pnl"], bins=bins)
for i in range(len(counts)):
    bar = "#" * max(1, counts[i] // 2)
    print(f"    {edges[i]:+6.0f} → {edges[i+1]:+6.0f}: {counts[i]:4d} {bar}")

print("\nDone.")
