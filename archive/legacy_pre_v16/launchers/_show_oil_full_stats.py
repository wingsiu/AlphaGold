#!/usr/bin/env python3
"""Full stats for Oil Combined v29 backtest — v14-style format (complete).

Covers every section that run_hybrid_backtest.py produces:
  Full Statistics, By Type, LONG/SHORT, Exit Breakdown,
  Profit Factor, Daily DD, Streaks, Risk-Adjusted,
  Monthly/Yearly tables, Weekday breakdown, Session heatmaps,
  Last trades table (HKT), PnL distribution.
"""
import pandas as pd
import numpy as np

CSV = "runtime/oil_combined_backtest_trades.csv"

# ── Load ──────────────────────────────────────────────────
tdf = pd.read_csv(CSV)
tdf["pnl"] = tdf["pnl"].astype(float)
tdf["entry_time"] = pd.to_datetime(tdf["entry"], utc=True)
tdf["exit_time"] = pd.to_datetime(tdf["exit"], utc=True)
tdf["exit_reason"] = tdf["reason"]
tdf["side_num"] = tdf["side"].map({1: 1, -1: -1})
tdf["side_label"] = tdf["side_num"].map({1: "up", -1: "down"})
tdf["dir"] = tdf["side_label"]

# ── Basic Stats ──────────────────────────────────────────
n = len(tdf)
wins = int((tdf["pnl"] > 0).sum())
net_pnl = float(tdf["pnl"].sum())
wr = wins / n * 100
cum = tdf["pnl"].cumsum()
max_dd = float((cum - cum.cummax()).min())

print("=" * 60)
print("  OIL COMBINED v29 — Full Backtest")
print("  2024-07-01 → 2026-06-18  |  ML-Filtered Three Legs")
print("=" * 60)
print(f"  Trades       : {n}  (W:{wins}  L:{n - wins})")
print(f"  Win Rate     : {wr:.1f}%")
print(f"  Net PnL      : {net_pnl:+.2f}")
print(f"  Avg/Trade    : {net_pnl / n:+.2f}")
print(f"  Max DD       : {max_dd:+.2f}")

# ── By Type ──────────────────────────────────────────────
print()
print("  By type:")
for name, grp in tdf.groupby("type", dropna=False):
    pnl_s = grp["pnl"]
    print(f"    {name:20s}: {len(grp):4d}  PnL={pnl_s.sum():+.1f}  WR={(pnl_s>0).mean()*100:.0f}%  avg={pnl_s.mean():+.2f}")

# ── LONG / SHORT ─────────────────────────────────────────
print()
long_t = tdf[tdf["side_num"] == 1]
short_t = tdf[tdf["side_num"] == -1]
if len(long_t):
    print(f"  LONG : {len(long_t):4d}  PnL={long_t['pnl'].sum():+.1f}  WR={(long_t['pnl']>0).mean()*100:.0f}%  avg={long_t['pnl'].mean():+.2f}")
if len(short_t):
    print(f"  SHORT: {len(short_t):4d}  PnL={short_t['pnl'].sum():+.1f}  WR={(short_t['pnl']>0).mean()*100:.0f}%  avg={short_t['pnl'].mean():+.2f}")

# ── Exit Breakdown ───────────────────────────────────────
print("\n  Exit Breakdown:")
for reason, grp in tdf.groupby("exit_reason"):
    rw = (grp["pnl"] > 0).mean() * 100
    print(f"    {reason:18s}: {len(grp):4d}  WR={rw:5.1f}%  avg={grp['pnl'].mean():+7.2f}")

# ── Durations ────────────────────────────────────────────
tdf["dur_min"] = (tdf["exit_time"] - tdf["entry_time"]).dt.total_seconds() / 60.0
avg_dur = float(tdf["dur_min"].mean())
med_dur = float(tdf["dur_min"].median())

# ── Profit Factor, DD, Daily Stats ───────────────────────
pnl_vals = tdf["pnl"].values
wins_arr = pnl_vals[pnl_vals > 0]
losses_arr = pnl_vals[pnl_vals < 0]
gross_w = float(wins_arr.sum())
gross_l = abs(float(losses_arr.sum()))
pf = gross_w / gross_l if gross_l > 0 else float("inf")
avg_win = float(wins_arr.mean()) if len(wins_arr) else 0.0
avg_loss = float(losses_arr.mean()) if len(losses_arr) else 0.0

tdf["trade_day"] = tdf["entry_time"].dt.tz_convert("America/New_York").dt.floor("D")
daily_pnl = tdf.groupby("trade_day")["pnl"].sum().astype(float)
n_days = len(daily_pnl)
pos_days = float((daily_pnl > 0).mean() * 100) if n_days else 0.0
mean_day = float(daily_pnl.mean()) if n_days else 0.0
daily_dd = float((daily_pnl.cumsum() - daily_pnl.cumsum().cummax()).min())
td_per_day = n / n_days if n_days else 0.0

print()
print(f"  Profit Factor   : {pf:.2f}")
print(f"  Daily Drawdown  : {daily_dd:.1f}")
print(f"  Avg Day PnL     : {mean_day:.1f}")
print(f"  Positive Days   : {pos_days:.1f}%")
print(f"  Trades/Day      : {td_per_day:.1f}")
print(f"  Avg Duration    : {avg_dur:.1f} min")
print(f"  Median Duration : {med_dur:.1f} min")

# ── Streaks ──────────────────────────────────────────────
streaks_win = 0; streaks_loss = 0; cur_win = 0; cur_loss = 0
for v in pnl_vals:
    if v > 0:
        cur_win += 1; cur_loss = 0
        streaks_win = max(streaks_win, cur_win)
    elif v < 0:
        cur_loss += 1; cur_win = 0
        streaks_loss = max(streaks_loss, cur_loss)

print()
print("  Streaks:")
print(f"    Max Win Streak   : {streaks_win}")
print(f"    Max Loss Streak  : {streaks_loss}")

# ── Risk-Adjusted ────────────────────────────────────────
std_day = float(daily_pnl.std(ddof=1)) if n_days > 1 else 0.0
downside = daily_pnl[daily_pnl < 0]
downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
sharpe = (mean_day / std_day) * np.sqrt(252.0) if std_day > 0 else 0.0
sortino = (mean_day / downside_std) * np.sqrt(252.0) if downside_std > 0 else 0.0
expectancy = (wr / 100.0) * avg_win + (1.0 - wr / 100.0) * avg_loss
recovery_factor = (net_pnl / abs(max_dd)) if max_dd < 0 else float("inf")

print()
print("  Risk-Adjusted:")
print(f"    Expectancy/Trade   : {expectancy:.2f}")
print(f"    Expectancy/Day     : {expectancy * td_per_day:.2f}")
print(f"    Recovery Factor    : {recovery_factor:.3f}")
print(f"    Sharpe  (annualized): {sharpe:.2f}")
print(f"    Sortino (annualized): {sortino:.2f}")

# ── Exit Reason Details ──────────────────────────────────
print("\n  Exit Reason Details:")
for rsn, grp in tdf.groupby("exit_reason"):
    rw = (grp["pnl"] > 0).mean() * 100
    print(f"    {rsn:18s}: {len(grp):4d}t  WR={rw:5.1f}%  avg={grp['pnl'].mean():+.2f}")

# ── MONTHLY STATISTICS (table) ───────────────────────────
mdf = tdf.copy()
mdf["entry_time"] = pd.to_datetime(mdf["entry_time"], utc=True)
mdf["month"] = (
    mdf["entry_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)
)
monthly = (
    mdf.groupby("month")["pnl"]
    .agg(
        trades="size",
        total_pnl="sum",
        avg_trade="mean",
        win_rate=lambda s: (s > 0).mean() * 100.0,
    )
    .reset_index()
    .sort_values("month")
)
monthly["win_rate"] = monthly["win_rate"].map(lambda v: f"{v:.1f}%")

print(f"\n{'='*60}")
print("  MONTHLY STATISTICS")
print(f"{'='*60}")
print(monthly.to_string(index=False))

# ── YEARLY STATISTICS (table) ────────────────────────────
ydf = tdf.copy()
ydf["entry_time"] = pd.to_datetime(ydf["entry_time"], utc=True)
ydf["year"] = ydf["entry_time"].dt.tz_convert("UTC").dt.year
yearly = (
    ydf.groupby("year")["pnl"]
    .agg(
        trades="size",
        total_pnl="sum",
        avg_trade="mean",
        win_rate=lambda s: (s > 0).mean() * 100.0,
    )
    .reset_index()
)
yearly["win_rate"] = yearly["win_rate"].map(lambda v: f"{v:.1f}%")
print(f"\n{'='*60}")
print("  YEARLY STATISTICS")
print(f"{'='*60}")
print(yearly.to_string(index=False))

# ── Weekday (UTC+2) ──────────────────────────────────────
wdf = tdf.copy()
wdf["entry_time"] = pd.to_datetime(wdf["entry_time"], utc=True)
wdf["weekday_utc2"] = (wdf["entry_time"] + pd.Timedelta(hours=2)).dt.day_name()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
wd_available = [d for d in weekday_order if d in wdf["weekday_utc2"].unique()]
weekday_table = (
    wdf.groupby("weekday_utc2")["pnl"]
    .agg(
        trades="size",
        total_pnl="sum",
        avg_trade="mean",
        win_rate_pct=lambda s: (s > 0).mean() * 100.0,
    )
    .reindex(wd_available)
    .reset_index()
)
weekday_table["win_rate_pct"] = weekday_table["win_rate_pct"].map(lambda v: f"{v:.1f}%")
print("\n  Weekday (UTC+2):")
print(weekday_table.to_string(index=False))

# ── Monthly WF cycles + session heatmaps (gold-style) ────
from oil.backtest_stats import print_monthly_wf_cycle_stats, print_session_heatmaps

print_monthly_wf_cycle_stats(tdf)
print_session_heatmaps(tdf)

# ── LAST 30 TRADES (HKT) ─────────────────────────────────
view = tdf.copy()
view["entry_hkt"] = (
    pd.to_datetime(view["entry_time"], utc=True)
    .dt.tz_convert("Asia/Hong_Kong")
    .dt.strftime("%m-%d %H:%M")
)
view["exit_hkt"] = (
    pd.to_datetime(view["exit_time"], utc=True)
    .dt.tz_convert("Asia/Hong_Kong")
    .dt.strftime("%H:%M")
)
show = view.sort_values("entry_time").tail(30)
print(f"\n{'─'*72}")
print(f"  LAST {len(show)} TRADES (HKT)")
print(f"{'─'*72}")
cols = ["entry_hkt", "exit_hkt", "dir", "entry_price", "exit_price", "pnl", "exit_reason", "type"]
available = [c for c in cols if c in show.columns]
print(show[available].to_string(index=False))

# ── Trade PnL Distribution ───────────────────────────────
print(f"\n{'='*60}")
print("  TRADE PNL DISTRIBUTION")
print(f"{'='*60}")
bins = [-100, -80, -60, -40, -20, -10, 0, 10, 20, 40, 60, 80, 100, 200, 500]
counts, edges = np.histogram(tdf["pnl"], bins=bins)
for i in range(len(counts)):
    bar = "#" * max(1, counts[i] // 2)
    print(f"    {edges[i]:+6.0f} → {edges[i+1]:+6.0f}: {counts[i]:4d} {bar}")

print("\nDone.")
