import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl

out_path = PROJECT_ROOT / "runtime" / "v14_backtest_trades.csv"
tdf = pd.read_csv(out_path)

wins = int((tdf['pnl'] > 0).sum())
losses = int((tdf['pnl'] <= 0).sum())
net_pnl = float(tdf['pnl'].sum())
wr = float(wins / len(tdf) * 100.0) if len(tdf) > 0 else 0.0

print(f"============================================================")
print(f"  AlphaGold v14 Backtest (30 Horizon / 30 TP / 25 SL)")
print(f"  Period : 2026-01-01 → 2026-05-21")
print(f"============================================================")
print(f"  Trades       : {len(tdf)}  (W:{wins}  L:{losses})")
print(f"  Win Rate     : {wr:.1f}%")
print(f"  Net PnL      : {net_pnl:.1f}")
print(f"  Avg Trade    : {net_pnl/len(tdf):.2f}")

tdf_long  = tdf[tdf['side'] == 'up']
tdf_short = tdf[tdf['side'] == 'down']
if len(tdf_long) > 0:
    print(f"\n  LONG : {len(tdf_long):4d} trades  PnL={tdf_long['pnl'].sum():.1f}  "
          f"WR={(tdf_long['pnl']>0).mean()*100:.1f}%  avg={tdf_long['pnl'].mean():.2f}")
else:
    print(f"\n  LONG :    0 trades  PnL=0.0  WR=0.0%  avg=0.00")

if len(tdf_short) > 0:
    print(f"  SHORT: {len(tdf_short):4d} trades  PnL={tdf_short['pnl'].sum():.1f}  "
          f"WR={(tdf_short['pnl']>0).mean()*100:.1f}%  avg={tdf_short['pnl'].mean():.2f}")
else:
    print(f"  SHORT:    0 trades  PnL=0.0  WR=0.0%  avg=0.00")

try:
    stats = rebuild_directional_pnl(out_path)
except Exception as exc:
    stats = None
    print(f"\n  Warning: extended stats unavailable ({exc})")

print(f"\n  Exit Breakdown:")
for reason, grp in tdf.groupby('exit_reason'):
    wr_r = (grp['pnl'] > 0).mean() * 100
    print(f"    {reason:18s}: {len(grp):4d}  WR={wr_r:5.1f}%  avg={grp['pnl'].mean():7.2f}")

if stats is not None:
    all_stats = stats.get('all', {})
    gross_win = float(all_stats.get('gross_profit') or 0.0)
    gross_loss = abs(float(all_stats.get('gross_loss') or 0.0))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float('inf')
    max_dd = float(all_stats.get('trade_max_drawdown') or 0.0)

    print(f"\n  Profit Factor   : {pf:.3f}")
    print(f"  Max Drawdown    : {max_dd:.1f}")
    print(f"  Daily Drawdown  : {float(all_stats.get('daily_max_drawdown') or 0.0):.1f}")
    print(f"  Avg Day PnL     : {float(all_stats.get('avg_day') or 0.0):.1f}")
    print(f"  Positive Days   : {float(all_stats.get('positive_days_pct') or 0.0):.1f}%")
    print(f"  Trades/Day      : {float(all_stats.get('avg_trades_per_day') or 0.0):.1f}")
    print(f"  Avg Duration    : {float(all_stats.get('avg_duration_min') or 0.0):.1f} min")

    st = stats.get('streaks', {})
    print(f"\n  Streaks:")
    print(f"    Max Win Streak   : {int(st.get('max_win_streak', 0))}")
    print(f"    Max Loss Streak  : {int(st.get('max_loss_streak', 0))}")
    print(f"    Current Win      : {int(st.get('current_win_streak', 0))}")
    print(f"    Current Loss     : {int(st.get('current_loss_streak', 0))}")

    avg_win = all_stats.get('avg_win')
    avg_loss = all_stats.get('avg_loss')
    avg_win = float(avg_win) if avg_win is not None else 0.0
    avg_loss = float(avg_loss) if avg_loss is not None else 0.0
    expectancy = (wr / 100.0) * avg_win + (1.0 - wr / 100.0) * avg_loss

    trades_per_day = float(all_stats.get('avg_trades_per_day') or 0.0)
    expectancy_per_day = expectancy * trades_per_day
    recovery_factor = (net_pnl / abs(max_dd)) if max_dd < 0 else float('inf')

    tdf_daily = tdf.copy()
    tdf_daily['entry_time'] = pd.to_datetime(tdf_daily['entry_time'], utc=True)
    tdf_daily['trade_day'] = tdf_daily['entry_time'].dt.tz_convert('America/New_York').dt.floor('D')
    daily_pnl = tdf_daily.groupby('trade_day')['pnl'].sum().astype(float)
    mean_day = float(daily_pnl.mean()) if len(daily_pnl) else 0.0
    std_day = float(daily_pnl.std(ddof=1)) if len(daily_pnl) > 1 else 0.0
    downside = daily_pnl[daily_pnl < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sharpe = (mean_day / std_day) * np.sqrt(252.0) if std_day > 0 else 0.0
    sortino = (mean_day / downside_std) * np.sqrt(252.0) if downside_std > 0 else 0.0

    print(f"\n  Risk-Adjusted:")
    print(f"    Expectancy/Trade   : {expectancy:.2f}")
    print(f"    Expectancy/Day     : {expectancy_per_day:.2f}")
    print(f"    Recovery Factor    : {recovery_factor:.3f}")
    print(f"    Sharpe  (annualized, √252) : {sharpe:.2f}")
    print(f"    Sortino (annualized, √252) : {sortino:.2f}")

    target_hit_stats = stats.get('target_hit', {})
    reverse_stats = stats.get('reverse_signal', {})
    timeout_stats = stats.get('timeout', {})

    print(f"\n  Exit Reason Details:")
    print(f"    Target Hit   : {int(target_hit_stats.get('trades', 0))} trades  avg={float(target_hit_stats.get('avg_pnl') or 0.0):.2f}")
    print(f"    Reverse Sig  : {int(reverse_stats.get('trades', 0))} trades  WR={float(reverse_stats.get('win_rate_pct') or 0.0):.1f}%  avg={float(reverse_stats.get('avg_pnl') or 0.0):.2f}")
    print(f"    Timeout      : {int(timeout_stats.get('trades', 0))} trades  WR={float(timeout_stats.get('win_rate_pct') or 0.0):.1f}%  avg={float(timeout_stats.get('avg_pnl') or 0.0):.2f}")

    print(f"\n  Target Updates:")
    print(f"    Mean={float(all_stats.get('target_updates_mean') or 0.0):.2f}  "
          f"Median={float(all_stats.get('target_updates_median') or 0.0):.2f}  "
          f"Max={int(all_stats.get('target_updates_max') or 0)}")

    mdf = tdf.copy()
    mdf['entry_time'] = pd.to_datetime(mdf['entry_time'], utc=True)
    mdf['month'] = mdf['entry_time'].dt.tz_convert('UTC').dt.tz_localize(None).dt.to_period('M').astype(str)
    monthly = mdf.groupby('month')['pnl'].agg(
        trades='size',
        total_pnl='sum',
        avg_trade='mean',
        win_rate=lambda s: (s > 0).mean() * 100.0,
    ).reset_index()
    monthly['win_rate'] = monthly['win_rate'].map(lambda v: f"{v:.1f}%")

    print(f"\n{'='*60}")
    print("  MONTHLY STATISTICS")
    print(f"{'='*60}")
    print(monthly.to_string(index=False))

    ydf = tdf.copy()
    ydf['entry_time'] = pd.to_datetime(ydf['entry_time'], utc=True)
    ydf['year'] = ydf['entry_time'].dt.tz_convert('UTC').dt.year
    yearly = ydf.groupby('year')['pnl'].agg(
        trades='size',
        total_pnl='sum',
        avg_trade='mean',
        win_rate=lambda s: (s > 0).mean() * 100.0,
    ).reset_index()
    yearly['win_rate'] = yearly['win_rate'].map(lambda v: f"{v:.1f}%")
    print(f"\n{'='*60}")
    print("  YEARLY STATISTICS")
    print(f"{'='*60}")
    print(yearly.to_string(index=False))

    wdf = tdf.copy()
    wdf['entry_time'] = pd.to_datetime(wdf['entry_time'], utc=True)
    wdf['weekday_utc2'] = (wdf['entry_time'] + pd.Timedelta(hours=2)).dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_utc2 = wdf.groupby('weekday_utc2')['pnl'].agg(
        trades='size',
        total_pnl='sum',
        avg_trade='mean',
        win_rate_pct=lambda s: (s > 0).mean() * 100.0,
    ).reindex([d for d in weekday_order if d in wdf['weekday_utc2'].unique()]).reset_index()
    weekday_utc2['win_rate_pct'] = weekday_utc2['win_rate_pct'].map(lambda v: f"{v:.1f}%")
    print(f"\n  Weekday (UTC+2):")
    print(weekday_utc2.to_string(index=False))

    td = all_stats.get('time_distribution', {})
    session_rows = td.get('by_session', [])
    if session_rows:
        session_df = pd.DataFrame(session_rows)
        print(f"\n  Session Breakdown:")
        print(session_df[['session', 'trades', 'total_pnl', 'avg_trade', 'win_rate_pct']].to_string(index=False))

    sh = td.get('session_heatmaps', {})
    if sh:
        print(f"\n{'='*60}")
        print("  SESSION HEATMAPS")
        print(f"{'='*60}")
        for sess in ('hkt', 'london', 'ny'):
            sess_block = sh.get(sess)
            if not sess_block:
                continue
            rendered = sess_block.get('rendered_tables', {})
            for metric_key in ('total_pnl', 'win_rate_pct', 'trade_count'):
                table = rendered.get(metric_key)
                if table:
                    print(f"\n{table}")
