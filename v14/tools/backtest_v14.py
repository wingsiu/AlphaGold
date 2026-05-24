#!/usr/bin/env python3
"""
AlphaGold v14 — Unified Backtest
=================================
Usage:
  python3 backtest_v14.py                         # full history (Jan 2025 → today, WF cycle models)
  python3 backtest_v14.py 30                      # last 30 days (WF cycle models)
"""
import os
import sys
from pathlib import Path
from datetime import date, timedelta

from v14._paths import PROJECT_ROOT

import pandas as pd
import numpy as np
import joblib

from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14
from xgboost_filter_model.train_filter_1min import load_price_data
from config.v14_config import ENERGETIC_EXECUTION_CONFIG, WF_CONFIG, TIME_FILTER_CONFIG

EXECUTION_CONFIG = ENERGETIC_EXECUTION_CONFIG
from v14.backtest.backtest_core import simulate_v13_core
from xgboost_filter_model.time_slot_filter import load_weak_filter, resolve_v14_time_filter_path

args = sys.argv[1:]
today_str      = date.today().strftime("%Y-%m-%d")
full_start     = WF_CONFIG["full_start"]
wf_start       = WF_CONFIG["wf_start"]

days_back  = None
bt_start   = wf_start          # default: full history
bt_end     = today_str

if len(args) == 1:
    if args[0].isdigit():
        days_back = int(args[0])
        bt_start  = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    else:
        bt_start = args[0]      # custom start date
elif len(args) == 2:
    bt_start, bt_end = args[0], args[1]

print(f"\n{'='*60}")
print(f"  AlphaGold v14 Backtest")
print(f"  Period : {bt_start} → {bt_end}")
print(f"  Models : Walk-Forward cycle")
print(f"  Exec   : energetic S1/S2  close_on_reverse={EXECUTION_CONFIG['close_on_reverse']}  "
      f"same_dir_refresh={EXECUTION_CONFIG['same_dir_refresh']}")
print(f"{'='*60}\n")

warmup_days = int(WF_CONFIG.get("feature_warmup_days", 120))
load_start_dt = pd.to_datetime(bt_start)
if load_start_dt.tzinfo is not None:
    load_start_dt = load_start_dt.tz_localize(None)
full_start_dt = pd.to_datetime(full_start)
if full_start_dt.tzinfo is not None:
    full_start_dt = full_start_dt.tz_localize(None)
warmup_start_dt = load_start_dt - pd.Timedelta(days=warmup_days)
load_start_dt = max(full_start_dt, warmup_start_dt)
load_start = load_start_dt.strftime("%Y-%m-%d")
if 'T' in bt_end:
    bt_end_date = bt_end.split('T')[0]
else:
    bt_end_date = bt_end
load_end = (pd.to_datetime(bt_end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

if 'T' in bt_start:
    bt_start_date = bt_start.split('T')[0]
else:
    bt_start_date = bt_start
print(f"Loading data from {load_start} to {bt_end} (loader end-exclusive {load_end})…")

df = prepare_data_v14(start_date=load_start, end_date=load_end)
df = prepare_directional_data_v14(df)

exclude = {
    'open','high','low','close','volume','timestamp',
    'trend_label','target_v10', 'target_v14', 'is_trend','atr','day_utc2',
    'future_max_move','future_min_move','future_er','atr_threshold',
    'bar_move','hour','day_id','day_high','day_low','high_90','low_90',
    'closePrice_ask','closePrice_bid','highPrice_ask','lowPrice_bid',
    'closePrice','lowPrice','open_price','highPrice_bid','lowPrice_ask',
    'openPrice_bid','openPrice_ask',
    'day_open', 'day_high_rolling', 'day_low_rolling',
    'hmm_regime', 'high_60m', 'low_60m', 'low_15m', 'high_15m', 'ma_60m',
    'daily_poc', 'daily_vwap', 'rolling_poc_4h', 'dynamic_tp', 'dynamic_sl',
    'fvg_bull_bottom', 'fvg_bull_top', 'fvg_bear_top', 'fvg_bear_bottom',
}
s2_extra = {
    'directional_change_15','directional_change_30','directional_change_90',
    'wick_ratio_15','wick_ratio_30','wick_ratio_90',
    'price_vs_ma_10','price_vs_ma_30','price_vs_ma_90',
    'ma_10_vs_30','ma_30_vs_90',
    'rsi_14','rsi_30','macd','macd_signal','macd_diff',
    'roc_15','roc_30','roc_60',
}
features    = [c for c in df.columns if c not in exclude]
s1_features = [f for f in features if f not in s2_extra]

bt_start_dt = pd.to_datetime(bt_start)
if bt_start_dt.tzinfo is None:
    bt_start_dt = bt_start_dt.tz_localize('UTC')
else:
    bt_start_dt = bt_start_dt.tz_convert('UTC')
    
df_test = df[df.index >= bt_start_dt].copy()
print(f"Energetic bars in test window: {len(df_test)}")

if df_test.empty:
    print("No energetic bars in the test window. Check date range and bar_move/volume filters.")
    sys.exit(0)

prod_s1 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v14_wf.joblib")
prod_s2 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v14_wf.joblib")

df_test['s1_prob'] = np.nan
df_test['s2_prob'] = np.nan

wf_dir = PROJECT_ROOT / os.environ.get(
    "V14_MODEL_OUTPUT_DIR",
    WF_CONFIG.get("model_output_dir", "runtime/bot_assets/wf_models_v14"),
)
retrain_days = WF_CONFIG.get("retrain_days", 14)
wf_anchor = pd.to_datetime(wf_start)
if wf_anchor.tzinfo is None:
    wf_anchor = wf_anchor.tz_localize('UTC')
else:
    wf_anchor = wf_anchor.tz_convert('UTC')
run_start = pd.to_datetime(bt_start)
if run_start.tzinfo is None:
    run_start = run_start.tz_localize('UTC')
else:
    run_start = run_start.tz_convert('UTC')
end_dt = pd.to_datetime(bt_end).tz_localize('UTC') + pd.Timedelta(days=1)

elapsed_days = max(0, (run_start - wf_anchor).days)
skip_cycles = elapsed_days // retrain_days
cycle = 1 + skip_cycles
current_start = wf_anchor + pd.Timedelta(days=skip_cycles * retrain_days)

wf_chunks = 0
prod_chunks = 0
print("Scoring with walk-forward cycle models…")
df_test['cycle_id'] = np.nan
df_test['model_path'] = None

present_models = []
missing_models = []

while current_start < end_dt:
    current_end = min(current_start + pd.Timedelta(days=retrain_days), end_dt)
    s1_path = wf_dir / f"filter_v14_cycle_{cycle}_{current_start.date()}.joblib"
    chunk = (df_test.index >= current_start) & (df_test.index < current_end)
    if chunk.any():
        if s1_path.exists():
            s1 = joblib.load(s1_path)
            wf_chunks += 1
            model_id = f"cycle_{cycle}_{current_start.date()}"
            model_path = str(s1_path)
            present_models.append(str(s1_path.name))
        else:
            s1 = prod_s1
            prod_chunks += 1
            model_id = "prod"
            model_path = str(PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v14_wf.joblib")
            missing_models.append(str(s1_path.name))
            
        df_test.loc[chunk, 's1_prob'] = s1.predict_proba(df_test.loc[chunk, s1_features])[:, 1]
        df_test.loc[chunk, 'cycle_id'] = model_id
        df_test.loc[chunk, 'model_path'] = model_path

        s2_path = wf_dir / f"directional_v14_cycle_{cycle}_{current_start.date()}.joblib"
        if s2_path.exists():
            s2 = joblib.load(s2_path)
        else:
            s2 = prod_s2
        s1_pass = chunk & (df_test['s1_prob'] >= EXECUTION_CONFIG["s1_threshold"])
        if s1_pass.any():
            df_test.loc[s1_pass, 's2_prob'] = s2.predict_proba(df_test.loc[s1_pass, features])[:, 1]
    current_start = current_end
    cycle += 1

s1_thresh    = EXECUTION_CONFIG["s1_threshold"]
s2_base      = EXECUTION_CONFIG["s2_threshold"]
s2_increment = EXECUTION_CONFIG["s2_loss_increment"]
s2_max       = EXECUTION_CONFIG["s2_max_threshold"]

trend_mask = df_test['s1_prob'] >= s1_thresh
df_test['side_signal'] = 0
df_test.loc[trend_mask & (df_test['s2_prob'] >= s2_base), 'side_signal']         =  1
df_test.loc[trend_mask & (df_test['s2_prob'] <= (1.0 - s2_base)), 'side_signal'] = -1

sig_count = (df_test['side_signal'] != 0).sum()
print(f"S1 ≥ {s1_thresh} bars : {trend_mask.sum()}")
print(f"Entry signals        : {sig_count}  "
      f"(LONG={(df_test['side_signal']==1).sum()}  SHORT={(df_test['side_signal']==-1).sum()})")

if sig_count == 0:
    print("\nNo signals in window.")
    sys.exit(0)

print("Loading full 1-min bars for exit simulation…")
raw_df = load_price_data(start_date=bt_start_date, end_date=load_end)
raw_df = raw_df[raw_df.index >= bt_start_dt].copy()

sim_df = raw_df[['open','high','low','close']].copy()
for col in ['side_signal','s1_prob','s2_prob','cycle_id','model_path']:
    sim_df[col] = df_test[col] if col in df_test.columns else np.nan
sim_df['side_signal'] = sim_df['side_signal'].fillna(0).astype(int)

tp = EXECUTION_CONFIG["tp"]
sl = EXECUTION_CONFIG["sl"]
horizon = EXECUTION_CONFIG["horizon"]

weak_cells = None
_filter_path = resolve_v14_time_filter_path(PROJECT_ROOT)
if _filter_path:
    weak_cells = load_weak_filter(_filter_path)
    print(f"Time filter: blocking {len(weak_cells)} slots from {_filter_path}")

all_trades = simulate_v13_core(
    sim_df, tp, sl, horizon, config=EXECUTION_CONFIG, weak_period_cells=weak_cells
)

if not all_trades:
    print("\nSignals found but no trades closed in the window.")
    sys.exit(0)

tdf = pd.DataFrame(all_trades)
tdf['side_str'] = tdf['side'].map({1: 'up', -1: 'down'})

out_path = PROJECT_ROOT / "runtime" / "v14_backtest_trades.csv"
tdf.drop(columns=['side_str']).rename(columns={'side': 'side_int'}).assign(
    side=tdf['side_str']
).drop(columns=['side_int']).to_csv(out_path, index=False)

wins    = int((tdf['pnl'] > 0).sum())
losses  = int((tdf['pnl'] <= 0).sum())
net_pnl = float(tdf['pnl'].sum())
wr      = float(wins / len(tdf) * 100.0)

print(f"\n{'='*60}")
print(f"  BACKTEST RESULTS  |  {bt_start} → {bt_end}")
print(f"{'='*60}")
print(f"  Trades       : {len(tdf)}  (W:{wins}  L:{losses})")
print(f"  Win Rate     : {wr:.1f}%")
print(f"  Net PnL      : {net_pnl:.1f}")
print(f"  Avg Trade    : {net_pnl/len(tdf):.2f}")

tdf_long  = tdf[tdf['side'] == 1]
tdf_short = tdf[tdf['side'] == -1]
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

print(f"\n  Exit Breakdown:")
for reason, grp in tdf.groupby('exit_reason'):
    wr_r = (grp['pnl'] > 0).mean() * 100
    print(f"    {reason:18s}: {len(grp):4d}  WR={wr_r:5.1f}%  avg={grp['pnl'].mean():7.2f}")

print(f"\n  Performance Metrics:")
cum_pnl = tdf['pnl'].cumsum()
max_dd = (cum_pnl.cummax() - cum_pnl).max()
gross_win = tdf[tdf['pnl'] > 0]['pnl'].sum()
gross_loss = abs(tdf[tdf['pnl'] <= 0]['pnl'].sum())
pf = (gross_win / gross_loss) if gross_loss > 0 else float('inf')
print(f"    Max Drawdown : -{max_dd:.1f}")
print(f"    Profit Factor: {pf:.2f}")
print(f"    Largest Win  : +{tdf['pnl'].max():.1f}")
print(f"    Largest Loss : {tdf['pnl'].min():.1f}")

try:
    from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl
    stats = rebuild_directional_pnl(out_path)
except Exception as exc:
    stats = None
    print(f"\n  Warning: extended stats unavailable ({exc})")

if stats is not None:
    all_stats = stats.get('all', {})
    
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

print(f"{'='*60}\n")

tdf_view = tdf.copy()
tdf_view['entry_hkt'] = pd.to_datetime(tdf_view['entry_time'], utc=True).dt.tz_convert('Asia/Hong_Kong').dt.strftime('%m-%d %H:%M')
tdf_view['exit_hkt']  = pd.to_datetime(tdf_view['exit_time'],  utc=True).dt.tz_convert('Asia/Hong_Kong').dt.strftime('%H:%M')
tdf_view['s1']  = tdf_view['s1_prob'].round(3)
tdf_view['s2']  = tdf_view['s2_prob'].round(3)
tdf_view['dir'] = tdf_view['side']

show = tdf_view if days_back else tdf_view.tail(30)
header = "ALL TRADES (HKT)" if days_back else "LAST 30 TRADES (HKT)"
print(f"\n{'─'*60}\n  {header}\n{'─'*60}")
print(show[['entry_hkt','exit_hkt','dir','entry_price','exit_price','pnl','exit_reason','s1','s2']].to_string(index=False))

s1_thresh = EXECUTION_CONFIG["s1_threshold"]
s2_base = EXECUTION_CONFIG["s2_threshold"]
sig_bars = df_test[df_test['s1_prob'] >= s1_thresh].tail(20).copy()
if not sig_bars.empty:
    sig_bars['time_hkt'] = sig_bars.index.tz_convert('Asia/Hong_Kong').strftime('%m-%d %H:%M')
    sig_bars['s1'] = sig_bars['s1_prob'].round(3)
    sig_bars['s2'] = sig_bars['s2_prob'].round(3)
    sig_bars['dir'] = 'flat'
    sig_bars.loc[sig_bars['s2_prob'] >= s2_base, 'dir']         = 'LONG'
    sig_bars.loc[sig_bars['s2_prob'] <= (1.0 - s2_base), 'dir'] = 'SHORT'
    print(f"\n{'─'*60}\n  LAST 20 S1 SIGNAL BARS (HKT)\n{'─'*60}")
    print(sig_bars[['time_hkt','close','s1','s2','dir']].to_string(index=False))
