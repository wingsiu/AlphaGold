#!/usr/bin/env python3
"""
AlphaGold v13 — Unified Backtest
=================================
Usage:
  python3 backtest.py                         # full history (Jan 2025 → today, WF cycle models)
    python3 backtest.py 7                       # last 7 days  (WF cycle models)
    python3 backtest.py 30                      # last 30 days (WF cycle models)
  python3 backtest.py 2025-03-01              # custom start → today (WF cycle models)
  python3 backtest.py 2025-03-01 2025-04-30   # custom range (WF cycle models)
"""
import sys
from pathlib import Path
from datetime import date, timedelta

from V13._paths import PROJECT_ROOT

import pandas as pd
import numpy as np
import joblib

from xgboost_filter_model.train_filter_v13_wf_image import prepare_data_v13
from xgboost_filter_model.train_filter_1min import load_price_data
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgboost_filter_model.train_directional_model_v9 import add_momentum_features
from config.v13_config import EXECUTION_CONFIG, WF_CONFIG
from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl

# ── Parse CLI args ──────────────────────────────────────────────────────────────
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
print(f"  AlphaGold v13 Backtest")
print(f"  Period : {bt_start} → {bt_end}")
print(f"  Models : Walk-Forward cycle")
print(f"{'='*60}\n")

# ── 1. Load & prepare data ──────────────────────────────────────────────────────
warmup_days = int(WF_CONFIG.get("feature_warmup_days", 120))
load_start_dt = pd.to_datetime(bt_start)
full_start_dt = pd.to_datetime(full_start)
warmup_start_dt = load_start_dt - pd.Timedelta(days=warmup_days)
load_start_dt = max(full_start_dt, warmup_start_dt)
load_start = load_start_dt.strftime("%Y-%m-%d")

# data_loader treats end_date as EXCLUSIVE at the table's trading-day boundary
# (HKT 06:00 for IG tables). Passing today_str would cut off the entire current
# trading day. Pass bt_end + 1 day so the bt_end day's bars are INCLUDED.
load_end = (pd.to_datetime(bt_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
print(f"Loading data from {load_start} to {bt_end} (loader end-exclusive {load_end})…")
df = prepare_data_v13(start_date=load_start, end_date=load_end)
df = add_directional_features(df)
df = add_ma_features(df)
df = add_momentum_features(df)
df.dropna(inplace=True)

# ── 2. Feature columns ──────────────────────────────────────────────────────────
exclude = {
    'open','high','low','close','volume','timestamp',
    'day_high_rolling','day_low_rolling','day_open',
    'Dchange_utc2_rel','Dupper_wick_utc2_rel','Dlower_wick_utc2_rel',
    'trend_label','target_v10','is_trend','atr','day_utc2',
    'future_max_move','future_min_move','future_er','atr_threshold',
    'bar_move','hour','day_id','day_high','day_low','high_90','low_90',
    'closePrice_ask','closePrice_bid','highPrice_ask','lowPrice_bid',
    'closePrice','lowPrice','open_price','highPrice_bid','lowPrice_ask',
    'openPrice_bid','openPrice_ask',
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

# ── 3. Slice test window ────────────────────────────────────────────────────────
df_test = df[df.index >= pd.to_datetime(bt_start).tz_localize('UTC')].copy()
print(f"Energetic bars in test window: {len(df_test)}")

if df_test.empty:
    print("No energetic bars in the test window. Check date range and bar_move/volume filters.")
    sys.exit(0)

# ── 4. Score with models ────────────────────────────────────────────────────────
prod_s1 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v13_wf_image.joblib")
prod_s2 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v13_wf.joblib")

df_test['s1_prob'] = np.nan
df_test['s2_prob'] = np.nan

# Walk-forward: use cycle-specific models, fall back to production
wf_dir = PROJECT_ROOT / WF_CONFIG.get("model_output_dir", "runtime/bot_assets/wf_models_v13")
retrain_days = WF_CONFIG.get("retrain_days", 14)
wf_anchor = pd.to_datetime(wf_start)
if wf_anchor.tzinfo is None:
    wf_anchor = wf_anchor.tz_localize('UTC')
else:
    wf_anchor = wf_anchor.tz_convert('UTC')
run_start = pd.to_datetime(bt_start).tz_localize('UTC')
# Use end-of-day (exclusive) so bt_end day's bars are INCLUDED in the loop.
# `pd.to_datetime("2026-05-15")` is 2026-05-15 00:00 UTC; without +1 day the
# loop's `current_start < end_dt` cut would drop the entire bt_end trading day.
end_dt        = pd.to_datetime(bt_end).tz_localize('UTC') + pd.Timedelta(days=1)

# Anchor cycle numbering to wf_start so custom windows map to the correct saved WF cycle models.
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
    s1_path = wf_dir / f"filter_v13_cycle_{cycle}_{current_start.date()}.joblib"
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
            model_path = str(PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v13_wf_image.joblib")
            missing_models.append(str(s1_path.name))
            print(f"[INFO] Using PROD model for {current_start.date()} (cycle {cycle}) — missing: {s1_path.name}")

        # Score S1
        df_test.loc[chunk, 's1_prob'] = s1.predict_proba(df_test.loc[chunk, s1_features])[:, 1]
        df_test.loc[chunk, 'cycle_id'] = model_id
        df_test.loc[chunk, 'model_path'] = model_path

        # Score S2 with cycle-specific directional model if available
        s2_path = wf_dir / f"directional_v13_cycle_{cycle}_{current_start.date()}.joblib"
        if s2_path.exists():
            s2 = joblib.load(s2_path)
        else:
            s2 = prod_s2
        s1_pass = chunk & (df_test['s1_prob'] >= EXECUTION_CONFIG["s1_threshold"])
        if s1_pass.any():
            df_test.loc[s1_pass, 's2_prob'] = s2.predict_proba(df_test.loc[s1_pass, features])[:, 1]
    current_start = current_end
    cycle += 1
print(f"WF chunks={wf_chunks}  fallback(prod) chunks={prod_chunks}")
if prod_chunks > 0:
    print("\n[SUMMARY] The following cycle models were missing and replaced with PROD model:")
    for m in missing_models:
        print(f"  - {m}")
if present_models:
    print("\n[SUMMARY] The following cycle models were found and used:")
    for m in present_models:
        print(f"  - {m}")

# ── 5. Pre-compute side_signal with BASE threshold (roll/extend logic uses this) ─
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

# ── 6. Bid/ask prices ──────────────────────────────────────────────────────────
spread = EXECUTION_CONFIG["spread_default"]
if 'openPrice_ask' in df_test.columns:
    df_test['open_ask']  = df_test['openPrice_ask']
    df_test['open_bid']  = df_test['openPrice_bid']
    df_test['close_ask'] = df_test['closePrice_ask']
    df_test['close_bid'] = df_test['closePrice_bid']
    df_test['high_ask']  = df_test['highPrice_ask']
    df_test['low_bid']   = df_test['lowPrice_bid']
else:
    df_test['open_ask']  = df_test['open']  + spread
    df_test['open_bid']  = df_test['open']  - spread
    df_test['close_ask'] = df_test['close'] + spread
    df_test['close_bid'] = df_test['close'] - spread
    df_test['high_ask']  = df_test['high']  + spread
    df_test['low_bid']   = df_test['low']   - spread

# ── 7. Trade simulation ────────────────────────────────────────────────────────
tp      = EXECUTION_CONFIG["tp"]
sl      = EXECUTION_CONFIG["sl"]
horizon = EXECUTION_CONFIG["horizon"]

# Build a full-resolution 1-min simulation frame so exits (SL/TP/timeout) fire
# even on bars that didn't pass the energetic-bar filter. Entries/reverses/rolls
# remain gated on energetic bars (where side_signal is defined).
print("Loading full 1-min bars for exit simulation…")
raw_df = load_price_data(start_date=bt_start, end_date=load_end)
raw_df = raw_df[raw_df.index >= pd.to_datetime(bt_start).tz_localize('UTC')].copy()

# Ask/bid columns on raw 1-min frame
if 'openPrice_ask' in raw_df.columns:
    raw_df['open_ask']  = raw_df['openPrice_ask']
    raw_df['open_bid']  = raw_df['openPrice_bid']
    raw_df['close_ask'] = raw_df['closePrice_ask']
    raw_df['close_bid'] = raw_df['closePrice_bid']
    raw_df['high_ask']  = raw_df['highPrice_ask']
    raw_df['low_bid']   = raw_df['lowPrice_bid']
else:
    raw_df['open_ask']  = raw_df['open']  + spread
    raw_df['open_bid']  = raw_df['open']  - spread
    raw_df['close_ask'] = raw_df['close'] + spread
    raw_df['close_bid'] = raw_df['close'] - spread
    raw_df['high_ask']  = raw_df['high']  + spread
    raw_df['low_bid']   = raw_df['low']   - spread

# Merge signal/probs/cycle from energetic df_test onto full 1-min index
sim_df = raw_df[['open','high','low','close',
                 'open_ask','open_bid','close_ask','close_bid',
                 'high_ask','low_bid']].copy()
for col in ['side_signal','s1_prob','s2_prob','cycle_id','model_path']:
    sim_df[col] = df_test[col] if col in df_test.columns else np.nan
sim_df['side_signal'] = sim_df['side_signal'].fillna(0).astype(int)

all_trades, active_pos = [], None
consecutive_losses = 0

print(f"Simulating trades on {len(sim_df)} 1-min bars (energetic signal bars: {len(df_test)})…")
for i in range(len(sim_df) - 1):
    row      = sim_df.iloc[i]
    next_row = sim_df.iloc[i + 1]
    now_ts   = row.name
    sig      = int(row['side_signal'])  # 0 on non-energetic bars

    # --- 1. Exit (checked EVERY 1-min bar) ---
    if active_pos:
        s = active_pos['side']
        exit_info = None
        if s == 1:
            if   row['low_bid']  <= active_pos['stop']:   exit_info = (active_pos['stop'],   'stop_loss')
            elif row['high_ask'] >= active_pos['target']: exit_info = (active_pos['target'], 'target_hit')
            elif now_ts          >= active_pos['timeout']: exit_info = (row['close_bid'],     'timeout')
        else:
            if   row['high_ask'] >= active_pos['stop']:   exit_info = (active_pos['stop'],   'stop_loss')
            elif row['low_bid']  <= active_pos['target']: exit_info = (active_pos['target'], 'target_hit')
            elif now_ts          >= active_pos['timeout']: exit_info = (row['close_ask'],     'timeout')
        if exit_info:
            px, reason = exit_info
            pnl = (px - active_pos['entry_price']) * s
            all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': px,
                                'exit_reason': reason, 'pnl': pnl,
                                'cycle_id': active_pos.get('cycle_id'),
                                'model_path': active_pos.get('model_path')})
            consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
            active_pos = None

    # --- 2. Reverse / Roll (only on energetic signal bars) ---
    if active_pos and sig != 0:
        s = active_pos['side']
        if sig == -s:
            px  = row['close_bid'] if s == 1 else row['close_ask']
            pnl = (px - active_pos['entry_price']) * s
            all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': px,
                                'exit_reason': 'reverse_signal', 'pnl': pnl,
                                'cycle_id': active_pos.get('cycle_id'),
                                'model_path': active_pos.get('model_path')})
            consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
            active_pos = None
        elif sig == s:
            active_pos['timeout'] = now_ts + pd.Timedelta(minutes=horizon)
            active_pos['target_updates'] += 1
            new_t = row['close'] + (tp if s == 1 else -tp)
            if (s == 1 and new_t > active_pos['target']) or (s == -1 and new_t < active_pos['target']):
                active_pos['target'] = new_t

    # --- 3. Entry (dynamic S2 gates new entries only; only on energetic bars) ---
    if active_pos is None and sig != 0:
        dynamic_s2 = min(s2_max, s2_base + consecutive_losses * s2_increment)
        s2_p = row['s2_prob']
        passes = (sig == 1 and s2_p >= dynamic_s2) or (sig == -1 and s2_p <= (1.0 - dynamic_s2))
        if passes:
            ep = next_row['open_ask'] if sig == 1 else next_row['open_bid']
            active_pos = {
                'side': sig,
                'entry_time': next_row.name,
                'entry_price': ep,
                'stop':    ep - sl if sig == 1 else ep + sl,
                'target':  ep + tp if sig == 1 else ep - tp,
                'timeout': next_row.name + pd.Timedelta(minutes=horizon),
                'target_updates': 0,
                's1_prob': row['s1_prob'],
                's2_prob': s2_p,
                'cycle_id': row['cycle_id'] if 'cycle_id' in row else None,
                'model_path': row['model_path'] if 'model_path' in row else None,
            }

# ── 8. Results ─────────────────────────────────────────────────────────────────
if not all_trades:
    print("\nSignals found but no trades closed in the window.")
    sys.exit(0)

tdf = pd.DataFrame(all_trades)
tdf['side_str'] = tdf['side'].map({1: 'up', -1: 'down'})

# Save trades
if days_back:
    out_path = PROJECT_ROOT / "runtime" / "recent_backtest_trades.csv"
else:
    out_path = PROJECT_ROOT / "xgboost_filter_model" / "v13_backtest_trades.csv"
tdf.drop(columns=['side_str']).rename(columns={'side': 'side_int'}).assign(
    side=tdf['side_str']
).drop(columns=['side_int']).to_csv(out_path, index=False)

# ── Expanded stats (includes suite-v13 blocks + additional analytics) ──
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

tdf_long  = tdf[tdf['side'] ==  1]
tdf_short = tdf[tdf['side'] == -1]
print(f"\n  LONG : {len(tdf_long):4d} trades  PnL={tdf_long['pnl'].sum():.1f}  "
      f"WR={(tdf_long['pnl']>0).mean()*100:.1f}%  avg={tdf_long['pnl'].mean():.2f}")
print(f"  SHORT: {len(tdf_short):4d} trades  PnL={tdf_short['pnl'].sum():.1f}  "
      f"WR={(tdf_short['pnl']>0).mean()*100:.1f}%  avg={tdf_short['pnl'].mean():.2f}")

try:
    stats = rebuild_directional_pnl(out_path)
except Exception as exc:
    stats = None
    print(f"\n  Warning: extended stats unavailable ({exc})")

print(f"\n  Exit Breakdown:")
for reason, grp in tdf.groupby('exit_reason'):
    wr_r = (grp['pnl'] > 0).mean() * 100
    print(f"    {reason:18s}: {len(grp):4d}  WR={wr_r:5.1f}%  avg={grp['pnl'].mean():7.2f}")

# --- Retrain cycle stats block (group by cycle_id) ---
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

    # Additional metrics beyond suite_v13: expectancy and risk-adjusted daily stats
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

    # Monthly statistics
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


    # Yearly statistics
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

    # Retrain cycle statistics (cycle stat)
    from config.v13_config import WF_CONFIG
    retrain_days = WF_CONFIG["retrain_days"]
    wf_start = pd.to_datetime(WF_CONFIG["wf_start"])
    wf_end = pd.to_datetime(WF_CONFIG["wf_end"])
    tdf['entry_time'] = pd.to_datetime(tdf['entry_time'], utc=True)
    # Compute cycle boundaries
    # Make cycle_starts tz-aware (UTC)
    wf_start_utc = wf_start.tz_localize('UTC') if wf_start.tzinfo is None else wf_start.tz_convert('UTC')
    wf_end_utc = wf_end.tz_localize('UTC') if wf_end.tzinfo is None else wf_end.tz_convert('UTC')
    cycle_starts = [wf_start_utc]
    while cycle_starts[-1] < wf_end_utc:
        cycle_starts.append(cycle_starts[-1] + pd.Timedelta(days=retrain_days))
    cycle_labels = []
    for i in range(len(cycle_starts)-1):
        s = cycle_starts[i]
        e = cycle_starts[i+1]
        cycle_labels.append(f"{s.date()} to {e.date()}")
    def assign_cycle(ts):
        ts_utc = ts.tz_convert('UTC') if ts.tzinfo is not None else ts.tz_localize('UTC')
        for i in range(len(cycle_starts)-1):
            if cycle_starts[i] <= ts_utc < cycle_starts[i+1]:
                return cycle_labels[i]
        return cycle_labels[-1]
    tdf['retrain_cycle'] = tdf['entry_time'].apply(assign_cycle)
    cycle_stats = tdf.groupby('retrain_cycle')['pnl'].agg(
        trades='size',
        total_pnl='sum',
        avg_trade='mean',
        win_rate=lambda s: (s > 0).mean() * 100.0,
    ).reset_index()
    cycle_stats['win_rate'] = cycle_stats['win_rate'].map(lambda v: f"{v:.1f}%")
    print(f"\n{'='*60}")
    print("  RETRAIN CYCLE STATISTICS")
    print(f"{'='*60}")
    print(cycle_stats.to_string(index=False))

    # Weekday (UTC+2) — aligns with trading-day cutoff used elsewhere in v13
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

    # Session summaries + heatmaps (ported from suite_v13)
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
else:
    gross_win  = tdf.loc[tdf['pnl'] > 0, 'pnl'].sum()
    gross_loss = tdf.loc[tdf['pnl'] <= 0, 'pnl'].abs().sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    equity = tdf['pnl'].cumsum()
    max_dd = (equity - equity.cummax()).min()
    print(f"\n  Profit Factor   : {pf:.3f}")
    print(f"  Max Drawdown    : {max_dd:.1f}")

# Trade table
tdf_view = tdf.copy()
tdf_view['entry_hkt'] = pd.to_datetime(tdf_view['entry_time'], utc=True).dt.tz_convert('Asia/Hong_Kong').dt.strftime('%m-%d %H:%M')
tdf_view['exit_hkt']  = pd.to_datetime(tdf_view['exit_time'],  utc=True).dt.tz_convert('Asia/Hong_Kong').dt.strftime('%H:%M')
tdf_view['s1']  = tdf_view['s1_prob'].round(3)
tdf_view['s2']  = tdf_view['s2_prob'].round(3)
tdf_view['dir'] = tdf_view['side'].map({1: 'up', -1: 'down'})

show = tdf_view if days_back else tdf_view.tail(30)
header = "ALL TRADES (HKT)" if days_back else "LAST 30 TRADES (HKT)"
print(f"\n{'─'*60}\n  {header}\n{'─'*60}")
print(show[['entry_hkt','exit_hkt','dir','entry_price','exit_price','pnl','exit_reason','s1','s2']].to_string(index=False))

# Recent S1 signals
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

print(f"\n  Trades saved → {out_path}")


# ── Reusable simulation core ────────────────────────────────────────────────────
# Moved to `backtest_core.py` so importers (daily_reconciliation, sweeps) don't
# trigger the full backtest script body (which parses sys.argv at module load).
from v14.backtest.backtest_core import simulate_v13_core  # noqa: E402,F401


