#!/usr/bin/env python3
"""
AlphaGold v13 — Unified Backtest
=================================
Usage:
  python3 backtest.py                         # full history (Jan 2025 → today, WF cycle models)
  python3 backtest.py 7                       # last 7 days  (production models, fast)
  python3 backtest.py 30                      # last 30 days (production models)
  python3 backtest.py 2025-03-01              # custom start → today (WF cycle models)
  python3 backtest.py 2025-03-01 2025-04-30   # custom range (WF cycle models)
"""
import sys
from pathlib import Path
from datetime import date, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib

from xgboost_filter_model.train_filter_v13_wf_image import prepare_data_v13
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgboost_filter_model.train_directional_model_v9 import add_momentum_features
from config.v13_config import EXECUTION_CONFIG, WF_CONFIG
from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl

# ── Parse CLI args ──────────────────────────────────────────────────────────────
args = sys.argv[1:]
today_str      = date.today().strftime("%Y-%m-%d")
tomorrow_str   = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
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

use_wf_models = (days_back is None)   # quick N-day runs use production models (faster)

print(f"\n{'='*60}")
print(f"  AlphaGold v13 Backtest")
print(f"  Period : {bt_start} → {bt_end}")
print(f"  Models : {'Walk-Forward cycle' if use_wf_models else 'Production (fast)'}")
print(f"{'='*60}\n")

# ── 1. Load & prepare data ──────────────────────────────────────────────────────
print(f"Loading data from {full_start} to {bt_end}…")
df = prepare_data_v13(start_date=full_start, end_date=bt_end if not days_back else tomorrow_str)
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

if use_wf_models:
    # Walk-forward: use cycle-specific models, fall back to production
    wf_dir = PROJECT_ROOT / "runtime" / "bot_assets" / "wf_models_v13"
    retrain_days = WF_CONFIG.get("retrain_days", 14)
    current_start = pd.to_datetime(bt_start).tz_localize('UTC')
    end_dt        = pd.to_datetime(bt_end).tz_localize('UTC')
    cycle = 1
    print("Scoring with walk-forward cycle models…")
    while current_start < end_dt:
        current_end  = current_start + pd.Timedelta(days=retrain_days)
        s1_path = wf_dir / f"filter_v13_cycle_{cycle}_{current_start.date()}.joblib"
        chunk   = (df_test.index >= current_start) & (df_test.index < current_end)
        if chunk.any():
            s1 = joblib.load(s1_path) if s1_path.exists() else prod_s1
            df_test.loc[chunk, 's1_prob'] = s1.predict_proba(df_test.loc[chunk, s1_features])[:, 1]
            s1_pass = chunk & (df_test['s1_prob'] >= EXECUTION_CONFIG["s1_threshold"])
            if s1_pass.any():
                df_test.loc[s1_pass, 's2_prob'] = prod_s2.predict_proba(df_test.loc[s1_pass, features])[:, 1]
        current_start = current_end
        cycle += 1
else:
    # Production models only (fast path for recent N-day runs)
    print("Scoring with production models…")
    df_test['s1_prob'] = prod_s1.predict_proba(df_test[s1_features])[:, 1]
    s1_pass = df_test['s1_prob'] >= EXECUTION_CONFIG["s1_threshold"]
    if s1_pass.any():
        df_test.loc[s1_pass, 's2_prob'] = prod_s2.predict_proba(df_test.loc[s1_pass, features])[:, 1]

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

all_trades, active_pos = [], None
consecutive_losses = 0

print(f"Simulating trades on {len(df_test)} bars…")
for i in range(len(df_test) - 1):
    row      = df_test.iloc[i]
    next_row = df_test.iloc[i + 1]
    now_ts   = row.name
    sig      = int(row['side_signal'])

    # --- 1. Exit ---
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
                                'exit_reason': reason, 'pnl': pnl})
            consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
            active_pos = None

    # --- 2. Reverse / Roll (reads base-threshold side_signal — always consistent) ---
    if active_pos:
        s = active_pos['side']
        if sig != 0 and sig == -s:
            px  = row['close_bid'] if s == 1 else row['close_ask']
            pnl = (px - active_pos['entry_price']) * s
            all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': px,
                                'exit_reason': 'reverse_signal', 'pnl': pnl})
            consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
            active_pos = None
        elif sig == s:
            active_pos['timeout'] = now_ts + pd.Timedelta(minutes=horizon)
            active_pos['target_updates'] += 1
            new_t = row['close'] + (tp if s == 1 else -tp)
            if (s == 1 and new_t > active_pos['target']) or (s == -1 and new_t < active_pos['target']):
                active_pos['target'] = new_t

    # --- 3. Entry (dynamic S2 gates new entries only) ---
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

# ── Summary stats ──
wins    = (tdf['pnl'] > 0).sum()
losses  = (tdf['pnl'] <= 0).sum()
net_pnl = tdf['pnl'].sum()
wr      = wins / len(tdf) * 100

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

print(f"\n  Exit Breakdown:")
for reason, grp in tdf.groupby('exit_reason'):
    wr_r = (grp['pnl'] > 0).mean() * 100
    print(f"    {reason:18s}: {len(grp):4d}  WR={wr_r:5.1f}%  avg={grp['pnl'].mean():7.2f}")

# Profit factor
gross_win  = tdf.loc[tdf['pnl'] > 0, 'pnl'].sum()
gross_loss = tdf.loc[tdf['pnl'] <= 0, 'pnl'].abs().sum()
pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

equity = tdf['pnl'].cumsum()
max_dd = (equity - equity.cummax()).min()

print(f"\n  Profit Factor  : {pf:.3f}")
print(f"  Max Drawdown   : {max_dd:.1f}")

# Full stats (only for full-history runs)
if use_wf_models:
    try:
        stats = rebuild_directional_pnl(out_path)
        print(f"\n  Avg Day PnL    : {stats.get('avg_day', 0):.1f}")
        print(f"  Positive Days  : {stats.get('positive_days_pct', 0):.1f}%")
        print(f"  Trades/Day     : {stats.get('avg_trades_per_day', 0):.1f}")
        print(f"  Avg Duration   : {stats['all'].get('avg_duration_min', 0):.1f} min")
        st = stats['streaks']
        print(f"  Max Win Streak : {st['max_win_streak']}")
        print(f"  Max Loss Streak: {st['max_loss_streak']}")

        # Monthly breakdown
        mdf = tdf.copy()
        mdf['entry_time'] = pd.to_datetime(mdf['entry_time'], utc=True)
        mdf['month'] = mdf['entry_time'].dt.tz_convert('UTC').dt.tz_localize(None).dt.to_period('M').astype(str)
        monthly = mdf.groupby('month')['pnl'].agg(
            trades='size', total_pnl='sum',
            win_rate=lambda s: f"{(s>0).mean()*100:.1f}%"
        ).reset_index()
        print(f"\n  Monthly PnL:\n{monthly.to_string(index=False)}")
    except Exception:
        pass

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


# ── Reusable simulation core (used by parameter_sweep_tp_sl and daily_reconciliation) ──
def simulate_v13_core(df_test, tp, sl, horizon_minutes):
    """
    Extracted trade simulation loop for parameter sweeping.
    Expects df_test to already have side_signal, s1_prob, s2_prob columns.
    Uses real bid/ask columns when available, else synthetic spread.
    """
    _spread = EXECUTION_CONFIG["spread_default"]
    if 'open_ask' not in df_test.columns:
        if 'openPrice_ask' in df_test.columns:
            df_test = df_test.copy()
            df_test['open_ask']  = df_test['openPrice_ask']
            df_test['open_bid']  = df_test['openPrice_bid']
            df_test['close_ask'] = df_test['closePrice_ask']
            df_test['close_bid'] = df_test['closePrice_bid']
            df_test['high_ask']  = df_test['highPrice_ask']
            df_test['low_bid']   = df_test['lowPrice_bid']
        else:
            df_test = df_test.copy()
            df_test['open_ask']  = df_test['open']  + _spread
            df_test['open_bid']  = df_test['open']  - _spread
            df_test['close_ask'] = df_test['close'] + _spread
            df_test['close_bid'] = df_test['close'] - _spread
            df_test['high_ask']  = df_test['high']  + _spread
            df_test['low_bid']   = df_test['low']   - _spread

    all_trades, active_pos = [], None
    for i in range(len(df_test) - 1):
        row      = df_test.iloc[i]
        next_row = df_test.iloc[i + 1]
        now_ts   = row.name
        sig      = int(row['side_signal'])

        if active_pos:
            s = active_pos['side']
            exit_info = None
            if s == 1:
                if   row['low_bid']  <= active_pos['stop']:    exit_info = (active_pos['stop'],   'stop_loss')
                elif row['high_ask'] >= active_pos['target']:  exit_info = (active_pos['target'], 'target_hit')
                elif now_ts          >= active_pos['timeout']: exit_info = (row['close_bid'],     'timeout')
            else:
                if   row['high_ask'] >= active_pos['stop']:    exit_info = (active_pos['stop'],   'stop_loss')
                elif row['low_bid']  <= active_pos['target']:  exit_info = (active_pos['target'], 'target_hit')
                elif now_ts          >= active_pos['timeout']: exit_info = (row['close_ask'],     'timeout')
            if exit_info:
                px, reason = exit_info
                pnl = (px - active_pos['entry_price']) * s
                all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': px,
                                    'exit_reason': reason, 'pnl': pnl})
                active_pos = None

        if active_pos:
            s = active_pos['side']
            if sig != 0 and sig == -s:
                px  = row['close_bid'] if s == 1 else row['close_ask']
                pnl = (px - active_pos['entry_price']) * s
                all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': px,
                                    'exit_reason': 'reverse_signal', 'pnl': pnl})
                active_pos = None
            elif sig == s:
                active_pos['timeout'] = now_ts + pd.Timedelta(minutes=horizon_minutes)
                active_pos['target_updates'] += 1
                new_t = row['close'] + (tp if s == 1 else -tp)
                if (s == 1 and new_t > active_pos['target']) or (s == -1 and new_t < active_pos['target']):
                    active_pos['target'] = new_t

        if active_pos is None and sig != 0:
            ep = next_row['open_ask'] if sig == 1 else next_row['open_bid']
            active_pos = {
                'side': sig,
                'entry_time': next_row.name,
                'entry_price': ep,
                'stop':    ep - sl  if sig == 1 else ep + sl,
                'target':  ep + tp  if sig == 1 else ep - tp,
                'timeout': next_row.name + pd.Timedelta(minutes=horizon_minutes),
                'target_updates': 0,
                's1_prob': row['s1_prob'],
                's2_prob': row['s2_prob'],
            }

    return all_trades


