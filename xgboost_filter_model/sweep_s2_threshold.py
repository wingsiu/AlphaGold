#!/usr/bin/env python3
"""
Sweep script for Stage 2 (v13 Directional) threshold.
Sweeps S2_THRESHOLD from 0.55 to 0.58 in 0.01 increments,
keeping S1_THRESHOLD fixed at 0.55.
"""
import sys
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import timedelta

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports from existing backtest and training modules
from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl
from xgboost_filter_model.train_filter_v13_wf_image import prepare_data_v13
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgboost_filter_model.train_directional_model_v9 import add_momentum_features
from config.v13_config import WF_CONFIG, EXECUTION_CONFIG

def run_s2_threshold_sweep(s2_thresholds, s1_fixed=0.55):
    # 1. Config
    FULL_START = WF_CONFIG["full_start"]
    BACKTEST_START = WF_CONFIG["wf_start"]
    BACKTEST_END = WF_CONFIG["wf_end"]
    RETRAIN_DAYS = WF_CONFIG.get("retrain_days", 14)

    print(f"--- Starting v13 Stage 2 Threshold Sweep ({BACKTEST_START} to {BACKTEST_END}) ---")
    print(f"Fixed S1_THRESHOLD: {s1_fixed}")

    # 2. Load and Prepare Full Dataset (only once)
    df = prepare_data_v13(start_date=FULL_START, end_date=BACKTEST_END)
    df = add_directional_features(df)
    df = add_ma_features(df)
    df = add_momentum_features(df)
    df.dropna(inplace=True)

    # 3. Define Features
    exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
               'day_high_rolling', 'day_low_rolling', 'day_open',
               'Dchange_utc2_rel', 'Dupper_wick_utc2_rel', 'Dlower_wick_utc2_rel',
               'trend_label', 'target_v10', 'is_trend', 'atr', 'day_utc2',
               'future_max_move', 'future_min_move', 'future_er', 'atr_threshold',
               'bar_move', 'hour', 'day_id', 'day_high', 'day_low']

    exclude += ['day_high', 'day_low', 'day_open', 'high_90', 'low_90',
                'closePrice_ask', 'closePrice_bid', 'highPrice_ask', 'lowPrice_bid',
                'closePrice', 'lowPrice', 'open_price',
                'highPrice_bid', 'lowPrice_ask', 'openPrice_bid', 'openPrice_ask']

    features = [c for c in df.columns if c not in exclude]

    s1_features = [f for f in features if f not in [
        'directional_change_15', 'directional_change_30', 'directional_change_90',
        'wick_ratio_15', 'wick_ratio_30', 'wick_ratio_90',
        'price_vs_ma_10', 'price_vs_ma_30', 'price_vs_ma_90', 'ma_10_vs_30', 'ma_30_vs_90',
        'rsi_14', 'rsi_30', 'macd', 'macd_signal', 'macd_diff', 'roc_15', 'roc_30', 'roc_60'
    ]]

    # 4. Pre-calculate s1_prob and s2_prob using cycle-specific models (only once)
    df_test = df[df.index >= pd.to_datetime(BACKTEST_START).tz_localize('UTC')].copy()
    df_test['s1_prob'] = np.nan
    df_test['s2_prob'] = np.nan

    current_test_start = pd.to_datetime(BACKTEST_START).tz_localize('UTC')
    end_dt = pd.to_datetime(BACKTEST_END).tz_localize('UTC')
    cycle = 1
    models_dir = PROJECT_ROOT / "runtime" / "bot_assets" / "wf_models_v13"

    print("Pre-calculating signals using Cycle-specific models...")
    prod_s2 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v13_wf.joblib")

    while current_test_start < end_dt:
        current_test_end = current_test_start + timedelta(days=RETRAIN_DAYS)
        s1_model_name = f"filter_v13_cycle_{cycle}_{current_test_start.date()}.joblib"
        s1_model_path = models_dir / s1_model_name

        chunk_mask = (df_test.index >= current_test_start) & (df_test.index < current_test_end)

        if chunk_mask.any():
            if s1_model_path.exists():
                s1_model = joblib.load(s1_model_path)
                df_test.loc[chunk_mask, 's1_prob'] = s1_model.predict_proba(df_test.loc[chunk_mask, s1_features])[:, 1]
            else:
                prod_s1 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v13_wf_image.joblib")
                df_test.loc[chunk_mask, 's1_prob'] = prod_s1.predict_proba(df_test.loc[chunk_mask, s1_features])[:, 1]

            df_test.loc[chunk_mask, 's2_prob'] = prod_s2.predict_proba(df_test.loc[chunk_mask, features])[:, 1]

        current_test_start = current_test_end
        cycle += 1

    # Prepare for trade simulation
    spread_shift = EXECUTION_CONFIG["spread_default"]
    if 'openPrice_ask' not in df_test.columns:
        df_test['open_ask'] = df_test['open'] + spread_shift
        df_test['open_bid'] = df_test['open'] - spread_shift
        df_test['close_ask'] = df_test['close'] + spread_shift
        df_test['close_bid'] = df_test['close'] - spread_shift
        df_test['high_ask'] = df_test['high'] + spread_shift
        df_test['low_bid'] = df_test['low'] - spread_shift
    else:
        df_test['open_ask'] = df_test['openPrice_ask']
        df_test['open_bid'] = df_test['openPrice_bid']
        df_test['close_ask'] = df_test['closePrice_ask']
        df_test['close_bid'] = df_test['closePrice_bid']
        df_test['high_ask'] = df_test['highPrice_ask']
        df_test['low_bid'] = df_test['lowPrice_bid']

    results = []

    for s2_thresh in s2_thresholds:
        print(f"\n--- Testing S2_THRESHOLD = {s2_thresh} (S1={s1_fixed}) ---")

        # Reset side_signal
        df_test['side_signal'] = 0
        trend_mask = df_test['s1_prob'] >= s1_fixed
        df_test.loc[trend_mask & (df_test['s2_prob'] >= s2_thresh), 'side_signal'] = 1
        df_test.loc[trend_mask & (df_test['s2_prob'] <= (1.0 - s2_thresh)), 'side_signal'] = -1

        # Trade Logic (same as backtest_v13.py)
        all_trades = []
        active_pos = None
        target_dist = EXECUTION_CONFIG["tp"]
        stop_dist = EXECUTION_CONFIG["sl"]
        horizon_minutes = EXECUTION_CONFIG["horizon"]

        for i in range(len(df_test) - 1):
            row = df_test.iloc[i]
            next_row = df_test.iloc[i+1]
            now_ts = row.name

            if active_pos is not None:
                side = active_pos['side']
                if side == 1:
                    if row['low_bid'] <= active_pos['stop']:
                        all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': active_pos['stop'], 'exit_reason': 'stop_loss'})
                        active_pos = None
                    elif row['high_ask'] >= active_pos['target']:
                        all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': active_pos['target'], 'exit_reason': 'target_hit'})
                        active_pos = None
                    elif now_ts >= active_pos['timeout']:
                        all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': row['close_bid'], 'exit_reason': 'timeout'})
                        active_pos = None
                else: # side == -1
                    if row['high_ask'] >= active_pos['stop']:
                        all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': active_pos['stop'], 'exit_reason': 'stop_loss'})
                        active_pos = None
                    elif row['low_bid'] <= active_pos['target']:
                        all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': active_pos['target'], 'exit_reason': 'target_hit'})
                        active_pos = None
                    elif now_ts >= active_pos['timeout']:
                        all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': row['close_ask'], 'exit_reason': 'timeout'})
                        active_pos = None

            if active_pos is not None:
                side = active_pos['side']
                current_signal = row['side_signal']
                if current_signal != 0 and current_signal == -side:
                    exit_px = row['close_bid'] if side == 1 else row['close_ask']
                    all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': exit_px, 'exit_reason': 'reverse_signal'})
                    active_pos = None
                elif current_signal == side:
                    active_pos['timeout'] = now_ts + pd.Timedelta(minutes=horizon_minutes)
                    active_pos['target_updates'] += 1
                    new_target = row['close'] + (target_dist if side == 1 else -target_dist)
                    if side == 1:
                        if new_target > active_pos['target']: active_pos['target'] = new_target
                    else:
                        if new_target < active_pos['target']: active_pos['target'] = new_target

            if active_pos is None and row['side_signal'] != 0:
                side = int(row['side_signal'])
                entry_time = next_row.name
                if side == 1:
                    entry_price, stop_px, target_px = next_row['open_ask'], next_row['open_ask'] - stop_dist, next_row['open_ask'] + target_dist
                else:
                    entry_price, stop_px, target_px = next_row['open_bid'], next_row['open_bid'] + stop_dist, next_row['open_bid'] - target_dist
                active_pos = {'side': side, 'entry_time': entry_time, 'entry_price': entry_price, 'stop': stop_px, 'target': target_px,
                              'timeout': entry_time + pd.Timedelta(minutes=horizon_minutes), 's1_prob': row['s1_prob'], 's2_prob': row['s2_prob'], 'target_updates': 0}

        if all_trades:
            tdf = pd.DataFrame(all_trades)
            tdf['side'] = tdf['side'].map({1: 'up', -1: 'down'})
            tdf['pnl'] = (tdf['exit_price'] - tdf['entry_price']) * (tdf['side'].map({'up': 1, 'down': -1}))

            temp_csv = PROJECT_ROOT / "xgboost_filter_model" / f"temp_sweep_s2_{s2_thresh}.csv"
            tdf.to_csv(temp_csv, index=False)
            stats = rebuild_directional_pnl(temp_csv)
            os.remove(temp_csv)

            results.append({
                's2_thresh': s2_thresh,
                'trades': stats['trades'],
                'pnl': stats['total_pnl'],
                'pf': stats['all'].get('profit_factor', 0),
                'win_rate': stats['all'].get('win_rate_pct', 0),
                'max_dd': stats.get('max_drawdown', 0)
            })
            print(f"Result: PnL={stats['total_pnl']:.1f}, MaxDD={stats.get('max_drawdown', 0):.1f}, Trades={stats['trades']}, PF={stats['all'].get('profit_factor', 0):.3f}")
        else:
            print("No trades executed.")

    print("\n" + "="*80)
    print(f"{'S2 Thresh':10s} | {'Trades':6s} | {'PnL':8s} | {'PF':6s} | {'Win%':6s} | {'MaxDD':8s}")
    print("-" * 80)
    for r in results:
        print(f"{r['s2_thresh']:10.2f} | {r['trades']:6d} | {r['pnl']:8.1f} | {r['pf']:6.3f} | {r['win_rate']:6.1f} | {r['max_dd']:8.1f}")
    print("="*80)

if __name__ == "__main__":
    # Sweeping S2 with higher values while S1 is fixed at 0.5
    s2_thresholds = [0.55, 0.57, 0.6, 0.65, 0.7]
    run_s2_threshold_sweep(s2_thresholds, s1_fixed=0.5)

