#!/usr/bin/env python3
"""
Backtest for Dynamic Stage 2 Threshold:
S2_THRESHOLD = BASE_S2 + (consecutive_losses * 0.01)
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

from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl
from xgboost_filter_model.train_filter_v13_wf_image import prepare_data_v13
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgboost_filter_model.train_directional_model_v9 import add_momentum_features
from config.v13_config import WF_CONFIG, EXECUTION_CONFIG

def run_dynamic_s2_backtest(base_s2=0.5, step=0.01, max_s2=0.7):
    # 1. Setup
    FULL_START = WF_CONFIG["full_start"]
    BACKTEST_START = WF_CONFIG["wf_start"]
    BACKTEST_END = WF_CONFIG["wf_end"]
    RETRAIN_DAYS = WF_CONFIG.get("retrain_days", 14)

    print(f"--- Starting Dynamic S2 Backtest (S2 = {base_s2} + losses * {step}) ---")

    # 2. Data
    df = prepare_data_v13(start_date=FULL_START, end_date=BACKTEST_END)
    df = add_directional_features(df)
    df = add_ma_features(df)
    df = add_momentum_features(df)
    df.dropna(inplace=True)

    # 3. Features
    exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'day_high_rolling', 'day_low_rolling', 'day_open',
               'Dchange_utc2_rel', 'Dupper_wick_utc2_rel', 'Dlower_wick_utc2_rel', 'trend_label', 'target_v10', 'is_trend',
               'atr', 'day_utc2', 'future_max_move', 'future_min_move', 'future_er', 'atr_threshold', 'bar_move', 'hour',
               'day_id', 'day_high', 'day_low', 'high_90', 'low_90', 'closePrice_ask', 'closePrice_bid', 'highPrice_ask',
               'lowPrice_bid', 'closePrice', 'lowPrice', 'open_price', 'highPrice_bid', 'lowPrice_ask', 'openPrice_bid', 'openPrice_ask']
    features = [c for c in df.columns if c not in exclude]
    s1_features = [f for f in features if f not in [
        'directional_change_15', 'directional_change_30', 'directional_change_90',
        'wick_ratio_15', 'wick_ratio_30', 'wick_ratio_90', 'price_vs_ma_10', 'price_vs_ma_30', 'price_vs_ma_90',
        'ma_10_vs_30', 'ma_30_vs_90', 'rsi_14', 'rsi_30', 'macd', 'macd_signal', 'macd_diff', 'roc_15', 'roc_30', 'roc_60'
    ]]

    # 4. Probabilities
    df_test = df[df.index >= pd.to_datetime(BACKTEST_START).tz_localize('UTC')].copy()
    df_test['s1_prob'] = np.nan
    df_test['s2_prob'] = np.nan

    current_test_start = pd.to_datetime(BACKTEST_START).tz_localize('UTC')
    end_dt = pd.to_datetime(BACKTEST_END).tz_localize('UTC')
    cycle = 1
    models_dir = PROJECT_ROOT / "runtime" / "bot_assets" / "wf_models_v13"
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

    # Bid/Ask
    spread_shift = EXECUTION_CONFIG["spread_default"]
    df_test['open_ask'] = df_test['open'] + spread_shift
    df_test['open_bid'] = df_test['open'] - spread_shift
    df_test['close_ask'] = df_test['close'] + spread_shift
    df_test['close_bid'] = df_test['close'] - spread_shift
    df_test['high_ask'] = df_test['high'] + spread_shift
    df_test['low_bid'] = df_test['low'] - spread_shift

    # 5. Simulation Loop with Dynamic S2
    all_trades = []
    active_pos = None
    consecutive_losses = 0
    s1_thresh = 0.5  # Fixed S1 as requested
    target_dist = EXECUTION_CONFIG["tp"]
    stop_dist = EXECUTION_CONFIG["sl"]
    horizon_minutes = EXECUTION_CONFIG["horizon"]

    for i in range(len(df_test) - 1):
        row = df_test.iloc[i]
        next_row = df_test.iloc[i+1]
        now_ts = row.name

        # Calculate dynamic S2
        current_s2_thresh = min(max_s2, base_s2 + (consecutive_losses * step))

        # Signal Generation (at current thresh)
        side_signal = 0
        if row['s1_prob'] >= s1_thresh:
            if row['s2_prob'] >= current_s2_thresh:
                side_signal = 1
            elif row['s2_prob'] <= (1.0 - current_s2_thresh):
                side_signal = -1

        # 1. Exit Logic
        if active_pos is not None:
            side = active_pos['side']
            exit_info = None
            if side == 1:
                if row['low_bid'] <= active_pos['stop']:
                    exit_info = {'p': active_pos['stop'], 'r': 'stop_loss'}
                elif row['high_ask'] >= active_pos['target']:
                    exit_info = {'p': active_pos['target'], 'r': 'target_hit'}
                elif now_ts >= active_pos['timeout']:
                    exit_info = {'p': row['close_bid'], 'r': 'timeout'}
            else:
                if row['high_ask'] >= active_pos['stop']:
                    exit_info = {'p': active_pos['stop'], 'r': 'stop_loss'}
                elif row['low_bid'] <= active_pos['target']:
                    exit_info = {'p': active_pos['target'], 'r': 'target_hit'}
                elif now_ts >= active_pos['timeout']:
                    exit_info = {'p': row['close_ask'], 'r': 'timeout'}

            if exit_info:
                pnl = (exit_info['p'] - active_pos['entry_price']) * side
                all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': exit_info['p'], 'exit_reason': exit_info['r'], 'pnl': pnl, 's2_thresh_used': current_s2_thresh})
                active_pos = None
                # Update loss tracker
                if pnl <= 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

        # 2. Reverse Signal
        if active_pos is not None:
            side = active_pos['side']
            if side_signal != 0 and side_signal == -side:
                exit_px = row['close_bid'] if side == 1 else row['close_ask']
                pnl = (exit_px - active_pos['entry_price']) * side
                all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': exit_px, 'exit_reason': 'reverse_signal', 'pnl': pnl, 's2_thresh_used': current_s2_thresh})
                active_pos = None
                if pnl <= 0: consecutive_losses += 1
                else: consecutive_losses = 0
            elif side_signal == side:
                active_pos['timeout'] = now_ts + pd.Timedelta(minutes=horizon_minutes)
                active_pos['target_updates'] += 1
                new_target = row['close'] + (target_dist if side == 1 else -target_dist)
                if side == 1:
                    if new_target > active_pos['target']: active_pos['target'] = new_target
                else:
                    if new_target < active_pos['target']: active_pos['target'] = new_target

        # 3. Entry
        if active_pos is None and side_signal != 0:
            side = int(side_signal)
            entry_time = next_row.name
            if side == 1:
                ep, sp, tp = next_row['open_ask'], next_row['open_ask'] - stop_dist, next_row['open_ask'] + target_dist
            else:
                ep, sp, tp = next_row['open_bid'], next_row['open_bid'] + stop_dist, next_row['open_bid'] - target_dist
            active_pos = {'side': side, 'entry_time': entry_time, 'entry_price': ep, 'stop': sp, 'target': tp,
                          'timeout': entry_time + pd.Timedelta(minutes=horizon_minutes), 's1_prob': row['s1_prob'], 's2_prob': row['s2_prob'], 'target_updates': 0}

    # Result Stats
    if all_trades:
        tdf = pd.DataFrame(all_trades)
        tdf['side'] = tdf['side'].map({1: 'up', -1: 'down'})
        temp_csv = PROJECT_ROOT / "xgboost_filter_model" / "temp_dynamic_s2.csv"
        tdf.to_csv(temp_csv, index=False)
        stats = rebuild_directional_pnl(temp_csv)
        os.remove(temp_csv)
        print(f"\nDYNAMIC S2 RESULTS:")
        print(f"Total PnL:     {stats['total_pnl']:.1f}")
        print(f"Max Drawdown:  {stats.get('max_drawdown', 0):.1f}")
        print(f"Trades:        {stats['trades']}")
        print(f"Profit Factor: {stats['all'].get('profit_factor', 0):.3f}")
        print(f"Win Rate:      {stats['all'].get('win_rate_pct', 0):.1f}%")
        print(f"Avg S2 Thresh: {tdf['s2_thresh_used'].mean():.3f}")
    else:
        print("No trades.")

if __name__ == "__main__":
    for c in [0.01, 0.015, 0.02]:
        print(f"\n>>> TESTING c = {c}")
        run_dynamic_s2_backtest(base_s2=0.55, step=c)

