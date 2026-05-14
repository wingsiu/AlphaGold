#!/usr/bin/env python3
"""
Parameter sweep for Target (tp) and Stop (sl) using v13 model signals.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import simulation core from unified backtest
from backtest import simulate_v13_core
from xgboost_filter_model.train_filter_v13_wf_image import prepare_data_v13
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgboost_filter_model.train_directional_model_v9 import add_momentum_features
from config.v13_config import WF_CONFIG, EXECUTION_CONFIG

def run_tp_sl_sweep():
    # 1. Config
    FULL_START = WF_CONFIG["full_start"]
    BACKTEST_START = WF_CONFIG["wf_start"]
    BACKTEST_END = WF_CONFIG["wf_end"]
    RETRAIN_DAYS = WF_CONFIG.get("retrain_days", 14)

    print(f"--- Preparing Data for Sweep ({BACKTEST_START} to {BACKTEST_END}) ---")

    # 2. Load and Prepare Full Dataset (Cached for all iterations)
    df = prepare_data_v13(start_date=FULL_START, end_date=BACKTEST_END)
    df = add_directional_features(df)
    df = add_ma_features(df)
    df = add_momentum_features(df)
    df.dropna(inplace=True)

    # 3. Define Features (consistent with backtest_v13)
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

    # 4. Generate Signals (Once)
    df_test = df[df.index >= pd.to_datetime(BACKTEST_START).tz_localize('UTC')].copy()
    df_test['s1_prob'] = np.nan
    df_test['s2_prob'] = np.nan

    current_test_start = pd.to_datetime(BACKTEST_START).tz_localize('UTC')
    end_dt = pd.to_datetime(BACKTEST_END).tz_localize('UTC')
    cycle = 1
    models_dir = PROJECT_ROOT / "runtime" / "bot_assets" / "wf_models_v13"

    print("Generating Stage 1 & Stage 2 signals...")
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

            s1_trend_mask = chunk_mask & (df_test['s1_prob'] >= EXECUTION_CONFIG["s1_threshold"])
            if s1_trend_mask.any():
                df_test.loc[s1_trend_mask, 's2_prob'] = prod_s2.predict_proba(df_test.loc[s1_trend_mask, features])[:, 1]

        current_test_start = current_test_end
        cycle += 1

    # side: 1 for Up, -1 for Down
    df_test['side_signal'] = 0
    trend_mask = df_test['s1_prob'] >= EXECUTION_CONFIG["s1_threshold"]
    s2_thresh = EXECUTION_CONFIG["s2_threshold"]
    df_test.loc[trend_mask & (df_test['s2_prob'] >= s2_thresh), 'side_signal'] = 1
    df_test.loc[trend_mask & (df_test['s2_prob'] <= (1.0 - s2_thresh)), 'side_signal'] = -1

    # Pre-calculate bid/ask in df_test
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

    # 5. Define Sweep Parameters
    tp_range = np.arange(10, 41, 5)  # 10, 15, ..., 40
    sl_range = np.arange(5, 26, 5)   # 5, 10, ..., 25
    horizon = EXECUTION_CONFIG["horizon"]

    results = []

    print(f"Starting sweep over {len(tp_range) * len(sl_range)} combinations...")

    for tp in tqdm(tp_range):
        for sl in sl_range:
            trades = simulate_v13_core(df_test, tp, sl, horizon)
            if not trades:
                results.append({'tp': tp, 'sl': sl, 'trades': 0, 'total_pnl': 0, 'win_rate': 0, 'profit_factor': 0})
                continue

            tdf = pd.DataFrame(trades)
            tdf['side_val'] = tdf['side'].map({1: 1, -1: -1})
            tdf['pnl'] = (tdf['exit_price'] - tdf['entry_price']) * tdf['side_val']

            total_pnl = tdf['pnl'].sum()
            win_rate = (tdf['pnl'] > 0).mean() * 100

            wins = tdf[tdf['pnl'] > 0]['pnl'].sum()
            losses = abs(tdf[tdf['pnl'] < 0]['pnl'].sum())
            pf = wins / losses if losses > 0 else 999

            # Max Drawdown
            equity = tdf['pnl'].cumsum()
            md = (equity - equity.cummax()).min()

            results.append({
                'tp': tp,
                'sl': sl,
                'trades': len(tdf),
                'total_pnl': round(total_pnl, 1),
                'win_rate': round(win_rate, 1),
                'profit_factor': round(pf, 3),
                'max_dd': round(md, 1)
            })

    # 6. Save and Display Results
    res_df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "xgboost_filter_model" / "tp_sl_sweep_results.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nSweep results saved to {out_path}")

    # Display Top 10 by Total PnL
    print("\n--- Top 10 Combinations (by Total PnL) ---")
    print(res_df.sort_values("total_pnl", ascending=False).head(10).to_string(index=False))

    # Display Top 10 by Profit Factor (minimum 20 trades)
    print("\n--- Top 10 Combinations (by Profit Factor, min 20 trades) ---")
    print(res_df[res_df['trades'] >= 20].sort_values("profit_factor", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    run_tp_sl_sweep()

