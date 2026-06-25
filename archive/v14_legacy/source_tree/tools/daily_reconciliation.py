#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.v13_config import WF_CONFIG
from backtest.core import simulate_v13_core
from xgboost_filter_model.train_filter_v13_wf_image import prepare_data_v13
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgboost_filter_model.train_directional_model_v9 import add_momentum_features
from config.v13_config import EXECUTION_CONFIG

def run_daily_reconciliation(target_date_str=None, show_weekly=False, show_cycle=False):
    """
    Runs a backtest for the specified date (defaulting to yesterday if not provided)
    and compares it with the bot's actual trade log.
    """
    if target_date_str is None:
        # Default to yesterday to ensure the trading day is fully concluded
        target_date = datetime.now(timezone.utc) - timedelta(days=1)
        target_date_str = target_date.strftime('%Y-%m-%d')

    print(f"--- Starting Daily Reconciliation for {target_date_str} ---")

    # 1. Load Bot Trades
    bot_trades_path = PROJECT_ROOT / "runtime" / "trading_bot_trades.csv"
    if not bot_trades_path.exists():
        print("Bot trade log not found.")
        return

    bot_df = pd.read_csv(bot_trades_path)
    bot_df['entry_time'] = pd.to_datetime(bot_df['entry_time'], utc=True)

    # Filter bot trades for the target date (using UTC for simplicity)
    day_start = pd.to_datetime(target_date_str).tz_localize('UTC')
    day_end = day_start + timedelta(days=1)
    bot_day_trades = bot_df[(bot_df['entry_time'] >= day_start) & (bot_df['entry_time'] < day_end)].copy()

    # 2. Run Backtest for the same day
    # We need a bit of buffer data before the start to calculate features
    buffer_start = (day_start - timedelta(days=5)).strftime('%Y-%m-%d')
    backtest_end = day_end.strftime('%Y-%m-%d')

    print(f"Fetching data for backtest: {buffer_start} to {backtest_end}")
    df = prepare_data_v13(start_date=buffer_start, end_date=backtest_end)
    df = add_directional_features(df)
    df = add_ma_features(df)
    df = add_momentum_features(df)
    df.dropna(inplace=True)

    # Get model signals
    # For daily recon, we use production models for simplicity, or we could load cycle models
    s1_model = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v13_wf_image.joblib")
    s2_model = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v13_wf.joblib")

    # Identify features
    exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'day_high_rolling', 'day_low_rolling', 'day_open']
    exclude += ['Dchange_utc2_rel', 'Dupper_wick_utc2_rel', 'Dlower_wick_utc2_rel', 'trend_label', 'target_v10', 'is_trend', 'atr', 'day_utc2']
    exclude += ['future_max_move', 'future_min_move', 'future_er', 'atr_threshold', 'bar_move', 'hour', 'day_id', 'day_high', 'day_low']
    exclude += ['day_high', 'day_low', 'day_open', 'high_90', 'low_90', 'closePrice_ask', 'closePrice_bid', 'highPrice_ask', 'lowPrice_bid', 'closePrice', 'lowPrice', 'open_price', 'highPrice_bid', 'lowPrice_ask', 'openPrice_bid', 'openPrice_ask']
    features = [c for c in df.columns if c not in exclude]
    s1_features = [f for f in features if f not in [
        'directional_change_15', 'directional_change_30', 'directional_change_90',
        'wick_ratio_15', 'wick_ratio_30', 'wick_ratio_90', 'price_vs_ma_10', 'price_vs_ma_30', 'price_vs_ma_90',
        'ma_10_vs_30', 'ma_30_vs_90', 'rsi_14', 'rsi_30', 'macd', 'macd_signal', 'macd_diff', 'roc_15', 'roc_30', 'roc_60'
    ]]

    # Filter to test window
    df_test = df[df.index >= day_start].copy()
    if df_test.empty:
         print(f"No market data found for {target_date_str}")
         return

    df_test['s1_prob'] = s1_model.predict_proba(df_test[s1_features])[:, 1]
    df_test['s2_prob'] = np.nan
    trend_mask = df_test['s1_prob'] >= EXECUTION_CONFIG["s1_threshold"]
    if trend_mask.any():
        df_test.loc[trend_mask, 's2_prob'] = s2_model.predict_proba(df_test.loc[trend_mask, features])[:, 1]

    # Side signal
    df_test['side_signal'] = 0
    s2_thresh = EXECUTION_CONFIG["s2_threshold"]
    df_test.loc[trend_mask & (df_test['s2_prob'] >= s2_thresh), 'side_signal'] = 1
    df_test.loc[trend_mask & (df_test['s2_prob'] <= (1.0 - s2_thresh)), 'side_signal'] = -1

    # Bid/Ask
    spread = EXECUTION_CONFIG["spread_default"]
    df_test['open_ask'] = df_test['open'] + spread
    df_test['open_bid'] = df_test['open'] - spread
    df_test['close_ask'] = df_test['close'] + spread
    df_test['close_bid'] = df_test['close'] - spread
    df_test['high_ask'] = df_test['high'] + spread
    df_test['low_bid'] = df_test['low'] - spread

    # Simulate
    tp = EXECUTION_CONFIG["tp"]
    sl = EXECUTION_CONFIG["sl"]
    horizon = EXECUTION_CONFIG["horizon"]

    backtest_trades = simulate_v13_core(df_test, tp, sl, horizon)
    bt_df = pd.DataFrame(backtest_trades)
    if not bt_df.empty:
        bt_df['entry_time'] = pd.to_datetime(bt_df['entry_time'], utc=True)
        bt_df['pnl'] = (bt_df['exit_price'] - bt_df['entry_price']) * bt_df['side']

    # 3. Compare (Reorganized to show summary last)
    bot_trades_count = len(bot_day_trades)
    bt_trades_count = len(bt_df)

    bot_pnl = bot_day_trades['pnl_usd'].sum() if not bot_day_trades.empty else 0.0
    bt_pnl = bt_df['pnl'].sum() if not bt_df.empty else 0.0

    bot_win_rate = (bot_day_trades['pnl_usd'] > 0).mean() * 100 if not bot_day_trades.empty else 0.0
    bt_win_rate = (bt_df['pnl'] > 0).mean() * 100 if not bt_df.empty else 0.0

    print("\n--- Trade Matching Detail ---")
    matches = pd.DataFrame()
    if not bot_day_trades.empty and not bt_df.empty:
        bot_short = bot_day_trades[['entry_time', 'direction', 'pnl_usd']].copy()
        bot_short['entry_time_min'] = bot_short['entry_time'].dt.floor('min')
        bot_short['side'] = bot_short['direction'].map({'LONG': 1, 'SHORT': -1})

        bt_short = bt_df[['entry_time', 'side', 'pnl']].copy()
        bt_short['entry_time_min'] = bt_short['entry_time'].dt.floor('min')

        merged = pd.merge(bt_short, bot_short, on=['entry_time_min', 'side'], how='outer', suffixes=('_bt', '_bot'))

        matches = merged.dropna(subset=['entry_time_bt', 'entry_time_bot'])
        only_bt = merged[merged['entry_time_bot'].isna()]
        only_bot = merged[merged['entry_time_bt'].isna()]

        print(f"Matches: {len(matches)}")
        print(f"Missed by Bot: {len(only_bt)}")
        print(f"Ghost Bot Trades: {len(only_bot)}")

        if not only_bt.empty:
            print("\nTrades in Backtest but NOT in Bot:")
            print(only_bt[['entry_time_min', 'side', 'pnl']].to_string(index=False))

        if not only_bot.empty:
            print("\nTrades in Bot but NOT in Backtest:")
            print(only_bot[['entry_time_min', 'side', 'pnl_usd']].to_string(index=False))

        # PnL Comparison for matches
        if not matches.empty:
            matches['pnl_diff'] = matches['pnl'] - matches['pnl_usd']
            print(f"\nAverage PnL Difference for Matched Trades (BT - Bot): {matches['pnl_diff'].mean():.2f}")

    print("\n" + "="*60)
    print(f" FINAL RECONCILIATION SUMMARY FOR {target_date_str}")
    print("="*60)
    print(f"{'Metric':<20} | {'Bot Actual':<15} | {'Backtest':<15} | {'Difference':<15}")
    print("-" * 70)
    print(f"{'Total Trades':<20} | {bot_trades_count:<15} | {bt_trades_count:<15} | {bot_trades_count - bt_trades_count:<15}")
    print(f"{'Total PnL':<20} | {bot_pnl:<15.2f} | {bt_pnl:<15.2f} | {bot_pnl - bt_pnl:<15.2f}")
    print(f"{'Win Rate %':<20} | {bot_win_rate:<15.1f} | {bt_win_rate:<15.1f} | {bot_win_rate - bt_win_rate:<15.1f}")
    if bt_trades_count > 0 and bot_trades_count > 0:
        print(f"{'Avg Trade PnL':<20} | {bot_pnl/bot_trades_count:<15.2f} | {bt_pnl/bt_trades_count:<15.2f} | {(bot_pnl/bot_trades_count) - (bt_pnl/bt_trades_count):<15.2f}")
    print("="*60)

    # Save results to a log
    recon_log = PROJECT_ROOT / "runtime" / "reconciliation_log.csv"
    log_exists = recon_log.exists()

    summary = {
        'date': target_date_str,
        'bot_count': bot_trades_count,
        'bt_count': bt_trades_count,
        'matches': len(matches) if not (bot_day_trades.empty or bt_df.empty) else 0,
        'bot_pnl': round(bot_pnl, 2),
        'bt_pnl': round(bt_pnl, 2),
        'pnl_diff': round(bot_pnl - bt_pnl, 2),
        'bot_win_rate': round(bot_win_rate, 1),
        'bt_win_rate': round(bt_win_rate, 1),
        'recon_time': datetime.now(timezone.utc).isoformat()
    }

    pd.DataFrame([summary]).to_csv(recon_log, mode='a', index=False, header=not log_exists)
    print(f"\nReconciliation report saved to {recon_log}")

    # 4. Period Summaries (Optional via flags)
    if show_weekly or show_cycle:
        try:
            log_df = pd.read_csv(recon_log)
            if not log_df.empty:
                print("\n" + "="*60)
                print(" PERIOD RECONCILIATION TRENDS")
                print("="*60)

                if show_weekly:
                    # Weekly (Last 7 concluded days)
                    weekly_df = log_df.tail(7)
                    w_bot_pnl = weekly_df['bot_pnl'].sum()
                    w_bt_pnl = weekly_df['bt_pnl'].sum()
                    w_diff = w_bot_pnl - w_bt_pnl
                    print(f"LAST 7 DAYS   | Bot PnL: {w_bot_pnl:>8.2f} | BT PnL: {w_bt_pnl:>8.2f} | Diff: {w_diff:>8.2f}")

                if show_cycle:
                    # Training Cycle (WF_CONFIG["retrain_days"])
                    retrain_days = WF_CONFIG.get("retrain_days", 14)
                    cycle_df = log_df.tail(retrain_days)
                    c_bot_pnl = cycle_df['bot_pnl'].sum()
                    c_bt_pnl = cycle_df['bt_pnl'].sum()
                    c_diff = c_bot_pnl - c_bt_pnl
                    print(f"WF CYCLE ({retrain_days}d) | Bot PnL: {c_bot_pnl:>8.2f} | BT PnL: {c_bt_pnl:>8.2f} | Diff: {c_diff:>8.2f}")
                print("="*60)
        except Exception as e:
            print(f"Could not generate period summaries: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AlphaGold Daily Reconciliation Tool")
    parser.add_argument("date", nargs="?", help="Target date (YYYY-MM-DD), defaults to yesterday")
    parser.add_argument("--weekly", action="store_true", help="Show cumulative 7-day report")
    parser.add_argument("--cycle", action="store_true", help="Show cumulative training cycle report")

    args = parser.parse_args()
    run_daily_reconciliation(args.date, show_weekly=args.weekly, show_cycle=args.cycle)
