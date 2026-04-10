#!/usr/bin/env python3
"""
ML UPTREND STRATEGY - COMPLETE OPTIMIZATION WORKFLOW
====================================================

This script combines 3 steps into one:
1. Optimize stop/target/hold parameters (from optimize_stop_target_fixed.py)
2. Find profitable time slots (from analyze_time_slots_proper.py)
3. Apply and compare with time filters (from apply_time_filters.py)

Usage:
    python ml_complete_optimizer.py

Features:
- Uses cached predictions (ml_predictions_cache.csv)
- Fixed position size (1.0 unit)
- One position at a time (no overlaps)
- Adjusts target when new signals appear
- Proper trading day definition (UTC+6)
- Two sessions: Asia (HKT 6-19) and NY (NY 6-17)
- Interactive prompt to save config

Author: Combined from working scripts
Date: March 2, 2026
"""

import pandas as pd
import numpy as np
import sys
import os
import json
import pytz
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_utils import load_data_from_mysql
from ml_utils import print_profitable_slots_summary, calculate_statistics

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data settings
START_DATE = '2025-05-20'
END_DATE = '2026-02-04'  # Exclusive: loads data through 2026-02-04 23:59
PREDICTIONS_CACHE = 'ml_predictions_cache.csv'

# Trading settings
INITIAL_CAPITAL = 1000
POSITION_SIZE = 1.0  # Fixed: 1.0 unit for all trades
# No commission - using bid/ask spread instead

# Optimization parameter ranges
STOP_VALUES = [15, 16, 17, 18, 19, 20, 25]
TARGET_VALUES = [30, 35, 40, 45, 50]  # Fixed target values in dollars
HOLD_TIME_VALUES = [50, 60, 70, 80, 90,120, 150,  180, 240]
PROBABILITY_THRESHOLDS = [0.70, 0.65, 0.60, 0.55, 0.50]  # Match optimize_stop_target_fixed.py

# Time slot analysis criteria (MUST match analyze_time_slots_ny.py!)
MIN_TRADES_FOR_SLOT = 1  # Minimum 1 trade per slot
MIN_WIN_RATE = 35.0  # % (strict > not >=)
MIN_TOTAL_PNL = 0.0  # $ Total P&L (strict > not >=)

# Session definitions
ASIA_SESSION_START = 6   # HKT 6:00 AM
ASIA_SESSION_END = 19    # HKT 7:00 PM
NY_SESSION_START = 6     # NY 6:00 AM
NY_SESSION_END = 17      # NY 5:00 PM

# ============================================================================
# STEP 1: OPTIMIZE PARAMETERS
# ============================================================================

def optimize_parameters():
    """Optimize stop/target/hold parameters"""
    print("=" * 120)
    print("STEP 1: OPTIMIZE STOP/TARGET/HOLD PARAMETERS")
    print("=" * 120)
    print()

    # Check for predictions cache
    if not os.path.exists(PREDICTIONS_CACHE):
        print(f"❌ Error: {PREDICTIONS_CACHE} not found!")
        print("   Please run train_uptrend_ml_model.py first to generate predictions.")
        sys.exit(1)

    print(f"Loading cached predictions from {PREDICTIONS_CACHE}...")
    predictions_df = pd.read_csv(PREDICTIONS_CACHE)
    print(f"✅ Loaded {len(predictions_df):,} cached predictions")
    print()

    # Load price data
    print("Loading price data...")
    df_raw = load_data_from_mysql(
        start_date=START_DATE,
        end_date=END_DATE,
        verbose=False
    )

    df = pd.DataFrame({
        'open': df_raw['open'],
        'open_ask': df_raw['open_ask'],
        'open_bid': df_raw['open_bid'],
        'high': df_raw['high'],
        'high_ask': df_raw['high_ask'],
        'high_bid': df_raw['high_bid'],
        'low': df_raw['low'],
        'low_ask': df_raw['low_ask'],
        'low_bid': df_raw['low_bid'],
        'close': df_raw['close'],
        'close_ask': df_raw['close_ask'],
        'close_bid': df_raw['close_bid'],
    })

    if 'datetime' in df_raw.columns:
        df['datetime'] = df_raw['datetime']
        df = df.set_index('datetime')

    print(f"✅ Loaded {len(df):,} bars")
    print(f"   Period: {START_DATE} to {END_DATE} (data through {df.index.max().date()})")
    print()

    # Run optimization
    total_combos = len(STOP_VALUES) * len(TARGET_VALUES) * len(HOLD_TIME_VALUES) * len(PROBABILITY_THRESHOLDS)
    print(f"Testing {total_combos} combinations...")
    print(f"  Stops: {STOP_VALUES}")
    print(f"  Targets: {TARGET_VALUES}")
    print(f"  Hold Times: {HOLD_TIME_VALUES}")
    print(f"  Prob Thresholds: {[f'{p:.0%}' for p in PROBABILITY_THRESHOLDS]}")
    print()

    results = []
    combo_num = 0

    for prob_thresh in PROBABILITY_THRESHOLDS:
        print(f"Testing Probability Threshold: {prob_thresh:.0%}")
        print("-" * 120)

        for stop in STOP_VALUES:
            for target in TARGET_VALUES:
                rr_ratio = target / stop  # Calculate ratio from target and stop

                for hold_time in HOLD_TIME_VALUES:
                    combo_num += 1

                    # Run backtest
                    result = run_backtest_optimized(
                        predictions_df, df, prob_thresh, stop, target, hold_time
                    )

                    if result:
                        results.append(result)

                        # Progress indicator
                        pct = (combo_num / total_combos) * 100
                        print(f"  [{combo_num}/{total_combos}] ({pct:.0f}%) "
                              f"Stop=${stop}, Target=${target:.0f} ({rr_ratio:.2f}:1), "
                              f"Hold={hold_time}bars... "
                              f"✓ Trades: {result['total_trades']}, "
                              f"Total Profits: {result['total_pnl']:.1f}, "
                              f"Return: {result['total_return']:.1f}%, "
                              f"WR: {result['win_rate']:.1f}%")
                        sys.stdout.flush()

        print()

    if not results:
        print("❌ No valid results found!")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    results_df.to_csv('ml_optimization_results.csv', index=False)
    print(f"✅ Saved optimization results to ml_optimization_results.csv")
    print()

    # Find best configuration
    results_df['sharpe_ratio'] = results_df['total_return'] / results_df['max_drawdown'].abs()
    results_df['sharpe_ratio'] = results_df['sharpe_ratio'].replace([np.inf, -np.inf], 0)

    # Score: return * sharpe * win_rate
    results_df['score'] = (results_df['total_return'] *
                           results_df['sharpe_ratio'] *
                           results_df['win_rate'] / 100)

    best_idx = results_df['score'].idxmax()
    best = results_df.loc[best_idx]

    print("=" * 120)
    print("🏆 BEST CONFIGURATION FOUND")
    print("=" * 120)
    print(f"  Probability Threshold: {best['probability_threshold']:.0%}")
    print(f"  Stop Loss:             ${best['stop_loss']:.0f}")
    print(f"  Take Profit:           ${best['target_profit']:.2f}")
    print(f"  Reward/Risk Ratio:     {best['reward_risk_ratio']:.2f}:1")
    print(f"  Max Hold Time:         {best['max_hold_bars']:.0f} bars")
    print()
    print(f"  Total Trades:          {best['total_trades']:.0f}")
    print(f"  Win Rate:              {best['win_rate']:.1f}%")
    print(f"  Total Return:          {best['total_return']:.1f}%")
    print(f"  Total Profit:          ${best['total_pnl']:.2f}")
    print(f"  Profit Factor:         {best['profit_factor']:.2f}")
    print(f"  Max Drawdown:          {best['max_drawdown']:.1f}%")
    print(f"  Sharpe Ratio:          {best['sharpe_ratio']:.2f}")
    print("=" * 120)
    print()

    # Save best config trades for time slot analysis
    trades_file = (f"ml_best_config_trades_"
                   f"prob{best['probability_threshold']:.0%}_"
                   f"stop{best['stop_loss']:.0f}_"
                   f"target{best['target_profit']:.0f}.csv")

    print(f"Running backtest with best config to save trades...")
    best_result = run_backtest_with_trades(
        predictions_df, df,
        float(best['probability_threshold']),
        float(best['stop_loss']),
        float(best['target_profit']),
        int(best['max_hold_bars'])  # Convert to int
    )

    if best_result:
        best_result['trades_df'].to_csv(trades_file, index=False)
        print(f"✅ Saved {len(best_result['trades_df']):,} trades to {trades_file}")
        print()

    return best, trades_file


def run_backtest_optimized(predictions_df, df, probability_threshold, stop_loss, target_profit, max_hold_bars):
    """Run backtest with specific parameters (optimized version)"""
    signals = predictions_df[predictions_df['probability'] >= probability_threshold].copy()

    if len(signals) == 0:
        return None

    signal_indices = signals['idx'].values
    highs_bid = df['high_bid'].values  # Use bid for exit (we sell at bid)
    lows_bid = df['low_bid'].values
    opens_ask = df['open_ask'].values  # Use ask for entry (we buy at ask)
    closes_bid = df['close_bid'].values  # Use bid for exit
    closes_ask = df['close_ask'].values  # Use ask for adjusting targets
    #prob = signals['probability'].values  # Get probabilities for dynamic target adjustment
    stop_distance = stop_loss / POSITION_SIZE
    target_distance = target_profit / POSITION_SIZE

    signal_set = set(signal_indices)

    trades = []
    capital = INITIAL_CAPITAL

    signal_idx = 0
    current_bar = int(signal_indices[0])

    while signal_idx < len(signal_indices) and current_bar < len(df) - 1:
        entry_idx = current_bar + 1
        entry_price = opens_ask[entry_idx]  # Buy at ask

        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance #*prob[signal_idx]  # Dynamic target based on probability

        end_idx = min(entry_idx + max_hold_bars, len(df))

        exit_idx = None
        exit_price = None
        exit_reason = None

        for current_idx in range(entry_idx, end_idx):
            current_high_bid = highs_bid[current_idx]  # Sell at bid
            current_low_bid = lows_bid[current_idx]

            if current_low_bid <= stop_price:
                exit_idx = current_idx
                exit_price = stop_price  # Exit at stop (assumes we can sell at bid <= stop)
                exit_reason = 'STOP'
                break
            elif current_high_bid >= target_price:
                exit_idx = current_idx
                exit_price = target_price  # Exit at target (assumes we can sell at bid >= target)
                exit_reason = 'TARGET'
                break


            if current_idx in signal_set and current_idx != entry_idx:
                current_price_ask = closes_ask[current_idx]  # New signal uses ask price
                new_target = current_price_ask + target_distance #*prob[signal_idx]  # Dynamic target based on probability

                if new_target > target_price:
                    target_price = new_target
                    #end_idx = min(current_idx + max_hold_bars, len(df)) # by alpha

        if exit_idx is None:
            exit_idx = end_idx - 1
            exit_price = closes_bid[exit_idx]  # Exit at bid
            exit_reason = 'TIME'

        bars_held = exit_idx - entry_idx

        # Calculate P&L: exit_price (bid) - entry_price (ask)
        # The spread cost is already included in the price difference
        if exit_reason == 'TARGET':
            pnl_dollars = (exit_price - entry_price) * POSITION_SIZE
        elif exit_reason == 'STOP':
            pnl_dollars = (exit_price - entry_price) * POSITION_SIZE
        else:
            pnl_dollars = (exit_price - entry_price) * POSITION_SIZE

        capital += pnl_dollars

        trades.append({
            'pnl_dollars': pnl_dollars,
            'capital': capital,
            'is_winner': pnl_dollars > 0
        })

        while signal_idx < len(signal_indices) and signal_indices[signal_idx] <= exit_idx:
            signal_idx += 1

        if signal_idx < len(signal_indices):
            current_bar = int(signal_indices[signal_idx])
        else:
            break

    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)

    total_trades = len(trades_df)
    winners = trades_df[trades_df['is_winner']]
    losers = trades_df[~trades_df['is_winner']]

    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
    total_pnl = trades_df['pnl_dollars'].sum()

    profit_factor = abs(winners['pnl_dollars'].sum() / losers['pnl_dollars'].sum()) if len(losers) > 0 and losers['pnl_dollars'].sum() != 0 else float('inf')

    total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    trades_df['peak'] = trades_df['capital'].cummax()
    trades_df['drawdown'] = trades_df['capital'] - trades_df['peak']
    trades_df['drawdown_pct'] = (trades_df['drawdown'] / trades_df['peak']) * 100
    max_drawdown = trades_df['drawdown_pct'].min()

    return {
        'probability_threshold': probability_threshold,
        'stop_loss': stop_loss,
        'target_profit': target_profit,
        'reward_risk_ratio': target_profit / stop_loss,
        'max_hold_bars': max_hold_bars,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'max_drawdown': max_drawdown
    }


def run_backtest_with_trades(predictions_df, df, probability_threshold, stop_loss, target_profit, max_hold_bars):
    """Run backtest and return full trade details"""
    # Ensure max_hold_bars is an integer
    max_hold_bars = int(max_hold_bars)

    signals = predictions_df[predictions_df['probability'] >= probability_threshold].copy()

    if len(signals) == 0:
        return None

    signal_indices = signals['idx'].values
    highs_bid = df['high_bid'].values  # Use bid for exit (we sell at bid)
    lows_bid = df['low_bid'].values
    opens_ask = df['open_ask'].values  # Use ask for entry (we buy at ask)
    closes_bid = df['close_bid'].values  # Use bid for exit
    closes_ask = df['close_ask'].values  # Use ask for adjusting targets

    stop_distance = stop_loss / POSITION_SIZE
    target_distance = target_profit / POSITION_SIZE

    signal_set = set(signal_indices)

    trades = []
    capital = INITIAL_CAPITAL

    signal_idx = 0
    current_bar = int(signal_indices[0])

    while signal_idx < len(signal_indices) and current_bar < len(df) - 1:
        entry_idx = current_bar + 1
        entry_price = opens_ask[entry_idx]  # Buy at ask
        entry_time = df.index[entry_idx]

        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance

        end_idx = min(entry_idx + max_hold_bars, len(df))

        exit_idx = None
        exit_price = None
        exit_reason = None

        for current_idx in range(entry_idx, end_idx):
            current_high_bid = highs_bid[current_idx]  # Sell at bid
            current_low_bid = lows_bid[current_idx]

            if current_low_bid <= stop_price:
                exit_idx = current_idx
                exit_price = stop_price  # Exit at stop (assumes we can sell at bid <= stop)
                exit_reason = 'STOP'
                break
            elif current_high_bid >= target_price:
                exit_idx = current_idx
                exit_price = target_price  # Exit at target (assumes we can sell at bid >= target)
                exit_reason = 'TARGET'
                break

            if current_idx in signal_set and current_idx != entry_idx:
                current_price_ask = closes_ask[current_idx]  # New signal uses ask price
                new_target = current_price_ask + target_distance

                if new_target > target_price:
                    target_price = new_target
                    #end_idx = min(current_idx + max_hold_bars, len(df))  # by alpha

        if exit_idx is None:
            exit_idx = end_idx - 1
            exit_price = closes_bid[exit_idx]  # Exit at bid
            exit_reason = 'TIME'

        exit_time = df.index[exit_idx]
        bars_held = exit_idx - entry_idx

        # Calculate P&L: exit_price (bid) - entry_price (ask)
        # The spread cost is already included in the price difference
        pnl_dollars = (exit_price - entry_price) * POSITION_SIZE

        capital += pnl_dollars

        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'pnl_dollars': pnl_dollars,
            'cumulative_pnl': capital - INITIAL_CAPITAL,
            'is_winner': pnl_dollars > 0
        })

        while signal_idx < len(signal_indices) and signal_indices[signal_idx] <= exit_idx:
            signal_idx += 1

        if signal_idx < len(signal_indices):
            current_bar = int(signal_indices[signal_idx])
        else:
            break

    if len(trades) == 0:
        return None

    return {'trades_df': pd.DataFrame(trades)}


# ============================================================================
# STEP 2: FIND PROFITABLE TIME SLOTS
# ============================================================================

def find_profitable_time_slots(trades_file):
    """Analyze trades and find profitable time slots"""
    print("=" * 120)
    print("STEP 2: FIND PROFITABLE TIME SLOTS")
    print("=" * 120)
    print()

    if not os.path.exists(trades_file):
        print(f"❌ Error: {trades_file} not found!")
        return None

    print(f"Loading trades from {trades_file}...")
    df = pd.read_csv(trades_file)
    df['entry_time'] = pd.to_datetime(df['entry_time'])

    if df['entry_time'].dt.tz is None:
        df['entry_time'] = df['entry_time'].dt.tz_localize('UTC')

    # Rename pnl_dollars to pnl for compatibility
    if 'pnl_dollars' in df.columns and 'pnl' not in df.columns:
        df['pnl'] = df['pnl_dollars']

    print(f"✅ Loaded {len(df):,} trades")
    print()

    # Define trading day at UTC+6 (HKT)
    hkt_tz = pytz.timezone('Asia/Hong_Kong')
    ny_tz = pytz.timezone('America/New_York')

    df['entry_hkt'] = df['entry_time'].dt.tz_convert(hkt_tz)
    df['entry_ny'] = df['entry_time'].dt.tz_convert(ny_tz)

    # Trading day starts at HKT 6:00 AM
    df['trading_day_hkt'] = df['entry_hkt'].apply(
        lambda x: (x - pd.Timedelta(hours=6)).date()
    )
    df['trading_day_of_week'] = pd.to_datetime(df['trading_day_hkt']).dt.dayofweek
    df['trading_day_name'] = pd.to_datetime(df['trading_day_hkt']).dt.day_name()

    # Remove Sunday (weekday=6)
    df = df[df['trading_day_of_week'] != 6].copy()
    print(f"After removing Sunday: {len(df):,} trades")
    print()

    df['hour_hkt'] = df['entry_hkt'].dt.hour
    df['hour_ny'] = df['entry_ny'].dt.hour

    # Assign session
    def assign_session(row):
        hour_hkt = row['hour_hkt']
        hour_ny = row['hour_ny']

        if ASIA_SESSION_START <= hour_hkt <= ASIA_SESSION_END:
            return 'Asia'
        elif NY_SESSION_START <= hour_ny <= NY_SESSION_END:
            return 'NY'
        else:
            return 'Other'

    df['session'] = df.apply(assign_session, axis=1)

    # Analyze both sessions
    saved_filters = {'asia': {}, 'ny': {}}

    for session_name, time_col in [('Asia', 'hour_hkt'), ('NY', 'hour_ny')]:
        print(f"Analyzing {session_name} Session ({time_col.replace('hour_', '').upper()} {ASIA_SESSION_START if session_name == 'Asia' else NY_SESSION_START}:00 - {ASIA_SESSION_END if session_name == 'Asia' else NY_SESSION_END}:00)...")

        session_df = df[df['session'] == session_name]
        print(f"  Session trades: {len(session_df):,}")

        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        session_key = session_name.lower()
        profitable_count = 0

        for day in days_order:
            day_data = session_df[session_df['trading_day_name'] == day]
            if len(day_data) == 0:
                saved_filters[session_key][day] = []
                continue

            day_profitable = []

            for hour in sorted(session_df[time_col].unique()):
                hour_data = day_data[day_data[time_col] == hour]
                if len(hour_data) >= MIN_TRADES_FOR_SLOT:
                    wr = (hour_data['pnl'] > 0).sum() / len(hour_data) * 100
                    total_pnl = hour_data['pnl'].sum()

                    # STRICT > not >=
                    if wr > MIN_WIN_RATE and total_pnl > MIN_TOTAL_PNL:
                        # Convert numpy int32 to Python int for JSON serialization
                        day_profitable.append(int(hour))
                        profitable_count += 1

            saved_filters[session_key][day] = day_profitable

        # Print profitable slots summary
        print_profitable_slots_summary(session_key, profitable_count, saved_filters)

    # Generate heatmaps
    generate_session_heatmap(df[df['session'] == 'Asia'], 'Asia', 'Asia/Hong_Kong', ASIA_SESSION_START, ASIA_SESSION_END)
    generate_session_heatmap(df[df['session'] == 'NY'], 'NY', 'America/New_York', NY_SESSION_START, NY_SESSION_END)

    # Save filters to ml_config.json (SINGLE SOURCE OF TRUTH)
    import json
    config_file = 'ml_config.json'

    # Load existing config or create new one
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    # Update time_filters section
    config['time_filters'] = saved_filters

    # Save back to ml_config.json
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Saved time filters to {config_file}")
    print()

    return saved_filters

    # Group by day and hour
    grouped = session_df.groupby(['session_weekday', 'session_hour']).agg({
        'pnl': ['count', 'sum', 'mean'],
        'is_winner': 'sum'
    }).reset_index()

    grouped.columns = ['weekday', 'hour', 'trades', 'total_pnl', 'avg_pnl', 'winners']
    grouped['win_rate'] = (grouped['winners'] / grouped['trades'] * 100).round(1)

    # Filter profitable slots (MUST match analyze_time_slots_ny.py criteria!)
    # Note: Using STRICT > not >=
    profitable = grouped[
        (grouped['win_rate'] > MIN_WIN_RATE) &  # STRICT >
        (grouped['total_pnl'] > MIN_TOTAL_PNL)  # STRICT > (uses TOTAL not avg)
    ].copy()

    print(f"  Profitable slots found: {len(profitable)}")
    print()

    # Convert to time filter format
    time_filters = {}
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for day in range(7):
        day_slots = profitable[profitable['weekday'] == day]
        if len(day_slots) > 0:
            hours = sorted(day_slots['hour'].tolist())
            time_filters[day] = hours
            print(f"  {day_names[day]:10s}: {hours}")

    return time_filters


def generate_session_heatmap(df, session_name, timezone, session_start, session_end):
    """Generate text heatmap for a session (same logic as analyze_time_slots_proper.py)"""
    tz = pytz.timezone(timezone)
    df = df.copy()
    df['session_time'] = df['entry_time'].dt.tz_convert(tz)
    df['session_hour'] = df['session_time'].dt.hour
    df['trading_day_name'] = df['session_time'].dt.day_name()

    # Filter for session (including end hour)
    session_df = df[(df['session_hour'] >= session_start) & (df['session_hour'] <= session_end)]

    if len(session_df) == 0:
        return

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    hours = sorted(session_df['session_hour'].unique())

    print("=" * 120)
    print(f"{'🌏' if session_name == 'Asia' else '🗽'} {session_name.upper()} SESSION HEATMAP")
    print("=" * 120)
    print()

    # Trade Count
    print(f"🔢 TRADE COUNT BY DAY × HOUR ({session_name})")
    print("-" * 120)
    print(f"{'Day':<12}", end='')
    for hour in hours:
        print(f"{hour:02d}:00".rjust(6), end='')
    print()
    print("-" * 120)

    for day in days_order:
        day_data = session_df[session_df['trading_day_name'] == day]
        if len(day_data) == 0:
            continue
        print(f"{day:<12}", end='')
        for hour in hours:
            count = len(day_data[day_data['session_hour'] == hour])
            if count > 0:
                print(f"{count:6d}", end='')
            else:
                print(f"{'--':>6}", end='')
        print()
    print()

    # Win Rate
    print(f"📊 WIN RATE % BY DAY × HOUR ({session_name}, min 1 trade)")
    print("-" * 120)
    print(f"{'Day':<12}", end='')
    for hour in hours:
        print(f"{hour:02d}:00".rjust(6), end='')
    print()
    print("-" * 120)

    for day in days_order:
        day_data = session_df[session_df['trading_day_name'] == day]
        if len(day_data) == 0:
            continue
        print(f"{day:<12}", end='')
        for hour in hours:
            hour_data = day_data[day_data['session_hour'] == hour]
            if len(hour_data) >= MIN_TRADES_FOR_SLOT:
                wr = (hour_data['pnl_dollars'] > 0).sum() / len(hour_data) * 100
                print(f"{wr:5.1f}%", end='')
            else:
                print(f"{'--':>6}", end='')
        print()
    print()

    # Avg P&L
    print(f"💰 AVG P&L BY DAY × HOUR ({session_name}, min 1 trade, 🟢=profit, 🔴=loss)")
    print("-" * 120)
    print(f"{'Day':<12}", end='')
    for hour in hours:
        print(f"{hour:02d}:00".rjust(9), end='')
    print()
    print("-" * 120)

    for day in days_order:
        day_data = session_df[session_df['trading_day_name'] == day]
        if len(day_data) == 0:
            continue
        print(f"{day:<12}", end='')
        for hour in hours:
            hour_data = day_data[day_data['session_hour'] == hour]
            if len(hour_data) >= MIN_TRADES_FOR_SLOT:
                avg_pnl = hour_data['pnl'].mean()
                marker = '🟢' if avg_pnl > 0 else '🔴'
                print(f"{marker}${avg_pnl:5.1f}", end='')
            else:
                print(f"{'--':>9}", end='')
        print()
    print()


# ============================================================================
# STEP 3: APPLY AND COMPARE WITH TIME FILTERS
# ============================================================================

def apply_and_compare(trades_file, time_filters):
    """Apply time filters and show comparison"""
    print("=" * 120)
    print("STEP 3: APPLY TIME FILTERS AND COMPARE RESULTS")
    print("=" * 120)
    print()

    print(f"Loading trades from {trades_file}...")
    df = pd.read_csv(trades_file)
    df['entry_time'] = pd.to_datetime(df['entry_time'])

    if df['entry_time'].dt.tz is None:
        df['entry_time'] = df['entry_time'].dt.tz_localize('UTC')

    # Rename pnl_dollars to pnl for compatibility
    if 'pnl_dollars' in df.columns and 'pnl' not in df.columns:
        df['pnl'] = df['pnl_dollars']

    print(f"✅ Loaded {len(df):,} trades")
    print()

    # Define trading day and sessions (same logic as analyze_time_slots_proper.py)
    hkt_tz = pytz.timezone('Asia/Hong_Kong')
    ny_tz = pytz.timezone('America/New_York')

    df['entry_hkt'] = df['entry_time'].dt.tz_convert(hkt_tz)
    df['entry_ny'] = df['entry_time'].dt.tz_convert(ny_tz)

    # Trading day starts at HKT 6:00 AM
    df['trading_day_hkt'] = df['entry_hkt'].apply(
        lambda x: (x - pd.Timedelta(hours=6)).date()
    )
    df['trading_day_of_week'] = pd.to_datetime(df['trading_day_hkt']).dt.dayofweek
    df['trading_day_name'] = pd.to_datetime(df['trading_day_hkt']).dt.day_name()

    df['hour_hkt'] = df['entry_hkt'].dt.hour
    df['hour_ny'] = df['entry_ny'].dt.hour

    # Assign session
    def assign_session(row):
        hour_hkt = row['hour_hkt']
        hour_ny = row['hour_ny']

        if ASIA_SESSION_START <= hour_hkt <= ASIA_SESSION_END:
            return 'Asia'
        elif NY_SESSION_START <= hour_ny <= NY_SESSION_END:
            return 'NY'
        else:
            return 'Other'

    df['session'] = df.apply(assign_session, axis=1)

    # Original statistics
    orig_stats = calculate_statistics(df, "ORIGINAL (No Filters)")

    # Apply session-specific filters
    def passes_filter(row):
        session = row['session']
        day = row['trading_day_name']

        if session == 'Asia':
            hour = row['hour_hkt']
            allowed_hours = time_filters.get('asia', {}).get(day, [])
        elif session == 'NY':
            hour = row['hour_ny']
            allowed_hours = time_filters.get('ny', {}).get(day, [])
        else:
            return False

        return hour in allowed_hours

    # Get filtered trades for each session
    asia_filtered = df[(df['session'] == 'Asia') & df.apply(passes_filter, axis=1)]
    ny_filtered = df[(df['session'] == 'NY') & df.apply(passes_filter, axis=1)]

    # Calculate stats for each session
    asia_stats = calculate_statistics(asia_filtered, "ASIA SESSION (Filtered)")
    ny_stats = calculate_statistics(ny_filtered, "NY SESSION (Filtered)")

    # Combined filtered
    filtered_df = pd.concat([asia_filtered, ny_filtered], ignore_index=True)
    filtered_stats = calculate_statistics(filtered_df, "COMBINED (Asia + NY)")

    # Show comparison
    print()
    print("=" * 120)
    print("📊 COMPARISON SUMMARY")
    print("=" * 120)
    print()

    print(f"Original Trades:     {len(df):,}")
    print(f"Filtered Trades:     {len(filtered_df):,} ({len(filtered_df)/len(df)*100:.1f}%)")
    print()
    print(f"Original Win Rate:   {orig_stats['win_rate']:.1f}%")
    print(f"Filtered Win Rate:   {filtered_stats['win_rate']:.1f}% ({filtered_stats['win_rate']-orig_stats['win_rate']:+.1f}%)")
    print()
    print(f"Original P&L:        ${orig_stats['total_pnl']:.2f}")
    print(f"Filtered P&L:        ${filtered_stats['total_pnl']:.2f} ({(filtered_stats['total_pnl']-orig_stats['total_pnl'])/orig_stats['total_pnl']*100:+.1f}%)")
    print()
    print(f"Original Avg P&L:    ${orig_stats['avg_pnl']:.2f}")
    print(f"Filtered Avg P&L:    ${filtered_stats['avg_pnl']:.2f} ({filtered_stats['avg_pnl']-orig_stats['avg_pnl']:+.2f})")
    print()

    return orig_stats, filtered_stats


def apply_filters(df, time_filters, timezone, session_start, session_end):
    """Apply time filters to dataframe (deprecated - kept for compatibility)"""
    tz = pytz.timezone(timezone)
    df = df.copy()
    df['filter_time'] = df['entry_time'].dt.tz_convert(tz)
    df['filter_hour'] = df['filter_time'].dt.hour
    df['filter_weekday'] = df['filter_time'].dt.dayofweek

    # Filter for session hours first
    df = df[(df['filter_hour'] >= session_start) & (df['filter_hour'] < session_end)]

    # Apply time slot filters
    mask = pd.Series(False, index=df.index)
    for day, hours in time_filters.items():
        mask |= (df['filter_weekday'] == day) & (df['filter_hour'].isin(hours))

    return df[mask]



# ============================================================================
# SAVE CONFIGURATION
# ============================================================================

def save_configuration(best_config, time_filters, orig_stats=None, filtered_stats=None):
    """Save configuration to JSON file"""
    config = {
        'optimization': {
            'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_period': {
                'start': START_DATE,
                'end': END_DATE,  # Use the actual END_DATE from optimization
                'note': f'Optimized on data from {START_DATE} to {END_DATE} (inclusive based on load_data_from_mysql)'
            },
            'best_parameters': {
                'probability_threshold': float(best_config['probability_threshold']),
                'stop_loss': float(best_config['stop_loss']),
                'take_profit': float(best_config['target_profit']),
                'reward_risk_ratio': float(best_config['reward_risk_ratio']),
                'max_hold_bars': int(best_config['max_hold_bars'])
            },
            'performance_unfiltered': {
                'total_trades': int(best_config['total_trades']),
                'win_rate': float(best_config['win_rate']),
                'total_return': float(best_config['total_return']),
                'total_pnl': float(best_config['total_pnl']),
                'profit_factor': float(best_config['profit_factor']),
                'max_drawdown': float(best_config['max_drawdown']),
                'sharpe_ratio': float(best_config['sharpe_ratio'])
            }
        },
        'time_filters': time_filters,
        'trading_rules': {
            'position_size': POSITION_SIZE,
            'cost_model': 'bid_ask_spread',
            'trading_day_start': 'UTC+6',
            'sessions': {
                'asia': {
                    'timezone': 'Asia/Hong_Kong',
                    'hours': f"{ASIA_SESSION_START}:00 - {ASIA_SESSION_END}:00"
                },
                'ny': {
                    'timezone': 'America/New_York',
                    'hours': f"{NY_SESSION_START}:00 - {NY_SESSION_END}:00"
                }
            }
        }
    }

    # Add filtered performance if available
    if filtered_stats is not None:
        config['optimization']['performance_filtered'] = {
            'total_trades': int(filtered_stats['total_trades']),
            'win_rate': float(filtered_stats['win_rate']),
            'total_pnl': float(filtered_stats['total_pnl']),
            'avg_pnl': float(filtered_stats['avg_pnl']),
            'total_return': float(filtered_stats['total_pnl'] / 10000 * 100)  # Calculate from P&L
        }

        # Calculate profit factor if we have the data
        if orig_stats:
            # Note: Simplified profit factor calculation
            gross_profit = filtered_stats['total_pnl'] if filtered_stats['total_pnl'] > 0 else 0
            # Estimate gross loss based on win rate
            estimated_losers = filtered_stats['total_trades'] * (1 - filtered_stats['win_rate'] / 100)
            if estimated_losers > 0:
                avg_loss_estimate = (orig_stats['total_pnl'] - filtered_stats['total_pnl']) / estimated_losers if estimated_losers > 0 else 0
                gross_loss = abs(avg_loss_estimate * estimated_losers)
                config['optimization']['performance_filtered']['profit_factor'] = float(gross_profit / gross_loss if gross_loss > 0 else 999)

    config_file = 'ml_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Configuration saved to {config_file}")
    print()


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Run complete optimization workflow"""
    print()
    print("╔" + "=" * 118 + "╗")
    print("║" + " " * 35 + "ML UPTREND STRATEGY - COMPLETE OPTIMIZER" + " " * 42 + "║")
    print("╚" + "=" * 118 + "╝")
    print()
    print("This will run 3 steps:")
    print("  1. Optimize stop/target/hold parameters")
    print("  2. Find profitable time slots")
    print("  3. Apply and compare with time filters")
    print()

    try:
        # Step 1: Optimize
        best_config, trades_file = optimize_parameters()

        # Step 2: Find time slots
        time_filters = find_profitable_time_slots(trades_file)

        if time_filters is None:
            print("⚠️  Could not find time slots, skipping filter comparison")
            time_filters = {'asia_filters': {}, 'ny_filters': {}}

        # Step 3: Apply and compare
        orig_stats, filtered_stats = apply_and_compare(trades_file, time_filters)

        # Prompt to save configuration
        print("=" * 120)
        save_prompt = input("💾 Do you want to save this configuration to ml_config.json? (y/n): ").strip().lower()

        if save_prompt == 'y' or save_prompt == 'yes':
            save_configuration(best_config, time_filters, orig_stats, filtered_stats)
            print()
            print("✅ Configuration saved successfully!")
        else:
            print()
            print("ℹ️  Configuration not saved.")

        print()
        print("=" * 120)
        print("✅ OPTIMIZATION COMPLETE")
        print("=" * 120)
        print()
        print("Generated files:")
        print(f"  • ml_optimization_results.csv")
        print(f"  • {trades_file}")
        if save_prompt == 'y' or save_prompt == 'yes':
            print(f"  • ml_config.json")
        print()

    except KeyboardInterrupt:
        print()
        print("⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

