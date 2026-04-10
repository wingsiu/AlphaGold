#!/usr/bin/env python3
"""
ML UPTREND STRATEGY - COMPLETE OPTIMIZATION WORKFLOW
====================================================

Steps:
1. Optimize stop/target/hold parameters
2. Find profitable time slots (Asia HKT, NY)
3. Apply and compare with time filters

Usage (from project root):
    python3 training/ml_complete_optimizer.py

Predictions cache:
    Reads training/full_period_signals_from_training.csv (produced by training/training.py).
    Set PREDICTION_MODEL below to 'rf' or 'gb'.
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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data import DataLoader
try:
    from training.training import prepare_gold_data
except (ModuleNotFoundError, ImportError):
    from training import prepare_gold_data

# ============================================================================
# CONFIGURATION
# ============================================================================

START_DATE = '2025-05-20'
END_DATE   = '2026-02-04'

# Which model probability column to use from the signals cache.
# 'rf' -> rf_probability,  'gb' -> gb_probability
PREDICTION_MODEL = 'rf'

PREDICTIONS_CACHE = str(ROOT_DIR / 'training' / 'full_period_signals_from_training.csv')

INITIAL_CAPITAL = 1000
POSITION_SIZE   = 1.0

STOP_VALUES          = [15, 17, 20, 25]
TARGET_VALUES        = [30, 35, 40, 45, 50]
HOLD_TIME_VALUES     = [50, 60, 80, 120]
PROBABILITY_THRESHOLDS = [0.70, 0.65, 0.60, 0.55, 0.50]

MIN_TRADES_FOR_SLOT = 1
MIN_WIN_RATE  = 35.0
MIN_TOTAL_PNL = 0.0

ASIA_SESSION_START = 6
ASIA_SESSION_END   = 19
NY_SESSION_START   = 6
NY_SESSION_END     = 17


# ============================================================================
# ML UTILS (inlined — replaces old ml_utils import)
# ============================================================================

def calculate_statistics(df: pd.DataFrame, label: str) -> dict:
    """Calculate and print trade statistics for a trades dataframe."""
    pnl_col = 'pnl' if 'pnl' in df.columns else 'pnl_dollars'
    if len(df) == 0:
        stats = dict(total_trades=0, win_rate=0.0, total_pnl=0.0, avg_pnl=0.0)
    else:
        wins = (df[pnl_col] > 0).sum()
        stats = dict(
            total_trades=len(df),
            win_rate=wins / len(df) * 100,
            total_pnl=float(df[pnl_col].sum()),
            avg_pnl=float(df[pnl_col].mean()),
        )

    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(f"  Trades   : {stats['total_trades']}")
    print(f"  Win Rate : {stats['win_rate']:.1f}%")
    print(f"  Total P&L: ${stats['total_pnl']:.2f}")
    print(f"  Avg  P&L : ${stats['avg_pnl']:.2f}")
    return stats


def print_profitable_slots_summary(session_key: str, profitable_count: int, saved_filters: dict) -> None:
    """Print a summary of profitable time slots found for a session."""
    filters = saved_filters.get(session_key, {})
    tz_label = 'HKT' if session_key == 'asia' else 'NY'
    print(f"\n  Profitable {session_key.upper()} slots found: {profitable_count}")
    for day, hours in filters.items():
        if hours:
            hour_strs = ', '.join(f'{h:02d}:00 {tz_label}' for h in hours)
            print(f"    {day:10s}: {hour_strs}")


# ============================================================================
# DATA LOADING HELPER
# ============================================================================

def _load_price_df() -> pd.DataFrame:
    """Load price data using project DataLoader and return with bid/ask columns."""
    print("Loading price data...")
    raw = DataLoader().load_data('gold_prices', start_date=START_DATE, end_date=END_DATE)

    col = lambda ask, bid, mid_a, mid_b: (
        raw[ask] if ask in raw.columns else
        raw[bid] if bid in raw.columns else
        raw[mid_a] if mid_a in raw.columns else
        raw[mid_b]
    )

    df = pd.DataFrame({
        'open':      raw.get('openPrice',   raw.get('open')),
        'open_ask':  col('openPrice_ask',  'openPrice_bid',  'openPrice',  'open'),
        'open_bid':  col('openPrice_bid',  'openPrice_ask',  'openPrice',  'open'),
        'high':      raw.get('highPrice',   raw.get('high')),
        'high_ask':  col('highPrice_ask',  'highPrice_bid',  'highPrice',  'high'),
        'high_bid':  col('highPrice_bid',  'highPrice_ask',  'highPrice',  'high'),
        'low':       raw.get('lowPrice',    raw.get('low')),
        'low_ask':   col('lowPrice_ask',   'lowPrice_bid',   'lowPrice',   'low'),
        'low_bid':   col('lowPrice_bid',   'lowPrice_ask',   'lowPrice',   'low'),
        'close':     raw.get('closePrice',  raw.get('close')),
        'close_ask': col('closePrice_ask', 'closePrice_bid', 'closePrice', 'close'),
        'close_bid': col('closePrice_bid', 'closePrice_ask', 'closePrice', 'close'),
    })

    if 'timestamp' in raw.columns:
        df.index = pd.to_datetime(raw['timestamp'], unit='ms', utc=True)
    elif 'snapshotTimeUTC' in raw.columns:
        df.index = pd.to_datetime(raw['snapshotTimeUTC'], utc=True)

    print(f"✅ Loaded {len(df):,} bars  ({df.index.min().date()} → {df.index.max().date()})")
    return df


# ============================================================================
# STEP 1: OPTIMIZE PARAMETERS
# ============================================================================

def optimize_parameters():
    if not os.path.exists(PREDICTIONS_CACHE):
        print(f"❌ {PREDICTIONS_CACHE} not found!")
        print("   Run python3 training/training.py first to generate the signal cache.")
        sys.exit(1)

    print(f"Loading cached predictions from {PREDICTIONS_CACHE}...")
    predictions_raw = pd.read_csv(PREDICTIONS_CACHE)

    prob_col = 'rf_probability' if PREDICTION_MODEL == 'rf' else 'gb_probability'
    if prob_col not in predictions_raw.columns:
        raise ValueError(f"Column '{prob_col}' not found. Available: {list(predictions_raw.columns)}")

    predictions_df = predictions_raw[['idx', prob_col]].rename(columns={prob_col: 'probability'})
    print(f"✅ Loaded {len(predictions_df):,} cached predictions  (model={PREDICTION_MODEL})")

    df = _load_price_df()

    total_combos = (len(STOP_VALUES) * len(TARGET_VALUES) *
                    len(HOLD_TIME_VALUES) * len(PROBABILITY_THRESHOLDS))
    print(f"\nTesting {total_combos} combinations...")

    results = []
    combo_num = 0

    for prob_thresh in PROBABILITY_THRESHOLDS:
        print(f"\nProb threshold: {prob_thresh:.0%}")
        for stop in STOP_VALUES:
            for target in TARGET_VALUES:
                for hold_time in HOLD_TIME_VALUES:
                    combo_num += 1
                    result = run_backtest_optimized(predictions_df, df, prob_thresh, stop, target, hold_time)
                    if result:
                        results.append(result)
                        pct = combo_num / total_combos * 100
                        print(f"  [{combo_num}/{total_combos}] ({pct:.0f}%) "
                              f"Stop=${stop} Target=${target} Hold={hold_time} "
                              f"Trades:{result['total_trades']} "
                              f"P&L:${result['total_pnl']:.1f} "
                              f"WR:{result['win_rate']:.1f}%")
                        sys.stdout.flush()

    if not results:
        print("❌ No valid results found!")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    out_csv = str(ROOT_DIR / 'training' / 'ml_optimization_results.csv')
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved optimization results to {out_csv}")

    results_df['sharpe_ratio'] = (
        results_df['total_return'] / results_df['max_drawdown'].abs()
    ).replace([np.inf, -np.inf], 0)
    results_df['score'] = (results_df['total_return'] *
                           results_df['sharpe_ratio'] *
                           results_df['win_rate'] / 100)

    best = results_df.loc[results_df['score'].idxmax()]

    print("\n" + "=" * 80)
    print("🏆 BEST CONFIGURATION")
    print("=" * 80)
    print(f"  Prob Threshold : {best['probability_threshold']:.0%}")
    print(f"  Stop Loss      : ${best['stop_loss']:.0f}")
    print(f"  Take Profit    : ${best['target_profit']:.0f}")
    print(f"  RR Ratio       : {best['reward_risk_ratio']:.2f}:1")
    print(f"  Max Hold       : {best['max_hold_bars']:.0f} bars")
    print(f"  Trades         : {best['total_trades']:.0f}")
    print(f"  Win Rate       : {best['win_rate']:.1f}%")
    print(f"  Total P&L      : ${best['total_pnl']:.2f}")
    print(f"  Max Drawdown   : {best['max_drawdown']:.1f}%")
    print(f"  Sharpe Ratio   : {best['sharpe_ratio']:.2f}")

    trades_file = str(ROOT_DIR / 'training' /
                      f"ml_best_trades_prob{best['probability_threshold']:.0%}"
                      f"_stop{best['stop_loss']:.0f}"
                      f"_target{best['target_profit']:.0f}.csv")

    best_result = run_backtest_with_trades(
        predictions_df, df,
        float(best['probability_threshold']),
        float(best['stop_loss']),
        float(best['target_profit']),
        int(best['max_hold_bars']),
    )
    if best_result:
        best_result['trades_df'].to_csv(trades_file, index=False)
        print(f"✅ Saved best-config trades to {trades_file}")

    return best, trades_file


def run_backtest_optimized(predictions_df, df, probability_threshold, stop_loss, target_profit, max_hold_bars):
    signals = predictions_df[predictions_df['probability'] >= probability_threshold].copy()
    if len(signals) == 0:
        return None

    signal_indices  = signals['idx'].values
    highs_bid  = df['high_bid'].values
    lows_bid   = df['low_bid'].values
    opens_ask  = df['open_ask'].values
    closes_bid = df['close_bid'].values
    closes_ask = df['close_ask'].values

    stop_distance   = stop_loss   / POSITION_SIZE
    target_distance = target_profit / POSITION_SIZE
    signal_set = set(signal_indices)

    trades = []
    capital = INITIAL_CAPITAL
    signal_idx = 0
    current_bar = int(signal_indices[0])

    while signal_idx < len(signal_indices) and current_bar < len(df) - 1:
        entry_idx   = current_bar + 1
        entry_price = opens_ask[entry_idx]
        stop_price  = entry_price - stop_distance
        target_price= entry_price + target_distance
        end_idx     = min(entry_idx + max_hold_bars, len(df))

        exit_idx = exit_price = exit_reason = None

        for ci in range(entry_idx, end_idx):
            if lows_bid[ci] <= stop_price:
                exit_idx, exit_price, exit_reason = ci, stop_price, 'STOP'; break
            elif highs_bid[ci] >= target_price:
                exit_idx, exit_price, exit_reason = ci, target_price, 'TARGET'; break
            if ci in signal_set and ci != entry_idx:
                new_target = closes_ask[ci] + target_distance
                if new_target > target_price:
                    target_price = new_target

        if exit_idx is None:
            exit_idx   = end_idx - 1
            exit_price = closes_bid[exit_idx]
            exit_reason = 'TIME'

        pnl = (exit_price - entry_price) * POSITION_SIZE
        capital += pnl
        trades.append({'pnl_dollars': pnl, 'capital': capital, 'is_winner': pnl > 0})

        while signal_idx < len(signal_indices) and signal_indices[signal_idx] <= exit_idx:
            signal_idx += 1
        if signal_idx < len(signal_indices):
            current_bar = int(signal_indices[signal_idx])
        else:
            break

    if not trades:
        return None

    t = pd.DataFrame(trades)
    wins   = t[t['is_winner']]
    losers = t[~t['is_winner']]
    win_rate   = len(wins) / len(t) * 100
    total_pnl  = t['pnl_dollars'].sum()
    pf = abs(wins['pnl_dollars'].sum() / losers['pnl_dollars'].sum()) if len(losers) and losers['pnl_dollars'].sum() != 0 else float('inf')
    total_ret  = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    t['peak']  = t['capital'].cummax()
    t['dd_pct']= (t['capital'] - t['peak']) / t['peak'] * 100
    max_dd     = t['dd_pct'].min()

    return dict(probability_threshold=probability_threshold, stop_loss=stop_loss,
                target_profit=target_profit, reward_risk_ratio=target_profit/stop_loss,
                max_hold_bars=max_hold_bars, total_trades=len(t), win_rate=win_rate,
                total_pnl=total_pnl, profit_factor=pf, total_return=total_ret, max_drawdown=max_dd)


def run_backtest_with_trades(predictions_df, df, probability_threshold, stop_loss, target_profit, max_hold_bars):
    max_hold_bars = int(max_hold_bars)
    signals = predictions_df[predictions_df['probability'] >= probability_threshold].copy()
    if len(signals) == 0:
        return None

    signal_indices  = signals['idx'].values
    highs_bid  = df['high_bid'].values
    lows_bid   = df['low_bid'].values
    opens_ask  = df['open_ask'].values
    closes_bid = df['close_bid'].values
    closes_ask = df['close_ask'].values

    stop_distance   = stop_loss   / POSITION_SIZE
    target_distance = target_profit / POSITION_SIZE
    signal_set = set(signal_indices)

    trades = []
    capital = INITIAL_CAPITAL
    signal_idx = 0
    current_bar = int(signal_indices[0])

    while signal_idx < len(signal_indices) and current_bar < len(df) - 1:
        entry_idx    = current_bar + 1
        entry_price  = opens_ask[entry_idx]
        entry_time   = df.index[entry_idx]
        stop_price   = entry_price - stop_distance
        target_price = entry_price + target_distance
        end_idx      = min(entry_idx + max_hold_bars, len(df))

        exit_idx = exit_price = exit_reason = None

        for ci in range(entry_idx, end_idx):
            if lows_bid[ci] <= stop_price:
                exit_idx, exit_price, exit_reason = ci, stop_price, 'STOP'; break
            elif highs_bid[ci] >= target_price:
                exit_idx, exit_price, exit_reason = ci, target_price, 'TARGET'; break
            if ci in signal_set and ci != entry_idx:
                new_target = closes_ask[ci] + target_distance
                if new_target > target_price:
                    target_price = new_target

        if exit_idx is None:
            exit_idx    = end_idx - 1
            exit_price  = closes_bid[exit_idx]
            exit_reason = 'TIME'

        pnl = (exit_price - entry_price) * POSITION_SIZE
        capital += pnl
        trades.append(dict(
            entry_time=entry_time, exit_time=df.index[exit_idx],
            entry_price=entry_price, exit_price=exit_price,
            stop_price=stop_price, target_price=target_price,
            exit_reason=exit_reason, bars_held=exit_idx - entry_idx,
            pnl_dollars=pnl, cumulative_pnl=capital - INITIAL_CAPITAL,
            is_winner=pnl > 0,
        ))

        while signal_idx < len(signal_indices) and signal_indices[signal_idx] <= exit_idx:
            signal_idx += 1
        if signal_idx < len(signal_indices):
            current_bar = int(signal_indices[signal_idx])
        else:
            break

    return {'trades_df': pd.DataFrame(trades)} if trades else None


# ============================================================================
# STEP 2: FIND PROFITABLE TIME SLOTS
# ============================================================================

def find_profitable_time_slots(trades_file):
    print("\n" + "=" * 80)
    print("STEP 2: FIND PROFITABLE TIME SLOTS")
    print("=" * 80)

    df = pd.read_csv(trades_file)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    if df['entry_time'].dt.tz is None:
        df['entry_time'] = df['entry_time'].dt.tz_localize('UTC')
    if 'pnl_dollars' in df.columns and 'pnl' not in df.columns:
        df['pnl'] = df['pnl_dollars']

    print(f"✅ Loaded {len(df):,} trades")

    hkt_tz = pytz.timezone('Asia/Hong_Kong')
    ny_tz  = pytz.timezone('America/New_York')

    df['entry_hkt'] = df['entry_time'].dt.tz_convert(hkt_tz)
    df['entry_ny']  = df['entry_time'].dt.tz_convert(ny_tz)
    df['trading_day_hkt'] = df['entry_hkt'].apply(lambda x: (x - pd.Timedelta(hours=6)).date())
    df['trading_day_of_week'] = pd.to_datetime(df['trading_day_hkt']).dt.dayofweek
    df['trading_day_name']    = pd.to_datetime(df['trading_day_hkt']).dt.day_name()
    df = df[df['trading_day_of_week'] != 6].copy()
    df['hour_hkt'] = df['entry_hkt'].dt.hour
    df['hour_ny']  = df['entry_ny'].dt.hour

    def assign_session(row):
        if ASIA_SESSION_START <= row['hour_hkt'] <= ASIA_SESSION_END:
            return 'Asia'
        elif NY_SESSION_START <= row['hour_ny'] <= NY_SESSION_END:
            return 'NY'
        return 'Other'

    df['session'] = df.apply(assign_session, axis=1)

    saved_filters = {'asia': {}, 'ny': {}}
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    for session_name, time_col in [('Asia', 'hour_hkt'), ('NY', 'hour_ny')]:
        session_df  = df[df['session'] == session_name]
        session_key = session_name.lower()
        profitable_count = 0

        print(f"\nAnalyzing {session_name} session — {len(session_df):,} trades")
        for day in days_order:
            day_data = session_df[session_df['trading_day_name'] == day]
            day_profitable = []
            for hour in sorted(session_df[time_col].unique()):
                hd = day_data[day_data[time_col] == hour]
                if len(hd) >= MIN_TRADES_FOR_SLOT:
                    wr = (hd['pnl'] > 0).sum() / len(hd) * 100
                    tp = hd['pnl'].sum()
                    if wr > MIN_WIN_RATE and tp > MIN_TOTAL_PNL:
                        day_profitable.append(int(hour))
                        profitable_count += 1
            saved_filters[session_key][day] = day_profitable

        print_profitable_slots_summary(session_key, profitable_count, saved_filters)
        _print_session_heatmap(df[df['session'] == session_name], session_name,
                               'Asia/Hong_Kong' if session_name == 'Asia' else 'America/New_York',
                               ASIA_SESSION_START if session_name == 'Asia' else NY_SESSION_START,
                               ASIA_SESSION_END   if session_name == 'Asia' else NY_SESSION_END)

    config_file = str(ROOT_DIR / 'ml_config.json')
    config = json.load(open(config_file)) if os.path.exists(config_file) else {}
    config['time_filters'] = saved_filters
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✅ Saved time filters to {config_file}")
    return saved_filters


def _print_session_heatmap(df, session_name, timezone, session_start, session_end):
    tz   = pytz.timezone(timezone)
    dfc  = df.copy()
    dfc['session_time'] = dfc['entry_time'].dt.tz_convert(tz)
    dfc['session_hour'] = dfc['session_time'].dt.hour
    dfc['day_name']     = dfc['session_time'].dt.day_name()
    session_df = dfc[(dfc['session_hour'] >= session_start) & (dfc['session_hour'] <= session_end)]
    if len(session_df) == 0:
        return

    pnl_col = 'pnl' if 'pnl' in session_df.columns else 'pnl_dollars'
    days  = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    hours = sorted(session_df['session_hour'].unique())

    print(f"\n{'='*100}")
    print(f"  {session_name.upper()} SESSION HEATMAP — Win Rate %")
    print(f"{'='*100}")
    print(f"{'Day':<12}", end='')
    for h in hours:
        print(f"{h:02d}:00".rjust(7), end='')
    print()
    for day in days:
        dd = session_df[session_df['day_name'] == day]
        if len(dd) == 0:
            continue
        print(f"{day:<12}", end='')
        for h in hours:
            hd = dd[dd['session_hour'] == h]
            if len(hd) >= MIN_TRADES_FOR_SLOT:
                wr = (hd[pnl_col] > 0).sum() / len(hd) * 100
                print(f"{wr:6.1f}%", end='')
            else:
                print(f"{'--':>7}", end='')
        print()
    print()


# ============================================================================
# STEP 3: APPLY AND COMPARE WITH TIME FILTERS
# ============================================================================

def apply_and_compare(trades_file, time_filters):
    print("\n" + "=" * 80)
    print("STEP 3: APPLY TIME FILTERS AND COMPARE")
    print("=" * 80)

    df = pd.read_csv(trades_file)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    if df['entry_time'].dt.tz is None:
        df['entry_time'] = df['entry_time'].dt.tz_localize('UTC')
    if 'pnl_dollars' in df.columns and 'pnl' not in df.columns:
        df['pnl'] = df['pnl_dollars']

    hkt_tz = pytz.timezone('Asia/Hong_Kong')
    ny_tz  = pytz.timezone('America/New_York')
    df['entry_hkt'] = df['entry_time'].dt.tz_convert(hkt_tz)
    df['entry_ny']  = df['entry_time'].dt.tz_convert(ny_tz)
    df['trading_day_hkt'] = df['entry_hkt'].apply(lambda x: (x - pd.Timedelta(hours=6)).date())
    df['trading_day_name']= pd.to_datetime(df['trading_day_hkt']).dt.day_name()
    df['trading_day_of_week'] = pd.to_datetime(df['trading_day_hkt']).dt.dayofweek
    df = df[df['trading_day_of_week'] != 6].copy()
    df['hour_hkt'] = df['entry_hkt'].dt.hour
    df['hour_ny']  = df['entry_ny'].dt.hour

    def assign_session(row):
        if ASIA_SESSION_START <= row['hour_hkt'] <= ASIA_SESSION_END:
            return 'Asia'
        elif NY_SESSION_START <= row['hour_ny'] <= NY_SESSION_END:
            return 'NY'
        return 'Other'

    df['session'] = df.apply(assign_session, axis=1)

    orig_stats = calculate_statistics(df, "ORIGINAL (No Filters)")

    def passes_filter(row):
        day = row['trading_day_name']
        if row['session'] == 'Asia':
            return row['hour_hkt'] in time_filters.get('asia', {}).get(day, [])
        elif row['session'] == 'NY':
            return row['hour_ny']  in time_filters.get('ny',   {}).get(day, [])
        return False

    asia_f = df[(df['session'] == 'Asia') & df.apply(passes_filter, axis=1)]
    ny_f   = df[(df['session'] == 'NY')   & df.apply(passes_filter, axis=1)]
    filtered_df = pd.concat([asia_f, ny_f], ignore_index=True)

    calculate_statistics(asia_f,       "ASIA SESSION (Filtered)")
    calculate_statistics(ny_f,         "NY SESSION (Filtered)")
    filtered_stats = calculate_statistics(filtered_df, "COMBINED (Asia + NY Filtered)")

    print(f"\n{'='*80}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  Trades  : {len(df):,}  →  {len(filtered_df):,} ({len(filtered_df)/max(1,len(df))*100:.1f}%)")
    print(f"  Win Rate: {orig_stats['win_rate']:.1f}%  →  {filtered_stats['win_rate']:.1f}%  ({filtered_stats['win_rate']-orig_stats['win_rate']:+.1f}%)")
    print(f"  P&L     : ${orig_stats['total_pnl']:.2f}  →  ${filtered_stats['total_pnl']:.2f}")

    return orig_stats, filtered_stats


# ============================================================================
# SAVE CONFIGURATION
# ============================================================================

def save_configuration(best_config, time_filters, orig_stats=None, filtered_stats=None):
    config = {
        'optimization': {
            'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_period': {'start': START_DATE, 'end': END_DATE},
            'prediction_model': PREDICTION_MODEL,
            'best_parameters': {
                'probability_threshold': float(best_config['probability_threshold']),
                'stop_loss':   float(best_config['stop_loss']),
                'take_profit': float(best_config['target_profit']),
                'reward_risk_ratio': float(best_config['reward_risk_ratio']),
                'max_hold_bars': int(best_config['max_hold_bars']),
            },
            'performance_unfiltered': {
                'total_trades': int(best_config['total_trades']),
                'win_rate': float(best_config['win_rate']),
                'total_pnl': float(best_config['total_pnl']),
                'profit_factor': float(best_config['profit_factor']),
                'max_drawdown': float(best_config['max_drawdown']),
                'sharpe_ratio': float(best_config['sharpe_ratio']),
            },
        },
        'time_filters': time_filters,
        'trading_rules': {
            'position_size': POSITION_SIZE,
            'cost_model': 'bid_ask_spread',
            'trading_day_start': 'UTC+6',
            'sessions': {
                'asia': {'timezone': 'Asia/Hong_Kong',    'hours': f'{ASIA_SESSION_START}:00-{ASIA_SESSION_END}:00'},
                'ny':   {'timezone': 'America/New_York',  'hours': f'{NY_SESSION_START}:00-{NY_SESSION_END}:00'},
            },
        },
    }

    if filtered_stats:
        config['optimization']['performance_filtered'] = {
            'total_trades': int(filtered_stats['total_trades']),
            'win_rate': float(filtered_stats['win_rate']),
            'total_pnl': float(filtered_stats['total_pnl']),
            'avg_pnl':   float(filtered_stats['avg_pnl']),
        }

    config_file = str(ROOT_DIR / 'ml_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Configuration saved to {config_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n╔" + "=" * 78 + "╗")
    print("║" + "  ML UPTREND STRATEGY — COMPLETE OPTIMIZER".center(78) + "║")
    print("╚" + "=" * 78 + "╝\n")
    print("Steps: 1) Optimize params  2) Find time slots  3) Apply filters\n")

    try:
        best_config, trades_file = optimize_parameters()
        time_filters = find_profitable_time_slots(trades_file)
        if time_filters is None:
            time_filters = {'asia': {}, 'ny': {}}
        orig_stats, filtered_stats = apply_and_compare(trades_file, time_filters)

        print("\n" + "=" * 80)
        answer = input("💾 Save configuration to ml_config.json? (y/n): ").strip().lower()
        if answer in ('y', 'yes'):
            save_configuration(best_config, time_filters, orig_stats, filtered_stats)

        print("\n✅ OPTIMIZATION COMPLETE")
        print(f"   Results CSV : training/ml_optimization_results.csv")
        print(f"   Best trades : {trades_file}")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

