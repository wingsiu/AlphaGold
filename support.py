#!/usr/bin/env python3
"""
Support Detection and Trading Utilities
- Shared functions for support detection
- Shared configuration loading
- Shared data loading
- No look-ahead bias
"""

import json
from pathlib import Path
import pandas as pd
import pytz
import warnings
import mysql.connector
import os
from dotenv import load_dotenv

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='.*SQLAlchemy.*')

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME', 'gold_data'),
}


def load_data_from_mysql(start_date='2025-05-01', end_date='2026-02-04', verbose=True):
    """
    Load gold 1-minute bar data from MySQL database

    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format (default: 2025-05-01)
    end_date : str
        End date in YYYY-MM-DD format (default: 2026-02-04)
    verbose : bool
        Whether to print loading information (default: True)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: datetime, open, high, low, close
        Index: datetime (datetime64[ns, UTC])
    """
    conn = mysql.connector.connect(**DB_CONFIG)

    query = """
    SELECT timestamp, openPrice, highPrice, lowPrice, closePrice, lastTradedVolume
    FROM gold_prices
    ORDER BY timestamp
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Convert timestamp to datetime with UTC timezone
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close','volume', 'datetime']

    # Filter to specified period
    start = pd.Timestamp(start_date, tz='UTC')
    end = pd.Timestamp(end_date, tz='UTC')
    df = df[(df['datetime'] >= start) & (df['datetime'] <= end)].copy()

    # Add UTC+6 columns for day grouping
    from datetime import timedelta
    df['datetime_utc6'] = df['datetime'] + timedelta(hours=6)
    df['date_utc6'] = df['datetime_utc6'].dt.date

    if verbose:
        print(f"✅ Loaded {len(df)} 1-minute bars")
        print(f"📅 Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")

    return df

def load_config(config_path='support_bounce_config.json', verbose=True):
    """Load configuration from JSON file."""
    if not os.path.isabs(config_path):
        config_path = str(Path(__file__).resolve().parent / config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    if verbose:
        print("="*80)
        print("CONFIGURATION LOADED")
        print("="*80)
        print()
        print(f"Strategy: {config['strategy_name']}")
        print(f"Period: {config['period']['start_date']} to {config['period']['end_date']}")
        print()
        print("Support Detection:")
        print(f"  Min 15-min range: ${config['support_detection']['min_15min_range']}")
        print(f"  Min lower wick %: {config['support_detection']['min_lower_wick_pct']}%")
        print(f"  Min reversal size: ${config['support_detection']['min_reversal_size']}")
        print(f"  Check FVG: {config['support_detection']['check_fvg']}")
        if config['support_detection']['check_fvg']:
            min_gap = config['support_detection'].get('min_fvg_gap', 0.5)
            print(f"  Min FVG gap: ${min_gap}")
        print()
        print("Entry/Exit:")
        print(f"  Entry zone: Support + ${config['entry']['entry_zone_min']} to ${config['entry']['entry_zone_max']}")
        print(f"  Entry price: Support + ${config['entry']['entry_offset']}")
        print(f"  Stop: Support - ${config['exit']['stop_loss']}")
        print(f"  Max hold: {config['exit']['max_hold_bars']} bars")
        print()
        print("Targets:")
        print(f"  Fixed Target: ${config['exit']['fixed_target']}")
        print(f"  Use Dynamic: {config['exit']['use_dynamic_target']}")
        if config['exit']['use_dynamic_target']:
            print(f"  Dynamic - Range thresholds: {config['dynamic_target']['range_thresholds']}")
            print(f"  Dynamic - Targets: {config['dynamic_target']['targets']}")
        print()

    return config

def detect_support_formations(df_day, config, verbose=False):
    """
    Detect support formations - NO LOOK-AHEAD BIAS (OPTIMIZED)

    FIX: Only use data UP TO the reversal bar (no future data)
    - Build 15-min bars incrementally using rolling windows
    - At each point, only use bars that existed at that time
    - Added MA filter: MA7 > MA30 (bullish trend confirmation)

    Args:
        df_day: DataFrame with 1-minute bars for one day
        config: Configuration dict from load_config()
        verbose: If True, print detection info

    Returns:
        List of support dictionaries
    """
    # Get parameters from config
    min_range = config['support_detection']['min_15min_range']
    min_wick_pct = config['support_detection']['min_lower_wick_pct']
    min_reversal = config['support_detection']['min_reversal_size']
    check_fvg = config['support_detection']['check_fvg']
    min_fvg_gap = config['support_detection'].get('min_fvg_gap', 0.5)  # Default $0.5 if not set

    supports = []

    # Get day info for verbose output
    summary_day = df_day.iloc[0]['datetime'].strftime('%Y-%m-%d') if len(df_day) > 0 else 'unknown'

    # We need at least 45 minutes of data (3 x 15-min bars) to detect a support
    # Plus 30 bars for MA30 calculation
    if len(df_day) < 75:  # 45 for pattern + 30 for MA30
        return []

    # Calculate moving averages on the close prices (no look-ahead bias)
    # MA7 and MA30 are calculated incrementally
    df_day = df_day.copy()
    df_day['ma7'] = df_day['close'].rolling(window=7, min_periods=7).mean()
    df_day['ma30'] = df_day['close'].rolling(window=30, min_periods=30).mean()

    # Helper function to create 15-min bar from 1-min bars
    def make_15min_bar(bars):
        return {
            'datetime': bars.iloc[-1]['datetime'],  # End time
            'open': bars.iloc[0]['open'],
            'high': bars['high'].max(),
            'low': bars['low'].min(),
            'close': bars.iloc[-1]['close'],
            'volume' : bars['volume'].sum()
        }

    # Process in 15-minute windows, checking for support at each step
    # Naming convention:
    # b0 = first 15-min bar (context before the drop)
    # b1 = middle 15-min bar (the sharp drop with wick OR the FVG bar)
    # b2 = third 15-min bar (reversal confirmation)
    for start_idx in range(0, len(df_day) - 44, 15):
        # Get 3 consecutive 15-min bars (45 minutes total)
        b0_bars = df_day.iloc[start_idx:start_idx+15]
        b1_bars = df_day.iloc[start_idx+15:start_idx+30]
        b2_bars = df_day.iloc[start_idx+30:start_idx+45]

        # Need all 3 complete bars
        if len(b0_bars) < 15 or len(b1_bars) < 15 or len(b2_bars) < 15:
            continue

        # Create 15-min OHLC bars
        b0 = make_15min_bar(b0_bars)
        b1 = make_15min_bar(b1_bars)
        b2 = make_15min_bar(b2_bars)

        # Calculate metrics for b1 (the potential support bar)
        b1['range'] = b1['high'] - b1['low']
        b1['lower_wick'] = min(b1['open'], b1['close']) - b1['low']
        b1['lower_wick_pct'] = (b1['lower_wick'] / b1['range'] * 100) if b1['range'] > 0 else 0

        # FVG Detection: Gap between b0 and b2
        # FVG definition: b2.low > b0.high (bullish FVG - gap exists)
        # Trading zone: within the gap [b0.high, b2.low]
        fvg_size = 0.0
        fvg_present = False
        fvg_gap_low = 0.0
        fvg_gap_high = 0.0

        if check_fvg and b2['low'] > b0['high']:
            gap = b2['low'] - b0['high']
            if gap >= min_fvg_gap and (b1['volume'] > 1000):  # Only count if gap >= threshold (default $0.5)
                fvg_size = gap
                fvg_present = True
                fvg_gap_low = b0['high']  # Bottom of gap
                fvg_gap_high = b2['low']  # Top of gap

        # Check if b1 range meets threshold (only for long wick, not for FVG)
        # FVG doesn't need b1 to have large range - the gap is what matters

        # Support formation criteria:
        # 1. Long wick support: b1 has significant lower wick AND range >= threshold
        # 2. FVG support: gap exists between b0 and b2 (b1 range not important)
        is_long_wick = (b1['lower_wick_pct'] > min_wick_pct) and (b1['range'] >= min_range) and (b1['volume'] > 3000) #and (b1['close'] < b1['open'])
        has_fvg = fvg_present

        # FVG is valid if gap exists and meets threshold
        # No additional validation needed

        # Handle when BOTH long wick and FVG are present:
        # Priority: Prefer long wick (more reliable), but mark FVG presence for tracking
        # This way we don't create duplicate signals for the same formation
        if is_long_wick or has_fvg:
            # For long wick: reversal must close above b1.low + min_reversal
            # For FVG: reversal confirmed when b2 closes above the gap
            reversal_confirmed = False
            support_level = 0.0
            formation_type = 'long_wick'  # Default to long wick

            if is_long_wick:
                # Long wick support: use b1's low
                support_level = b1['low']
                if b2['close'] > b1['low'] + min_reversal:
                    reversal_confirmed = True
                formation_type = 'long_wick'
            elif has_fvg:
                # FVG support ONLY if no long wick
                # Use the gap midpoint as support level
                support_level = (fvg_gap_low + fvg_gap_high) / 2
                # For FVG, just need b2 to close above the gap bottom (already confirmed by bullish bars)
                # No additional reversal requirement since the gap + bullish bars already confirm bounce
                if b2['close'] > fvg_gap_low:
                    reversal_confirmed = True
                formation_type = 'fvg'

            if reversal_confirmed:
                signal_time = b2['datetime']  # Signal AFTER b2 completes

                # CHECK MA FILTER: MA7 > MA30 at signal time (bullish trend)
                # Get the last bar index in b2_bars to check MA values
                signal_bar_idx = start_idx + 44  # Last bar of b2 (0-based)

                # Ensure we have MA values at this point
                if pd.isna(df_day.iloc[signal_bar_idx]['ma7']) or pd.isna(df_day.iloc[signal_bar_idx]['ma30']):
                    continue  # Skip if MA not available yet

                ma7_value = df_day.iloc[signal_bar_idx]['ma7']
                ma30_value = df_day.iloc[signal_bar_idx]['ma30']

                # Only accept signal if MA7 > MA30 (bullish trend)
                if ma7_value + 4 <= ma30_value:
                    continue  # Skip this support - not in bullish trend

                # Add NY hour
                ny_tz = pytz.timezone('America/New_York')
                dt_ny = signal_time.tz_convert(ny_tz)

                # Check if we already added this support (avoid duplicates)
                if not any(abs((s['datetime'] - signal_time).total_seconds()) < 60 for s in supports):
                    support_info = {
                        'datetime': signal_time,
                        'support_level': support_level,
                        'range': b1['range'],
                        'lower_wick_pct': b1['lower_wick_pct'],
                        'fvg_present': has_fvg,  # Track if FVG was also present
                        'fvg_size': fvg_size,
                        'fvg_gap_low': fvg_gap_low if has_fvg else 0.0,
                        'fvg_gap_high': fvg_gap_high if has_fvg else 0.0,
                        'formation_type': formation_type,
                        'hour_ny': dt_ny.hour,
                        'reversal_size': b2['close'] - support_level
                    }

                    supports.append(support_info)

    # Print summary if verbose
    if verbose and len(supports) > 0:
        print(f"  {summary_day}: Found {len(supports)} support(s)")
        for s in supports:
            if s['formation_type'] == 'fvg':
                print(f"    • ${s['support_level']:.2f} at {s['datetime'].strftime('%H:%M')} "
                      f"(FVG Gap: ${s['fvg_gap_low']:.2f}-${s['fvg_gap_high']:.2f}, "
                      f"Size: ${s['fvg_size']:.2f})")
            else:
                print(f"    • ${s['support_level']:.2f} at {s['datetime'].strftime('%H:%M')} "
                      f"(Range: ${s['range']:.2f}, Wick: {s['lower_wick_pct']:.1f}%)")

    return supports


def calculate_last_15min_range(df, idx):
    """
    Calculate range of last 15 minutes before index

    Parameters:
    -----------
    df : pandas.DataFrame
        Price data
    idx : int
        Current index position

    Returns:
    --------
    float or None
        Range (high - low) of last 15 bars, or None if not enough data
    """
    if idx < 15:
        return None

    last_15 = df.iloc[idx-15:idx]
    return last_15['high'].max() - last_15['low'].min()


def get_dynamic_target(last_15min_range, config):
    """
    Get dynamic target based on last 15-min range

    Parameters:
    -----------
    last_15min_range : float or None
        Range of last 15 minutes
    config : dict
        Configuration dictionary

    Returns:
    --------
    float
        Target amount in dollars
    """
    if last_15min_range is None:
        return config['exit']['fixed_target']

    thresholds = config['dynamic_target']['range_thresholds']
    targets = config['dynamic_target']['targets']

    # Use thresholds to determine target
    for i, threshold in enumerate(thresholds):
        if last_15min_range < threshold:
            return targets[i]

    # If range exceeds all thresholds, use last target
    return targets[-1]


def detect_fake_down_signals(df_day, verbose=False):
    """
    Detect "fake down" signals on 1-minute gold data

    Signal criteria:
    1. Change of first 20 mins < -20
    2. Change for last 3 bars > 5
    3. Average volume of last 5 bars > 1.1 * average volume of last 30 bars

    Args:
        df_day: DataFrame with 1-minute bars for one day
        verbose: If True, print detection info

    Returns:
        List of signal dictionaries
    """
    signals = []

    # Get day info for verbose output
    summary_day = df_day.iloc[0]['datetime'].strftime('%Y-%m-%d') if len(df_day) > 0 else 'unknown'

    if len(df_day) < 30:
        return []

    # Start checking from bar 30 onwards (need 20 bars for first criterion, 30 for volume)
    for i in range(30, len(df_day)):
        # Criterion 1: Change of first 20 mins < -20
        if i < 20:
            continue

        first_20_bars = df_day.iloc[i-19:i+1]  # Last 20 bars including current
        change_20mins = first_20_bars.iloc[-1]['close'] - first_20_bars.iloc[0]['open']

        if change_20mins >= -20:
            continue

        # Criterion 2: Change for last 3 bars > 5
        if i < 2:
            continue

        last_3_bars = df_day.iloc[i-2:i+1]  # Last 3 bars including current
        change_3bars = last_3_bars.iloc[-1]['close'] - last_3_bars.iloc[0]['open']

        if change_3bars <= 5:
            continue

        # Criterion 3: Average volume of last 5 bars > 1.1 * average volume of last 30 bars
        if i < 5:
            continue

        last_5_bars = df_day.iloc[i-4:i+1]  # Last 5 bars including current
        avg_vol_5 = last_5_bars['volume'].mean()

        last_30_bars = df_day.iloc[i-29:i+1]  # Last 30 bars including current
        avg_vol_30 = last_30_bars['volume'].mean()

        if avg_vol_5 <= 1.1 * avg_vol_30:
            continue

        # All criteria met - we have a signal
        signal_time = df_day.iloc[i]['datetime']
        signal_price = df_day.iloc[i]['close']

        # Add NY hour
        ny_tz = pytz.timezone('America/New_York')
        dt_ny = signal_time.tz_convert(ny_tz)

        # Check if we already added this signal (avoid duplicates within 5 minutes)
        if not any(abs((s['datetime'] - signal_time).total_seconds()) < 300 for s in signals):
            signal_info = {
                'datetime': signal_time,
                'price': signal_price,
                'change_20mins': change_20mins,
                'change_3bars': change_3bars,
                'avg_vol_5': avg_vol_5,
                'avg_vol_30': avg_vol_30,
                'vol_ratio': avg_vol_5 / avg_vol_30 if avg_vol_30 > 0 else 0,
                'hour_ny': dt_ny.hour
            }

            signals.append(signal_info)

    # Print summary if verbose
    if verbose and len(signals) > 0:
        print(f"  {summary_day}: Found {len(signals)} fake down signal(s)")
        for s in signals:
            print(f"    • ${s['price']:.2f} at {s['datetime'].strftime('%H:%M')} "
                  f"(20min: ${s['change_20mins']:.2f}, 3bar: ${s['change_3bars']:.2f}, "
                  f"Vol ratio: {s['vol_ratio']:.2f})")

    return signals


def backtest_trade(df, support, config, use_dynamic_target=False):
    """
    Backtest a single trade from a support level

    Args:
        df: DataFrame with 1-minute bars (sorted by datetime)
        support: Support dict from detect_support_formations()
        config: Configuration dict
        use_dynamic_target: Boolean, whether to use dynamic or fixed target

    Returns:
        Dict with trade results or None if no valid entry
    """
    formation_time = support['datetime']
    support_level = support['support_level']

    # Entry parameters
    entry_offset = config['entry']['entry_offset']
    entry_zone_max = config['entry']['entry_zone_max']

    # For FVG supports, use the gap zone as entry zone
    # For long wick supports, use traditional offset from support
    is_fvg = support.get('formation_type', 'long_wick') == 'fvg'

    if is_fvg and support.get('fvg_gap_low', 0) > 0:
        # FVG entry zone: anywhere within the gap
        entry_zone_min_price = support['fvg_gap_low']
        entry_zone_max_price = support['fvg_gap_high']
    else:
        # Long wick entry zone: traditional approach
        entry_zone_min_price = support_level + config['entry']['entry_zone_min']
        entry_zone_max_price = support_level + config['entry']['entry_zone_max']

    # Exit parameters (stop will be calculated after we know actual entry price)
    max_hold = config['exit']['max_hold_bars']

    # Find index after formation
    formation_ts = pd.Timestamp(formation_time)
    if formation_ts.tz is None:
        formation_ts = formation_ts.tz_localize('UTC')

    mask = df['datetime'] > formation_ts
    if not mask.any():
        return None

    start_idx = mask.idxmax()

    # Look for entry opportunity
    # Entry when price touches the entry zone (calculated above based on formation type)
    end_idx = min(start_idx + 60, len(df))
    entry_zone = df.iloc[start_idx:end_idx]


    # Entry condition: bar touches the entry zone
    # We enter when: low <= entry_zone_max AND high >= entry_zone_min
    # This means the bar overlaps with our entry zone
    entry_mask = (entry_zone['low'] <= entry_zone_max_price) & (entry_zone['high'] >= entry_zone_min_price)
    if not entry_mask.any():
        return None

    signal_idx = entry_zone[entry_mask].index[0]

    # IMPORTANT: Enter at NEXT bar after signal to avoid look-ahead bias
    # We can't enter on the same bar we detect the entry signal
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None

    entry_bar = df.loc[entry_idx]

    # CRITICAL FIX: Use actual entry bar's OPEN price, not theoretical price
    # This matches real trading where you enter at market on the next bar
    entry_price = entry_bar['open']

    # Calculate stop price based on ACTUAL entry price
    # Stop is placed X dollars below where we actually entered
    stop_price = entry_price - config['exit']['stop_loss']

    # Calculate last 15-min range for dynamic target
    last_15min_range = calculate_last_15min_range(df, entry_idx)

    if use_dynamic_target:
        target = get_dynamic_target(last_15min_range, config)
    else:
        target = config['exit']['fixed_target']

    target_price = entry_price + target

    # Get bars after entry (starting from NEXT bar after entry)
    # Entry and exit cannot happen on the same bar
    first_exit_check_idx = entry_idx + 1
    bars_after_entry = df.iloc[first_exit_check_idx:min(first_exit_check_idx + max_hold, len(df))]

    if len(bars_after_entry) == 0:
        return None

    # Check what happens
    for i, (idx, bar) in enumerate(bars_after_entry.iterrows()):
        bars_held = i + 1  # At least 1 bar held (can't exit on entry bar)

        # Check if both stop and target hit in same bar - treat as stop (conservative)
        if bar['low'] <= stop_price and bar['high'] >= target_price:
            pnl = stop_price - entry_price
            return {
                'entry_time': entry_bar['datetime'],
                'exit_time': bar['datetime'],
                'entry_price': entry_price,
                'exit_price': stop_price,
                'pnl': pnl,
                'bars_held': bars_held,
                'exit_reason': 'stop',
                'target': target,
                'last_15min_range': last_15min_range,
                'support_range': support['range'],
                'formation_type': support['formation_type']
            }

        # Check stop
        if bar['low'] <= stop_price:
            pnl = stop_price - entry_price
            return {
                'entry_time': entry_bar['datetime'],
                'exit_time': bar['datetime'],
                'entry_price': entry_price,
                'exit_price': stop_price,
                'pnl': pnl,
                'bars_held': bars_held,
                'exit_reason': 'stop',
                'target': target,
                'last_15min_range': last_15min_range,
                'support_range': support['range'],
                'formation_type': support['formation_type']
            }

        # Check target
        if bar['high'] >= target_price:
            pnl = target
            return {
                'entry_time': entry_bar['datetime'],
                'exit_time': bar['datetime'],
                'entry_price': entry_price,
                'exit_price': target_price,
                'pnl': pnl,
                'bars_held': bars_held,
                'exit_reason': 'target',
                'target': target,
                'last_15min_range': last_15min_range,
                'support_range': support['range'],
                'formation_type': support['formation_type']
            }

    # Timeout
    last_bar = bars_after_entry.iloc[-1]
    exit_price = last_bar['close']
    pnl = exit_price - entry_price

    return {
        'entry_time': entry_bar['datetime'],
        'exit_time': last_bar['datetime'],
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl': pnl,
        'bars_held': len(bars_after_entry),
        'exit_reason': 'timeout',
        'target': target,
        'last_15min_range': last_15min_range,
        'support_range': support['range'],
        'formation_type': support['formation_type']
    }

