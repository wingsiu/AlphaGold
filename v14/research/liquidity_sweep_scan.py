#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
from tqdm import tqdm

# Add project root to sys.path
from v14._paths import PROJECT_ROOT

from data.data_loader import DataLoader

def run_liquidity_sweep_scan():
    print("--- Starting Liquidity Sweep Scan (UTC+2 Day Start) ---")

    # 1. Load Data
    dl = DataLoader()
    # Loading last 30 days of data for the scan
    df = dl.load_data(table_name="gold_prices", start_date="2026-04-01")
    if df.empty:
        print("No data found.")
        return

    # Create DateTime index from timestamp column (assuming ms as in data_loader.py)
    if 'timestamp' in df.columns:
        df.index = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    elif 'time' in df.columns:
        df.index = pd.to_datetime(df['time'], utc=True)

    # Convert to UTC+2
    UTC2 = ZoneInfo("Etc/GMT-2") # Etc/GMT-2 is UTC+2
    df.index = df.index.tz_convert(UTC2)

    # 2. Session Definitions (from image_trend_ml.py or standard)
    # Asian Session: roughly 08:00 - 16:00 HKT
    # London: 08:00 - 16:30 London
    # NY: 09:30 - 16:00 NY

    HK_TZ = ZoneInfo("Asia/Hong_Kong")
    LONDON_TZ = ZoneInfo("Europe/London")
    NY_TZ = ZoneInfo("America/New_York")

    SWEEP_TOLERANCE = 2.0  # $2 tolerance for "Small Tolerance" as requested

    # 3. Process by Day (starting at 00:00 UTC+2)
    daily_groups = df.groupby(df.index.date)

    results = []

    for day, day_df in tqdm(daily_groups):
        if len(day_df) < 100: continue

        # Identify "Asian Range" - relative to day start in UTC+2
        # Typically Asian session is early in the UTC+2 day.
        # Let's define Asian Session as 00:00 to 07:00 UTC+2 for this liquidity check
        asian_range_df = day_df.between_time("00:00", "07:00")
        if asian_range_df.empty: continue

        asian_high = asian_range_df['highPrice'].max()
        asian_low = asian_range_df['lowPrice'].min()

        # Scan for sweeps in London/NY hours (approx 08:00 - 20:00 UTC+2)
        monitoring_df = day_df.between_time("07:01", "21:00")

        for ts, row in monitoring_df.iterrows():
            # SSL Sweep (Long Opportunity)
            if row['lowPrice'] < (asian_low - SWEEP_TOLERANCE):
                results.append({
                    'time': ts,
                    'type': 'SSL_SWEEP (Below Low)',
                    'price': row['closePrice'],
                    'asian_low': asian_low,
                    'sweep_depth': asian_low - row['lowPrice']
                })

            # BSL Sweep (Short Opportunity)
            if row['highPrice'] > (asian_high + SWEEP_TOLERANCE):
                results.append({
                    'time': ts,
                    'type': 'BSL_SWEEP (Above High)',
                    'price': row['closePrice'],
                    'asian_high': asian_high,
                    'sweep_depth': row['highPrice'] - asian_high
                })

    if not results:
        print("No liquidity sweeps detected with current parameters.")
    else:
        rdf = pd.DataFrame(results)
        print("\n--- Detected Liquidity Sweeps ---")
        print(rdf.tail(20))

        out_path = PROJECT_ROOT / "liquidity_sweep_test.csv"
        rdf.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    run_liquidity_sweep_scan()

