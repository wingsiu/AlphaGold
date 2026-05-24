import pandas as pd
import numpy as np
import sys
from pathlib import Path

from v14._paths import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from xgboost_filter_model.train_filter_1min import load_price_data

print("Loading 1 year of data...")
# Load roughly 1 year of data
df = load_price_data(start_date='2025-05-01', end_date='2026-05-21')

# Ensure index is datetime
df.index = pd.to_datetime(df.index)

# Convert to NY time to align with trading days
df_ny = df.copy()
if df_ny.index.tzinfo is None:
    df_ny.index = df_ny.index.tz_localize('UTC')
df_ny.index = df_ny.index.tz_convert('America/New_York')

# Resample to daily
daily = df_ny.resample('B').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

daily['range'] = daily['high'] - daily['low']
daily['net_move'] = (daily['close'] - daily['open']).abs()
daily['dir_move_up'] = daily['high'] - daily['open']
daily['dir_move_down'] = daily['open'] - daily['low']
daily['max_dir_move'] = daily[['dir_move_up', 'dir_move_down']].max(axis=1)

print("\n=== Daily Gold Move Analysis (NY Trading Days) ===")
print(f"Total Trading Days Analyzed: {len(daily)}")
print(f"Average Daily Range (High - Low): {daily['range'].mean():.2f}")
print(f"Median Daily Range: {daily['range'].median():.2f}")
print(f"75th Percentile Range: {daily['range'].quantile(0.75):.2f}")
print(f"90th Percentile Range: {daily['range'].quantile(0.90):.2f}")
print("-" * 40)
print(f"% of days with Range >= 60: {(daily['range'] >= 60).mean() * 100:.1f}%")
print(f"% of days with Range >= 50: {(daily['range'] >= 50).mean() * 100:.1f}%")
print(f"% of days with Range >= 40: {(daily['range'] >= 40).mean() * 100:.1f}%")
print(f"% of days with Range >= 30: {(daily['range'] >= 30).mean() * 100:.1f}%")
print("-" * 40)
print(f"Average Max Directional Move from Open: {daily['max_dir_move'].mean():.2f}")
print(f"Median Max Directional Move from Open: {daily['max_dir_move'].median():.2f}")
print(f"% of days with Max Dir Move >= 60: {(daily['max_dir_move'] >= 60).mean() * 100:.1f}%")
print(f"% of days with Max Dir Move >= 40: {(daily['max_dir_move'] >= 40).mean() * 100:.1f}%")
print(f"% of days with Max Dir Move >= 30: {(daily['max_dir_move'] >= 30).mean() * 100:.1f}%")

