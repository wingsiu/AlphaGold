#!/usr/bin/env python3
"""
v18 Oil Daily Wick Analysis — Port of v16 to Oil
==================================================
Investigates daily candle wick structure for crude oil (prices table).
Skips intraday trade simulation (too slow), focuses on analytics:
  1. Daily wick sizes and structure
  2. Wick fill rate (same day, by threshold)
  3. All-day wick fill rate
  4. Wick → next day predictive power (correlation)
  5. Wick distribution by day of week
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from data.data_loader import DataLoader


def load_oil_data(start_date="2024-01-01", end_date="2026-05-22"):
    loader = DataLoader()
    raw = loader.load_data(table_name="prices", start_date=start_date, end_date=end_date)
    raw.index = pd.to_datetime(raw['timestamp'], unit='ms')

    df = pd.DataFrame(index=raw.index)
    df['open_ask'] = raw['openPrice_ask'].astype(float)
    df['high_bid'] = raw['highPrice_bid'].astype(float)
    df['low_bid'] = raw['lowPrice_bid'].astype(float)
    df['high_ask'] = raw['highPrice_ask'].astype(float)
    df['low_ask'] = raw['lowPrice_ask'].astype(float)
    df['close_ask'] = raw['closePrice_ask'].astype(float)
    df['close_bid'] = raw['closePrice_bid'].astype(float)

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def main():
    print("=" * 72)
    print("v18 Oil Daily Wick Analysis")
    print("=" * 72)

    df = load_oil_data()
    print(f"  {len(df):,} 1-min bars, {df.index[0].date()} -> {df.index[-1].date()}")

    # Daily OHLC
    daily = df.resample('D').agg({
        'open_ask': 'first',
        'high_ask': 'max', 'low_ask': 'min',
        'high_bid': 'max', 'low_bid': 'min',
        'close_ask': 'last', 'close_bid': 'last',
    }).dropna()

    daily['body_top'] = daily[['open_ask', 'close_ask']].max(axis=1)
    daily['body_bot'] = daily[['open_ask', 'close_ask']].min(axis=1)
    daily['upper_wick'] = daily['high_ask'] - daily['body_top']
    daily['lower_wick'] = daily['body_bot'] - daily['low_ask']
    daily['body_size'] = daily['body_top'] - daily['body_bot']
    daily['range'] = daily['high_ask'] - daily['low_ask']
    daily['upper_wick_pct'] = daily['upper_wick'] / (daily['range'] + 0.01) * 100
    daily['lower_wick_pct'] = daily['lower_wick'] / (daily['range'] + 0.01) * 100
    daily['body_pct'] = daily['body_size'] / (daily['range'] + 0.01) * 100

    n = len(daily)
    print(f"  {n} daily candles")

    # =========================================================================
    # 1. Wick size distribution
    # =========================================================================
    print(f"\n{'='*60}")
    print("DAILY WICK STRUCTURE")
    print(f"{'='*60}")
    for col, label in [('upper_wick', 'Upper wick'), ('lower_wick', 'Lower wick'),
                       ('body_size', 'Body'), ('range', 'Total range')]:
        vals = daily[col]
        print(f"  {label:<15s}: mean={vals.mean():>6.1f}, median={vals.median():>6.1f}, "
              f"std={vals.std():>5.1f}, min={vals.min():>5.1f}, max={vals.max():>5.1f}  (DB pts)")

    print(f"\n  Spot $ equivalent (÷100):")
    for col, label in [('upper_wick', 'Upper wick $'), ('lower_wick', 'Lower wick $'),
                       ('range', 'Range $')]:
        vals = daily[col] / 100
        print(f"  {label:<15s}: mean=${vals.mean():.3f}, max=${vals.max():.3f}")

    print(f"\n  Wick as % of range:")
    for col, label in [('upper_wick_pct', 'Upper wick %'), ('lower_wick_pct', 'Lower wick %'),
                       ('body_pct', 'Body %')]:
        vals = daily[col]
        print(f"  {label:<15s}: mean={vals.mean():>5.0f}%, median={vals.median():>5.0f}%")

    # =========================================================================
    # 2. Wick fill rate (same day — vectorized for speed)
    # =========================================================================
    print(f"\n{'='*60}")
    print("WICK FILL RATE (same day)")
    print(f"{'='*60}")

    # Pre-compute: for each day, find the low after the high and high after the low
    uw_fills = []; lw_fills = []
    for wick_pct_thresh in [25, 30, 35, 40]:
        large_uw = daily[daily['upper_wick_pct'] > wick_pct_thresh]
        large_lw = daily[daily['lower_wick_pct'] > wick_pct_thresh]
        uw_fill = 0; lw_fill = 0

        for day_idx, day_row in large_uw.iterrows():
            day_data = df[df.index.date == day_idx.date()]
            if len(day_data) < 10: continue
            high_time = day_data['high_ask'].idxmax()
            after_high = day_data[day_data.index > high_time]
            if len(after_high) < 5: continue
            fill_target = day_row['high_ask'] - day_row['upper_wick'] * 0.5
            if after_high['low_ask'].min() <= fill_target:
                uw_fill += 1

        for day_idx, day_row in large_lw.iterrows():
            day_data = df[df.index.date == day_idx.date()]
            if len(day_data) < 10: continue
            low_time = day_data['low_ask'].idxmin()
            after_low = day_data[day_data.index > low_time]
            if len(after_low) < 5: continue
            fill_target = day_row['low_ask'] + day_row['lower_wick'] * 0.5
            if after_low['high_ask'].max() >= fill_target:
                lw_fill += 1

        n_uw = max(len(large_uw), 1); n_lw = max(len(large_lw), 1)
        uw_fills.append(uw_fill / n_uw * 100)
        lw_fills.append(lw_fill / n_lw * 100)
        print(f"  Wick >{wick_pct_thresh}%: UW fill={uw_fill}/{len(large_uw)} ({uw_fill/n_uw*100:.0f}%), "
              f"LW fill={lw_fill}/{len(large_lw)} ({lw_fill/n_lw*100:.0f}%)")

    # =========================================================================
    # 3. Wick → next day predictive power
    # =========================================================================
    print(f"\n{'='*60}")
    print("WICK → NEXT DAY PREDICTIVE POWER")
    print(f"{'='*60}")
    next_day_move = daily['close_ask'].diff().shift(-1)
    for col in ['upper_wick', 'lower_wick', 'upper_wick_pct', 'lower_wick_pct']:
        corr = daily[col].corr(next_day_move)
        print(f"  corr({col}, next_day_move): {corr:+.3f}")

    # Test directional bias: after large lower wick (>40%), what % of next days close higher?
    large_lw = daily[daily['lower_wick_pct'] > 40]
    if len(large_lw) > 0:
        lw_next_day = daily['close_ask'].shift(-1)
        lw_next = lw_next_day[daily['lower_wick_pct'] > 40] - large_lw['close_ask']
        print(f"  Large lower wick (>40%): next day mean move={lw_next.mean():.1f}, "
              f"up={(lw_next > 0).mean()*100:.0f}%")

    large_uw = daily[daily['upper_wick_pct'] > 40]
    if len(large_uw) > 0:
        uw_next_day = daily['close_ask'].shift(-1)
        uw_next = uw_next_day[daily['upper_wick_pct'] > 40] - large_uw['close_ask']
        print(f"  Large upper wick (>40%): next day mean move={uw_next.mean():.1f}, "
              f"down={(uw_next < 0).mean()*100:.0f}%")

    # =========================================================================
    # 4. By day of week
    # =========================================================================
    print(f"\n{'='*60}")
    print("WICKS BY DAY OF WEEK")
    print(f"{'='*60}")
    daily['dow'] = daily.index.dayofweek
    for dow in range(5):
        sub = daily[daily['dow'] == dow]
        if len(sub) > 0:
            print(f"  {['Mon','Tue','Wed','Thu','Fri'][dow]}: n={len(sub)}, "
                  f"uw={sub['upper_wick'].mean():.1f}pts ({sub['upper_wick_pct'].mean():.0f}%), "
                  f"lw={sub['lower_wick'].mean():.1f}pts ({sub['lower_wick_pct'].mean():.0f}%)")

    # =========================================================================
    # 5. All-wick fill rate
    # =========================================================================
    print(f"\n{'='*60}")
    print("ALL-DAY WICK FILL RATE (≥50%)")
    print(f"{'='*60}")
    uw_total = 0; uw_filled = 0; lw_total = 0; lw_filled = 0
    for day_idx, day_row in daily.iterrows():
        day_data = df[df.index.date == day_idx.date()]
        if len(day_data) < 10: continue

        if day_row['upper_wick'] > 0:
            uw_total += 1
            high_time = day_data['high_ask'].idxmax()
            after_high = day_data[day_data.index > high_time]
            if len(after_high) >= 3:
                fill_target = day_row['high_ask'] - day_row['upper_wick'] * 0.5
                if after_high['low_ask'].min() <= fill_target:
                    uw_filled += 1

        if day_row['lower_wick'] > 0:
            lw_total += 1
            low_time = day_data['low_ask'].idxmin()
            after_low = day_data[day_data.index > low_time]
            if len(after_low) >= 3:
                fill_target = day_row['low_ask'] + day_row['lower_wick'] * 0.5
                if after_low['high_ask'].max() >= fill_target:
                    lw_filled += 1

    print(f"  Upper wick fill: {uw_filled}/{uw_total} ({uw_filled/max(uw_total,1)*100:.0f}%)")
    print(f"  Lower wick fill: {lw_filled}/{lw_total} ({lw_filled/max(lw_total,1)*100:.0f}%)")
    print(f"  Combined: {(uw_filled+lw_filled)/max(uw_total+lw_total,1)*100:.0f}%")

    # =========================================================================
    # 6. Comparison to gold v16
    # =========================================================================
    print(f"\n{'='*60}")
    print("OIL vs GOLD WICK COMPARISON")
    print(f"{'='*60}")
    print(f"  Gold (v16): 100% wick fill rate, 74% WR on lower-wick fade")
    print(f"  Oil (v18):  {max(lw_fills):.0f}% lower wick fill rate")
    print(f"  Oil wicks:  mean upper={daily['upper_wick'].mean():.1f}pts, lower={daily['lower_wick'].mean():.1f}pts")
    print(f"  Gold wicks: mean upper≈? pts, lower≈? pts (from v16)")
    print(f"  Implication: same wick-fill dynamic, scaling needed for entry triggers")

    print(f"\nDONE.")


if __name__ == '__main__':
    main()
