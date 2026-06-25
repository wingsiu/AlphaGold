#!/usr/bin/env python3
"""
v15 Daily Range Potential — What's Actually Capturable?
=========================================================
Checks: if daily range is ~$80 in May-June 2026, why can't we capture $10/day?

Tests:
  1. Daily OHLC stats: open, high, low, close, range, spread
  2. Perfect foresight: best possible long/short on each day (hindsight PnL)
  3. Open-to-close: simple buy-open sell-close
  4. Range fade with perfect timing: buy exact low, sell exact high
  5. 1-min bar distribution: how many bars move >0.5 pts vs noise
  
Goal: identify the structural friction preventing $10/day capture.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from data.data_loader import DataLoader


def load_data(start_date="2026-05-01", end_date="2026-06-09"):
    loader = DataLoader()
    raw = loader.load_data(table_name="gold_prices", start_date=start_date, end_date=end_date)
    raw.index = pd.to_datetime(raw['timestamp'], unit='ms')
    df = pd.DataFrame(index=raw.index)
    df['open_ask'] = raw['openPrice_ask'].astype(float)
    df['high_bid'] = raw['highPrice_bid'].astype(float)
    df['low_bid'] = raw['lowPrice_bid'].astype(float)
    df['high_ask'] = raw['highPrice_ask'].astype(float)
    df['low_ask'] = raw['lowPrice_ask'].astype(float)
    df['close_ask'] = raw['closePrice_ask'].astype(float)
    df['close_bid'] = raw['closePrice_bid'].astype(float)
    df['close'] = df['close_ask']
    df['spread'] = df['close_ask'] - df['close_bid']
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def main():
    print("=" * 72)
    print("v15 Daily Range Potential — May-June 2026")
    print("=" * 72)

    df = load_data()
    print(f"  {len(df):,} 1-min bars, {df.index[0]} → {df.index[-1]}")

    # Daily resample
    daily = df.resample('D').agg({
        'open_ask': 'first',
        'high_bid': 'max',
        'low_bid': 'min',
        'high_ask': 'max',
        'low_ask': 'min',
        'close_ask': 'last',
        'close_bid': 'last',
        'spread': 'mean',
    }).dropna()

    daily['range_ask'] = daily['high_ask'] - daily['low_ask']       # full ask range
    daily['range_bid'] = daily['high_bid'] - daily['low_bid']        # full bid range
    daily['open_close'] = daily['close_ask'] - daily['open_ask']    # buy open sell close
    daily['open_close_bid'] = daily['close_bid'] - daily['open_ask'] # buy ask sell bid (realistic)

    # Perfect foresight: best possible trade each day
    # Best long: buy at open_ask, sell at high_bid = high_bid - open_ask
    # Best short: sell at open_bid (~open_ask-spread), cover at low_ask = open_bid - low_ask
    daily['best_long'] = daily['high_bid'] - daily['open_ask']
    daily['best_short'] = daily['open_ask'] - daily['low_ask'] - daily['spread'].mean()  # approx

    # Fade the full range: buy low_ask, sell high_bid
    daily['perfect_fade'] = daily['high_bid'] - daily['low_ask']

    # If we could capture 1/8th of the range perfectly
    daily['capture_12pct'] = daily['range_ask'] * 0.125

    n_days = len(daily)

    print(f"\n  {n_days} trading days")
    print(f"\n{'='*60}")
    print("DAILY STATISTICS (ask prices)")
    print(f"{'='*60}")
    for col in ['range_ask', 'range_bid', 'open_close', 'open_close_bid']:
        vals = daily[col]
        print(f"  {col:<20s}: mean={vals.mean():.2f}, median={vals.median():.2f}, "
              f"std={vals.std():.2f}, min={vals.min():.2f}, max={vals.max():.2f}")

    print(f"\n{'='*60}")
    print("PERFECT FORESIGHT TRADING (impossible in reality)")
    print(f"{'='*60}")
    for col, label in [('best_long', 'Buy open→sell high'), ('best_short', 'Sell open→cover low'),
                       ('perfect_fade', 'Buy low→sell high')]:
        vals = daily[col]
        pos_days = (vals > 0).sum()
        neg_days = (vals < 0).sum()
        print(f"  {label:<25s}: mean={vals.mean():>+6.2f}, "
              f"median={vals.median():>+6.2f}, "
              f"positive={pos_days}/{n_days} days, "
              f"sum={vals.sum():>+8.1f}")

    # Net: could we make $10/day with perfect timing?
    daily['best_trade'] = np.maximum(daily['best_long'], daily['best_short'])
    print(f"\n  Best possible (long OR short, perfect timing):")
    print(f"    Mean: {daily['best_trade'].mean():.2f} pts/day")
    print(f"    Total: {daily['best_trade'].sum():.1f} pts")
    print(f"    Days >10 pts: {(daily['best_trade']>10).sum()}/{n_days}")
    print(f"    Days >5 pts:  {(daily['best_trade']>5).sum()}/{n_days}")
    print(f"    Days >3 pts:  {(daily['best_trade']>3).sum()}/{n_days}")

    # What % of days does a simple buy-open-sell-close work?
    print(f"\n{'='*60}")
    print("SIMPLE BUY-OPEN SELL-CLOSE (no timing skill)")
    print(f"{'='*60}")
    oc = daily['open_close_bid']
    wins = (oc > 0).sum()
    print(f"  Mean: {oc.mean():+.2f} pts, Wins: {wins}/{n_days} ({wins/n_days*100:.1f}%)")
    print(f"  Total PnL: {oc.sum():+.1f} pts")
    print(f"  Spread cost/day: {daily['spread'].mean():.2f} pts")

    # Realistic capture rate analysis
    print(f"\n{'='*60}")
    print("REALITY CHECK: Why $10/day is hard")
    print(f"{'='*60}")
    print(f"  Daily spread cost: ~{daily['spread'].mean():.2f} pts (one round trip)")
    print(f"  If we trade 2x/day (enter+exit): ~{daily['spread'].mean()*2:.2f} pts spread cost")
    print(f"  To make $10 (10 pts) net, need ~{10+daily['spread'].mean()*2:.1f} pts gross edge/day")

    # 1-min bar movement distribution
    bar_moves = abs(df['close'].diff()).dropna()
    print(f"\n{'='*60}")
    print("1-MIN BAR MOVEMENTS")
    print(f"{'='*60}")
    for thresh in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        pct = (bar_moves > thresh).mean() * 100
        print(f"  Bars >{thresh:.1f} pts: {pct:.1f}%")
    print(f"  Mean |move|: {bar_moves.mean():.3f} pts")
    print(f"  Median |move|: {bar_moves.median():.3f} pts")

    # How many 1-min bars are "tradable" (move > spread)?
    tradable = (bar_moves > daily['spread'].mean()).mean() * 100
    print(f"  Bars with |move| > spread: {tradable:.1f}%")
    print(f"  → ~{100-tradable:.0f}% of bars are pure noise (< spread)")

    # Intraday range capture simulation
    print(f"\n{'='*60}")
    print("RANGE CAPTURE SIMULATION")
    print(f"{'='*60}")
    
    # Simulate: buy at a random time, TP=10, SL=15, exit at EOD
    # This tests whether a basic scalp can survive spread + randomness
    np.random.seed(42)
    n_sims = 1000
    sim_pnls = []
    for _ in range(n_sims):
        entry_i = np.random.randint(0, len(df) - 100)
        entry_ask = df.iloc[entry_i]['close_ask']
        # Exit: check next 100 bars for TP=10 or SL=15
        tp_hit = False; sl_hit = False
        for j in range(1, min(100, len(df) - entry_i - 1)):
            bar = df.iloc[entry_i + j]
            if bar['high_bid'] >= entry_ask + 10:
                sim_pnls.append(10.0); tp_hit = True; break
            if bar['low_bid'] <= entry_ask - 15:
                sim_pnls.append(-15.0); sl_hit = True; break
        if not tp_hit and not sl_hit:
            # Close at bid after 100 bars
            sim_pnls.append(df.iloc[entry_i + min(100, len(df)-entry_i-1)]['close_bid'] - entry_ask)

    sim_a = np.array(sim_pnls)
    print(f"  Random entry TP=10 SL=15 (100 bar max):")
    print(f"    Mean: {sim_a.mean():+.2f} pts, WR: {(sim_a>0).mean()*100:.1f}%")
    print(f"    Total: {sim_a.sum():+.1f} over {n_sims} random entries")

    # Conclusion
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    avg_range = daily['range_ask'].mean()
    avg_spread = daily['spread'].mean()
    theoretical_max = daily['best_trade'].mean()
    print(f"  Daily range (ask): {avg_range:.1f} pts")
    print(f"  Daily spread: {avg_spread:.2f} pts")
    print(f"  Theoretical max with perfect foresight: {theoretical_max:.1f} pts/day")
    print(f"  Spread as % of range: {avg_spread/avg_range*100:.1f}%")
    print(f"  Random TP=10 capture rate: {sim_a.mean():+.2f} pts/attempt")
    print(f"")
    if theoretical_max < 10:
        print(f"  → Even with PERFECT hindsight, max daily edge is only {theoretical_max:.1f} pts")
        print(f"  → This is less than the $10 target — the range ISN'T big enough")
    else:
        print(f"  → Range IS big enough (max={theoretical_max:.1f}), but spread eats ~{avg_spread:.1f}% of range")
        print(f"  → Need {10/theoretical_max*100:.0f}% capture rate of max move — extremely demanding")
    print(f"\nDONE.")


if __name__ == '__main__':
    main()
