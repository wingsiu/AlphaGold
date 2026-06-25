#!/usr/bin/env python3
"""
v16 Daily Wick Analysis — Can we use daily upper/lower wicks?
===============================================================
Investigates daily candle wick structure in May-June 2026:
  1. Daily wick sizes: upper wick, lower wick, body, wick/range ratio
  2. Wick fill rate: does the wick get "filled" (= price returns to wick zone)?
  3. Intraday timing: when do wicks form? (session-specific)
  4. Wick fade strategy: if daily has large lower wick, buy at next day's open?
  5. Same-day wick fade: enter when wick exceeds threshold intraday
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
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def main():
    print("=" * 72)
    print("v16 Daily Wick Analysis — May-June 2026")
    print("=" * 72)

    df = load_data()
    print(f"  {len(df):,} 1-min bars")

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
              f"std={vals.std():>5.1f}, min={vals.min():>5.1f}, max={vals.max():>5.1f}")

    print(f"\n  Wick as % of range:")
    for col, label in [('upper_wick_pct', 'Upper wick %'), ('lower_wick_pct', 'Lower wick %'),
                       ('body_pct', 'Body %')]:
        vals = daily[col]
        print(f"  {label:<15s}: mean={vals.mean():>5.0f}%, median={vals.median():>5.0f}%")

    # =========================================================================
    # 2. Wick fill rate: does price return to the wick zone?
    # =========================================================================
    print(f"\n{'='*60}")
    print("WICK FILL RATE (same day)")
    print(f"{'='*60}")

    # On days with large upper wick (>30% of range), does price come back down
    # to fill at least 50% of the wick before close?
    large_uw = daily[daily['upper_wick_pct'] > 30]
    large_lw = daily[daily['lower_wick_pct'] > 30]

    # For each large-upper-wick day, check intraday: after the high was made,
    # did price retrace at least 50% of the wick distance?
    uw_fill_count = 0
    lw_fill_count = 0

    for day_idx, day_row in large_uw.iterrows():
        day_data = df[df.index.date == day_idx.date()]
        if len(day_data) < 10:
            continue
        high_time = day_data['high_ask'].idxmax()
        after_high = day_data[day_data.index > high_time]
        if len(after_high) < 5:
            continue
        wick_size = day_row['upper_wick']
        fill_target = day_row['high_ask'] - wick_size * 0.5  # 50% fill
        if after_high['low_ask'].min() <= fill_target:
            uw_fill_count += 1

    for day_idx, day_row in large_lw.iterrows():
        day_data = df[df.index.date == day_idx.date()]
        if len(day_data) < 10:
            continue
        low_time = day_data['low_ask'].idxmin()
        after_low = day_data[day_data.index > low_time]
        if len(after_low) < 5:
            continue
        wick_size = day_row['lower_wick']
        fill_target = day_row['low_ask'] + wick_size * 0.5  # 50% fill
        if after_low['high_ask'].max() >= fill_target:
            lw_fill_count += 1

    print(f"  Days with upper wick >30%: {len(large_uw)}/{n}")
    print(f"    Wick fills ≥50% same day: {uw_fill_count}/{len(large_uw)} ({uw_fill_count/max(len(large_uw),1)*100:.0f}%)")
    print(f"  Days with lower wick >30%: {len(large_lw)}/{n}")
    print(f"    Wick fills ≥50% same day: {lw_fill_count}/{len(large_lw)} ({lw_fill_count/max(len(large_lw),1)*100:.0f}%)")

    # =========================================================================
    # 3. Intraday wick fade simulation
    # =========================================================================
    print(f"\n{'='*60}")
    print("INTRADAY WICK FADE STRATEGY")
    print(f"{'='*60}")
    print(f"  Rule: When intraday high - body_top > 15 pts (large upper wick forming),")
    print(f"        SHORT at body_top, TP=10, SL=15, exit by EOD")
    print(f"        When body_bot - intraday low > 15 pts (large lower wick),")
    print(f"        LONG at body_bot, TP=10, SL=15, exit by EOD")

    # Track intraday rolling high/low vs rolling body
    trades = []
    for day_idx, day_row in daily.iterrows():
        day_data = df[df.index.date == day_idx.date()]
        if len(day_data) < 30:
            continue

        # Rolling metrics
        rolling_high = day_data['high_ask'].cummax()
        rolling_low = day_data['low_ask'].cummin()
        rolling_open = day_data['open_ask'].iloc[0]

        for i in range(20, len(day_data)):
            current_body_top = max(rolling_open, day_data['close_ask'].iloc[i])
            current_body_bot = min(rolling_open, day_data['close_ask'].iloc[i])
            upper_wick_so_far = rolling_high.iloc[i] - current_body_top
            lower_wick_so_far = current_body_bot - rolling_low.iloc[i]

            # Active trades (simple: only one at a time)
            already_in = any(t['day'] == str(day_idx.date()) for t in trades[-5:] if 'day' in t)

            if not already_in:
                if upper_wick_so_far > 15:
                    # Short at current body_top
                    entry = current_body_top  # ask for short? use close_ask
                    # Simulate exit on remaining bars
                    remaining = day_data.iloc[i+1:]
                    for j, (_, bar) in enumerate(remaining.iterrows()):
                        if bar['low_ask'] <= entry - 10:  # TP hit
                            trades.append({'day': str(day_idx.date()), 'side': 'short',
                                          'pnl': 10.0, 'entry_time': day_data.index[i]})
                            break
                        if bar['high_ask'] >= entry + 15:  # SL hit
                            trades.append({'day': str(day_idx.date()), 'side': 'short',
                                          'pnl': -15.0, 'entry_time': day_data.index[i]})
                            break
                    else:
                        # timeout at close
                        last_bid = remaining['close_bid'].iloc[-1] if len(remaining) > 0 else day_data['close_bid'].iloc[-1]
                        trades.append({'day': str(day_idx.date()), 'side': 'short',
                                      'pnl': entry - last_bid, 'entry_time': day_data.index[i]})
                    break  # one trade per day for this sim

                elif lower_wick_so_far > 15:
                    # Long at current body_bot
                    entry = current_body_bot
                    remaining = day_data.iloc[i+1:]
                    for j, (_, bar) in enumerate(remaining.iterrows()):
                        if bar['high_bid'] >= entry + 10:
                            trades.append({'day': str(day_idx.date()), 'side': 'long',
                                          'pnl': 10.0, 'entry_time': day_data.index[i]})
                            break
                        if bar['low_bid'] <= entry - 15:
                            trades.append({'day': str(day_idx.date()), 'side': 'long',
                                          'pnl': -15.0, 'entry_time': day_data.index[i]})
                            break
                    else:
                        last_bid = remaining['close_bid'].iloc[-1] if len(remaining) > 0 else day_data['close_bid'].iloc[-1]
                        trades.append({'day': str(day_idx.date()), 'side': 'long',
                                      'pnl': last_bid - entry, 'entry_time': day_data.index[i]})
                    break

    TD = pd.DataFrame(trades)
    if len(TD) > 0:
        print(f"  Trades: {len(TD)}")
        print(f"  PnL: {TD['pnl'].sum():+.1f} pts")
        print(f"  WR: {(TD['pnl']>0).mean()*100:.1f}%")
        print(f"  Avg: {TD['pnl'].mean():+.2f} pts/trade")
        longs = TD[TD['side']=='long']; shorts = TD[TD['side']=='short']
        if len(longs):
            print(f"  Longs: {len(longs)}, PnL={longs['pnl'].sum():+.1f}, WR={(longs['pnl']>0).mean()*100:.0f}%")
        if len(shorts):
            print(f"  Shorts: {len(shorts)}, PnL={shorts['pnl'].sum():+.1f}, WR={(shorts['pnl']>0).mean()*100:.0f}%")

    # =========================================================================
    # 4. Overnight wick fade (next day reversal)
    # =========================================================================
    print(f"\n{'='*60}")
    print("OVERNIGHT WICK FADE")
    print(f"{'='*60}")
    print(f"  Rule: If prev day had large upper wick >40% range → short next day open")
    print(f"        If prev day had large lower wick >40% range → long next day open")
    print(f"        TP=15, SL=20, close at EOD")

    ot_trades = []
    for i in range(1, len(daily)):
        prev = daily.iloc[i-1]; curr_day_start = daily.index[i]
        curr_data = df[df.index.date == curr_day_start.date()]
        if len(curr_data) < 30:
            continue

        entry = curr_data['open_ask'].iloc[0]

        if prev['upper_wick_pct'] > 40:
            # Short at open
            for j, (_, bar) in enumerate(curr_data.iterrows()):
                if bar['low_ask'] <= entry - 15:  # TP
                    ot_trades.append({'day': str(curr_day_start.date()), 'side': 'short', 'pnl': 15.0,
                                     'prev_uw_pct': prev['upper_wick_pct']})
                    break
                if bar['high_ask'] >= entry + 20:  # SL
                    ot_trades.append({'day': str(curr_day_start.date()), 'side': 'short', 'pnl': -20.0,
                                     'prev_uw_pct': prev['upper_wick_pct']})
                    break
            else:
                last = curr_data['close_bid'].iloc[-1]
                ot_trades.append({'day': str(curr_day_start.date()), 'side': 'short', 'pnl': entry - last,
                                 'prev_uw_pct': prev['upper_wick_pct']})

        if prev['lower_wick_pct'] > 40:
            # Long at open
            for j, (_, bar) in enumerate(curr_data.iterrows()):
                if bar['high_bid'] >= entry + 15:
                    ot_trades.append({'day': str(curr_day_start.date()), 'side': 'long', 'pnl': 15.0,
                                     'prev_lw_pct': prev['lower_wick_pct']})
                    break
                if bar['low_bid'] <= entry - 20:
                    ot_trades.append({'day': str(curr_day_start.date()), 'side': 'long', 'pnl': -20.0,
                                     'prev_lw_pct': prev['lower_wick_pct']})
                    break
            else:
                last = curr_data['close_bid'].iloc[-1]
                ot_trades.append({'day': str(curr_day_start.date()), 'side': 'long', 'pnl': last - entry,
                                 'prev_lw_pct': prev['lower_wick_pct']})

    OTD = pd.DataFrame(ot_trades)
    if len(OTD) > 0:
        print(f"  Trades: {len(OTD)}")
        print(f"  PnL: {OTD['pnl'].sum():+.1f} pts")
        print(f"  WR: {(OTD['pnl']>0).mean()*100:.1f}%")
        for side in ['long', 'short']:
            ss = OTD[OTD['side']==side]
            if len(ss):
                print(f"  {side}: {len(ss)}, PnL={ss['pnl'].sum():+.1f}, WR={(ss['pnl']>0).mean()*100:.0f}%")

    # =========================================================================
    # 5. Wick size vs next day move correlation
    # =========================================================================
    print(f"\n{'='*60}")
    print("WICK → NEXT DAY PREDICTIVE POWER")
    print(f"{'='*60}")
    next_day_move = daily['close_ask'].diff().shift(-1)  # tomorrow's close - today's close
    for col in ['upper_wick', 'lower_wick', 'upper_wick_pct', 'lower_wick_pct']:
        corr = daily[col].corr(next_day_move)
        print(f"  corr({col}, next_day_move): {corr:+.3f}")

    # =========================================================================
    # 6. Wick clustering by day of week
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

    print(f"\nDONE.")


if __name__ == '__main__':
    main()
