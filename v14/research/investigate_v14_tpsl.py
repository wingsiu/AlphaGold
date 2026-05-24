import pandas as pd
import numpy as np
import sys
from pathlib import Path

from v14._paths import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from xgboost_filter_model.train_filter_1min import load_price_data

print("Loading 1 year of 1-min data...")
df = load_price_data(start_date='2025-05-01', end_date='2026-05-21')
df.index = pd.to_datetime(df.index)

# Convert to NY time
df_ny = df.copy()
if df_ny.index.tzinfo is None:
    df_ny.index = df_ny.index.tz_localize('UTC')
df_ny.index = df_ny.index.tz_convert('America/New_York')

# Define trading days (starts at 17:00 NY time previous day)
# We shift by 7 hours forward so that 17:00 NY becomes 00:00 of the "trading day"
df_ny['trading_day'] = (df_ny.index + pd.Timedelta(hours=7)).floor('D')

print("Simulating Trend Breakouts...")
results = []

# Test different TP/SL pairs
tp_sl_pairs = [(20, 10), (30, 15), (40, 20), (50, 25), (60, 30), (40, 25)]

# Group by trading day
grouped = df_ny.groupby('trading_day')

total_days = 0
triggered_days = 0

for day, group in grouped:
    if len(group) < 100: continue # Skip partial days
    total_days += 1
    
    # Daily open is the first 'open' of the trading day
    daily_open = group['open'].iloc[0]
    
    # Find the first bar that moves 15 points away from the open
    trigger_threshold = 15.0
    
    group = group.copy()
    group['dist_up'] = group['high'] - daily_open
    group['dist_down'] = daily_open - group['low']
    
    # Find index where it breaks up or down
    up_breaks = group[group['dist_up'] >= trigger_threshold]
    down_breaks = group[group['dist_down'] >= trigger_threshold]
    
    trigger_idx = None
    direction = 0
    
    if not up_breaks.empty and not down_breaks.empty:
        if up_breaks.index[0] < down_breaks.index[0]:
            trigger_idx = up_breaks.index[0]
            direction = 1
        else:
            trigger_idx = down_breaks.index[0]
            direction = -1
    elif not up_breaks.empty:
        trigger_idx = up_breaks.index[0]
        direction = 1
    elif not down_breaks.empty:
        trigger_idx = down_breaks.index[0]
        direction = -1
        
    if trigger_idx is not None:
        triggered_days += 1
        # The entry price is exactly the breakout level
        entry_price = daily_open + trigger_threshold if direction == 1 else daily_open - trigger_threshold
        
        # Get all bars strictly AFTER the trigger to prevent lookahead bias
        forward_bars = group.loc[trigger_idx:].iloc[1:] 
        
        day_results = {'day': day, 'direction': direction}
        
        for tp, sl in tp_sl_pairs:
            pnl = 0
            reason = 'timeout'
            
            if direction == 1:
                target_price = entry_price + tp
                stop_price = entry_price - sl
                
                for _, bar in forward_bars.iterrows():
                    if bar['low'] <= stop_price:
                        pnl = -sl
                        reason = 'sl'
                        break
                    elif bar['high'] >= target_price:
                        pnl = tp
                        reason = 'tp'
                        break
            else:
                target_price = entry_price - tp
                stop_price = entry_price + sl
                
                for _, bar in forward_bars.iterrows():
                    if bar['high'] >= stop_price:
                        pnl = -sl
                        reason = 'sl'
                        break
                    elif bar['low'] <= target_price:
                        pnl = tp
                        reason = 'tp'
                        break
            
            # If timeout, close at the last close of the day
            if reason == 'timeout' and not forward_bars.empty:
                last_close = forward_bars['close'].iloc[-1]
                pnl = (last_close - entry_price) * direction
                
            day_results[f'pnl_{tp}_{sl}'] = pnl
            
        results.append(day_results)

res_df = pd.DataFrame(results)

print(f"\n=== Pure Trend Breakout Simulation (No ML, Just Price Action) ===")
print(f"Total Days: {total_days}")
print(f"Days with 15-point Breakout: {triggered_days} ({(triggered_days/total_days)*100:.1f}%)")
print("\nPerformance of different TP/SL pairs on Breakout Days:")
print(f"{'TP/SL':<10} | {'Win Rate':<10} | {'Avg PnL':<10} | {'Total PnL':<10} | {'Profit Factor'}")
print("-" * 65)

for tp, sl in tp_sl_pairs:
    col = f'pnl_{tp}_{sl}'
    wins = res_df[res_df[col] > 0]
    losses = res_df[res_df[col] <= 0]
    
    win_rate = len(wins) / len(res_df) * 100
    avg_pnl = res_df[col].mean()
    total_pnl = res_df[col].sum()
    
    gross_profit = wins[col].sum()
    gross_loss = abs(losses[col].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print(f"{tp}/{sl:<7} | {win_rate:>5.1f}%    | {avg_pnl:>7.2f}    | {total_pnl:>8.1f}   | {pf:.2f}")

