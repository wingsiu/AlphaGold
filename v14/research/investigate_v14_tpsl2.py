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

df_ny = df.copy()
if df_ny.index.tzinfo is None:
    df_ny.index = df_ny.index.tz_localize('UTC')
df_ny.index = df_ny.index.tz_convert('America/New_York')

df_ny['trading_day'] = (df_ny.index + pd.Timedelta(hours=7)).floor('D')

results = []
# Test different breakout thresholds and TP/SL
scenarios = [
    (20, 40, 20), # Breakout 20, TP 40, SL 20
    (20, 30, 15),
    (25, 40, 20),
    (30, 30, 20),
    (30, 40, 20)
]

grouped = df_ny.groupby('trading_day')

for day, group in grouped:
    if len(group) < 100: continue
    
    daily_open = group['open'].iloc[0]
    group = group.copy()
    group['dist_up'] = group['high'] - daily_open
    group['dist_down'] = daily_open - group['low']
    
    day_results = {'day': day}
    
    for trigger_threshold, tp, sl in scenarios:
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
            entry_price = daily_open + trigger_threshold if direction == 1 else daily_open - trigger_threshold
            forward_bars = group.loc[trigger_idx:].iloc[1:] 
            
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
            
            if reason == 'timeout' and not forward_bars.empty:
                last_close = forward_bars['close'].iloc[-1]
                pnl = (last_close - entry_price) * direction
                
            day_results[f'pnl_{trigger_threshold}_{tp}_{sl}'] = pnl
        else:
            day_results[f'pnl_{trigger_threshold}_{tp}_{sl}'] = 0 # No trade
            
    results.append(day_results)

res_df = pd.DataFrame(results)

print(f"\n=== Breakout + TP/SL Scenarios ===")
print(f"{'Breakout/TP/SL':<15} | {'Win Rate':<10} | {'Avg PnL':<10} | {'Total PnL':<10} | {'Profit Factor'}")
print("-" * 70)

for trigger_threshold, tp, sl in scenarios:
    col = f'pnl_{trigger_threshold}_{tp}_{sl}'
    trades = res_df[res_df[col] != 0] # Only count days where a trade happened
    wins = trades[trades[col] > 0]
    losses = trades[trades[col] <= 0]
    
    if len(trades) == 0: continue
    
    win_rate = len(wins) / len(trades) * 100
    avg_pnl = trades[col].mean()
    total_pnl = trades[col].sum()
    
    gross_profit = wins[col].sum()
    gross_loss = abs(losses[col].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print(f"{trigger_threshold}/{tp}/{sl:<9} | {win_rate:>5.1f}%    | {avg_pnl:>7.2f}    | {total_pnl:>8.1f}   | {pf:.2f}")

