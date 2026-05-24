import sys
from pathlib import Path
import pandas as pd
import numpy as np

from v14._paths import PROJECT_ROOT

from xgboost_filter_model.train_filter_1min import load_price_data

def run_investigation():
    print("Loading 1-min data...")
    df = load_price_data(start_date='2025-01-01', end_date='2026-05-21')
    
    # Resample to 1min to ensure no gaps mess up our positional indexing
    print("Resampling to fill gaps...")
    df = df.resample('1min').ffill()
    
    print("Extracting 15-minute blocks...")
    blocks = []
    for name, group in df.groupby(df.index.floor('15min')):
        if len(group) != 15: 
            continue
        
        # Calculate returns (Close - Open)
        ret_full = group['close'].iloc[-1] - group['open'].iloc[0]
        ret_first3 = group['close'].iloc[2] - group['open'].iloc[0]
        ret_last3 = group['close'].iloc[14] - group['open'].iloc[12]
        
        # Calculate ranges (High - Low)
        range_full = group['high'].max() - group['low'].min()
        range_last3 = group['high'].iloc[12:15].max() - group['low'].iloc[12:15].min()
        
        blocks.append({
            'time': name,
            'ret_full': ret_full,
            'ret_first3': ret_first3,
            'ret_last3': ret_last3,
            'range_full': range_full,
            'range_last3': range_last3
        })
        
    df_blocks = pd.DataFrame(blocks).set_index('time')
    
    # Calculate future returns
    df_blocks['next_15m_ret'] = df_blocks['ret_full'].shift(-1)
    df_blocks['next_30m_ret'] = df_blocks['ret_full'].rolling(2).sum().shift(-2)
    df_blocks['next_60m_ret'] = df_blocks['ret_full'].rolling(4).sum().shift(-4)
    df_blocks.dropna(inplace=True)
    
    print("\n=== Correlation with Future Returns ===")
    corr = df_blocks[['ret_first3', 'ret_last3', 'ret_full', 'next_15m_ret', 'next_30m_ret', 'next_60m_ret']].corr()
    print(corr.loc[['ret_first3', 'ret_last3', 'ret_full'], ['next_15m_ret', 'next_30m_ret', 'next_60m_ret']].round(4))
    
    print("\n=== Directional Continuation (Hit Rate for Next 30 Mins) ===")
    # Look at moves greater than 1.0 point to filter out noise
    threshold = 1.0
    
    for col in ['ret_first3', 'ret_last3', 'ret_full']:
        mask_up = df_blocks[col] > threshold
        mask_down = df_blocks[col] < -threshold
        
        hit_up = (df_blocks.loc[mask_up, 'next_30m_ret'] > 0).mean() * 100
        hit_down = (df_blocks.loc[mask_down, 'next_30m_ret'] < 0).mean() * 100
        avg_hit = (hit_up + hit_down) / 2
        
        print(f"\n{col} (> {threshold} pt move):")
        print(f"  Up continuation:   {hit_up:.1f}% (N={mask_up.sum()})")
        print(f"  Down continuation: {hit_down:.1f}% (N={mask_down.sum()})")
        print(f"  Average Edge:      {avg_hit:.1f}%")

    print("\n=== What if the Last 3 Mins reverse the First 12 Mins? ===")
    # e.g., first 12 mins went up, but last 3 mins went down sharply
    df_blocks['ret_first12'] = df_blocks['ret_full'] - df_blocks['ret_last3']
    
    # Reversal setup: First 12m up > 1.5, Last 3m down < -1.0
    rev_down = (df_blocks['ret_first12'] > 1.5) & (df_blocks['ret_last3'] < -1.0)
    rev_up = (df_blocks['ret_first12'] < -1.5) & (df_blocks['ret_last3'] > 1.0)
    
    hit_rev_down = (df_blocks.loc[rev_down, 'next_30m_ret'] < 0).mean() * 100
    hit_rev_up = (df_blocks.loc[rev_up, 'next_30m_ret'] > 0).mean() * 100
    
    print(f"  Last 3m Reverses Down: {hit_rev_down:.1f}% continuation DOWN (N={rev_down.sum()})")
    print(f"  Last 3m Reverses Up:   {hit_rev_up:.1f}% continuation UP (N={rev_up.sum()})")

if __name__ == "__main__":
    run_investigation()