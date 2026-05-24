import numpy as np
import pandas as pd
from numba import njit

@njit
def _calc_vp_numba(prices, volumes, day_ids, bin_size=0.5):
    n = len(prices)
    daily_poc = np.zeros(n)
    daily_vwap = np.zeros(n)
    rolling_poc = np.zeros(n)
    
    # Bins for daily
    d_bins = np.zeros(20000)
    # Bins for rolling 240
    r_bins = np.zeros(20000)
    
    current_day = day_ids[0]
    d_max_vol = 0.0
    d_current_poc = prices[0]
    
    cum_vol = 0.0
    cum_vol_price = 0.0
    
    window = 240
    
    for i in range(n):
        p = prices[i]
        v = volumes[i]
        
        # --- Daily POC & VWAP ---
        if day_ids[i] != current_day:
            d_bins[:] = 0.0
            current_day = day_ids[i]
            d_max_vol = 0.0
            d_current_poc = p
            cum_vol = 0.0
            cum_vol_price = 0.0
            
        cum_vol += v
        cum_vol_price += p * v
        if cum_vol > 0:
            daily_vwap[i] = cum_vol_price / cum_vol
        else:
            daily_vwap[i] = p
            
        bin_idx = int(p / bin_size)
        if 0 <= bin_idx < len(d_bins):
            d_bins[bin_idx] += v
            if d_bins[bin_idx] > d_max_vol:
                d_max_vol = d_bins[bin_idx]
                d_current_poc = bin_idx * bin_size
        
        daily_poc[i] = d_current_poc
        
        # --- Rolling 240m POC ---
        if 0 <= bin_idx < len(r_bins):
            r_bins[bin_idx] += v
            
        if i >= window:
            p_rem = prices[i - window]
            v_rem = volumes[i - window]
            idx_rem = int(p_rem / bin_size)
            if 0 <= idx_rem < len(r_bins):
                r_bins[idx_rem] -= v_rem
        
        # Find max for rolling
        # Optimization: only scan around the current price +/- 200 bins (100 points)
        search_start = max(0, bin_idx - 200)
        search_end = min(len(r_bins), bin_idx + 200)
        
        r_max_v = -1.0
        r_best_idx = bin_idx
        for j in range(search_start, search_end):
            if r_bins[j] > r_max_v:
                r_max_v = r_bins[j]
                r_best_idx = j
        
        rolling_poc[i] = r_best_idx * bin_size
        
    return daily_poc, daily_vwap, rolling_poc

def add_volume_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Volume Profile features using fast Numba compilation.
    - Daily VWAP
    - Daily Point of Control (POC)
    - Rolling 4-hour Point of Control (POC)
    """
    df = df.copy()
    
    # Typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    prices = typical_price.values
    volumes = df['volume'].values
    
    # Create integer day_id
    day_ids = df.index.year * 1000 + df.index.dayofyear
    day_ids = day_ids.values
    
    daily_poc, daily_vwap, rolling_poc = _calc_vp_numba(prices, volumes, day_ids, bin_size=0.5)
    
    df['daily_poc'] = daily_poc
    df['daily_vwap'] = daily_vwap
    df['rolling_poc_4h'] = rolling_poc
    
    # Create distance features (these are what the model actually uses)
    df['dist_daily_poc'] = df['close'] - df['daily_poc']
    df['dist_daily_vwap'] = df['close'] - df['daily_vwap']
    df['dist_rolling_poc_4h'] = df['close'] - df['rolling_poc_4h']
    
    return df
