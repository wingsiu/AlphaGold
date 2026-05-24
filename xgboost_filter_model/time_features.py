import numpy as np
import pandas as pd
from numba import njit

@njit
def _calc_time_from_extremes(highs, lows, window=60):
    n = len(highs)
    time_from_max = np.zeros(n)
    time_from_min = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        
        # Find argmax (most recent if equal)
        max_val = -np.inf
        max_idx = i
        for j in range(start_idx, i + 1):
            if highs[j] >= max_val:
                max_val = highs[j]
                max_idx = j
        time_from_max[i] = (i - max_idx) / window
        
        # Find argmin (most recent if equal)
        min_val = np.inf
        min_idx = i
        for j in range(start_idx, i + 1):
            if lows[j] <= min_val:
                min_val = lows[j]
                min_idx = j
        time_from_min[i] = (i - min_idx) / window
        
    return time_from_max, time_from_min

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based features:
    1. time_from_15m: Position of the current minute within the 15-minute block (0 to 14 / 15)
    2. time_from_max: Bars since the 60-minute high / 60
    3. time_from_min: Bars since the 60-minute low / 60
    """
    df = df.copy()
    
    # 1. time from 15 mins
    df['time_from_15m'] = (df.index.minute % 15) / 15.0
    
    # 2 & 3. time from max/min (60m)
    highs = df['high'].values
    lows = df['low'].values
    
    time_from_max, time_from_min = _calc_time_from_extremes(highs, lows, window=60)
    
    df['time_from_max'] = time_from_max
    df['time_from_min'] = time_from_min
    
    return df
