import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from hmmlearn import hmm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hmmlearn"])
    from hmmlearn import hmm

from v14._paths import PROJECT_ROOT

from xgboost_filter_model.train_filter_1min import load_price_data

def run_hmm_analysis():
    print("Loading 1-min data...")
    df = load_price_data(start_date='2025-01-01', end_date='2026-05-21')
    
    # Resample to 1-hour for regime detection (better for daily trends)
    print("Resampling to 1-hour bars...")
    df_hourly = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Calculate features for HMM
    # We use Log Returns and True Range (normalized) as features
    df_hourly['log_return'] = np.log(df_hourly['close'] / df_hourly['close'].shift(1)) * 1000 # Scale for HMM
    df_hourly['range'] = (df_hourly['high'] - df_hourly['low']) / df_hourly['open'] * 1000
    df_hourly.dropna(inplace=True)

    X = df_hourly[['log_return', 'range']].values

    print("Fitting Hidden Markov Model (HMM) with 3 regimes...")
    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(X)
    hidden_states = model.predict(X)
    df_hourly['regime'] = hidden_states

    # Analyze regimes
    print("\n=== Regime Characteristics ===")
    
    stats = []
    for i in range(3):
        mask = df_hourly['regime'] == i
        state_data = df_hourly[mask]
        
        freq = len(state_data) / len(df_hourly) * 100
        mean_ret = state_data['log_return'].mean()
        std_ret = state_data['log_return'].std()
        mean_range = state_data['range'].mean()
        
        # Calculate average duration
        blocks = (df_hourly['regime'] != df_hourly['regime'].shift(1)).cumsum()
        state_blocks = blocks[mask]
        avg_duration = state_blocks.value_counts().mean()
        
        print(f"\nRegime {i}:")
        print(f"  Frequency: {freq:.1f}% ({len(state_data)} hours)")
        print(f"  Mean Log Return (scaled): {mean_ret:.4f}")
        print(f"  Volatility (Std Dev): {std_ret:.4f}")
        print(f"  Mean Range (scaled): {mean_range:.4f}")
        print(f"  Avg Duration: {avg_duration:.1f} hours")
        
        stats.append({
            "regime": i,
            "frequency_pct": freq,
            "mean_return": mean_ret,
            "volatility": std_ret,
            "mean_range": mean_range,
            "avg_duration_hours": avg_duration
        })
        
    # Save to CSV for visualization
    out_dir = PROJECT_ROOT / "runtime"
    out_dir.mkdir(exist_ok=True)
    
    # Save a sample of the last 30 days for plotting
    recent_df = df_hourly.tail(24 * 30)
    recent_df[['open', 'high', 'low', 'close', 'regime']].to_csv(out_dir / 'hmm_recent_prices.csv')
    
    import json
    with open(out_dir / 'hmm_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"\nSaved regime data to {out_dir / 'hmm_recent_prices.csv'}")

if __name__ == "__main__":
    run_hmm_analysis()