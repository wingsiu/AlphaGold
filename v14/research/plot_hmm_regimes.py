import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from pathlib import Path

PROJECT_ROOT = Path('/Users/alpha/Desktop/python/AlphaGold')
csv_path = PROJECT_ROOT / 'runtime' / 'hmm_recent_prices.csv'
out_path = PROJECT_ROOT / 'runtime' / 'hmm_regimes_sample.png'

def plot_candlesticks_with_regimes():
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Take the last 14 days for a clear view
    df_plot = df.tail(24 * 14)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 1. Plot Candlesticks manually for full control
    up = df_plot[df_plot['close'] >= df_plot['open']]
    down = df_plot[df_plot['close'] < df_plot['open']]
    
    # Plot wicks
    ax.vlines(up.index, up['low'], up['high'], color='green', linewidth=1)
    ax.vlines(down.index, down['low'], down['high'], color='red', linewidth=1)
    
    # Plot bodies
    ax.vlines(up.index, up['open'], up['close'], color='green', linewidth=4)
    ax.vlines(down.index, down['open'], down['close'], color='red', linewidth=4)
    
    # 2. Color backgrounds based on regime
    # 0: Low Volatility (Gray)
    # 1: Steady Trend (Light Blue)
    # 2: Shock/High Vol (Light Coral)
    colors = {0: 'lightgray', 1: 'lightblue', 2: 'lightcoral'}
    
    start_idx = df_plot.index[0]
    current_regime = df_plot['regime'].iloc[0]
    
    for i in range(1, len(df_plot)):
        if df_plot['regime'].iloc[i] != current_regime:
            end_idx = df_plot.index[i]
            ax.axvspan(start_idx, end_idx, color=colors[current_regime], alpha=0.3)
            start_idx = end_idx
            current_regime = df_plot['regime'].iloc[i]
    
    # Add the last span
    ax.axvspan(start_idx, df_plot.index[-1], color=colors[current_regime], alpha=0.3)
    
    # 3. Formatting
    ax.set_title("Gold 1-Hour Candlesticks with HMM Regimes (Last 14 Days)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Gold Price", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    
    # 4. Create custom legend
    legend_elements = [
        Patch(facecolor='lightgray', alpha=0.5, label='Regime 0: Ranging / Low Volatility'),
        Patch(facecolor='lightblue', alpha=0.5, label='Regime 1: Steady Trend'),
        Patch(facecolor='lightcoral', alpha=0.5, label='Regime 2: Shock / High Volatility Breakout')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Chart successfully saved to {out_path}")

if __name__ == "__main__":
    plot_candlesticks_with_regimes()
