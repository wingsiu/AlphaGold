import pandas as pd

df = pd.read_csv('/Users/alpha/Desktop/python/AlphaGold/runtime/v14_backtest_trades.csv')

print("\n" + "="*60)
print("  AlphaGold v14 Backtest (30 Horizon / 30 TP / 25 SL)")
print("  Period : 2026-01-01 -> 2026-05-21")
print("="*60)

wins = len(df[df["pnl"] > 0])
losses = len(df[df["pnl"] <= 0])
win_rate = (wins / len(df)) * 100

print(f"  Total Trades : {len(df)}")
print(f"  Win Rate     : {win_rate:.1f}%  ({wins} Wins / {losses} Losses)")
print(f"  Net PnL      : {df['pnl'].sum():+.1f} points\n")

cum_pnl = df["pnl"].cumsum()
max_dd = (cum_pnl.cummax() - cum_pnl).max()

print(f"  Max Drawdown : -{max_dd:.1f} points")
print(f"  Average Win  : +{df[df['pnl'] > 0]['pnl'].mean():.1f} points")
print(f"  Average Loss : {df[df['pnl'] <= 0]['pnl'].mean():.1f} points")
print(f"  Max Win      : +{df['pnl'].max():.1f} points")
print(f"  Max Loss     : {df['pnl'].min():.1f} points\n")

print(f"  Long Trades  : {len(df[df['side'] == 'up'])}")
print(f"  Short Trades : {len(df[df['side'] == 'down'])}")
print("="*60 + "\n")
