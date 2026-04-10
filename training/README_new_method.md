# New Gold Method Re-analysis

This is a fresh, independent method (not the prior ML setups):

- 15-minute volatility-adjusted breakout
- EMA trend filter (20/60)
- ATR-based TP/SL
- Chronological 70/30 train-test split
- Grid-search on train only, fixed-parameter OOS evaluation on test

## Script

- `training/reanalyse_gold_new_method.py`

## Quick run

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/reanalyse_gold_new_method.py
```

## Custom range

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/reanalyse_gold_new_method.py --start-date 2025-05-20 --end-date 2026-04-10
```

## Output

- OOS test trades CSV: `training/backtest_trades_new_method.csv`
- Console stats for train (optimized) and test (out-of-sample)

