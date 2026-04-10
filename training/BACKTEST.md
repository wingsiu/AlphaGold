# Backtest Quickstart

This backtest uses your trained model files in `training/ml_models` and runs a simple long-only simulation:

- signal at candle `i`
- entry at candle `i+1` open
- exit by take-profit / stop-loss / max holding bars

## Run

```bash
cd "/Users/alpha/Desktop/python/AlphaGold"
python3 training/backtest.py \
  --start-date 2026-01-20 \
  --end-date 2026-02-04 \
  --model-type random_forest \
  --take-profit-pct 0.30 \
  --stop-loss-usd 10 \
  --max-hold-bars 30
```

Use `--stop-loss-pct` for percentage-based stops, or `--stop-loss-usd` for absolute dollar distance per 1-unit trade.

## Output

- Console summary: trades, win rate, avg pnl, trades/day.
- Trade log CSV (default): `training/backtest_trades.csv`

## Notes

- `--cutoff` can override model default decision threshold.
- `training/test_period_signals.csv` should be regenerated from the latest training run; it is now produced from the chronological test holdout rather than a random split.
- Use `--allow-overlap` to allow concurrent entries.

