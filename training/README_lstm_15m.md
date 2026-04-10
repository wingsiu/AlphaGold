# LSTM 15m Experiment

This is a parallel LSTM branch for your 15-minute workflow.

## Target rule
A sample is positive when, over the next `forward_bars` (default 4):

- `max(high) >= close * (1 + threshold_pct/100)`
- and `min(low) >= close * (1 - max_adverse_low_pct/100)`

Defaults:
- `threshold_pct = 0.60`
- `max_adverse_low_pct = 0.40`
- `forward_bars = 4`
- `sequence_length = 32`
- `feature_selection = target_corr`
- `feature_top_n = 32`

## Files
- `training/ml_lstm_15m.py` - LSTM dataset/model utility layer
- `training/train_lstm_15m.py` - train + evaluate + export caches
- `training/backtest_lstm_15m.py` - backtest with cached or live LSTM signals
- `training/smoke_test_lstm_15m.py` - quick data-pipeline validation

## Quick smoke test
```bash
python3 /Users/alpha/Desktop/python/AlphaGold/training/smoke_test_lstm_15m.py \
  --start-date 2025-05-20 \
  --end-date 2025-05-23
```

## Train
```bash
python3 -u /Users/alpha/Desktop/python/AlphaGold/training/train_lstm_15m.py \
  --start-date 2025-05-20 \
  --end-date 2026-02-04 \
  --threshold-pct 0.60 \
  --max-adverse-low-pct 0.40 \
  --forward-bars 4 \
  --sequence-length 32 \
  --feature-selection target_corr \
  --feature-top-n 32 \
  --class-weight-pos 1.2 \
  --class-weight-neg 1.0 \
  --epochs 15 \
  --batch-size 128 \
  --model-dir training/ml_models_lstm_15m \
  --test-cache-out training/test_period_signals_lstm_15m.csv \
  --full-cache-out training/full_period_signals_lstm_15m.csv
```

## Backtest (using cache)
```bash
python3 -u /Users/alpha/Desktop/python/AlphaGold/training/backtest_lstm_15m.py \
  --start-date 2025-05-20 \
  --end-date 2026-02-04 \
  --signals-file training/full_period_signals_lstm_15m.csv \
  --take-profit-pct 0.60 \
  --stop-loss-pct 0.40 \
  --max-hold-bars 4 \
  --optimize-cutoff \
  --cutoff-min 0.05 \
  --cutoff-max 0.50 \
  --cutoff-step 0.05 \
  --out training/backtest_trades_lstm_15m.csv
```

## Notes
- TensorFlow is required for train/inference (`tensorflow` package).
- If TensorFlow is unavailable, smoke-test still validates data and sequence construction.
- LSTM is experimental: compare with GB baseline using the same backtest settings.
- Start with fewer features first (default 32 by correlation), then scale up if needed.
- You can tune decision aggressiveness via class weights (`--class-weight-pos`, `--class-weight-neg`).

