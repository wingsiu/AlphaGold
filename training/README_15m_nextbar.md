# 15m Multi-Bar Target Model

This workflow trains a model on **15-minute candles**.

## Target
A sample is labeled positive when, over the next 4 bars:

- current bar close = `C_t`
- max future high over next 4 bars = `max(H_{t+1..t+4})`
- min future low over next 4 bars = `min(L_{t+1..t+4})`
- positive if
  - `((max(H_{t+1..t+4}) - C_t) / C_t) * 100 >= 0.60`
  - and `((min(L_{t+1..t+4}) - C_t) / C_t) * 100 >= -0.40`

## Files
- `training/ml_alpha_model_15m_nextbar.py` — model/data-prep module
- `training/training_15m_nextbar.py` — training entrypoint
- `training/backtest_15m_nextbar.py` — backtest entrypoint
- `training/smoke_test_15m_nextbar.py` — quick pipeline validation

## Quick smoke test
```bash
python3 /Users/alpha/Desktop/python/AlphaGold/training/smoke_test_15m_nextbar.py \
  --start-date 2025-05-20 \
  --end-date 2025-05-23 \
  --threshold-pct 0.60 \
  --max-adverse-low-pct 0.40 \
  --forward-bars 4
```

## Train
```bash
python3 /Users/alpha/Desktop/python/AlphaGold/training/training_15m_nextbar.py \
  --start-date 2025-05-20 \
  --end-date 2026-02-04 \
  --threshold-pct 0.60 \
  --max-adverse-low-pct 0.40 \
  --forward-bars 4 \
  --correlation-analysis \
  --corr-threshold 0.90 \
  --corr-top-n 25 \
  --model-dir ml_models_15m_nextbar \
  --artifacts-dir training/artifacts_15m_nextbar \
  --test-cache-out training/test_period_signals_15m_nextbar.csv \
  --full-cache-out training/full_period_signals_15m_nextbar.csv
```

## Backtest
```bash
python3 /Users/alpha/Desktop/python/AlphaGold/training/backtest_15m_nextbar.py \
  --start-date 2025-05-20 \
  --end-date 2026-02-04 \
  --model-type gradient_boosting \
  --signals-file training/full_period_signals_15m_nextbar.csv \
  --take-profit-pct 0.60 \
  --stop-loss-pct 0.40 \
  --max-hold-bars 4 \
  --optimize-cutoff \
  --cutoff-min 0.05 \
  --cutoff-max 0.50 \
  --cutoff-step 0.05 \
  --out training/backtest_trades_15m_nextbar.csv
```

## Notes
- The backtest still enters on the **next bar open**, so it remains bias-safe.
- The label uses a **4-bar horizon** with both upside and adverse-low constraints.
- `max_hold_bars=4` is the natural starting point for this target.
- Correlation outputs are saved to artifacts:
  - `feature_target_correlation_15m_nextbar.csv`
  - `highly_correlated_feature_pairs_15m_nextbar.csv`
  - `correlation_matrix_top_features_15m_nextbar.png`

