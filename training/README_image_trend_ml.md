# Image-Style Trend ML (Gold)

This pipeline converts rolling candle windows into image-like tensors and trains a trend classifier.

## What it does

- Uses `gold_prices` from MySQL via `DataLoader`
- Uses 1-minute candles by default
- Filters training images to keep only windows where `max(high)-min(low) > 40` (configurable)
- Converts each rolling window (default `150` bars) into a tensor of shape:
  - `9 channels x 150 width`
  - Channels include OHLC shape + explicit volume channels (`volume_z`, `volume_rel`, `volume_change`)
- Labels trend `down / flat / up` using future return over `horizon` bars
- Target rule (per sample):
  - `up` if `(future_high - current_close) / current_close > 0.4%` and `(current_close - future_low) < 15`
  - `down` if `(current_close - future_low) / current_close > 0.4%` and `(future_high - current_close) < 15`
  - otherwise `flat`
- Trains a multiclass logistic regression model
- Uses a two-stage model (`flat vs trend`, then `down vs up`) with optional confidence gating
- Optional `--optimize-prob` tunes stage confidence gates on a train-validation split
- Evaluates chronologically on out-of-sample data
- Saves model and JSON report

## Files

- `training/image_trend_ml.py`
- `training/run_image_trend_smoke.py`

## Quick smoke run

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/run_image_trend_smoke.py
```

## Full run

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/image_trend_ml.py
```

## Useful options

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/image_trend_ml.py \
  --start-date 2025-05-20 \
  --end-date 2026-04-10 \
  --timeframe 1min \
  --window 150 \
  --min-window-range 40 \
  --horizon 15 \
  --trend-threshold 0.004 \
  --adverse-limit 15 \
  --stage1-min-prob 0.55 \
  --stage2-min-prob 0.55 \
  --optimize-prob \
  --model-out training/image_trend_model.joblib \
  --report-out training/image_trend_report.json
```

