#!/usr/bin/env bash
set -euo pipefail
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u training/image_trend_ml.py \
  --start-date 2025-05-20 \
  --end-date 2026-04-10 \
  --timeframe 1min \
  --eval-mode single_split \
  --test-start-date 2025-11-25T17:02:00+00:00 \
  --test-size 0.4 \
  --disable-time-filter \
  --window 150 \
  --window-15m 0 \
  --min-window-range 40 \
  --min-15m-drop 15 \
  --min-15m-rise 0 \
  --horizon 25 \
  --trend-threshold 0.008 \
  --adverse-limit 15 \
  --long-target-threshold 0.006 \
  --short-target-threshold 0.008 \
  --long-adverse-limit 12 \
  --short-adverse-limit 18 \
  --classifier gradient_boosting \
  --use-state-features \
  --pred-history-len 150 \
  --max-flat-ratio 2.5 \
  --stage1-min-prob 0.48 \
  --stage2-min-prob 0.50 \
  --prep-cache-dir training/_prep_cache_single_split_aligned \
  --model-out training/backtest_model_best_base_corrected.joblib \
  --report-out training/backtest_report_best_base_corrected.json \
  --trades-out training/backtest_trades_best_base_corrected.csv

