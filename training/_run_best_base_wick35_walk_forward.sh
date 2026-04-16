#!/usr/bin/env bash
set -euo pipefail

cd /Users/alpha/Desktop/python/AlphaGold

PREP_CACHE_DIR="${PREP_CACHE_DIR:-training/_prep_cache_wf_wick35}"
WICK_FEATURE_MIN_RANGE="${WICK_FEATURE_MIN_RANGE:-40}"
WICK_FEATURE_MIN_PCT="${WICK_FEATURE_MIN_PCT:-35}"
WICK_FEATURE_MIN_VOLUME="${WICK_FEATURE_MIN_VOLUME:-3000}"
REFRESH_PREP_CACHE_FLAG=""
if [[ "${REFRESH_PREP_CACHE:-0}" == "1" ]]; then
  REFRESH_PREP_CACHE_FLAG="--refresh-prep-cache"
fi

python3 -u training/image_trend_ml.py \
  --eval-mode walk_forward \
  --disable-time-filter \
  --use-state-features \
  --use-15m-wick-features \
  --wick-feature-min-range "$WICK_FEATURE_MIN_RANGE" \
  --wick-feature-min-pct "$WICK_FEATURE_MIN_PCT" \
  --wick-feature-min-volume "$WICK_FEATURE_MIN_VOLUME" \
  --state-oof-splits 5 \
  --pred-history-len 150 \
  --start-date 2025-05-20 \
  --end-date 2026-04-10 \
  --test-size 0.4 \
  --window 150 \
  --window-15m 0 \
  --min-window-range 40 \
  --min-15m-drop 15 \
  --min-15m-rise 0 \
  --horizon 25 \
  --trend-threshold 0.008 \
  --long-target-threshold 0.006 \
  --short-target-threshold 0.008 \
  --adverse-limit 15 \
  --long-adverse-limit 12 \
  --short-adverse-limit 18 \
  --max-flat-ratio 2.5 \
  --classifier gradient_boosting \
  --stage1-min-prob 0.48 \
  --stage2-min-prob 0.50 \
  --wf-init-train-months 6 \
  --wf-retrain-days 14 \
  --wf-max-train-days 365 \
  --wf-min-train-samples 300 \
  --wf-anchor-mode weekend_fri_close \
  --wf-cycle-model-dir training/backtest_model_best_base_wick35_wf_cycles \
  --prep-cache-dir "$PREP_CACHE_DIR" \
  ${REFRESH_PREP_CACHE_FLAG:+$REFRESH_PREP_CACHE_FLAG} \
  --model-out training/backtest_model_best_base_wick35_wf.joblib \
  --report-out training/backtest_report_best_base_wick35_wf.json \
  --trades-out training/backtest_trades_best_base_wick35_wf.csv

echo
echo "Saved:"
echo "  training/backtest_model_best_base_wick35_wf.joblib"
echo "  training/backtest_report_best_base_wick35_wf.json"
echo "  training/backtest_trades_best_base_wick35_wf.csv"
echo "  training/backtest_model_best_base_wick35_wf_cycles/"

