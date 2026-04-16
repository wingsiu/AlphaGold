#!/usr/bin/env bash
set -euo pipefail

cd /Users/alpha/Desktop/python/AlphaGold

PREP_CACHE_DIR="${PREP_CACHE_DIR:-training/_prep_cache_single_split_aligned}"
WICK_FEATURE_MIN_RANGE="${WICK_FEATURE_MIN_RANGE:-40}"
WICK_FEATURE_MIN_PCT="${WICK_FEATURE_MIN_PCT:-35}"
WICK_FEATURE_MIN_VOLUME="${WICK_FEATURE_MIN_VOLUME:-3000}"
REFRESH_PREP_CACHE_FLAG=""
if [[ "${REFRESH_PREP_CACHE:-0}" == "1" ]]; then
  REFRESH_PREP_CACHE_FLAG="--refresh-prep-cache"
fi

ALIGNED_TEST_START="$(python3 -u training/resolve_aligned_single_split.py \
  --start-date 2025-05-20 \
  --end-date 2026-04-10 \
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
  --wf-init-train-months 6 \
  --wf-retrain-days 14 \
  --wf-max-train-days 365 \
  --wf-min-train-samples 300 \
  --wf-anchor-mode weekend_fri_close \
  --prep-cache-dir "$PREP_CACHE_DIR" \
  ${REFRESH_PREP_CACHE_FLAG:+$REFRESH_PREP_CACHE_FLAG} \
  --output-format ts_only)"

if [[ ! "$ALIGNED_TEST_START" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]]; then
  echo "ERROR: invalid aligned test start: $ALIGNED_TEST_START" >&2
  exit 1
fi

echo "Aligned single-split test start: $ALIGNED_TEST_START"

echo
python3 -u training/image_trend_ml.py \
  --eval-mode single_split \
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
  --test-start-date "$ALIGNED_TEST_START" \
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
  --stage1-min-prob 0.48 \
  --stage2-min-prob 0.50 \
  --prep-cache-dir "$PREP_CACHE_DIR" \
  ${REFRESH_PREP_CACHE_FLAG:+$REFRESH_PREP_CACHE_FLAG} \
  --model-out training/backtest_model_best_base_wick35.joblib \
  --report-out training/backtest_report_best_base_wick35.json \
  --trades-out training/backtest_trades_best_base_wick35.csv

echo
echo "Saved:"
echo "  training/backtest_model_best_base_wick35.joblib"
echo "  training/backtest_report_best_base_wick35.json"
echo "  training/backtest_trades_best_base_wick35.csv"

