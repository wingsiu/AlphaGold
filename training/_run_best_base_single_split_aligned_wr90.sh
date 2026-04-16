#!/usr/bin/env bash
set -euo pipefail

cd /Users/alpha/Desktop/python/AlphaGold

PREP_CACHE_DIR="${PREP_CACHE_DIR:-training/_prep_cache_single_split_aligned}"
REFRESH_PREP_CACHE_FLAG=""
if [[ "${REFRESH_PREP_CACHE:-0}" == "1" ]]; then
  REFRESH_PREP_CACHE_FLAG="--refresh-prep-cache"
fi

# Current "best base" convention:
# use the promoted best aligned single-split candidate, not the pre-stop base.
# Current promoted best base from 2026-04-15:
#   window=150 horizon=25 trend=0.008 min-window-range=40 min-15m-drop=15
#   long_target=0.006 short_target=0.008
#   long_stop=12 short_stop=18
#   max_flat_ratio=2.5 stage1=0.48 stage2=0.50
#   use_state_features=true pred_history_len=150
#   min-15m-rise=0/15 tied -> use 0 here.
# Test add-on:
#   stage1 also gets UTC+2 Dopen/DHigh/DLow features

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
  --start-date 2025-05-20 \
  --end-date 2026-04-10 \
  --timeframe 1min \
  --eval-mode single_split \
  --test-start-date "$ALIGNED_TEST_START" \
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
  --use-stage1-day-ohl-utc2 \
  --pred-history-len 150 \
  --max-flat-ratio 2.5 \
  --stage1-min-prob 0.48 \
  --stage2-min-prob 0.50 \
  --prep-cache-dir "$PREP_CACHE_DIR" \
  ${REFRESH_PREP_CACHE_FLAG:+$REFRESH_PREP_CACHE_FLAG} \
  --model-out training/backtest_model_best_base_wr90_filter.joblib \
  --report-out training/backtest_report_best_base_wr90_filter.json \
  --trades-out training/backtest_trades_best_base_wr90_filter.csv

echo
echo "Saved:"
echo "  training/backtest_model_best_base_wr90_filter.joblib"
echo "  training/backtest_report_best_base_wr90_filter.json"
echo "  training/backtest_trades_best_base_wr90_filter.csv"

