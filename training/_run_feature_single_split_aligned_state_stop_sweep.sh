#!/usr/bin/env bash
set -euo pipefail

cd /Users/alpha/Desktop/python/AlphaGold

PREP_CACHE_DIR="${PREP_CACHE_DIR:-training/_prep_cache_single_split_aligned}"
REFRESH_PREP_CACHE_FLAG=""
if [[ "${REFRESH_PREP_CACHE:-0}" == "1" ]]; then
  REFRESH_PREP_CACHE_FLAG="--refresh-prep-cache"
fi

# Best aligned state-feature base winner from 2026-04-15:
#   window=150 horizon=25 trend=0.008 min-window-range=40 min-15m-drop=15
#   long_target=0.006 short_target=0.008
#   max_flat_ratio=2.5 stage1=0.48 stage2=0.50
#   use_state_features=true pred_history_len=150
#   min-15m-rise=0/15 tied, so use 0 only for this narrower stop sweep.
#
# Override grids if needed:
#   LONG_STOP_VALUES=12,15,18 SHORT_STOP_VALUES=15,18,21 bash ...
LONG_STOP_VALUES="${LONG_STOP_VALUES:-12,15,18}"
SHORT_STOP_VALUES="${SHORT_STOP_VALUES:-15,18,21}"
LAST_BAR_WR90_HIGH_ARG=""
LAST_BAR_WR90_LOW_ARG=""
if [[ -n "${LAST_BAR_WR90_HIGH:-}" ]]; then
  LAST_BAR_WR90_HIGH_ARG="--last-bar-wr90-high $LAST_BAR_WR90_HIGH"
fi
if [[ -n "${LAST_BAR_WR90_LOW:-}" ]]; then
  LAST_BAR_WR90_LOW_ARG="--last-bar-wr90-low $LAST_BAR_WR90_LOW"
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
  --long-adverse-limit 15 \
  --short-adverse-limit 18 \
  --wf-init-train-months 6 \
  --wf-retrain-days 14 \
  --wf-max-train-days 365 \
  --wf-min-train-samples 300 \
  --wf-anchor-mode weekend_fri_close \
  ${LAST_BAR_WR90_HIGH_ARG:+$LAST_BAR_WR90_HIGH_ARG} \
  ${LAST_BAR_WR90_LOW_ARG:+$LAST_BAR_WR90_LOW_ARG} \
  --prep-cache-dir "$PREP_CACHE_DIR" \
  ${REFRESH_PREP_CACHE_FLAG:+$REFRESH_PREP_CACHE_FLAG} \
  --output-format ts_only)"

if [[ ! "$ALIGNED_TEST_START" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]]; then
  echo "ERROR: invalid aligned test start: $ALIGNED_TEST_START" >&2
  exit 1
fi

echo "Aligned single-split test start: $ALIGNED_TEST_START"
echo "Sweeping long stops:  $LONG_STOP_VALUES"
echo "Sweeping short stops: $SHORT_STOP_VALUES"

echo
python3 -u training/sweep_image_trend_ml.py \
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
  --adverse-limit 15 \
  --classifier gradient_boosting \
  --use-state-features \
  --pred-history-len 150 \
  --horizon-values 25 \
  --trend-threshold-values 0.008 \
  --long-target-values 0.006 \
  --short-target-values 0.008 \
  --long-stop-values "$LONG_STOP_VALUES" \
  --short-stop-values "$SHORT_STOP_VALUES" \
  --min-15m-drop-values 15 \
  --min-15m-rise-values 0 \
  ${LAST_BAR_WR90_HIGH_ARG:+$LAST_BAR_WR90_HIGH_ARG} \
  ${LAST_BAR_WR90_LOW_ARG:+$LAST_BAR_WR90_LOW_ARG} \
  --max-flat-ratio-values 2.5 \
  --stage1-prob-values 0.48 \
  --stage2-prob-values 0.50 \
  --prep-cache-dir "$PREP_CACHE_DIR" \
  ${REFRESH_PREP_CACHE_FLAG:+$REFRESH_PREP_CACHE_FLAG} \
  --tmp-dir training/_tmp_feature_single_split_aligned_state_stop_sweep \
  --out-csv training/feature_single_split_aligned_state_stop_sweep_results.csv \
  --out-txt training/feature_single_split_aligned_state_stop_sweep_summary.txt

echo
echo "Saved:"
echo "  training/feature_single_split_aligned_state_stop_sweep_results.csv"
echo "  training/feature_single_split_aligned_state_stop_sweep_summary.txt"

