#!/usr/bin/env bash
set -euo pipefail

cd /Users/alpha/Desktop/python/AlphaGold

PREP_CACHE_DIR="${PREP_CACHE_DIR:-training/_prep_cache_single_split_aligned}"
REFRESH_PREP_CACHE_FLAG=""
if [[ "${REFRESH_PREP_CACHE:-0}" == "1" ]]; then
  REFRESH_PREP_CACHE_FLAG="--refresh-prep-cache"
fi

# Phase 1: find a cheap base model first using a single aligned split.
# The split is anchored to the FIRST walk-forward test window under the current
# cycle logic, so this stays comparable to the later expensive WF reruns.

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
  --prep-cache-dir "$PREP_CACHE_DIR" \
  ${REFRESH_PREP_CACHE_FLAG:+$REFRESH_PREP_CACHE_FLAG} \
  --output-format ts_only)"

if [[ ! "$ALIGNED_TEST_START" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]]; then
  echo "ERROR: invalid aligned test start: $ALIGNED_TEST_START" >&2
  exit 1
fi

echo "Aligned single-split test start: $ALIGNED_TEST_START"

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
  --horizon-values 25 \
  --trend-threshold-values 0.008 \
  --long-target-values 0.006 \
  --short-target-values 0.008 \
  --long-stop-values 15 \
  --short-stop-values 18 \
  --min-15m-drop-values 15 \
  --min-15m-rise-values 0,15 \
  --max-flat-ratio-values 2.0,2.5,3.0,3.5,4.0 \
  --stage1-prob-values 0.48,0.50,0.52,0.55 \
  --stage2-prob-values 0.50,0.55,0.60 \
  --prep-cache-dir "$PREP_CACHE_DIR" \
  ${REFRESH_PREP_CACHE_FLAG:+$REFRESH_PREP_CACHE_FLAG} \
  --tmp-dir training/_tmp_base_single_split_aligned \
  --out-csv training/base_single_split_aligned_results.csv \
  --out-txt training/base_single_split_aligned_summary.txt

echo
echo "Saved:"
echo "  training/base_single_split_aligned_results.csv"
echo "  training/base_single_split_aligned_summary.txt"

