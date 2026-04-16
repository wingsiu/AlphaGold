#!/usr/bin/env bash
set -euo pipefail

cd /Users/alpha/Desktop/python/AlphaGold

# Requested native walk-forward sweep under current next-bar-open execution:
# - keep the Config C family structure
# - switch to r0-style filter: min-15m-rise = 0
# - sweep flat ratios: 3.0, 3.5, 4.0, 4.5, 5.0
# - sweep stage1 probs: 0.50, 0.55, 0.60
# - sweep directional stage2 up/down probs: 0.50, 0.55, 0.60, 0.65

MODEL_OUT="training/_wfC_r0_flat300_500_s150_60_u50_65_d50_65.joblib"
REPORT_OUT="training/_wfC_r0_flat300_500_s150_60_u50_65_d50_65.json"
TRADES_OUT="training/_wfC_r0_flat300_500_s150_60_u50_65_d50_65.csv"

python3 -u training/image_trend_ml.py \
  --eval-mode walk_forward \
  --disable-time-filter \
  --use-state-features \
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
  --long-adverse-limit 15 \
  --short-adverse-limit 18 \
  --max-flat-ratio 4 \
  --classifier gradient_boosting \
  --stage1-min-prob 0.57 \
  --stage2-min-prob 0.62 \
  --stage2-min-prob-up 0.62 \
  --stage2-min-prob-down 0.64 \
  --reverse-exit-prob 0.7 \
  --wf-init-train-months 6 \
  --wf-retrain-days 14 \
  --wf-max-train-days 365 \
  --wf-min-train-samples 300 \
  --wf-sweep-flat-ratios 3.0,3.5,4.0,4.5,5.0 \
  --wf-sweep-stage1-probs 0.5,0.55,0.6 \
  --wf-sweep-stage2-long-probs 0.5,0.55,0.6,0.65 \
  --wf-sweep-stage2-short-probs 0.5,0.55,0.6,0.65 \
  --wf-sweep-val-ratio 0.2 \
  --wf-sweep-min-val-samples 50 \
  --wf-anchor-mode weekend_fri_close \
  --model-out "$MODEL_OUT" \
  --report-out "$REPORT_OUT" \
  --trades-out "$TRADES_OUT"

echo
echo "Expected result files:"
echo "  model : $MODEL_OUT"
echo "  report: $REPORT_OUT"
echo "  trades: $TRADES_OUT"

