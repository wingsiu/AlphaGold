#!/usr/bin/env bash
set -euo pipefail

cd /Users/alpha/Desktop/python/AlphaGold

# Narrowed follow-up around the current best region:
# - min-15m-rise fixed to 0
# - flat ratio sweep: 1.8, 2.0, 2.2, 2.5
# - stage1 sweep: 0.46, 0.48, 0.50
# - asymmetric stage2 directional sweeps:
#     up   = 0.54, 0.56, 0.58, 0.60
#     down = 0.62, 0.64, 0.66

MODEL_OUT="training/_wfC_r0_narrow_flat180_250_s146_50_u54_60_d62_66.joblib"
REPORT_OUT="training/_wfC_r0_narrow_flat180_250_s146_50_u54_60_d62_66.json"
TRADES_OUT="training/_wfC_r0_narrow_flat180_250_s146_50_u54_60_d62_66.csv"

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
  --reverse-exit-prob 0.7 \
  --wf-init-train-months 6 \
  --wf-retrain-days 14 \
  --wf-max-train-days 365 \
  --wf-min-train-samples 300 \
  --wf-sweep-flat-ratios 1.8,2.0,2.2,2.5 \
  --wf-sweep-stage1-probs 0.46,0.48,0.50 \
  --wf-sweep-stage2-long-probs 0.54,0.56,0.58,0.60 \
  --wf-sweep-stage2-short-probs 0.62,0.64,0.66 \
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

