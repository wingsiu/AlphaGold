#!/usr/bin/env bash
set -euo pipefail

cd /Users/alpha/Desktop/python/AlphaGold

# Config-C-near follow-up for the new next-bar-open execution semantics:
# - keep the original Config C regime/filter family
# - keep Config C flat-ratio sweep: 3.0, 4.0, 5.0
# - keep Config C stage1 threshold fixed at 0.57
# - sweep directional stage2 gates around Config C's 0.62 / 0.64
# - walk-forward scoring/backtest now uses next-bar-open entry logic

MODEL_OUT="training/_wfC_nbopen_nearC_stage2ud.joblib"
REPORT_OUT="training/_wfC_nbopen_nearC_stage2ud.json"
TRADES_OUT="training/_wfC_nbopen_nearC_stage2ud.csv"

python3 -u training/image_trend_ml.py \
  --eval-mode walk_forward \
  --disable-time-filter \
  --use-state-features \
  --state-oof-splits 5 \
  --pred-history-len 150 \
  --start-date 2025-05-20 \
  --end-date 2026-04-10 \
  --test-size 0.3 \
  --window 150 \
  --window-15m 0 \
  --min-window-range 40 \
  --min-15m-drop 15 \
  --min-15m-rise 15 \
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
  --wf-sweep-flat-ratios 3.0,4.0,5.0 \
  --wf-sweep-stage2-long-probs 0.60,0.62,0.64 \
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

