#!/usr/bin/env bash
set -euo pipefail

cd /Users/alpha/Desktop/python/AlphaGold

# Historical Config C family regeneration:
# - same filter family as saved backtest_report_wf_state_sweep_C.json
# - same base config values
# - same walk-forward flat-ratio sweep 3.0,4.0,5.0
# - stage1/stage2 remain sweepable via the built-in default WF grids,
#   matching the historical C-family behavior

MODEL_OUT="training/backtest_model_wf_state_sweep_C_regen.joblib"
REPORT_OUT="training/backtest_report_wf_state_sweep_C_regen.json"
TRADES_OUT="training/backtest_trades_wf_state_sweep_C_regen.csv"

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
  --reverse-exit-prob 0.7 \
  --wf-init-train-months 6 \
  --wf-retrain-days 14 \
  --wf-max-train-days 365 \
  --wf-min-train-samples 300 \
  --wf-sweep-flat-ratios 3.0,4.0,5.0 \
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

