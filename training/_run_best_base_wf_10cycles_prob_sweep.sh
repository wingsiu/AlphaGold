#!/usr/bin/env bash
set -euo pipefail

cd /Users/alpha/Desktop/python/AlphaGold

# Requested threshold sweep for the promoted best base using state features.
# Keep the promoted best-base structure fixed:
#   window=150 horizon=25 trend=0.008 min-window-range=40 min-15m-drop=15
#   long_target=0.006 short_target=0.008
#   long_stop=12 short_stop=18
#   max_flat_ratio fixed at 2.5
#   use_state_features=true pred_history_len=150
#   min-15m-rise=0
#
# Sweep only:
#   stage1_min_prob        = 0.48, 0.50, 0.52
#   stage2_min_prob_up     = 0.50, 0.54, 0.58
#   stage2_min_prob_down   = 0.54, 0.58, 0.62
#
# Walk-forward schedule remains the same 10-cycle setup.

MODEL_OUT="training/backtest_model_best_base_wf_10cycles_prob_sweep.joblib"
REPORT_OUT="training/backtest_report_best_base_wf_10cycles_prob_sweep.json"
TRADES_OUT="training/backtest_trades_best_base_wf_10cycles_prob_sweep.csv"

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
  --long-adverse-limit 12 \
  --short-adverse-limit 18 \
  --max-flat-ratio 2.5 \
  --classifier gradient_boosting \
  --stage1-min-prob 0.48 \
  --stage2-min-prob 0.50 \
  --stage2-min-prob-up 0.50 \
  --stage2-min-prob-down 0.54 \
  --reverse-exit-prob 0.7 \
  --wf-init-train-months 6 \
  --wf-retrain-days 14 \
  --wf-max-train-days 365 \
  --wf-min-train-samples 300 \
  --wf-sweep-flat-ratios 2.5 \
  --wf-sweep-stage1-probs 0.48,0.50,0.52 \
  --wf-sweep-stage2-long-probs 0.50,0.54,0.58 \
  --wf-sweep-stage2-short-probs 0.54,0.58,0.62 \
  --wf-sweep-val-ratio 0.2 \
  --wf-sweep-min-val-samples 50 \
  --wf-anchor-mode weekend_fri_close \
  --model-out "$MODEL_OUT" \
  --report-out "$REPORT_OUT" \
  --trades-out "$TRADES_OUT"

echo
echo "Saved:"
echo "  $MODEL_OUT"
echo "  $REPORT_OUT"
echo "  $TRADES_OUT"

