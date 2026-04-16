#!/usr/bin/env bash
set -euo pipefail

cd /Users/alpha/Desktop/python/AlphaGold

# Current promoted best base (aligned single-split winner after stop retune):
#   window=150 horizon=25 trend=0.008 min-window-range=40 min-15m-drop=15
#   long_target=0.006 short_target=0.008
#   long_stop=12 short_stop=18
#   max_flat_ratio=2.5 stage1=0.48 stage2=0.50
#   use_state_features=true pred_history_len=150
#   min-15m-rise=0/15 tied -> use 0 here.
#
# Walk-forward schedule:
#   start=2025-05-20 end=2026-04-10
#   wf-init-train-months=6
#   wf-retrain-days=14
#   wf-max-train-days=365
#   wf-min-train-samples=300
#   wf-anchor-mode=weekend_fri_close
#
# On the current full-period schedule this setup yields 10 walk-forward cycles.

MODEL_OUT="training/backtest_model_best_base_wf_10cycles.joblib"
REPORT_OUT="training/backtest_report_best_base_wf_10cycles.json"
TRADES_OUT="training/backtest_trades_best_base_wf_10cycles.csv"

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
  --reverse-exit-prob 0.7 \
  --wf-init-train-months 6 \
  --wf-retrain-days 14 \
  --wf-max-train-days 365 \
  --wf-min-train-samples 300 \
  --wf-disable-sweep \
  --wf-anchor-mode weekend_fri_close \
  --model-out "$MODEL_OUT" \
  --report-out "$REPORT_OUT" \
  --trades-out "$TRADES_OUT"

echo
echo "Saved:"
echo "  $MODEL_OUT"
echo "  $REPORT_OUT"
echo "  $TRADES_OUT"

