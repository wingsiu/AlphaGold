#!/usr/bin/env bash
# Restore WF + pattern models from the May 2025 hybrid calibration (~3125 / ~4305).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "Restoring energetic WF models from git d64c541…"
git checkout d64c541 -- \
  runtime/bot_assets/wf_models_v14/ \
  xgboost_filter_model/filter_model_v14_wf.joblib \
  xgboost_filter_model/directional_model_v14_wf.joblib \
  config/v14_patterns.py

restore() {
  local name=$1 variant=$2 snap=$3
  local dest="$ROOT/runtime/bot_assets/wf_models_v14_patterns/$name/$variant"
  mkdir -p "$dest"
  cp -f "$snap"/* "$dest/"
  echo "  $name/$variant"
}

echo "Restoring pattern models from 2026-05-24 snapshots…"
restore uptrend_retrace h15_tp20_sl15 \
  "$ROOT/v14/runtime/model_snapshots/uptrend_retrace/h15_tp20_sl15/20260524_161134_h15_tp20_sl15"
restore downtrend_retrace h15_tp40_sl30 \
  "$ROOT/v14/runtime/model_snapshots/downtrend_retrace/h15_tp40_sl30/20260524_161144_h15_tp40_sl30"
restore breakthrough_long h15_tp40_sl20 \
  "$ROOT/v14/runtime/model_snapshots/breakthrough_long/h15_tp40_sl20/20260524_160914_h15_tp40_sl20"
restore breakthrough_short h30_tp40_sl30 \
  "$ROOT/v14/runtime/model_snapshots/breakthrough_short/h30_tp40_sl30/20260524_160919_h30_tp40_sl30"
restore reversal_fvg_long h15_tp20_sl15 \
  "$ROOT/v14/runtime/model_snapshots/reversal_fvg_long/h15_tp20_sl15/20260524_161008_h15_tp20_sl15"
restore reversal_fvg_short h15_tp40_sl30 \
  "$ROOT/v14/runtime/model_snapshots/reversal_fvg_short/h15_tp40_sl30/20260524_161057_h15_tp40_sl30"

echo "Done. Run:"
echo "  PYTHONPATH=$ROOT .venv/bin/python3 tools/run_hybrid_time_filter.py 2025-06-01 2026-05-23"
echo ""
echo "If cycle_37 was re-trained mid-cycle (e.g. 2026-05-30), restore that cycle from"
echo "  v14/runtime/model_snapshots/.../20260524_* before live uses it until 2026-06-06."
