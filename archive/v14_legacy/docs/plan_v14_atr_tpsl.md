# v14 Pattern Model: ATR-Based Relative TP/SL

## Problem
Current TP/SL is absolute (fixed dollar amounts like TP=$20, SL=$15) per pattern.
This doesn't adapt to market volatility — in volatile periods stops are too tight,
in quiet periods targets are too far.

## Solution
Switch to **ATR-based relative TP/SL**: each pattern's TP/SL becomes a multiplier of
1-min ATR(14) so that position sizing adapts to current volatility.

## Files to change
1. **config/v14_patterns.py** — add `target_mode: "atr"`, change to ATR multipliers
2. **xgboost_filter_model/hybrid_live.py** — `score_pattern()` converts ATR multipliers
   to absolute distances at signal time
3. **(verify) xgboost_filter_model/pattern_training.py** — already supports ATR via
   `execution_target_mode`, `dynamic_tp_sl_series`, `execution_tp_sl`
4. **(verify) v14/backtest/backtest_core.py** — `_exec_params_from_row` reads
   pre-computed `exec_tp`/`exec_sl` columns; ATR-scaled values are already set by
   `assign_exec_tp_sl` in the scoring loop
5. **(verify) v14/tools/train_patterns_v14.py** — calls `label_df_for_pattern` which
   routes through `apply_exec_labels(..., target_mode=execution_target_mode(ex))`
   — already ATR-aware
6. **(no change) v14/backtest/backtest_patterns_v14.py** — `assign_exec_tp_sl` already
   calls `dynamic_tp_sl_series` which returns ATR-scaled absolute distances
7. **(no change) trading_bot_hybrid_v14.py** — uses sig.tp/sig.sl as absolute distances;
   once live scorer returns ATR-scaled absolutes, this works unchanged

## ATR Multiplier Selection
Calibrated to match typical gold 1-min ATR(14) ≈ $1.2–1.8:

| Pattern | H | TP×ATR | SL×ATR | Approx abs @ $1.5 |
|---------|---|--------|--------|-------------------|
| uptrend_retrace | 15 | 13 | 10 | $19.50 / $15 |
| downtrend_retrace | 15 | 27 | 20 | $40.50 / $30 |
| breakthrough_long | 15 | 27 | 13 | $40.50 / $19.50 |
| breakthrough_short | 30 | 27 | 20 | $40.50 / $30 |
| reversal_wick_long | 15 | 13 | 10 | $19.50 / $15 |
| reversal_fvg_long | 15 | 13 | 10 | $19.50 / $15 |
| reversal_wick_short | 15 | 27 | 20 | $40.50 / $30 |
| reversal_fvg_short | 15 | 27 | 20 | $40.50 / $30 |

## Post-change steps
- Must retrain all pattern models (labels change with ATR-based targets)
- Should run hybrid backtest to verify no regression
