# AlphaGold — Current Status (2026-04-19)

## Active Model
```
runtime/backtest_model_best_base_weak_nostate.joblib
```
Gradient Boosting, single-position, no state features, weak-period filter applied.

## Best Base Parameters
```
--window 150
--horizon 25
--trend-threshold 0.008
--long-target-threshold 0.006   # long take-profit = entry × 1.006
--short-target-threshold 0.008  # short take-profit = entry × 0.992
--long-adverse-limit 12         # long stop = entry - 12
--short-adverse-limit 18        # short stop = entry + 18
--adverse-limit 15
--stage1-min-prob 0.48
--stage2-min-prob 0.50
--min-window-range 40
--min-15m-drop 15
--max-flat-ratio 2.5
--classifier gradient_boosting
--timeframe 1min
```

## Weak Period Filter
```
runtime/bot_assets/weak-filter.json  (19 cells)
```

## How to Run Backtest (no retrain)
```bash
python3 runtime/bot_assets/backtest_no_retrain.py \
  --option B --b-start-date 2026-04-13 --b-end-date 2026-04-19 \
  --non-interactive
```
Results saved to `runtime/backtest_no_retrain/` (gitignored).

## How to Run Live Bot
```bash
python3 run_live_bot.py
```

## Key Fix Applied Today (2026-04-19)
**Target-hit now triggers a real early exit.**

Previously: `target_abs` was computed but the trade always exited at
the horizon-bar close (`fut[i]`). The label `target_hit` was only
applied retroactively if `pnl >= target_abs`.

Fixed in: `training/image_trend_ml.py` — `_backtest_trades_df()`

Now: when `curr[i]` crosses `entry ± target_abs` during the simulation
loop, the trade exits immediately at the target price (same pattern as
the stop-loss check). This mirrors real trading behaviour.

**All backtest results produced before 2026-04-19 are invalid** and
have been deleted as part of the cleanup on this date.

## Improvement Ideas (to try next)
- Raise `--stage1-min-prob 0.60`, `--stage2-min-prob 0.58` → reduce 19.49 trades/day
- Raise `--long-adverse-limit 15` → give longs more room (win rate 49.8% → target 52%+)
- Raise `--long-target-threshold 0.008` → ride winners further once stops widened
- Raise `--min-window-range 50` → filter low-volatility entries

