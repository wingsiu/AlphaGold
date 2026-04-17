# backtest_no_retrain.py

Interactive runner for **best-base no-retrain / no-state-features** backtests with weak filters.

## What it supports

- Runs `training/image_trend_ml.py` with `--model-in` (no retrain)
- Keeps state features off (no `--use-state-features`)
- Uses weak filter json (`--weak-periods-json`)
- 3 data options:
  - `A`: defaults up to today
  - `B`: latest N days
  - `C`: full predefined range
- Shows full statistics
- Shows time-distribution heatmaps
- Prompts to save outputs (overwrite per option)
- Prompts to save weak filters (overwrite if yes)

## Quick run

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u runtime/bot_assets/backtest_no_retrain.py
```

## Non-interactive run

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u runtime/bot_assets/backtest_no_retrain.py --option C --non-interactive
```

## Notes

- Default weak filter path is `runtime/bot_assets/weak-filter.json`.
- If the weak filter file does not exist, the script creates an empty one:
  `{"weak_cells": []}`

