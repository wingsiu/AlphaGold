# Config Folder

Use this folder as the single source of truth for live/backtest artifact paths.

## Active paths

`active_paths.json` controls which files are used by:
- `run_live_bot.py`
- `trading_bot.py`
- `runtime/bot_assets/backtest_no_retrain.py`

Current defaults:
- `config/artifacts/model_bundle.joblib`
- `config/artifacts/weak-filter.json`

## Deployment workflow

1. Replace files in `config/artifacts/` with your new bundle and weak-filter.
2. Keep the same filenames (`model_bundle.joblib`, `weak-filter.json`) for zero code/XML changes.
3. Run backtest/live scripts as usual.

Optional: if you want custom names, edit only `config/active_paths.json`.

