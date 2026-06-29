# AlphaGold — agent guide

**Production: v15 hybrid bot.** The `v14/` folder has been retired (see `archive/v14_legacy/`).

## Entry points

| Goal | File |
|------|------|
| Live gold bot | `trading_bot_hybrid_v15.py` |
| Gold backtest / mobile compare | `v15/backtest/backtest_v15.py` or `run_pattern_backtest.py` |
| v16 scalp research (standalone) | `v16/research/scalp_scaleout_backtest.py` |
| Oil trading | `oil_live_bot.py` |
| Oil backtest | `oil/tools/backtest.py` → `oil/backtest/pattern_backtest.py` |
| Mobile API | `mobile_api/server.py` |
| WF retrain | `tools/retrain_hybrid_wf.py` |
| Launchd install | `scripts/install_launch_services.sh` |
| Cron watchdog | `watchdog_bots.sh` |

## Config (canonical names)

| Module | Purpose |
|--------|---------|
| `config/hybrid_config.py` | WF, execution, hybrid, time-filter |
| `config/pattern_registry.py` | Pattern definitions + model dir |
| `config/v14_config.py` | **Shim** → `hybrid_config` |
| `config/v14_patterns.py` | **Shim** → `pattern_registry` |

## Shared libraries

| Module | Purpose |
|--------|---------|
| `trading_bot_base.py` | IG execution base (subclassed by v15) |
| `backtest/core.py` | Trade simulation |
| `tools/` | WF retrain, time filter, pattern training |
| `scripts/` | Launchd + ops |

## Runtime paths

| Path | Notes |
|------|-------|
| `runtime/v15_backtest_trades.csv` | Gold backtest output |
| `runtime/hybrid_weak_time_slots.json` | Time filter (symlink from legacy name OK) |
| `runtime/bot_assets/wf_models/` | S1/S2 cycles (symlink → `wf_models_v14`) |
| `runtime/bot_assets/pattern_models/` | Pattern models (symlink → `wf_models_v14_patterns`) |

## v16 research (two winner lanes)

See **`v16/V16_WINNERS.md`** for full specs and OOS stats.

| Lane | Config | Best OOS (ML) |
|------|--------|----------------|
| Momentum pre-close | `v16_config.MOMENTUM_V16_WINNER_PRECLOSE` | ET p≥0.50, 14d WF → **+1,777** (254 tr) |
| Dip short rip | `v16_config.DIP_SHORT_RIP` | p≥0.70 → **~+918** (292 tr) |

Registry: `v16_config.V16_RESEARCH_WINNERS`

## Do NOT use

- `archive/v14_legacy/` for new work
- `trading_bot_hybrid_v14.py` (stub)
- `v14/` folder (removed — full tree in `archive/v14_legacy/source_tree/`)

## xgboost_filter_model (legacy filenames)

Training modules still named `train_filter_v14.py` etc. because `.joblib` cycle files embed `v14` in filenames. Do not rename without a dual-load migration.
