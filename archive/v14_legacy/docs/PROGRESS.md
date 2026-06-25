# v14 Reorganization — Progress Log

**Updated:** 2026-05-25  
**Goal:** Minimal project root — production scripts only; everything else under `V13/` or `v14/`.

---

## Project root (production only)

```
AlphaGold/
├── trading_bot_hybrid_v14.py   # Live hybrid bot
├── trading_bot_v14.py          # Energetic-only bot (hybrid dependency)
├── run_hybrid_backtest.py      # Hybrid backtest entry
├── run_pattern_backtest.py     # Pattern backtest + stats
├── requirements.txt
├── brokers/  config/  data/  execution/  ig_scripts/  training/
├── xgboost_filter_model/       # v14 features + train modules
├── runtime/                    # Active models, state, live outputs
├── V13/                        # Legacy v13 (see V13/README.md)
└── v14/                        # Everything else v14 (this doc)
```

---

## `v14/` layout

| Path | Contents |
|------|----------|
| **`v14/backtest/`** | `backtest_core.py`, `backtest_patterns_v14.py` |
| **`v14/tools/`** | `train_patterns_v14.py`, `backtest_v14.py`, time filters, `daily_reconciliation.py` |
| **`v14/research/`** | Sweeps, investigate, evaluate, ab tests |
| **`v14/scripts/`** | `run_all_parallel.sh`, `run_parallel_sweep.sh` |
| **`v14/docs/`** | Baseline docs + `legacy/` (old README, DEPLOY_SERVER) |
| **`v14/runtime/results/`** | Archived backtest CSVs, time-filter outputs |
| **`v14/runtime/logs/`** | Sweep logs, old root `*.log` / `*.csv` outputs |
| **`v14/runtime/figures/`** | PNG plots (boxplots, feature charts) |
| **`v14/runtime/experiments/`** | Experimental WF model dirs |
| **`v14/runtime/model_snapshots/`** | Pattern training snapshots |
| **`v14/archive/old_results/`** | Former `_archive_old_results/` |
| **`v14/_paths.py`** | `PROJECT_ROOT` helper for scripts under `v14/` |

---

## Commands

```bash
# Live
.venv/bin/python3 trading_bot_hybrid_v14.py

# Backtest
.venv/bin/python3 run_hybrid_backtest.py 2025-06-01 2026-05-23
.venv/bin/python3 run_pattern_backtest.py 2025-06-01 2026-05-23

# Retrain
.venv/bin/python3 xgboost_filter_model/train_filter_v14.py
.venv/bin/python3 xgboost_filter_model/train_stage2_v14_directional.py
.venv/bin/python3 v14/tools/train_patterns_v14.py

# Time filter rebuild
.venv/bin/python3 v14/tools/run_hybrid_time_filter.py 2025-06-01 2026-05-23

# Reconciliation
.venv/bin/python3 v14/tools/daily_reconciliation.py 2026-05-22
```

---

## Active `runtime/` (unchanged location)

| Path | Purpose |
|------|---------|
| `runtime/bot_assets/wf_models_v14/` | Energetic S1/S2 cycle models |
| `runtime/bot_assets/wf_models_v14_patterns/` | 6 pattern models |
| `runtime/bot_assets/hmm_model.joblib` | HMM |
| `runtime/v14_weak_time_slots.json` | Time filter |
| `runtime/v14_pattern_backtest_trades.csv` | Latest backtest trades |
| `runtime/retrain_logs/` | Retrain logs |

---

## What moved (2026-05-25 cleanup)

- Root `*.log`, `*.csv`, `*.png`, `*.txt` → `v14/archive/logs/` or `v14/archive/figures/`
- `backtest_core.py`, `backtest_patterns_v14.py` → `v14/backtest/`
- `daily_reconciliation.py` → `v14/tools/`
- `README_trading_bot.md`, `DEPLOY_SERVER.md` → `v14/docs/legacy/`
- `_archive_old_results/` → `v14/archive/old_results/`
- Shell scripts → `v14/scripts/`

All imports updated to `v14.backtest.backtest_core` and subprocess paths to `v14/backtest/backtest_patterns_v14.py`.
