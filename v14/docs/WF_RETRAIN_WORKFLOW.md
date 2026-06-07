# Walk-forward retrain workflow (v14 hybrid)

## Intended behaviour

1. **Every 14 days** on the WF grid (`retrain_days: 14`, anchor `2025-01-03`), train **one** new model per leg **after the prior cycle ends**:
   - Example: cycle 37 = 2026-05-22 → 2026-06-05; **cycle 38** model trains **on/after 2026-06-06** (calendar day after start 2026-06-05). Do **not** re-train on 2026-05-30 mid-cycle.
   - Do **not** set `V14_WF_FORCE_LATEST` on scheduled retrains (overwrites a good cycle file).
   - Uses **all bars strictly before** the new cycle start (cycle 38 → data `< 2026-06-05`).
   - **Does not** overwrite older cycle files or re-train the cycle still in progress.

2. **Training does not use** the weak time filter (filter is applied only in backtest/live simulation).

3. **After** models are updated, run a **hybrid backtest without** weak filter, then **rebuild** `runtime/v14_weak_time_slots.json` from those trades.

## Commands

```bash
# Normal bi-weekly (incremental + weak filter rebuild)
PYTHONPATH=. .venv/bin/python3 v14/tools/retrain_hybrid_wf.py 2025-06-01 2026-05-23

# First-time bootstrap (rewrite every cycle file — slow)
V14_WF_TRAIN_MODE=full PYTHONPATH=. .venv/bin/python3 v14/tools/train_patterns_v14.py
# … same for train_filter_v14.py, train_stage2_v14_directional.py

# Force re-train latest cycle only (even if file exists)
V14_WF_FORCE_LATEST=1 V14_WF_TRAIN_MODE=incremental …
```

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `V14_WF_TRAIN_MODE` | `incremental` | Only latest cycle; `full` = old behaviour (all cycles) |
| `V14_WF_FORCE_LATEST` | unset | Set `1` on scheduled retrain to refresh current cycle |
| `V14_WF_TRAIN_AS_OF` | now (UTC) | Pin “today” for which cycle is “latest” |

Live bot scheduled retrain calls `retrain_hybrid_wf.py` with incremental mode.
