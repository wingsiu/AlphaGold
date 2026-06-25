# Oil pattern models (v14 pipeline)

Same machinery as gold pattern specialists, isolated in this folder:

- **Data:** MySQL table `prices` (IG `Price.Oil` → `CC.D.CL.BMU.IP`)
- **Models:** `runtime/bot_assets/oil_pattern_models/`
- **Trades CSV:** `runtime/oil_pattern_backtest_trades.csv`

## Should you use WF on the first test?

**No.** For initial oil work use **one prod model** (`filter_prod.joblib`) on the full holdout window — same idea as pinning gold to a single cycle before trusting 14-day WF.

| Mode | When |
|------|------|
| **Prod-only (default)** | First pattern, compare to gold `2025-06-01 → 2026-05-23` holdout |
| **`--wf`** | After a pattern shows edge; writes `filter_cycle_*.joblib` like gold |

Backtest loads cycle files when present; otherwise it uses **prod** for the entire window.

## Test period (aligned with gold)

| Window | Dates |
|--------|--------|
| Feature / train data | `2022-06-01` → `2026-05-23` (DB has oil from ~2022-05-11) |
| **Prod train** | all pattern bars **before** `2025-06-01` |
| **OOS backtest** | `2025-06-01` → `2026-05-23` (gold baseline holdout) |
| WF calendar (optional) | anchor `2025-01-03T22:00:00Z`, 14-day cycles |

Gold reference: `v14/docs/v14_2398_baseline.md` (~713 filtered trades, +4305 PnL on that window).

## Add patterns one-by-one

1. Copy a gold pattern shape into `oil/patterns.py` → `PATTERN_REGISTRY`.
2. Scale absolute `$` rules with `THRESHOLD_SCALE` in `oil/config.py` (default `2.0` for ~2× gold price level in DB).
3. Train + backtest **only that name**.

Patterns (one at a time):

- **`oil_short_impulse`** — oil_trader **v2 rules** (tighter drop, spread/ATR/crowding filters). Default `prob: 0.0` = rule-only; set `prob: 0.55` to add XGB.
- **`oil_bar_drop_short`** — 1m bear body `bar_bear_drop > 15`, `volume > 900`, short H120 TP80 SL50 (fixed).
- **`oil_downtrend_retrace`** — 240m retrace rules (paused; ATR target experiment).

**oil_trader `short_impulse` research** (separate repo, rule backtest +$1.3k short-only 2025): see [`oil/docs/short_impulse_handoff.md`](docs/short_impulse_handoff.md).

## Commands (repo root)

```bash
# One pattern: train prod-only + backtest holdout (no gold weak-slot filter)
PYTHONPATH=. .venv/bin/python3 oil/tools/run_pattern.py oil_short_impulse

PYTHONPATH=. .venv/bin/python3 oil/tools/run_pattern.py oil_downtrend_retrace

# Train only
PYTHONPATH=. .venv/bin/python3 oil/tools/train.py oil_downtrend_retrace

# Backtest only (after train)
PYTHONPATH=. V14_NO_TIME_FILTER=1 .venv/bin/python3 oil/tools/backtest.py oil_downtrend_retrace

# Enable WF cycle models (later)
PYTHONPATH=. .venv/bin/python3 oil/tools/train.py oil_downtrend_retrace --wf
```

## Data check

Confirm `prices` matches your instrument (WTI CFD). Sync:

```bash
PYTHONPATH=. .venv/bin/python3 -c "from ig_scripts.ig_data_api import *; ..."
```

If levels look wrong (e.g. not ~$60–80 WTI), fix IG epic / table before trusting thresholds.

## Next patterns (suggested order)

1. `oil_downtrend_retrace` ✅ starter  
2. `oil_uptrend_retrace`  
3. FVG / breakthrough variants after retrace works  

Do **not** reuse `runtime/hybrid_weak_time_slots.json` (gold-specific). Build a new weak-slot file from oil baseline trades if you add a time filter later.
