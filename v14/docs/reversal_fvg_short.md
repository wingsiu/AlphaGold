# reversal_fvg_short (6th pattern)

Reference backtest: **2025-06-01 → 2026-05-23**, combined **6-pattern** router (2398 baseline + both FVG patterns).

| Metric | Target (6-pattern) |
|--------|-------------------|
| Trial-only (`reversal_fvg_short`) | **34 trades**, **+347 PnL**, WR ~44% |
| Δ vs 5-pattern (+ long only) | **+17 trades**, **+150 PnL** |

See [`docs/v14_6pattern_baseline.md`](v14_6pattern_baseline.md) for full combined stats.

Sim refresh: same as 2398 — `same_dir_refresh=entry`, `upgrade_stop=False`, `close_on_reverse=False`.

---

## Pattern definition

**15m bearish FVG** on XAUUSD 1m bars:

- **FVG detected:** `low[-2] > high[0]` on 15m (`V14_FVG_MIN_GAP=0`).
- **Direction:** short only.
- **Priority:** 8 (after wick short trial, before none).
- **Filter:** XGBoost prob ≥ 0.55.

### Train vs router (critical)

| Stage | Key | Rule | Purpose |
|-------|-----|------|---------|
| **Training** | `pattern` | `time_from_fvg_bear < 30` | Label + fit samples |
| **Backtest / live** | `router` | `time_from_fvg_bear < 60` | Route bars to specialist |

```python
"pattern": [{"feat": "time_from_fvg_bear", "op": "<", "val": 30.0}],
"router":  [{"feat": "time_from_fvg_bear", "op": "<", "val": 60.0}],
```

Env override for sweeps: `V14_REVERSAL_FVG_SHORT_TIME_FROM_FVG_BEAR_MAX=<minutes>`.

---

## Exec params

```python
reversal_fvg_short:  H=15  TP=40  SL=30
```

Model variant: `h15_tp40_sl30`

H/TP/SL sweep (pattern-only, `min_gap=0`) best: **H45/TP45/SL25** (+808 PnL, 119 trades). **Not used** for combined 6-pattern — kept **H15/TP40/SL30** (registry default; combined test with router<60 was better).

CSV: `runtime/sweep_reversal_fvg_short_exec.csv`

---

## Features

Same FVG PA group as long — `pa_groups=("fvg",)`, `feature_set=current` (102 cols). See [`docs/reversal_fvg_long.md`](reversal_fvg_long.md#features).

**Gap size:** `V14_FVG_MIN_GAP=0`. Sweep at min_gap>0 with tight time had high Δ but only **4 trades** — not production-ready.

Gap × time sweep CSV: `runtime/sweep_reversal_fvg_short.csv`

Best **min_gap=0** combined (5-pattern, short only): **time<60** → Δ **+51** vs baseline.

---

## Reproduce

```bash
V14_FVG_MIN_GAP=0 .venv/bin/python3 train_patterns_v14.py reversal_fvg_short

V14_FVG_MIN_GAP=0 .venv/bin/python3 run_pattern_backtest.py 2025-06-01 2026-05-23 \
  uptrend_retrace downtrend_retrace breakthrough_long breakthrough_short \
  reversal_fvg_long reversal_fvg_short
```

Model: `runtime/bot_assets/wf_models_v14_patterns/reversal_fvg_short/h15_tp40_sl30/filter_prod.joblib`

---

## Checklist

- [ ] `V14_FVG_MIN_GAP=0`
- [ ] Train on `pattern` (< 30 min), route on `router` (< 60 min)
- [ ] Exec **H15 / TP40 / SL30**; model `h15_tp40_sl30`
- [ ] In 6-pattern run: ~**34** short trades, ~**+347** short PnL

---

## Verified result (2026-05-24)

6-pattern combined backtest:

- reversal_fvg_short: **34 / +347.3**
- See full breakdown in [`docs/v14_6pattern_baseline.md`](v14_6pattern_baseline.md)
