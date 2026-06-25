# v14 six-pattern baseline (+2,764)

Reference backtest: **2025-06-01 → 2026-05-23**, combined router with **both FVG reversal** patterns.

| Metric | Target |
|--------|--------|
| Trades | **545** |
| Net PnL | **+2,764** |
| Sim refresh | `same_dir_refresh=entry`, `upgrade_stop=False`, `close_on_reverse=False` |

### vs other baselines

| Setup | Trades | Net PnL | Δ PnL vs 4-pattern |
|-------|--------|---------|-------------------|
| 4-pattern (2398) | 450 | +2,398 | — |
| 5-pattern (+ FVG long) | 528 | +2,614 | +216 |
| **6-pattern (+ FVG short)** | **545** | **+2,764** | **+366** |

---

## Active patterns

| Pattern | Priority | Train rule | Router rule | Exec H/TP/SL | Model |
|---------|----------|------------|-------------|--------------|-------|
| uptrend_retrace | 1 | (registry) | same | 15 / 20 / 15 | v2398 · 93 feat |
| downtrend_retrace | 2 | (registry) | same | 15 / 40 / 30 | v2398 · 93 feat |
| breakthrough_long | 3 | (registry) | same | 15 / 40 / 20 | current · 96 feat |
| breakthrough_short | 4 | (registry) | same | 30 / 40 / 30 | current · 96 feat |
| **reversal_fvg_long** | 6 | fvg bull **< 30m** | fvg bull **< 45m** | 15 / 20 / 15 | current · 102 feat |
| **reversal_fvg_short** | 8 | fvg bear **< 30m** | fvg bear **< 60m** | 15 / 40 / 30 | current · 102 feat |

Retrace + breakthrough: see [`docs/v14_2398_baseline.md`](v14_2398_baseline.md).

FVG long: [`docs/reversal_fvg_long.md`](reversal_fvg_long.md).

FVG short: [`docs/reversal_fvg_short.md`](reversal_fvg_short.md).

**Global FVG setting:** `V14_FVG_MIN_GAP=0` (required for both FVG patterns).

**Train vs router:** FVG patterns use separate `pattern` (training samples) and `router` (live routing) in `config/v14_patterns.py`. Implemented via `pattern_router.pattern_mask(..., training=True|False)`.

---

## Reproduce from scratch

```bash
# 1) Retrain 4-pattern baseline (hybrid feature sets)
V14_PATTERN_FEATURE_SET=v2398 .venv/bin/python3 train_patterns_v14.py \
  uptrend_retrace downtrend_retrace
.venv/bin/python3 train_patterns_v14.py breakthrough_long breakthrough_short

# 2) Retrain FVG patterns (min_gap=0; train uses pattern time<30)
V14_FVG_MIN_GAP=0 .venv/bin/python3 train_patterns_v14.py reversal_fvg_long reversal_fvg_short

# 3) Combined 6-pattern backtest + full stats
V14_FVG_MIN_GAP=0 .venv/bin/python3 run_pattern_backtest.py 2025-06-01 2026-05-23 \
  uptrend_retrace downtrend_retrace breakthrough_long breakthrough_short \
  reversal_fvg_long reversal_fvg_short
```

Trades CSV: `runtime/v14_pattern_backtest_trades.csv`

Backtest matrix: **`current` (96+ cols)** — widest set; retrace models use column subset via `feature_names_in_`.

---

## Verified breakdown (2026-05-24)

| Pattern | Trades | PnL |
|---------|--------|-----|
| uptrend_retrace | 342 | +1,070 |
| reversal_fvg_long | 87 | +438 |
| downtrend_retrace | 71 | +418 |
| reversal_fvg_short | 34 | +347 |
| breakthrough_long | 6 | +287 |
| breakthrough_short | 5 | +205 |
| **Total** | **545** | **+2,764** |

LONG: 435 trades · SHORT: 110 trades

---

## Checklist

- [ ] 4-pattern models correct (93/93/96/96 features)
- [ ] FVG long: train <30, route <45, H15/TP20/SL15, `h15_tp20_sl15`
- [ ] FVG short: train <30, route <60, H15/TP40/SL30, `h15_tp40_sl30`
- [ ] `V14_FVG_MIN_GAP=0` on train and backtest
- [ ] Entry refresh sim (2398 config)
- [ ] Combined: ~**545 trades**, ~**+2,764 PnL**

---

## Sweep artifacts

| Script | Output |
|--------|--------|
| `sweep_reversal_fvg_long.py` | `runtime/sweep_reversal_fvg_long.csv` |
| `sweep_reversal_fvg_long_exec.py` | `runtime/sweep_reversal_fvg_long_exec.csv` |
| `sweep_reversal_fvg_short.py` | `runtime/sweep_reversal_fvg_short.csv` |
| `sweep_reversal_fvg_short_exec.py` | `runtime/sweep_reversal_fvg_short_exec.csv` |
| `try_add_pattern.py` | `runtime/try_add_pattern_results.csv` |
