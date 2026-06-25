# v14 pattern baseline (+2,398 / ~450 trades)

Reference backtest: **2025-06-01 → 2026-05-23**, combined 4-pattern router.

| Metric | Target |
|--------|--------|
| Trades | **450** |
| Net PnL | **+2,398** |
| Sim refresh | `same_dir_refresh=entry`, `upgrade_stop=False`, `close_on_reverse=False` |

---

## Root causes (why reproduction failed for a day)

1. **Training labels used global 30/25/30** instead of each pattern’s exec H/TP/SL from `PATTERN_REGISTRY`.
2. **Feature matrix drift**: shared `add_pattern_features()` grew from **93 → 96** columns when breakthrough work added:
   - `rise_from_day_low`
   - `drop_from_day_high`
   - `wr_120_cross_up_10`
3. **Mixed feature sets**: retrace patterns (uptrend/downtrend) were trained on the **original 93**; breakthrough was trained on **96**. Retraining *everything* on 93 broke breakthrough (~−31 PnL).
4. **`c15_breakthrough_up/down` are router-only** (in `EXCLUDE_COLS`) — not XGB inputs. Breakthrough models still use `wr_90`, `ret_3m`, etc.
5. **Sweep order**: uptrend H/TP/SL must be swept under the correct feature set before locking exec params.
6. **Lost weights**: old prod models were overwritten during PA/sweep/retrain; reproduction requires retrain + correct config, not just reloading files.

---

## Hybrid feature sets (required for 2398)

| Pattern | `feature_set` | Model inputs | Notes |
|---------|---------------|--------------|-------|
| uptrend_retrace | **v2398** | **93** | H15 / TP20 / SL15 (sweep winner on v2398) |
| downtrend_retrace | **v2398** | **93** | H15 / TP40 / SL30 |
| breakthrough_long | **current** | **96** | H15 / TP40 / SL20 |
| breakthrough_short | **current** | **96** | H30 / TP40 / SL30 |

**Backtest / live inference:** always build the **`current` (96-column)** feature matrix. Retrace models use a **subset** of those columns via `model.feature_names_in_`.

Env overrides (optional):

- `V14_PATTERN_FEATURE_SET=v2398|current` — forces one set for all patterns (training/sweeps only).
- `V14_BACKTEST_FEATURE_SET=current` — default; do not use v2398 for combined backtest.

---

## Exec params (registry)

```python
uptrend_retrace:     H=15  TP=20  SL=15
downtrend_retrace:   H=15  TP=40  SL=30
breakthrough_long:   H=15  TP=40  SL=20
breakthrough_short:  H=30  TP=40  SL=30
```

Uptrend-only v2398 sweep best: **H15/TP20/SL15** → 381 trades, +1,207 (entry refresh).

---

## Reproduce from scratch

```bash
# 1) Train retrace patterns (93 features)
V14_PATTERN_FEATURE_SET=v2398 .venv/bin/python3 train_patterns_v14.py \
  uptrend_retrace downtrend_retrace

# 2) Train breakthrough patterns (96 features — default)
.venv/bin/python3 train_patterns_v14.py breakthrough_long breakthrough_short

# 3) Combined backtest (96-col matrix, do NOT set v2398)
.venv/bin/python3 run_pattern_backtest.py 2025-06-01 2026-05-23
```

Or use per-pattern `feature_set` in registry (no env) via updated `train_patterns_v14.py`:

```bash
.venv/bin/python3 train_patterns_v14.py   # trains each pattern with its registry feature_set
.venv/bin/python3 run_pattern_backtest.py 2025-06-01 2026-05-23
```

**Verify model feature counts:**

```bash
.venv/bin/python3 -c "
import joblib
from config.v14_patterns import PATTERN_REGISTRY, PATTERN_MODEL_DIR
from xgboost_filter_model.pattern_training import pattern_execution, pattern_variant_tag
for n, s in PATTERN_REGISTRY.items():
    ex = pattern_execution(n)
    v = pattern_variant_tag(ex['horizon'], ex['tp'], ex['sl'])
    m = joblib.load(PATTERN_MODEL_DIR / n / v / 'filter_prod.joblib')
    print(n, s.get('feature_set'), len(m.feature_names_in_))
"
```

Expected: uptrend/downtrend **93**, breakthrough **96**.

---

## Training labels

Each pattern must use **its own exec H/TP/SL** for labels (`label_df_for_pattern` / `apply_exec_labels`), **not** global `TARGET_CONFIG` 30/25/30.

Models save to: `runtime/bot_assets/wf_models_v14_patterns/<pattern>/h*_tp*_sl*/`

---

## Sim / refresh (do not change for 2398)

From `config/v14_config.py` → `EXECUTION_CONFIG`:

```python
"same_dir_refresh": "entry",   # trail target + extend timeout from entry TP/H only
"upgrade_stop": False,
"close_on_reverse": False,
```

Other refresh modes tested (exec + upgrade_stop, etc.) — worse or identical for uptrend-only when exec_* is constant per pattern.

---

## Sweeps

Uptrend retrace (v2398, entry refresh):

```bash
V14_PATTERN_FEATURE_SET=v2398 .venv/bin/python3 sweep_pattern_uptrend_retrain.py 2025-06-01 2026-05-23
# → runtime/sweep_uptrend_retrace_retrain_v2398.csv
```

---

## Checklist before trusting a retrain

- [ ] Per-pattern exec H/TP/SL in labels matches registry
- [ ] Retrace models: **93** features (`feature_set=v2398`)
- [ ] Breakthrough models: **96** features (`feature_set=current`)
- [ ] Combined backtest uses **96-col** feature prep
- [ ] Entry refresh sim config (not exec/stop trail unless explicitly A/B testing)
- [ ] Compare combined: ~**450 trades**, ~**+2,398 PnL**

---

## Verified result (2026-05-24)

Hybrid retrain + combined backtest:

- **450 trades**, **+2,398.2 PnL**
- uptrend 357 / +1,280 · downtrend 76 / +412 · breakthrough 17 / +706

---

## 5-pattern (+ reversal_fvg_long)

Add **`reversal_fvg_long`** per [`docs/reversal_fvg_long.md`](reversal_fvg_long.md):

- Combined target: **528 trades**, **+2,614 PnL** (Δ +216 vs baseline)
- Requires train/router split (`pattern` < 30 min, `router` < 45 min) and `V14_FVG_MIN_GAP=0`

---

## 6-pattern (+ reversal_fvg_short)

Add **`reversal_fvg_short`** on top of 5-pattern per [`docs/v14_6pattern_baseline.md`](v14_6pattern_baseline.md):

- Combined target: **545 trades**, **+2,764 PnL** (Δ **+366** vs 4-pattern, **+150** vs 5-pattern)
- FVG short: train `< 30 min`, router `< 60 min`, H15/TP40/SL30, `V14_FVG_MIN_GAP=0`
- Detail: [`docs/reversal_fvg_short.md`](reversal_fvg_short.md)
