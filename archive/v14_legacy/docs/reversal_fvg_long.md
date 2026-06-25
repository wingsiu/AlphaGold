# reversal_fvg_long (5th pattern)

Reference backtest: **2025-06-01 → 2026-05-23**, combined **5-pattern** router (2398 baseline + this pattern).

| Metric | Target |
|--------|--------|
| Trades | **528** |
| Net PnL | **+2,614** |
| Δ vs 4-pattern baseline | **+78 trades**, **+216 PnL** |
| Trial-only (`reversal_fvg_long`) | **86 trades**, **+444 PnL**, WR ~53% |

Sim refresh: same as 2398 — `same_dir_refresh=entry`, `upgrade_stop=False`, `close_on_reverse=False`.

---

## Pattern definition

**15m bullish FVG** on XAUUSD 1m bars:

- **FVG detected:** `low[0] > high[-2]` on 15m (gap size filter: `V14_FVG_MIN_GAP=0`, any positive gap).
- **Direction:** long only.
- **Priority:** 6 (between retrace and breakthrough patterns).
- **Filter:** XGBoost prob ≥ 0.55.

### Train vs router (critical)

The +2,614 result depends on **different time windows for training vs live routing**:

| Stage | Key | Rule | Purpose |
|-------|-----|------|---------|
| **Training** | `pattern` | `time_from_fvg_bull < 30` | Label + fit samples (~66,639 bars) |
| **Backtest / live** | `router` | `time_from_fvg_bull < 45` | Which bars route to this specialist (~72,828 routed) |

Do **not** set both to 45 — retraining on the wider window drops combined PnL (~+2,386).

Implemented in `config/v14_patterns.py`:

```python
"pattern": [{"feat": "time_from_fvg_bull", "op": "<", "val": 30.0}],
"router":  [{"feat": "time_from_fvg_bull", "op": "<", "val": 45.0}],
```

`pattern_router.pattern_mask(..., training=True)` uses `pattern`; `assign_patterns()` / live bot use `router`.

Env override for sweeps only: `V14_REVERSAL_FVG_LONG_TIME_FROM_FVG_BULL_MAX=<minutes>`.

---

## Exec params

```python
reversal_fvg_long:  H=15  TP=20  SL=15
```

Model variant: `h15_tp20_sl15`

H/TP/SL sweep (pattern-only) favoured **TP30/SL15**, but combined 5-pattern PnL was lower (+2,508). **TP20/SL15 kept** for max combined delta vs baseline.

---

## Features

| Setting | Value |
|---------|--------|
| `feature_set` | **current** (102 cols with FVG PA enabled) |
| `pa_groups` | `("fvg",)` |
| Backtest matrix | **current** (widest; same as other non-v2398 patterns) |

**FVG model inputs** (6 cols when `pa_groups` includes `fvg`):

- `dist_fvg_bull_bottom`, `dist_fvg_bull_top` — distance from close to gap edges
- `dist_fvg_bear_top`, `dist_fvg_bear_bottom`
- `time_from_fvg_bull`, `time_from_fvg_bear`

Router uses **time only**; distance columns are XGB inputs.

**Gap size:** `V14_FVG_MIN_GAP=0` (points). Sweep showed min_gap > 0 hurts combined PnL.

---

## Sweep results (2026-05-24)

### Gap × time (`sweep_reversal_fvg_long.py`)

Best combined: **min_gap=0**, **router time<45** → +2,614.

CSV: `runtime/sweep_reversal_fvg_long.csv`

### H / TP / SL (`sweep_reversal_fvg_long_exec.py`)

Best pattern-only: H15/TP30/SL15 (+478 PnL, 93 trades). Not used for production (see exec above).

CSV: `runtime/sweep_reversal_fvg_long_exec.csv`

### A/B vs 4-pattern baseline (`try_add_pattern.py`)

| Trial | Combined Δ PnL |
|-------|----------------|
| **reversal_fvg_long** (time<30 train, later router<45) | **+216** |
| reversal_fvg_short | −84 |
| reversal_wick_long | −9 |
| reversal_wick_short | −194 |

Only **reversal_fvg_long** added net value on combined metrics.

---

## Reproduce

```bash
# 1) Train (uses pattern time<30 for samples + labels H15/TP20/SL15)
V14_FVG_MIN_GAP=0 .venv/bin/python3 train_patterns_v14.py reversal_fvg_long

# 2) Combined 5-pattern backtest + full stats (uses router time<45)
V14_FVG_MIN_GAP=0 .venv/bin/python3 run_pattern_backtest.py 2025-06-01 2026-05-23 \
  uptrend_retrace downtrend_retrace breakthrough_long breakthrough_short reversal_fvg_long
```

Trades CSV: `runtime/v14_pattern_backtest_trades.csv`

**Verify model:**

```bash
.venv/bin/python3 -c "
import joblib
from config.v14_patterns import PATTERN_MODEL_DIR
m = joblib.load('runtime/bot_assets/wf_models_v14_patterns/reversal_fvg_long/h15_tp20_sl15/filter_prod.joblib')
print(len(m.feature_names_in_), 'features')
"
```

Expected: **102** features.

---

## Checklist

- [ ] `V14_FVG_MIN_GAP=0`
- [ ] Train on `pattern` (< 30 min), route on `router` (< 45 min)
- [ ] Exec **H15 / TP20 / SL15**; model `h15_tp20_sl15`
- [ ] `feature_set=current`; FVG PA group enabled
- [ ] Combined backtest: ~**528 trades**, ~**+2,614 PnL**

---

## Verified result (2026-05-24)

5-pattern combined backtest:

- **528 trades**, **+2,613.7 PnL**
- reversal_fvg_long: 86 / +444 · uptrend_retrace: 344 / +1,080 · downtrend: 84 / +498 · breakthrough: 14 / +618

4-pattern baseline unchanged: **450 trades**, **+2,398 PnL** (see `docs/v14_2398_baseline.md`).

With **reversal_fvg_short** in a 6-pattern run, see [`docs/v14_6pattern_baseline.md`](v14_6pattern_baseline.md) (**545 / +2,764**).
