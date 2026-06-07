# oil_trader `short_impulse` — handoff for AlphaGold oil ML

Copy-paste context for agents working on **AlphaGold** `oil/` pipeline.  
**Not the same codebase** as `oil_trader` (HMM rule backtester vs v14 XGB pattern router).

## Repos

| Repo | Path | Role |
|------|------|------|
| **oil_trader** | `/Users/alpha/Desktop/python/oil_trader` | HMM regime backtester; `short_impulse` in `backtester.py` → `define_signals()` |
| **AlphaGold** | `/Users/alpha/AlphaGold` | v14 oil ML: `oil/patterns.py`, `oil/tools/train.py`, `oil/tools/backtest.py` |

## How to run oil_trader (always venv)

```bash
cd /Users/alpha/Desktop/python/oil_trader
.venv/bin/python backtester.py                    # full strategy (long+short), default init_from_ig()
.venv/bin/python run_short_impulse_backtest.py --start 2025-01-01 --end 2025-12-31   # short-only
.venv/bin/python analyze_short_impulse_monthly.py   # month-by-month 2024 → now, short-only
```

Do **not** use system `python3` (missing `talib`).

## Entry rule (source: `backtester.py`)

`change` = 1m close − open (IG bid/close scale in oil_trader CSVs — **not** AlphaGold `prices` 100× DB).

```python
short_impulse = (
    (change < -14) & (change.shift(1) < 10) & (change.shift(1) > -14) &
    (close - low < 35) &
    (volume > 1000) &
    (up_count3_15min != -3) &
    (high_day - close < 180)
)
short_entry = short_impulse & (change > -50) & (is_us_time | is_uk_time)
```

- `up_count3_15min`: sum of 15m bar directions over last 3 bars (`data_loader.py`).
- `high_day` / daily cols: merged from `1d_data.csv` (prior day).
- `short_only=True` in `backtest()` sets `conditions['long'] = False`.

## Exits (oil_trader ticks on bid/ask)

| Leg | Rule |
|-----|------|
| Short TP | `closePrice_bid - 70` |
| Short SL | `closePrice_bid + 40` |

Comment in code: “WTI $0.08 = 8 ticks” — thresholds are **IG point/tick units** on their price series, not AlphaGold DB ÷100.

## Key backtest results (user summary)

### Full strategy 2025-10-01 → 2026-06-04 (long + short)

- 835 trades, final capital **−$3,443** from $1k  
- Longs **−$3,663**; shorts **−$780**

### Short-only 2025-01-01 → 2025-12-31

- **127** trades, **+$1,288** (+128.8%), WR **50.4%**, PF **1.58**, max DD **−15.8%**, Sharpe **32.44**
- Files: `trades_2025-01-01_to_2025-12-31_short_only.csv`, `capital_2025-01-01_to_2025-12-31_short_only.csv`

### Short-only month-by-month 2024-01-01 → 2026-06-04

- **29** months, **16** losing months, sum monthly PnL ≈ **+$1,410**
- File: `short_impulse_monthly_2024_now.csv`

## Weak periods

| Window | Notes |
|--------|--------|
| H1 2024 (Jan–Jul) | Mostly negative; WR 9–36%; SL rate 57–67% |
| 2024-10 | −$216, 32% WR, 64% SL |
| 2025-06, 07, 10, 12 | Negative or flat |
| **2026-02–05** (worst) | 2026-03: 59 trades, −$380, 31% WR, 69% SL; spread ~4.7–4.9 vs ~3.0 in good months |

## Strong periods

| Window | Notes |
|--------|--------|
| 2024-05, 08, 09, 11 | WR 58–70%, solid PnL |
| 2025-02, 04, 05 | WR 67–73%, PF 3.4+ |

## Bad vs good month patterns

| Signal | Bad months | Good months |
|--------|------------|-------------|
| SL rate | Often **>55%** | Lower |
| Spread | **~4.7–5** (esp. 2026) | **~3.0** |
| Impulse frequency | High (e.g. Mar 2026: **127** impulses, **42%** of bars) | Lower |
| ATR | **~6.5** | **~4.1** |

Tuning ideas from `analyze_short_impulse_monthly.py`: tighten `|change|` (e.g. −14 → −16), session filter, spread cap before entry.

## AlphaGold pattern (implemented)

**Name:** `oil_short_impulse` — **oil_trader v2 rules**, `prob: 0.0` (rule-only by default).  
**Features:** `oil/short_impulse_features.py`  
**Execution:** H90, fixed TP=70, SL=40.

See oil_trader analysis: `/Users/alpha/Desktop/python/oil_trader/docs/short_impulse_analysis.md`

### ML optional

- `thresholds.prob: 0.0` → take every rule hit (no XGB).
- Set `prob: 0.55` and retrain if you want a filter on top of v2 rules.

### Rule-only holdout (v2, 2025-06-01 → 2026-05-23)

~4–few trades (strict filters + different session flags vs oil_trader). Use oil_trader backtester for primary validation.

```bash
cd /Users/alpha/AlphaGold
PYTHONPATH=. .venv/bin/python3 oil/tools/run_pattern.py oil_short_impulse
```

## AlphaGold commands (reference)

```bash
cd /Users/alpha/AlphaGold
PYTHONPATH=. .venv/bin/python3 oil/tools/run_pattern.py oil_short_impulse
```

Oil table: MySQL `prices`; models under `runtime/bot_assets/oil_pattern_models/`.
