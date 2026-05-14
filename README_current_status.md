# AlphaGold v13 — Progress & Status Log

---

## 🗓️ Last Updated: 2026-05-15

---

## ✅ Current Config — Single Source of Truth (`config/v13_config.py`)

| Parameter | Value |
|-----------|-------|
| **TP** | 40.0 pts |
| **SL** | 25.0 pts |
| **Horizon (timeout)** | 45 min |
| **S1 threshold** | 0.50 |
| **S2 base threshold** | 0.55 |
| **S2 loss increment** | +0.01 per consecutive loss |
| **S2 max threshold** | 0.70 |
| **Spread** | 0.25 pts |
| **Trade size** | 2.0 lots |
| **Bar filter** | move > 3.0, volume > 250 |
| **Backtest start** | 2025-01-01 |
| **Data start** | 2020-01-01 |

> **Dynamic S2 logic:** `side_signal` is pre-computed with the BASE threshold (0.55)
> so the roll/extend logic always works. Dynamic threshold (+0.01 per consecutive loss,
> max 0.70) is applied **only when opening NEW positions** — not for position management.
>
> **All three files read from this config — no hardcoded values:**
> - `trading_bot_v13.py` (live bot)
> - `backtest.py` (unified backtest — full history, recent N days, or custom range)
> - `daily_reconciliation.py` (daily P&L check)

---

## 📊 Full Backtest Results — v13 (Dynamic S2 correct logic, Fresh run 2026-05-15)

**Period: 2025-01-01 → 2026-05-15**

| Metric | Value |
|--------|-------|
| **Total Trades** | 1,310 |
| **Net PnL** | **+15,082 pts** |
| **Avg Trade** | +11.51 pts |
| **Win Rate** | **69.0%** |
| **Profit Factor** | **3.322** |
| **Max Drawdown** | -185.0 pts |
| **Avg PnL / Day** | +81.5 pts |
| **Positive Days** | 75.7% |
| **Trades / Day** | 7.1 |
| **Avg Duration** | 128.3 min |

### Side Breakdown
| Side | Trades | Net PnL | Win Rate | Avg Trade |
|------|--------|---------|----------|-----------|
| LONG | 711 | +7,076.5 | 67.2% | +9.95 |
| SHORT | 599 | +8,005.0 | 71.1% | +13.36 |

### Exit Breakdown
| Exit Type | Trades | Win Rate | Avg PnL |
|-----------|--------|----------|---------|
| reverse_signal | 589 | 87.6% | +18.55 |
| timeout | 468 | 66.9% | +9.27 |
| stop_loss | 178 | 0% | -10.00 |
| target_hit | 75 | 100% | +56.92 |

### Streaks
- Max Win Streak: **17**
- Max Loss Streak: **7**
- Current: 1 consecutive win (as of 2026-05-15)

### Monthly PnL
| Month | Trades | PnL | Win Rate |
|-------|--------|-----|----------|
| 2025-06 | 8 | +93.2 | 75.0% |
| 2025-07 | 15 | +97.4 | 66.7% |
| 2025-08 | 10 | +78.6 | 70.0% |
| 2025-09 | 26 | +11.6 | 46.2% |
| 2025-10 | 202 | +1,941.5 | 67.3% |
| 2025-11 | 45 | +481.2 | 73.3% |
| 2025-12 | 54 | +621.6 | 77.8% |
| 2026-01 | 204 | +2,096.5 | 59.8% |
| 2026-02 | 162 | +1,875.3 | 67.3% |
| **2026-03** | 414 | **+6,405.8** | 76.1% |
| 2026-04 | 136 | +1,315.7 | 68.4% |
| 2026-05 (partial) | 34 | +63.2 | 55.9% |

> **Note:** May 2026 still recovering (+63 pts / 34 trades, 56% WR).
> Live bot had 0 trades for ~4 days (May 10–14) due to image feature bug (now fixed).

### Max Drawdown Event
- **DD Start:** 2026-01-29 23:14 HKT
- **DD Bottom:** 2026-01-29 23:33 HKT (-185.0 pts)
- **Recovery:** 2026-01-30 00:07 HKT (~1h recovery)

---

## 📉 Session Heatmaps

```
[HKT — Total PnL]
               08:00   09:00   10:00   11:00   12:00   13:00   14:00   15:00
Monday         554.0   416.6   162.0   -14.6   163.7   135.1     6.6   103.2
Tuesday        251.3   142.9    -5.2    13.2    68.2    42.2    46.7    87.8
Wednesday       97.4   158.7     9.9   -13.9    10.5    73.5   112.6    -6.6
Thursday       135.1   259.1    85.9    54.1    44.1   254.6   323.2    69.2
Friday           5.4   221.7   -32.9     9.9       -    68.6   125.8    35.0

[HKT — Win Rate %]
               08:00   09:00   10:00   11:00   12:00   13:00   14:00   15:00
Monday           93%     92%     75%     60%     67%     80%     64%    100%
Tuesday          67%     63%     50%    100%    100%     43%     78%     86%
Wednesday        67%     77%     67%     67%     50%     88%     70%     50%
Thursday         77%     68%     75%     80%     67%    100%     64%     75%
Friday           25%     72%     27%    100%       -    100%     83%     60%

[NY — Total PnL]
               08:00   09:00   10:00   11:00   12:00   13:00   14:00   15:00
Monday         521.8   223.6   272.5   101.8   135.2    92.9   140.0    34.1
Tuesday        296.7   323.0   101.1   155.9   166.1   130.8    56.5   187.4
Wednesday      216.8   260.8   238.5    98.8    39.3   123.4   145.7     6.7
Thursday       194.7   465.2   190.1   233.2   276.9    37.2    -5.7   125.6
Friday         412.1   396.8   481.2   340.5   -20.7   189.7    88.7    35.4

[NY — Win Rate %]
               08:00   09:00   10:00   11:00   12:00   13:00   14:00   15:00
Monday           76%     68%     75%     68%     75%    100%     69%     43%
Tuesday          71%     66%     48%     72%     73%     80%     83%     78%
Wednesday        63%     68%     70%     78%     62%     33%     73%     40%
Thursday         74%     73%     58%     53%     85%     73%     33%    100%
Friday           71%     66%     67%     94%     57%     50%     75%     50%
```

---

## 🔧 Bugs Fixed (History)

### Bug 1 — Image Feature Mismatch (Fixed ~2026-05-14)
- **Symptom:** Live bot's S1 score always 0.30–0.38, never reaching 0.50 threshold → zero trades for days.
- **Root Cause:** `prepare_base_features()` uses `shift(-45)` for future labels + `dropna()`, silently dropping the last 45 bars every poll. Bot was scoring a stale bar → wrong features → low S1.
- **Fix:** Added `for_live_inference=True` flag to skip future label computation in live inference mode.

### Bug 2 — Dynamic S2 broke roll/extend logic (Fixed 2026-05-15)
- **Symptom 1 (first attempt):** Dynamic S2 logic set `side_signal=0` everywhere, so roll/extend never triggered. PnL dropped from ~$15k to $10k.
- **Symptom 2 (root cause):** `side_signal` must be pre-computed with the BASE threshold (0.55) before the loop — this is what drives roll/extend (extending timeout when same-direction signal fires on an open trade).
- **Fix (correct design):**
  - `side_signal` pre-computed with base S2=0.55 → roll/extend logic always reads valid signals
  - Dynamic threshold (`base + 0.01 × consecutive_losses`, max 0.70) is applied **only to gate new trade entries** — not position management
  - After a win (or profitable exit), `consecutive_losses` resets to 0
  - All 3 files use identical logic from `EXECUTION_CONFIG`

---

## 📁 Key Files

```
AlphaGold/
├── trading_bot_v13.py              ← Live bot
├── backtest.py                     ← ✅ Unified backtest (replaces 3 old files)
├── daily_reconciliation.py        ← Daily P&L reconciliation
├── config/v13_config.py           ← ✅ SINGLE CONFIG for all params
├── xgboost_filter_model/
│   └── v13_backtest_trades.csv    ← Latest full backtest trades
├── runtime/
│   ├── bot_assets/                ← Live model files
│   ├── recent_backtest_trades.csv ← Latest quick backtest output
│   ├── trading_bot_v13.log        ← Live bot log
│   └── _archive_old_backtests/   ← All old backtest files archived here
└── training/
    └── image_trend_model.joblib   ← Image trend model (stage1 image scorer)
```

---

## 🚀 How to Run

```bash
# Full backtest (Jan 2025 → today, WF cycle models):
python3 backtest.py

# Quick recent backtest:
python3 backtest.py 7      # last 7 days
python3 backtest.py 15     # last 15 days

# Custom date range:
python3 backtest.py 2025-03-01
python3 backtest.py 2025-03-01 2025-04-30

# Start live bot:
python3 trading_bot_v13.py --mode live --signal-model-family best_base_state
```

---

## 📌 Open Items / Next Steps

- [ ] Monitor May 2026 live performance — bot now fixed, expect signals to resume
- [ ] Consider filtering Fri HKT 10:00 (-32.9 pts historically)
- [ ] Consider filtering HKT Tue 10:00 and Wed 11:00 (slightly negative)
- [ ] Investigate Sep 2025 underperformance (48% WR, +52 pts only)

---

*Last updated: 2026-05-15 by Copilot*
