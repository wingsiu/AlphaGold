# V15 Pattern Development — Progress Tracker

**Last updated:** 2026-06-14

---

## Requirements for ALL Patterns (Must Fulfill)

- [ ] **No HMM features** — `hmm_regime` column must be dropped before training
- [ ] **Use bid/ask prices** — `openPrice_ask`, `openPrice_bid`, `closePrice_ask`, `closePrice_bid`, `highPrice_ask`, `lowPrice_bid` mapped into simulation
- [ ] **Walk-Forward training** — per-cycle retrain: `train = df.index < cycle_start`, `test = [cycle_start, cycle_end)`
- [ ] **Tested standalone first** — pattern validated alone before combining with others
- [ ] **No future data leak** — features computed only from bars ≤ current timestamp; labels use future price moves (deterministic, no model leakage)
- [ ] **Single position per side** — `close_on_reverse=True`, `same_dir_refresh="entry"`

## WF Cycle Parameters

| Parameter | Value |
|-----------|-------|
| Test period | 2025-06-01 → 2026-06-14 |
| WF anchor | bi-weekly cycles |
| Min train bars | 200 (long), 400 (short) |
| Min test bars | 5 (long), 10 (short) |

---

## Pattern Status

### 1. Short Retrace (Downtrend) — ✅ READY

| Property | Value |
|----------|-------|
| **Definition** | WR-based: F15S60 ob=-5 cool=-20 os=-40 rec=-30 slk=10 rob=20 ros=10 |
| **Feature set** | `v2398` (93 cols, no HMM, no 15m PA) |
| **Horizon** | 480 min |
| **TP / SL** | 0.65×ATR5 / 0.30×ATR5 |
| **Prob threshold** | 0.40 |
| **Training** | Per-cycle WF retrain |
| **Standalone PnL** | **+1,368** |
| **Standalone WR** | 50.4% |
| **Standalone Trades** | 133 |

**Entry logic:**  
Fast WR(15) was overbought (≥ -5 in last 20 bars) → now cooled (≤ -20)  
Slow WR(60) was oversold (≤ -40 in last 10 bars) → now recovered (≥ -30)  
Fast WR slope over 10 bars is negative (bounce failing → short)

**Sweep scripts:** `/tmp/v38_relaxed_sweep.py`, `/tmp/v38_horizon_tpsl_sweep.py`  
**Full stats:** `/tmp/v38_best_full_stats.py`  
**Results CSV:** `runtime/v38_horizon_tpsl_sweep.csv`

---

### 2. Long Retrace (Uptrend) — 🔄 IN PROGRESS

#### V15 Daily ATR Approach (V32)

| Property | Value |
|----------|-------|
| **Definition** | Price-move: rise_from_low_240 ≥ daily_atr14 × 0.30, drop_from_high_240 ≥ daily_atr14 × 0.10, near_high_zone ≠ 1 |
| **Feature set** | `current` (96 cols, 15m PA wick) |
| **Horizon** | 120 min |
| **TP / SL** | 0.15×ATR14 / 0.112×ATR14 |
| **Prob threshold** | 0.45 |
| **Label** | Symmetric: `(fmax >= tp_abs) & (fmin <= sl_abs)` |

**Results (non-WF, single model all-data):**

| Version | Trades | PnL | WR | MaxDD |
|---------|--------|------|------|--------|
| V32 standalone | 800 | **+6,768** | 65.8% | -74 |
| V39v6 (single model) | 830 | **+9,062** | 78.0% | -88 |

**Results (strict WF per-cycle retrain):**

| Version | Trades | PnL | WR | MaxDD |
|---------|--------|------|------|--------|
| V39v7 (WF) | 1,091 | **-313** | 36.0% | -841 |

**Issue:** Strict WF degrades long retrace significantly. Early cycles have too few training bars (200-800) — XGBoost cannot learn effective signal. The non-WF version (train on all data) produces +6,768 to +9,062 PnL but uses feature distributions from test period.

**Next steps:**
- [ ] Sweep rise/drop thresholds for WF robustness
- [ ] Try WR-based definition for long (mirror of short)
- [ ] Try simpler model (max_depth=2-3, fewer estimators) for small-sample WF
- [ ] Consider earlier training start (2020-01-01) to accumulate more bars before test

#### WR-Based Long Approach (V38L)

| Property | Value |
|----------|-------|
| **Definition** | Fast WR was oversold → warmed, Slow WR was overbought → cooled, slope positive |
| **Best standalone** | +458 PnL (weak) |
| **Verdict** | ❌ Not viable — WR definition doesn't work well for long side in uptrend regime |

**Script:** `/tmp/v38_long_retrace_wr.py`  
**Results:** `runtime/v38_long_retrace_wr.csv`

---

### 3. Energetic Gate — ⏳ PENDING

| Property | Status |
|----------|--------|
| **Script** | `v15/energetic_gate.py` |
| **Definition** | Momentum/volatility breakout filter |
| **Standalone tested** | ❌ Not yet |
| **WF compliant** | ❌ Not verified |

**To do:**
- [ ] Audit energetic standalone performance with WF rules
- [ ] Determine if energetic should be a pre-filter or a standalone signal
- [ ] Integrate after long retrace is finalized

---

## Combination Progress

| Attempt | Long | Short | Combined PnL | Combined WR | MaxDD | Notes |
|---------|------|-------|-------------|-------------|-------|-------|
| V39v2 | V15 ATR (single model) | V38 WR (WF) | +3,807 | 45.3% | -148 | Mixed TP/SL bug |
| V39v6 | V15 ATR (single model, separate df) | V38 WR (WF, separate df) | **+10,430** | 74.1% | -88 | Long non-WF |
| V39v7 | V15 ATR (WF) | V38 WR (WF) | +1,055 | 37.6% | -841 | Both strict WF |

**Current combined script:** `/tmp/v39_combined_retrace.py`

---

## Process

1. ✅ Short retrace — standalone validated, WF compliant, **READY**
2. 🔄 Long retrace — find correct definition that survives WF
3. ⏳ Energetic — audit and integrate if ready
4. ⏳ Combine all ready patterns → final V15 combined

---

## Live Bot Architecture (recorded 16 Jun 2026)

### Running Bots
- **Gold v15** (`trading_bot_hybrid_v15.py`) — `:05` poll_trade, `:30` poll_db
- **Oil** (`oil_live_bot.py`) — `:06` fetch

### `:05` poll_trade (Gold v15 only)
Fetches Gold 1m data from IG → builds features → scores patterns/energetic → manages positions

### `:30` poll_db (Gold v15) — fetches ALL THREE to MySQL
- `Price.Gold` → `gold_prices`
- `Price.Oil` → `prices`
- `Price.AUD` → `aud_prices`

### Manual Trades Do NOT Block Bot Entries (fixed 16 Jun)
Both bots ignore untracked IG positions. Only block if bot's OWN tracked position is still open.

### IG API Limitation
20,000 point limit per request on chart/snapshot endpoint. `fetch_and_store_prices_from_latest` chunks into 14-day segments for large backfills.

### Known Issues
- Oil epic (`CC.D.CL.BMU.IP`) gets 502 errors during US peak hours — IG server-side, self-resolves within minutes.
- `oil_prices` table doesn't exist in MySQL — oil data lives in shared `prices` table (works fine).

### IG 1-Minute Data Retention
- **~38-40 days** max lookback for 1-minute granularity (tested 16 Jun 2026: data back to ~9 May 2026).
- Beyond that IG returns empty results — this is IG's retention policy, not a bug.

---

## Key Files

| Purpose | Path |
|---------|------|
| Short standalone sweeps | `/tmp/v38_*.py` |
| Long standalone | `/tmp/v39_long_retrace_fixed.py` |
| Combined backtest | `/tmp/v39_combined_retrace.py` |
| Short full statistics | `/tmp/v38_best_full_stats.py`, `/tmp/v38_best_h480_stats.py` |
| V32 reference (long) | `v15/research/v32_uptrend_retrace_v15_proper.py` |
| Energetic gate | `v15/energetic_gate.py` |
| Backtest core | `v14/backtest/backtest_core.py` |
| Feature definitions | `config/v14_patterns.py` |
