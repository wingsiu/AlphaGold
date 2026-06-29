# V16 research winners

**OOS window:** Jun 2025 → Jun 2026 (unless noted)  
**Production bot:** still v15 hybrid — these are the two v16 lanes to carry forward.

| Lane | Config key | Pattern |
|------|------------|---------|
| **1. Momentum pre-close** | `MOMENTUM_V16_WINNER_PRECLOSE` | `impulse_1m_15m` |
| **2. Dip short rip** | `DIP_SHORT_RIP` | `dip_short_rip` |

---

## 1. Momentum pre-close (`MOMENTUM_V16_WINNER_PRECLOSE`)

**Signal:** first 1m bar in 15m slot with \|body\| ≥ 3pt (London/NY)

**Entry**
- Breakout after impulse; bar *before* break closed within **10pt** of level
- Fill: **next_open**
- Gate: **with-trend** 15m structure (HH/HL long, LH/LL short)

**Exit (production candidate — struct-hold sweep Jun 2026)**
- **Struct-hold:** no TP, struct-exit always, **H=480–720** (sweep winner H=720 ML)
- Replaces R=3 H=120 for ML lane (+2,793 vs +1,777 OOS)
- SL: impulse H/L unchanged

**ML**
- Model: **Extra Trees (`et`)**
- Threshold: **p ≥ 0.50**
- Walk-forward: **v15 14-day grid** (`ML_CONFIG.retrain_freq: "14D"`)

**OOS results (Jun 2025→Jun 2026)**

| Mode | H | Trades | WR% | Net (pt) | Avg |
|------|--:|-------:|----:|---------:|----:|
| ML struct-hold | **720** | 174 | 41.4 | **+2,793** | +16.05 |
| ML struct-hold | 480 | 175 | 44.6 | +2,780 | +15.89 |
| ML baseline R=3 | 120 | 254 | 50.0 | +1,777 | +7.00 |
| Mech struct-hold | 600 | 493 | 22.9 | +1,819 | +3.69 |
| Mech baseline R=3 | 120 | 699 | 33.9 | +1,512 | +2.16 |

**Portfolio (v16 models — separate patterns, one gold position live)**

| Portfolio | Trades | Net (pt) | Notes |
|-----------|-------:|---------:|-------|
| **3-model:** preclose struct-hold ET + dip ML + open struct-hold LGB | 558 | **~+4,982** | sum of lanes (optimistic) |
| 2-model: preclose struct-hold + dip ML | 466 | **~+3,711** | recommended core |
| preclose + dip + open baseline LGB | 594 | ~+4,307 | open struct-hold beats open R=3 |

**3-model lanes (OOS Jun 2025→Jun 2026)**

| Lane | Config | Trades | Net |
|------|--------|-------:|----:|
| 1 Pre-close struct-hold | `MOMENTUM_V16_WINNER_PRECLOSE` ET p≥0.50 H=720 | ~174 | ~+2,793 |
| 2 Dip short rip | `DIP_SHORT_RIP` p≥0.70 | 292 | +918 |
| 3 Open struct-hold | `MOMENTUM_V16_WINNER` LGB p≥0.50 H=720 | 120 | +1,571 |

**Live caveat:** pre-close ↔ open momentum overlap **~58%** of entries within 60m — need a **priority router** (one position), not blind sum. Dip short is mostly orthogonal (~29% overlap with preclose).

**Do not** stack baseline R=3 + struct-hold on the **same** entry (one exit per trade).

Sweep: `runtime/v16_winner_struct_hold_sweep.csv` · `runtime/v16_winner_struct_hold_portfolio.csv`

**Research scripts**
- Backtest report: `v16/research/momentum_15m_hold_winner_report.py`
- ML sweep: `v16/research/momentum_15m_hold_winner_ml.py`
- Exit sweep: `v16/research/momentum_15m_hold_exit_sweep.py`
- TP sweep: `v16/research/momentum_15m_hold_winner_tpsl_struct_exit_sweep.py`

**Runtime artifacts**
- `runtime/v16_winner_preclose_et_trades.csv`
- `runtime/v16_winner_preclose_ml_sweep.csv`
- `runtime/v16_winner_exit_sweep.csv`

---

## 2. Dip short rip (`DIP_SHORT_RIP`)

**Signal:** prior 15m up + current slot up + price ripped **≥ 5pt** above slot open, minute in slot **< 10** (London/NY)

**Entry:** short at signal (next bar open)

**Exit (mechanical winner)**
- TP **40** / SL **30** / H **60**

**Exit (ML winner)**
- TP **35** / SL **35** / H **45**
- ML: **p ≥ 0.70** (`ml_prob` in config)
- Labels: **scaleout** (`ml_label_source`)

**OOS results** (from TP/SL sweep)

| Mode | Trades | Net (pt) |
|------|-------:|---------:|
| Mechanical (40/30/60) | ~797 | **~+1,216** |
| ML p≥0.70 (35/35/45) | ~292 | **~+918** |

**v15 behaviour:** single position; `same_dir_refresh: "entry"`; no `close_on_reverse` in this lane config.

**Research scripts**
- TP/SL sweep: `v16/research/dip_short_rip_tpsl_sweep.py`
- Prob sweep: `v16/research/dip_short_rip_prob_sweep.py`
- Full report: `v16/research/dip_short_rip_full_report.py`
- Run: `v16/backtest/dip_short_rip_run.py`

**Runtime artifacts**
- `runtime/v16_dip_short_rip_tpsl_sweep.csv`
- `runtime/v16_dip_short_rip_trades.csv`
- Models: `runtime/v16_models/dip_short_rip/`

**Note:** dip_short WF in `v16/backtest/ml.py` still uses **monthly** folds; align to v15 14d when wiring live retrain.

---

## 3. Structure trend hold — exploratory (`STRUCTURE_TREND_HOLD`)

**Goal:** ride multi-hour trends; tolerate 30–90 min retraces; **hold until structure breaks**.

**Entry (prototype):** 15m with-trend + pullback 15–65% + leg age 2–6 (30–90 min).  
**Exit:** swing break (trail last L/H) + struct_trend flip; 480m safety; no TP.

**First OOS:** 153 tr, -787 pt — prototype; tune TF / entry timing / atr_mult.

**Run:** `PYTHONPATH=. python3 v16/research/structure_trend_hold_backtest.py 2025-06-01 2026-06-25`

---

## Shared ML / WF conventions

| Setting | Momentum | Dip short |
|---------|----------|-----------|
| Min train window | 45 days | 45 days |
| Retrain grid | **v15 14d** | monthly (pending 14d) |
| Config module | `v16/config/v16_config.py` → `ML_CONFIG` | same |

**v15 production reference:** `config/hybrid_config.py` → `WF_CONFIG.retrain_days: 14`, `tools/retrain_hybrid_wf.py`

---

## Do not use (superseded for these lanes)

- Momentum: fixed-TP exit, reverse-signal exit, `MOMENTUM_VOL3_*` (below pre-close winner)
- Dip: execution-label ML (underperforms scaleout labels in sweep)
