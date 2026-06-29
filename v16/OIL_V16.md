# Oil v16 (crude)

Port of the oil three-leg system into v16 with gold-style ML and pattern upgrades.

## Lanes

| Lane | Pattern | ML | Exit |
|------|---------|-----|------|
| WR90 | Capitulation cluster long | 14D WF LGB | struct-hold (default) or fixed TP |
| Retrace (`ret`) | 15m long fade from Dlow (red bar) | 14D WF XGB | TP30/SL15 |
| **Retrace short (`ret_short`)** | 15m fade from Dlow (green extension bar) | 14D WF XGB ≥0.55 | TP30/SL15 |
| **Long retrace (`long_ret`)** | 15m bounce from Dhigh (green bar) | 14D WF XGB ≥0.50 | TP30/SL15 |
| SI | 1m short impulse | 14D WF ET | TP120/SL80 |
| Rip short | Slot rip ≥0.50 (v16 dip_short_rip port) | **Off by default** — weak OOS |

Single IG slot merge (same as live bot). Enable rip with `--rip` on the backtest CLI if experimenting.

## Run

```bash
# Phase A — model search (updates suggested OIL_LEG_MODELS)
PYTHONPATH=. python3 v16/research/oil_v16_ml_model_search.py 2024-01-01 2026-06-30

# Combined backtest (default: fixed TP WR90 + rip short)
PYTHONPATH=. python3 v16/research/oil_v16_combined_backtest.py 2024-01-01 2026-06-30

# Phase B — WR90 struct-hold exit (recommended — +7,193 combined OOS Jan24→Jun26)
PYTHONPATH=. python3 v16/research/oil_v16_combined_backtest.py --struct-hold

# Without rip-short lane
PYTHONPATH=. python3 v16/research/oil_v16_combined_backtest.py --no-rip
```

## Layout

- `v16/config/oil_config.py` — thresholds, ML, rip-short
- `v16/data/load_oil.py` — MySQL `prices` loader
- `v16/oil/wf_ml.py` — 14D walk-forward (xgb/et/lgb/hgb)
- `v16/oil/patterns.py` — signal collectors
- `v16/oil/structure_feats.py` — zigzag on 15m
- `v16/oil/struct_hold.py` — WR90 struct-hold sim
- `v16/oil/combined_run.py` — portfolio runner
- `v16/oil/sim_15m.py` — one-trade-per-signal 15m sim (no cascade)
- Models: `v16/oil/wf_models/{wr90|ret|ret_short|long_ret|si|rip}/`

## vs v15 oil live

Production bot still uses `oil/signal_engine.py` + monthly XGB in `v15/oil/wf_models/`.
v16 oil is research until backtest beats v15 and is wired to `oil_live_bot.py`.

## OOS results (2024-01-01 → 2026-06-30, single slot)

| Config | Combined PnL | Trades | vs v15 Option 1 (+4,236) |
|--------|-------------|--------|--------------------------|
| v16 5-leg (struct-hold WR90 + ret_short + long_ret) | **+10,566** | 1,230 | **+6,330** |
| v16 4-leg pre-fix (no ret_short) | +9,772 | 889 | +5,536 |
| v16 3-leg (no long_ret) | +7,193 | 419 | +2,957 |
| v16 fixed TP + 14D ML | +5,445 | 432 | +1,209 |

Best models (14D WF): WR90 LGB≥0.55, Ret XGB≥0.55, Ret-short XGB≥0.55, Long-ret XGB≥0.50, SI ET≥0.50.

Pre-merge leg PnL (5-leg, post bug-fix): WR90 +4,596 | ret +2,701 | ret_short +1,586 | long_ret +2,624 | SI +1,628.

Bug fixes in this run: `ret` uses long structure gate (not short); 15m legs use isolated sim (no target cascade); ML no pass-all fallback when WF untrained.

Rip-short lane needs threshold sweep (only ~9 signals at ML≥0.65).

## Long retrace (standalone research)

Run: `PYTHONPATH=. python3 v16/research/oil_long_retrace_backtest.py 2024-01-01 2026-06-30`

| Pattern | Description | OOS (Jan24→Jun26) |
|---------|-------------|-------------------|
| **Long retrace 15m** | Mirror of prod `ret` leg: pullback from **Dhigh** + green bar | ML XGB≥0.50: **+3,235** (586t, 45% WR) |
| Dip long 15m (gold port) | Prev 15m down + slot dip (gold `dip_long_15m`) | **Fails on oil** (~−11k mech) |

Long retrace 15m rules: `cah>20`, `avg_r3>30`, `bc>10`, `uw<16`, TP30/SL15 (same as ret).

Retrace short rules (mirror of ret, short side): `cad>20`, `avg_r3>30`, `bc>10`, `uw<16`, TP30/SL15.
