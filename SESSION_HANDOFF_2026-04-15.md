# AlphaGold handoff – 2026-04-15

## 1) Trading bot progress

### Current bot status
- Main file: `trading_bot.py`
- The bot now defaults to the promoted **best-base state-feature model**:
  - family: `best_base_state`
  - model: `training/_tmp_feature_single_split_aligned_state_stop_sweep/w150_h25_thr0.008_d15_lt0.006_st0.008_ls12_ss18_r0_f2.5_p10.48_p20.5_sf.joblib`
- Default mode is `signal_only`.
- It is **not live trading yet**.
- There is still **no live broker adapter** for IG order placement/position management in this repo, so best-base is currently used for signal generation only.

### What the bot already does
- Loads the promoted best-base model and generates causal 1-minute signals.
- Keeps a rolling in-memory prediction cache for gold minute data.
- Can split timing between:
  - `:05` prediction refresh from IG for signal generation
  - `:30` MySQL backup/snapshot sync for `gold`, `aud`, `oil`
- Persists bot state/status to:
  - `runtime/trading_bot_state.json`
  - `runtime/trading_bot_status.json`
- Supports a `--market-sync-only` mode.
- Keeps the old `legacy_15m_nextbar` paper-trading scaffold available separately.

### Recent verified state
From `runtime/trading_bot_status.json`:
- mode: `signal_only`
- signal family: `best_base_state`
- open position: none
- paper trades: `0`
- latest prediction cache bucket: `2026-04-15T09:41:00+00:00`
- cached gold rows: `10351`
- candidate best-base samples in cache: `3624`
- market data capture was disabled in that saved status snapshot

### Bot tests
- `test_trading_bot.py` has 25 tests passing.
- Latest saved run: `runtime/test_trading_bot_after_execution_hook.txt`
- Latest exit code: `runtime/test_trading_bot_after_execution_hook.exit.txt` = `0`
- Coverage includes:
  - best-base signal-only path
  - market-sync-only behavior
  - status persistence
  - poll timing helpers
  - paper broker helper behavior

### Useful bot commands
Run one cycle:
```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py --once
```

Run best-base signal only explicitly:
```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py \
  --signal-model-family best_base_state \
  --signal-model-path training/_tmp_feature_single_split_aligned_state_stop_sweep/w150_h25_thr0.008_d15_lt0.006_st0.008_ls12_ss18_r0_f2.5_p10.48_p20.5_sf.joblib \
  --mode signal_only \
  --once
```

Run data sync only:
```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py --market-sync-only --once
```

### Main unfinished bot items
- Add a real broker adapter / IG trading endpoints.
- Add proper live/paper execution engine for best-base exits/position handling.
- Compare live bot decisions against backtest/replay outputs.
- Add richer runtime monitoring and sizing/risk controls.

---

## 2) Finding models for trading

### Current promoted model
The current promoted **best base** is the aligned single-split **state-feature** winner with stop retune:
- `window=150`
- `horizon=25`
- `trend_threshold=0.008`
- `min_window_range=40`
- `min_15m_drop=15`
- `min_15m_rise=0`
- `long_target=0.006`
- `short_target=0.008`
- `long_stop=12`
- `short_stop=18`
- `max_flat_ratio=2.5`
- `stage1_min_prob=0.48`
- `stage2_min_prob=0.50`
- `use_state_features=true`
- `pred_history_len=150`

Primary artifact:
- `training/_tmp_feature_single_split_aligned_state_stop_sweep/w150_h25_thr0.008_d15_lt0.006_st0.008_ls12_ss18_r0_f2.5_p10.48_p20.5_sf.joblib`

### Why this model was promoted
From `training/feature_single_split_aligned_state_stop_sweep_summary.txt` and `training/README_image_trend_ml.md`:
- best stop pair from the aligned state-feature stop sweep: `long_stop=12`, `short_stop=18`
- promoted best-base snapshot:
  - trades: `1415`
  - total pnl: `6504.07`
  - avg day pnl: `91.61`
  - positive days: `67.61%`
  - stage2 balanced accuracy: `0.5560`
  - profit factor: `1.8391`
  - composite score: `7736.0837`

### Full single-split re-run / confirmation
The corrected rerun exists and finished successfully:
- script: `runtime/run_best_base_corrected.sh`
- wrapper output: `runtime/best_base_corrected.stdout.txt`
- exit code: `runtime/best_base_corrected.exit.txt` = `0`
- report: `training/backtest_report_best_base_corrected.json`
- trades: `training/backtest_trades_best_base_corrected.csv`

Confirmed rerun metrics:
- trades: `1415`
- total pnl: `6504.07`
- avg trade: `4.60`
- avg day: `91.61`
- positive days: `67.6%`
- long pnl: `4384.30`
- short pnl: `2119.77`
- overall profit factor: `1.8391`
- trade max drawdown: `-177.09`
- daily max drawdown: `-98.17`

### Walk-forward follow-up results
#### 10-cycle walk-forward using fixed promoted gates
Report:
- `training/backtest_report_best_base_wf_10cycles.json`

Key result:
- trades: `1328`
- total pnl: `3743.64`
- avg day: `58.49`
- positive days: `76.56%`
- profit factor: `1.4854`

Interpretation:
- still profitable, but much weaker than the single-split result
- suggests the promoted config may be over-optimistic vs stricter walk-forward validation

#### 10-cycle walk-forward with side-specific probability sweep
Report:
- `training/backtest_report_best_base_wf_10cycles_prob_sweep.json`

Key result:
- trades: `1294`
- total pnl: `3213.64`
- avg day: `49.44`
- positive days: `72.31%`
- profit factor: `1.4266`

Interpretation:
- the probability sweep did **not** improve the walk-forward result
- fixed promoted gates actually did a bit better than this follow-up WF sweep

### WR90 filter experiment
Report:
- `training/backtest_report_best_base_wr90_filter.json`

Key result:
- trades: `1400`
- total pnl: `4955.43`
- avg day: `71.82`
- positive days: `72.46%`
- profit factor: `1.5937`

Interpretation:
- interesting, but still below the promoted corrected best-base result (`6504.07`)
- not currently promoted over the `ls12/ss18` best-base artifact

### Bot-side replay after 2026-04-09
Replay tool:
- `run_best_base_mock_replay.py`

Recent outputs:
- `runtime/mock_best_base_after_2026-04-09_report.json`
- `runtime/mock_best_base_after_2026-04-09_report_v2.json`
- `runtime/mock_best_base_after_2026-04-09_trades.csv`
- `runtime/mock_best_base_after_2026-04-09_trades_v2.csv`

Replay summary on `2026-04-09` to `2026-04-15`:
- raw rows: `6027`
- candidate samples: `1212`
- tradable signals: `28`
- all tradable signals were `up`; no `down` signals
- executed trades: `8`
- total pnl: `-15.51`
- profit factor: `0.6365`

Interpretation:
- the promoted model looked strong on the broader backtest, but the short recent replay slice after Apr 9 was weak
- this is a useful warning sign and should probably be the next thing we investigate

### Best current conclusion on model search
- **Current promoted artifact remains the aligned state-feature stop-sweep winner** with `long_stop=12`, `short_stop=18`.
- It is the best **single-split** result we have confirmed.
- But walk-forward performance is materially weaker.
- And the short recent post-Apr-09 replay was negative.
- So the model is promising, but not yet something I would treat as “fully validated for live deployment”.

---

## Recommended next steps for the next chat
1. Re-check why post-Apr-09 replay is weak despite strong single-split metrics.
2. Compare bot-side replay vs training backtest trade-by-trade on the same slice.
3. Decide whether to trust the current best-base artifact, or promote a more robust walk-forward candidate instead.
4. If trading execution is next priority, build the IG broker adapter and keep best-base in signal-only until exits are implemented properly.

## Best files to open first next chat
- `SESSION_HANDOFF_2026-04-15.md`
- `README_trading_bot.md`
- `training/README_image_trend_ml.md`
- `trading_bot.py`
- `runtime/trading_bot_status.json`
- `training/backtest_report_best_base_corrected.json`
- `training/backtest_report_best_base_wf_10cycles.json`
- `training/backtest_report_best_base_wf_10cycles_prob_sweep.json`
- `runtime/mock_best_base_after_2026-04-09_report_v2.json`

## Exact command that reproduced the corrected best-base run
```bash
cd /Users/alpha/Desktop/python/AlphaGold
bash runtime/run_best_base_corrected.sh
```

