# Trading Bot (v1 scaffold)

This repo now includes `trading_bot.py`, a safe first trading-bot scaffold built from the current AlphaGold workspace.

## What it does

- loads recent `gold_prices` candles from MySQL via `DataLoader`
- by default loads the promoted **best base** `training/image_trend_ml.py` state-feature model artifact
- generates the latest best-base signal causally using the same rolling 1-minute image/state-feature logic
- prints a clearer best-base log line like `BEST_BASE SIGNAL: side=up prob=... ts=...`
- keeps the older 15-minute next-bar scaffold available as an optional legacy signal family
- legacy paper mode still uses the older 15-minute next-bar take-profit / stop-loss / timeout scaffold
- persists bot state to disk so repeated runs do not duplicate entries
- writes a runtime status summary with realized P&L / trade counts / open-position snapshot
- supports optional daily entry caps and cooldown-after-exit risk controls
- around `:05` each minute, refreshes a live in-memory **gold** minute cache from IG for best-base prediction
- around `:30` each minute, backfills/stores latest IG minute data for `gold`, `aud`, and `oil` into MySQL for backup/backtest usage later

## Current scope

This is intentionally a staged scaffold:

- `signal_only`: model signals + dry-run execution attempt records
- `paper`: legacy 15-minute next-bar paper execution
- `live`: best-base live order submit + open-position tracking + dynamic target/stop updates

It does **not** impose extra session-window filters by default. The intent is to
let the trading bot follow the model's own signal logic unless a separate risk
gate is explicitly requested later.

The workspace currently has historical IG access in `ig_scripts/ig_data_api.py`, but it does not yet expose the live trading methods used by `old_bot.py` (`place_order`, `get_positions`, `delete_position`, etc.).

So this version gives you:

- default: promoted best-base **signal generation** via `--mode signal_only`
- `--mode paper` → legacy 15-minute next-bar paper entries/exits + CSV trade log
- `--mode signal_only` → signal generation only, no paper position management
- `--mode live` (best-base only) → submits live IG market orders and tracks `deal_id`
- `--market-sync-only` → run only the IG latest-DB-to-now sync + snapshot task and write status/log output

## Default signal model

The bot defaults to:

- signal model family: `best_base_state`
- signal model path: `training/_tmp_feature_single_split_aligned_state_stop_sweep/w150_h25_thr0.008_d15_lt0.006_st0.008_ls12_ss18_r0_f2.5_p10.48_p20.5_sf.joblib`
- mode: `signal_only`

These artifacts already exist in the workspace.

## Run one cycle

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py --once
```

That default run now uses the promoted best-base state-feature model in signal-only mode.

At startup the bot now logs a `MODEL CONFIG:` line showing active family/mode/size and key model settings.

When the default best-base path is active, the log now emits lines like:

- `BEST_BASE INPUT: source=prediction_memory_cache reload_each_cycle=False ...`
- `BEST_BASE SIGNAL: side=up prob=0.8123 ts=2026-04-15T06:09:00+00:00 entry_ts=2026-04-15T06:10:00+00:00 ... market_open=1`

With the newer live minute split, the best-base path uses:

- `:05` prediction refresh → update in-memory gold minute cache from IG and generate the minute prediction
- `:30` MySQL backup refresh → store latest IG data into MySQL for later recovery/backtest usage

## Run continuously

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py --sleep-seconds 5
```

The continuous loop is aligned to real clock boundaries so the best-base prediction task can run around `HH:MM:05` and the MySQL backup task can run around `HH:MM:30`.

It also emits a periodic performance snapshot log at `:00` and `:30` each hour (once per bucket) with 30m/60m/today/week realized summaries.

Signal logs also include an easy-read `SIGNAL STATUS` line with:

- `hour=[OK]/[X]` (currently inside trading hours)
- `signal=[OK]/[X]` (qualified or not)
- `weak_filter=[OK]/[X]/[?]` (allowed/blocked/not-loaded)
- `freshness=[OK]/[X]` and `lag_min=<minutes>`

## Signal-only mode

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py --mode signal_only --once
```

## Live mode (best-base)

Live mode is supported only with `--signal-model-family best_base_state`.

Start with one cycle first:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py \
  --signal-model-family best_base_state \
  --mode live \
  --once
```

Run continuously:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py \
  --signal-model-family best_base_state \
  --mode live \
  --sleep-seconds 5
```

Disable dynamic target/stop updates if you want static exits from initial submit only:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py \
  --signal-model-family best_base_state \
  --mode live \
  --disable-dynamic-target-stop \
  --once
```

### One-click launcher (PyCharm friendly)

You can run live mode through the helper script:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u run_live_bot.py --once
python3 -u run_live_bot.py --sleep-seconds 5
```

`run_live_bot.py` defaults are now explicit:

- `--prediction-poll-second 5`
- `--market-data-poll-second 30`
- `--prediction-cache-max-rows 1200`

Weak-time filter support is also available via:

- `--weak-periods-json runtime/_tmp_weak_zone_gate_probs_v2.json`

If the weak-period JSON exists and contains `weak_cells`, live/signal entries whose entry minute lands in those cells are blocked.

Shared PyCharm Play-button configs are included under `.idea/runConfigurations/`:

- `Live Bot Once`
- `Live Bot Loop`

## Legacy paper mode

The old paper-trading scaffold remains available for the legacy 15-minute next-bar model family:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py \
  --signal-model-family legacy_15m_nextbar \
  --mode paper \
  --once
```

## Market-sync-only mode

Use this when you want to test only the data-sync task without loading the trading model or running entry/exit logic.

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py --market-sync-only --once
```

## Useful options

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py \
  --once \
  --signal-model-family legacy_15m_nextbar \
  --model-type gradient_boosting \
  --model-dir training/ml_models_15m_nextbar_060_corr \
  --probability-cutoff 0.50 \
  --take-profit-pct 0.60 \
  --stop-loss-pct 0.40 \
  --max-hold-bars 4 \
  --recent-days 10 \
  --max-trades-per-day 3 \
  --cooldown-bars-after-exit 2
```

Best-base signal generation explicitly:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py \
  --signal-model-family best_base_state \
  --signal-model-path training/_tmp_feature_single_split_aligned_state_stop_sweep/w150_h25_thr0.008_d15_lt0.006_st0.008_ls12_ss18_r0_f2.5_p10.48_p20.5_sf.joblib \
  --mode signal_only \
  --once
```

## Risk controls

- `--max-trades-per-day N`
  - optional cap on entries per trading day
  - trading day uses the same `17:00 America/New_York` cutoff as the ML reports
  - `0` disables the cap

- `--cooldown-bars-after-exit N`
  - optional wait time after a paper exit before a new entry is allowed
  - measured in completed 15-minute bars
  - `0` disables the cooldown

Notes:

- `--signal-model-family best_base_state` supports `signal_only` and `live`.
- `--signal-model-family best_base_state` does not support `paper`.
- While an open position exists, the bot skips new entries and updates open-position targets/stops dynamically when matching new signals arrive (unless disabled).

## IG market-data capture task

When enabled (default), each bot cycle also tries to:

- start from the latest timestamp already present in local MySQL for each instrument
- fetch minute data forward from that point up to the current bot time
- once the current second is at or past the configured poll second (default `:30`) and only once per minute bucket
- fetch latest IG snapshots for:
  - `gold`
  - `aud`
  - `oil`
- map each snapshot into the existing minute-bar schema and upsert it into:
  - `gold` -> `gold_prices`
  - `aud` -> `aud_prices`
  - `oil` -> `prices`

This task is intended to build local history for future model work and monitoring without introducing a separate snapshot-only table.

`snapshot_written` is now counted as logical snapshot rows (1 per snapshot). MySQL may report `affected_rows=2` on upsert updates; that DB count is logged separately for diagnostics.

The bot log now prints the requested sync period and resulting stored period for each instrument, for example:

- requested: latest MySQL timestamp -> current bot time
- resulting period: first returned IG row -> latest stored minute bucket

Useful options:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py \
  --sleep-seconds 30 \
  --market-data-poll-second 30
```

The default split is now already:

- `--prediction-poll-second 5`
- `--market-data-poll-second 30`

For the requested two-step behavior, use this instead:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py \
  --sleep-seconds 5 \
  --prediction-poll-second 5 \
  --market-data-poll-second 30
```

That makes the bot:

- at `HH:MM:05` or later: fetch the latest gold minute data from IG into the in-memory prediction cache and run the best-base prediction
- at `HH:MM:30` or later: fetch/store the latest IG data into MySQL for backup/backtest usage

## Best-base input storage / cache behavior

For the default `best_base_state` path, the bot now keeps a live **in-memory gold minute cache** for prediction.

At startup / when the cache is empty it bootstraps from recent MySQL rows, then on each prediction minute it:

- fetches the latest gold minute data from IG at the configured prediction second
- merges/deduplicates that data into the in-memory cache
- rebuilds the best-base 1-minute image windows and causal state features from that in-memory slice

So the effective prediction cache is an in-memory rolling minute frame: bootstrapped from recent MySQL history, then kept current with incremental IG minute fetches.

How many rows?

- controlled mainly by `--recent-days`
- actual row count depends on trading hours and gaps in the table
- the bot now writes the latest numbers into `runtime/trading_bot_status.json` under:
  - `input_data`
  - `best_base_runtime`

Those status fields include the latest raw row count, the raw start/end timestamps, the derived 1-minute bar count, and the number of candidate best-base samples built from that slice.

For a data-only sync pass:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py \
  --market-sync-only \
  --once \
  --market-data-poll-second 30
```

Disable the capture task if needed:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u trading_bot.py --disable-market-data-capture --once
```

`--market-sync-only` requires market-data capture to stay enabled.

## Files written

By default the bot writes:

- state: `runtime/trading_bot_state.json`
- status: `runtime/trading_bot_status.json`
- log: `runtime/trading_bot.log`
- paper trades: `runtime/trading_bot_trades.csv`
- lock: `/tmp/alphagold_trading_bot.lock`
- local MySQL live snapshot upserts: `gold_prices`, `aud_prices`, `prices`

The status file includes:

- current trading day label
- signal model family / signal model path
- total trades / wins / losses / breakeven
- realized P&L lifetime and current trading day
- last closed trade snapshot
- current open position snapshot
- configured position size (`configured_position_size`)
- configured daily cap / cooldown settings
- market-sync-only flag
- market-data capture enabled flag / target existing tables / poll second / last captured minute bucket
- last market-data error, if any
- per-instrument latest-sync summaries including requested period and resulting stored period
- current best-base input/runtime metadata under `input_data` and `best_base_runtime`
- rolling performance snapshots under `performance_windows`:
  - `30m` (last 30 minutes, realized exits)
  - `60m` (last 60 minutes, realized exits)
- `daily_summary` (NY 17:00 cutoff aligned):
  - `today`
  - `previous_day`
  - `trailing_7d`
- `weekly_summary` (calendar week on trading-day labels):
  - `this_week`
  - `previous_week`
- `last_periodic_report_bucket_utc` for the latest emitted `:00/:30` performance snapshot

## Mock replay on data after 9 Apr

There is now a small replay runner for the promoted best-base model:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u run_best_base_mock_replay.py \
  --start-date 2026-04-09 \
  --end-date 2026-04-15
```

By default it writes:

- JSON summary: `runtime/mock_best_base_after_2026-04-09_report.json`
- trades CSV: `runtime/mock_best_base_after_2026-04-09_trades.csv`

The replay uses the same bot-side causal best-base inference path rather than the old legacy paper model.

If IG authentication fails, `--market-sync-only` now exits as a failure instead of printing a misleading success message, and the error is also written into `market_data_last_error` in the status file.

## Next build steps

Good next upgrades:

1. harden closed-trade reconciliation from IG history for broker-triggered stop/limit exits
2. add explicit live risk kill-switches (max daily loss, max live trades/day, spread gate)
3. add replay mode that compares bot decisions to backtest output
4. add richer runtime status views (live amend history, close-source breakdown)
5. optionally swap in the newer `training/image_trend_ml.py` model family later

