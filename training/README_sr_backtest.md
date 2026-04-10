# Support/Resistance Walk-Forward Backtest

## Architecture (hybrid — no lookahead bias)

| Source | Role |
|--------|------|
| Databento `GC.c.0` trade ticks | Build volume-profile S/R levels (walk-forward) |
| MySQL `gold_prices` 1-minute bars | Price simulation for entry/exit (~253 k bars) |

Using the MySQL 1m bars instead of Databento bars eliminates the "too few trades"
problem caused by the sparse continuous-contract tick file (~37 k ticks / 9 months).

## Method

1. For each test day, build support/resistance from the **previous `N` calendar days
   of Databento ticks** using a volume profile (no future data leakage).
2. Scan every 1-minute MySQL bar against the active levels:
   - **Long**: bar low ≤ support + `touch_buffer` **and** close ≥ support + `reject_margin`
   - **Short**: bar high ≥ resistance − `touch_buffer` **and** close ≤ resistance − `reject_margin`
3. Exit by TP/SL from entry price or `max_hold_bars` timeout.
   - **TP**: `entry ± tp_span_frac × span`
     where `span` = distance to nearest paired level; fallback = `tp_span_frac × yesterday's bar range`
   - **SL**: `entry ∓ sl_pts` fixed dollar stop               (default 17.0 pts)

## Run smoke test

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -m unittest training/test_backtest_sr_strategy.py
```

## Run full backtest

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/backtest_sr_strategy.py \
  --trades-csv training/l2/trades_gc_c0_20250520_20260204.csv \
  --table gold_prices \
  --start-date 2025-05-20 \
  --end-date   2026-02-04 \
  --out-csv training/l2/backtest_sr_trades_gc_c0.csv \
  --lookback-days 20 \
  --tick-size 0.10 \
  --top-n 15 \
  --merge-ticks 30 \
  --min-prom-pct 0.02 \
  --touch-buffer 0.15 \
  --reject-margin 0.05 \
  --max-trades-per-level-per-day 2 \
  --tp-span-frac 0.60 \
  --sl-pts 17.0 \
  --max-hold-bars 30
```

## Main tuning levers

| Parameter | Effect |
|-----------|--------|
| `--lookback-days` | Larger = more stable levels, slower to adapt |
| `--merge-ticks` | Larger = fewer, stronger levels |
| `--touch-buffer` | Dollar tolerance for "price touched level" |
| `--reject-margin` | Dollar confirmation required above/below level |
| `--max-trades-per-level-per-day` | Cap entries per level per session |
| `--tp-span-frac` | Fraction of S/R span used as take-profit distance |
| `--sl-pts` | Fixed dollar stop-loss distance from entry |
| `--max-hold-bars` | Max minutes in trade (1m bars) |
