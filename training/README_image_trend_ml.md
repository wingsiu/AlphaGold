# Image-Style Trend ML (Gold)

This pipeline converts rolling candle windows into image-like tensors and trains a trend classifier.

## What it does

- Uses `gold_prices` from MySQL via `DataLoader`
- Uses 1-minute candles by default
- Filters training images to keep only windows where `max(high)-min(low) > 40` (configurable)
- Filters further to keep windows containing a **15-bar downside move** (`--min-15m-drop`) and optionally a **15-bar upside move** (`--min-15m-rise`)
- Converts each rolling window (default `150` bars) into a tensor of shape:
  - `9 channels x 150 width`
  - Channels include OHLC shape + explicit volume channels (`volume_z`, `volume_rel`, `volume_change`)
- **15-min candle image** (`--window-15m`, default `0`): disabled by default.
  Set `--window-15m 40` (or another positive value) only for explicit two-branch experiments.
- Labels trend `down / flat / up` using future return over `horizon` bars
- Target rule (per sample):
  - `up` if `(future_high - current_close) / current_close > 0.4%` and `(current_close - future_low) < 15`
  - `down` if `(current_close - future_low) / current_close > 0.4%` and `(future_high - current_close) < 15`
  - otherwise `flat`
- Trains a multiclass logistic regression model
- Uses a two-stage model (`flat vs trend`, then `down vs up`) with optional confidence gating
- Optional causal state features (`--use-state-features`) add previous-prediction continuity and
  trade-status inputs (`flat/long/short`, bars in position, unrealized return, target/stop hit flags)
  built strictly from past/current bars only (no look-ahead)
- By default, state mode encodes the previous `150` predictions via `--pred-history-len 150`
- Optional `--optimize-prob` tunes stage confidence gates on a train-validation split
- Evaluates chronologically on out-of-sample data
- Saves model + JSON report (includes `feature_shape` so inference knows the expected input width)

## Files

- `training/image_trend_ml.py`
- `training/run_image_trend_smoke.py`
- `training/sweep_image_trend_ml.py`

## Quick smoke run

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/run_image_trend_smoke.py
```

## Full run

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/image_trend_ml.py
```

## Sweep multiple configs

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/sweep_image_trend_ml.py \
  --start-date 2025-05-20 \
  --end-date 2026-04-10 \
  --window-15m 0 \
  --horizon-values 60 \
  --trend-threshold-values 0.01 \
  --min-15m-drop-values 10,15,20 \
  --min-15m-rise-values 0,15 \
  --stage1-prob-values 0.55,0.60,0.65 \
  --stage2-prob-values 0.60,0.65,0.70 \
  --use-optimize-prob
```

Outputs:

- `training/image_trend_sweep_results.csv`
- `training/image_trend_sweep_summary.txt`
- per-run artifacts under `training/_tmp_image_trend_sweep/`

## Useful options

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/image_trend_ml.py \
  --start-date 2025-05-20 \
  --end-date 2026-04-10 \
  --timeframe 1min \
  --window 150 \
  --window-15m 0 \
  --min-window-range 40 \
  --min-15m-drop 20 \
  --min-15m-rise 15 \
  --horizon 15 \
  --trend-threshold 0.004 \
  --adverse-limit 15 \
  --stage1-min-prob 0.55 \
  --stage2-min-prob 0.55 \
  --optimize-prob \
  --model-out training/image_trend_model.joblib \
  --report-out training/image_trend_report.json
```

## Causal state features (optional)

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/image_trend_ml.py \
  --start-date 2025-05-20 \
  --end-date 2026-04-10 \
  --timeframe 1min \
  --window 150 \
  --min-window-range 40 \
  --min-15m-drop 15 \
  --min-15m-rise 15 \
  --horizon 40 \
  --trend-threshold 0.01 \
  --stage1-min-prob 0.60 \
  --stage2-min-prob 0.70 \
  --use-state-features \
  --pred-history-len 150 \
  --state-oof-splits 5 \
  --model-out training/image_trend_model_state.joblib \
  --report-out training/image_trend_report_state.json
```

Current limitation: `--use-state-features` is supported in 1m-only mode (no `--two-branch`) and without `--optimize` / `--optimize-prob`.

## Baseline (2026-04-12)

### Best aligned base config (2026-04-15)

Terminology used below:

- **plain base** = no `--use-state-features`
- **state-feature base** = same aligned single-split search but with `--use-state-features`
- **best base** = the currently promoted best aligned single-split candidate after follow-up tuning
- as of `2026-04-15`, that promoted **best base** is the state-feature winner with the stop retune `long_stop=12`, `short_stop=18`

Cheapest aligned single-split base sweep (no state features) winner from:

- summary: `training/base_single_split_aligned_summary.txt`
- csv: `training/base_single_split_aligned_results.csv`
- temp artifacts: `training/_tmp_base_single_split_aligned/`

Best unique base config:

- mode: `single_split` + aligned `--test-start-date` from `training/resolve_aligned_single_split.py`
- features: **no** `--use-state-features`
- core params: `window=150`, `horizon=25`, `trend=0.008`, `min-window-range=40`, `min-15m-drop=15`
- directional filters: `min-15m-rise=0` and `15` tied on this split (future sweeps can usually keep just `0`)
- targets/stops: `long_target=0.006`, `short_target=0.008`, `long_stop=15`, `short_stop=18`
- best gates: `max_flat_ratio=2.0`, `stage1_min_prob=0.55`, `stage2_min_prob=0.60`

Best aligned base snapshot:

- trades: `1440`
- total pnl: `4808.94`
- avg day pnl: `66.7908`
- positive days: `77.7778%`
- stage2 balanced accuracy: `0.5349`
- composite score: `6121.6442`

Reproduce via runner:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
bash training/_run_base_single_split_aligned.sh
```

Requested walk-forward threshold sweep around the promoted best base:

- runner: `training/_run_best_base_wf_10cycles_prob_sweep.sh`
- keeps the promoted best-base structure fixed with `--use-state-features`
- fixed flat ratio: `2.5`
- sweeps only:
  - `stage1_min_prob = 0.48, 0.50, 0.52`
  - `stage2_min_prob_up = 0.50, 0.54, 0.58`
  - `stage2_min_prob_down = 0.54, 0.58, 0.62`

Run it with:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
bash training/_run_best_base_wf_10cycles_prob_sweep.sh
```

The aligned base sweep on 2026-04-15 used the shared prep cache plus gate-only model reuse and completed with:

- `runs=120`
- `trained=10`
- `auto_reused=110`

Narrow follow-up stop sweep from that best base config:

- runner: `training/_run_base_single_split_aligned_stop_sweep.sh`
- fixed base: `max_flat_ratio=2.0`, `stage1=0.55`, `stage2=0.60`, `min-15m-rise=0`
- default stop grids: `LONG_STOP_VALUES=12,15,18`, `SHORT_STOP_VALUES=15,18,21`

Run it with:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
bash training/_run_base_single_split_aligned_stop_sweep.sh
```

Optional custom stop grid:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
LONG_STOP_VALUES=12,15,18 SHORT_STOP_VALUES=12,15,18,21 bash training/_run_base_single_split_aligned_stop_sweep.sh
```

Stop-sweep winner from the plain-base best config:

- summary: `training/base_single_split_aligned_stop_sweep_summary.txt`
- csv: `training/base_single_split_aligned_stop_sweep_results.csv`
- best stop pair: `long_stop=12`, `short_stop=18`

Best plain-base + stop snapshot:

- fixed base: `max_flat_ratio=2.0`, `stage1=0.55`, `stage2=0.60`, `min-15m-rise=0`
- trades: `1508`
- total pnl: `5629.52`
- avg day pnl: `79.2890`
- positive days: `69.0141%`
- stage2 balanced accuracy: `0.5407`
- composite score: `6860.3633`

Single-split baseline used for current comparisons:

- Mode: `single_split` + `--disable-time-filter` + `--test-start-date ""` + `--test-size 0.4`
- Features: `--use-state-features --pred-history-len 150 --state-oof-splits 5`
- Core params: `window=150`, `horizon=25`, `trend=0.008`, `min-window-range=40`, `min-15m-drop=15`, `min-15m-rise=15`
- Directional targets/stops: `long_target=0.006`, `short_target=0.008`, `long_stop=15`, `short_stop=18`

Threshold sweep around baseline (`s1/s2 in {0.55,0.60,0.65}`):

- csv: `training/image_trend_sweep_probgrid_bestsf.csv`
- txt: `training/image_trend_sweep_probgrid_bestsf.txt`
- best pnl in this grid: `s1=0.55`, `s2=0.60` -> `trades=494`, `total_pnl=2105.05`

Best config recheck (`s1=0.55`, `s2=0.60`):

- report: `training/backtest_report_best_sf_nofilter_p055_p060_fullstats.json`
- model: `training/backtest_model_best_sf_nofilter_p055_p060_fullstats.joblib`
- trades: `training/backtest_trades_best_sf_nofilter_p055_p060_fullstats.csv`

### Full stats snapshot (best config)

- trades: `494`
- total pnl: `2105.05`
- avg trade: `4.2612`
- n_days: `28`
- avg trades/day: `17.6429`
- avg day pnl: `75.1804`
- positive days: `71.4286%`
- max drawdown: `-151.08` (daily `-80.63`)
- long: `278 trades`, `52.1583%` win, pnl `1435.50`
- short: `216 trades`, `48.6111%` win, pnl `669.55`
- exits (report schema): `reverse_signal=182`, `signal_target=159`, `stop_loss=153`

### No-retrain rebuilt directional stats (target_hit/timeout schema)

Use existing trades CSV to refresh `directional_pnl` without retraining:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/rebuild_directional_pnl_from_trades.py \
  --trades-csv training/backtest_trades_best_sf_nofilter_p055_p060_fullstats.csv \
  --report-in training/backtest_report_best_sf_nofilter_p055_p060_fullstats.json \
  --report-out training/backtest_report_best_sf_nofilter_p055_p060_fullstats_v2.json \
  --stats-out training/backtest_directional_pnl_best_sf_nofilter_p055_p060_v2.json
```

Rebuilt summary (`training/backtest_directional_pnl_best_sf_nofilter_p055_p060_v2.json`):

- reverse_signal: `182 trades`, `130 wins`, `52 losses`, `avg_pnl=9.6638`
- target_hit: `120 trades`, `avg_pnl=25.5924` (win-only by definition)
- timeout: `39 trades`, `0 wins`, `39 losses`, `avg_pnl=-6.0987`

Baseline reproduce command:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/image_trend_ml.py \
  --eval-mode single_split \
  --disable-time-filter \
  --use-state-features \
  --state-oof-splits 5 \
  --pred-history-len 150 \
  --start-date 2025-05-20 \
  --end-date 2026-04-10 \
  --test-start-date "" \
  --test-size 0.4 \
  --window 150 \
  --min-window-range 40 \
  --min-15m-drop 15 \
  --min-15m-rise 15 \
  --horizon 25 \
  --trend-threshold 0.008 \
  --long-target-threshold 0.006 \
  --short-target-threshold 0.008 \
  --adverse-limit 15 \
  --long-adverse-limit 15 \
  --short-adverse-limit 18 \
  --max-flat-ratio 4 \
  --stage1-min-prob 0.55 \
  --stage2-min-prob 0.60 \
  --report-out training/backtest_report_best_sf_nofilter_p055_p060_fullstats.json \
  --model-out training/backtest_model_best_sf_nofilter_p055_p060_fullstats.joblib \
  --trades-out training/backtest_trades_best_sf_nofilter_p055_p060_fullstats.csv
```

### Best aligned state-feature base before stop retune (2026-04-15)

Aligned state-feature sweep winner from:

- summary: `training/feature_single_split_aligned_state_summary.txt`
- csv: `training/feature_single_split_aligned_state_results.csv`
- temp artifacts: `training/_tmp_feature_single_split_aligned_state/`

Best unique state-feature base config:

- mode: `single_split` + aligned `--test-start-date` from `training/resolve_aligned_single_split.py`
- features: `--use-state-features --pred-history-len 150`
- core params: `window=150`, `horizon=25`, `trend=0.008`, `min-window-range=40`, `min-15m-drop=15`
- directional filters: `min-15m-rise=0` and `15` tied on this split
- targets/stops: `long_target=0.006`, `short_target=0.008`, `long_stop=15`, `short_stop=18`
- best gates: `max_flat_ratio=2.5`, `stage1_min_prob=0.48`, `stage2_min_prob=0.50`

Best aligned state-feature base snapshot:

- trades: `1413`
- total pnl: `5923.14`
- avg day pnl: `83.4245`
- positive days: `73.2394%`
- stage2 balanced accuracy: `0.5680`
- composite score: `7223.4919`

This beat the plain-base best (`4808.94`) and the plain-base stop-sweep best (`5629.52`), so this became the state-feature reference point **before** the later stop retune.

Next stop sweep from that best state-feature base:

- runner: `training/_run_feature_single_split_aligned_state_stop_sweep.sh`
- fixed base: `max_flat_ratio=2.5`, `stage1=0.48`, `stage2=0.50`, `min-15m-rise=0`, `--use-state-features`

State-feature stop-sweep winner from that best base:

- summary: `training/feature_single_split_aligned_state_stop_sweep_summary.txt`
- csv: `training/feature_single_split_aligned_state_stop_sweep_results.csv`
- best stop pair: `long_stop=12`, `short_stop=18`

Promoted best base snapshot:

- fixed base: `max_flat_ratio=2.5`, `stage1=0.48`, `stage2=0.50`, `min-15m-rise=0`, `--use-state-features`
- trades: `1415`
- total pnl: `6504.07`
- avg day pnl: `91.6066`
- positive days: `67.6056%`
- stage2 balanced accuracy: `0.5560`
- profit factor: `1.8391`
- composite score: `7736.0837`

This promoted best overall candidate is now the project's **best base** reference point.

### WR90 last-bar extreme filter (added 2026-04-15)

Optional dataset filter to reduce flat-class dominance by only keeping samples where the **last bar** in the input window has an extreme WR90 reading.

New trainer/sweep flags:

- `--last-bar-wr90-high <value>` keeps samples when last-bar `WR90 >= value`
- `--last-bar-wr90-low <value>` keeps samples when last-bar `WR90 <= value`
- if both are set, a sample is kept when **either** extreme matches

For the requested experiment, use:

- `--last-bar-wr90-high -30`
- `--last-bar-wr90-low -70`

The aligned state-feature runners also accept these via environment variables:

- `LAST_BAR_WR90_HIGH=-30`
- `LAST_BAR_WR90_LOW=-70`

Try the best-base stop sweep with the WR90 extreme filter:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
LAST_BAR_WR90_HIGH=-30 LAST_BAR_WR90_LOW=-70 bash training/_run_feature_single_split_aligned_state_stop_sweep.sh
```

Or run the broader state-feature aligned sweep with the same filter:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
LAST_BAR_WR90_HIGH=-30 LAST_BAR_WR90_LOW=-70 bash training/_run_feature_single_split_aligned_state.sh
```

## Walk-forward winner (state + sweep, 2026-04-12)

Winner selected from 4 runs:

- `training/backtest_report_wf_state_sweep_A.json`
- `training/backtest_report_wf_state_sweep_B.json`
- `training/backtest_report_wf_state_sweep_C.json`  <- winner
- `training/backtest_report_wf_state_sweep_D.json`

Winner artifacts:

- report: `training/backtest_report_wf_state_sweep_C.json`
- model: `training/backtest_model_wf_state_sweep_C.joblib`
- trades: `training/backtest_trades_wf_state_sweep_C.csv`
- per-cycle models: `training/backtest_model_wf_state_sweep_C_cycles/`

Winner config snapshot:

- mode: `walk_forward` + `--disable-time-filter` + `--use-state-features`
- core params: `window=150`, `horizon=25`, `trend=0.008`, `min-window-range=40`, `min-15m-drop=15`, `min-15m-rise=15`
- targets/stops: `long_target=0.006`, `short_target=0.008`, `long_stop=15`, `short_stop=18`
- base gates: `stage1=0.57`, `stage2=0.62`
- WF: `init_train_months=6`, `retrain_days=14`, `max_train_days=365`, `min_train_samples=300`, `anchor=weekend_fri_close`
- sweep: enabled, `wf_sweep_flat_ratios=3,4,5`, `wf_sweep_val_ratio=0.2`, `wf_sweep_min_val_samples=50`

Winner full stats snapshot (`directional_pnl.all`):

- trades: `1280`
- total pnl: `5019.81`
- avg trade: `3.9217`
- median trade: `-1.42`
- win rate: `47.4219%`
- gross profit / loss: `13138.09 / -8118.28`
- profit factor: `1.6183`
- best/worst trade: `229.98 / -18.0`
- avg win / avg loss: `21.6443 / -12.0628`
- trade max drawdown: `-313.86`
- daily max drawdown: `-64.20`
- n_days: `62`
- avg trades/day: `20.6452`
- avg day pnl: `80.9647`
- median day pnl: `30.04`
- positive days: `83.8710%`
- best/worst day: `700.82 / -64.20`

Winner side split:

- long: `724` trades, pnl `3011.25`, win `47.2376%`, PF `1.6572`
- short: `556` trades, pnl `2008.56`, win `47.6619%`, PF `1.5680`

Winner exit breakdown:

- reverse_signal: `566`, avg_pnl `8.7825`, win `65.1943%`
- target_hit: `238`, avg_pnl `29.4724`
- stop_loss: `409`
- timeout: `67`, avg_pnl `-5.8579`, win `0%`

### Next 3-run matrix (side-specific Stage-2 long/short gates)

Use winner config, keep sweep enabled, vary only side-specific Stage-2 gates:

```bash
cd /Users/alpha/Desktop/python/AlphaGold

python3 -u training/image_trend_ml.py \
  --eval-mode walk_forward --disable-time-filter --use-state-features --state-oof-splits 5 --pred-history-len 150 \
  --start-date 2025-05-20 --end-date 2026-04-10 --test-size 0.4 \
  --window 150 --window-15m 0 --min-window-range 40 --min-15m-drop 15 --min-15m-rise 15 \
  --horizon 25 --trend-threshold 0.008 \
  --long-target-threshold 0.006 --short-target-threshold 0.008 \
  --adverse-limit 15 --long-adverse-limit 15 --short-adverse-limit 18 \
  --max-flat-ratio 4 --classifier gradient_boosting \
  --stage1-min-prob 0.57 --stage2-min-prob 0.62 \
  --stage2-min-prob-long 0.62 --stage2-min-prob-short 0.62 \
  --reverse-exit-prob 0.7 \
  --wf-init-train-months 6 --wf-retrain-days 14 --wf-max-train-days 365 --wf-min-train-samples 300 \
  --wf-sweep-flat-ratios 3,4,5 --wf-sweep-val-ratio 0.2 --wf-sweep-min-val-samples 50 --wf-anchor-mode weekend_fri_close \
  --model-out training/backtest_model_wf_state_long62_short62.joblib \
  --report-out training/backtest_report_wf_state_long62_short62.json \
  --trades-out training/backtest_trades_wf_state_long62_short62.csv

python3 -u training/image_trend_ml.py \
  --eval-mode walk_forward --disable-time-filter --use-state-features --state-oof-splits 5 --pred-history-len 150 \
  --start-date 2025-05-20 --end-date 2026-04-10 --test-size 0.4 \
  --window 150 --window-15m 0 --min-window-range 40 --min-15m-drop 15 --min-15m-rise 15 \
  --horizon 25 --trend-threshold 0.008 \
  --long-target-threshold 0.006 --short-target-threshold 0.008 \
  --adverse-limit 15 --long-adverse-limit 15 --short-adverse-limit 18 \
  --max-flat-ratio 4 --classifier gradient_boosting \
  --stage1-min-prob 0.57 --stage2-min-prob 0.62 \
  --stage2-min-prob-long 0.64 --stage2-min-prob-short 0.62 \
  --reverse-exit-prob 0.7 \
  --wf-init-train-months 6 --wf-retrain-days 14 --wf-max-train-days 365 --wf-min-train-samples 300 \
  --wf-sweep-flat-ratios 3,4,5 --wf-sweep-val-ratio 0.2 --wf-sweep-min-val-samples 50 --wf-anchor-mode weekend_fri_close \
  --model-out training/backtest_model_wf_state_long64_short62.joblib \
  --report-out training/backtest_report_wf_state_long64_short62.json \
  --trades-out training/backtest_trades_wf_state_long64_short62.csv

python3 -u training/image_trend_ml.py \
  --eval-mode walk_forward --disable-time-filter --use-state-features --state-oof-splits 5 --pred-history-len 150 \
  --start-date 2025-05-20 --end-date 2026-04-10 --test-size 0.4 \
  --window 150 --window-15m 0 --min-window-range 40 --min-15m-drop 15 --min-15m-rise 15 \
  --horizon 25 --trend-threshold 0.008 \
  --long-target-threshold 0.006 --short-target-threshold 0.008 \
  --adverse-limit 15 --long-adverse-limit 15 --short-adverse-limit 18 \
  --max-flat-ratio 4 --classifier gradient_boosting \
  --stage1-min-prob 0.57 --stage2-min-prob 0.62 \
  --stage2-min-prob-long 0.62 --stage2-min-prob-short 0.64 \
  --reverse-exit-prob 0.7 \
  --wf-init-train-months 6 --wf-retrain-days 14 --wf-max-train-days 365 --wf-min-train-samples 300 \
  --wf-sweep-flat-ratios 3,4,5 --wf-sweep-val-ratio 0.2 --wf-sweep-min-val-samples 50 --wf-anchor-mode weekend_fri_close \
  --model-out training/backtest_model_wf_state_long62_short64.joblib \
  --report-out training/backtest_report_wf_state_long62_short64.json \
  --trades-out training/backtest_trades_wf_state_long62_short64.csv
```

## Tensor layout

| Slice | Timeframe | Channels | Width |
|-------|-----------|----------|-------|
| `[:, :window]` | 1-min | 9 | `window` (default 150) |
| `[:, window:]` | 15-min | 9 | `window_15m` (default 40) |

When `--window-15m 0` is passed the 15-min slice is omitted and total width = `window`.


## Files

- `training/image_trend_ml.py`
- `training/run_image_trend_smoke.py`
- `training/sweep_image_trend_ml.py`

## Quick smoke run

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/run_image_trend_smoke.py
```

## Full run

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/image_trend_ml.py
```

## Sweep multiple configs

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/sweep_image_trend_ml.py \
  --start-date 2025-05-20 \
  --end-date 2026-04-10 \
  --horizon-values 60 \
  --trend-threshold-values 0.01 \
  --min-15m-drop-values 10,15,20 \
  --min-15m-rise-values 0,15 \
  --stage1-prob-values 0.55,0.60,0.65 \
  --stage2-prob-values 0.60,0.65,0.70 \
  --use-optimize-prob
```

Outputs:

- `training/image_trend_sweep_results.csv`
- `training/image_trend_sweep_summary.txt`
- per-run artifacts under `training/_tmp_image_trend_sweep/`

## Useful options

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/image_trend_ml.py \
  --start-date 2025-05-20 \
  --end-date 2026-04-10 \
  --timeframe 1min \
  --window 150 \
  --min-window-range 40 \
  --min-15m-drop 20 \
  --min-15m-rise 15 \
  --horizon 15 \
  --trend-threshold 0.004 \
  --adverse-limit 15 \
  --stage1-min-prob 0.55 \
  --stage2-min-prob 0.55 \
  --optimize-prob \
  --model-out training/image_trend_model.joblib \
  --report-out training/image_trend_report.json
```
