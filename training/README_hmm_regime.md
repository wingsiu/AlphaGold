# HMM Regime Walk-Forward (No Lookahead)

This adds a bias-safe Hidden Markov Model regime predictor in `training/hmm_regime_walkforward.py`.

## What makes it bias-safe

- Each prediction is for bar `t + horizon_bars` (default 15).
- The HMM is fit only on rows up to bar `t`.
- State labels (bearish/neutral/bullish) are derived from training-window state returns only.
- No future rows are used for fitting, scaling, state mapping, or prediction.

## Quick smoke test

Run from project root:

```bash
python3 -m unittest training/test_hmm_regime_walkforward.py
```

## Run on your price CSV

```bash
python3 training/hmm_regime_walkforward.py \
  --input-csv training/your_prices.csv \
  --timestamp-column timestamp \
  --price-column close \
  --n-states 3 \
  --train-window 4000 \
  --min-train-rows 700 \
  --horizon-bars 15 \
  --retrain-step 5 \
  --regime-threshold-pct 0.03 \
  --out training/hmm_regime_predictions.csv
```

## Run on database prices

```bash
python3 training/hmm_regime_walkforward.py \
  --table gold_prices \
  --start-date 2025-05-20 \
  --end-date 2026-02-04 \
  --n-states 3 \
  --train-window 4000 \
  --min-train-rows 700 \
  --horizon-bars 15 \
  --retrain-step 250 \
  --hmm-n-iter 80 \
  --regime-threshold-pct 0.03 \
  --progress-every 1000 \
  --out training/hmm_regime_predictions.csv
```

## Speed tips for large 1m ranges

- Start with `--retrain-step 250` or `500` (instead of `5`).
- Lower `--hmm-n-iter` to `40-80` for faster fits.
- Reduce `--train-window` (for example `2000`) if you still need more speed.
- If progress is too chatty, increase `--progress-every`.

## Output columns

- `timestamp`: predicted bar timestamp (`t + horizon_bars`)
- `horizon_bars`: forecast horizon used for this row
- `predicted_regime`: model forecast (`bearish`, `neutral`, `bullish`)
- `bullish_prob`, `neutral_prob`, `bearish_prob`: next-state probabilities
- `actual_ret_h_pct`: realized horizon return (percent)
- `actual_regime`: realized regime by threshold
- `is_correct`: 1 when predicted and actual regimes match
