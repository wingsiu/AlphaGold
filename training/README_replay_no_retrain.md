# No-retrain walk-forward replay

Use `replay_wf_report_no_retrain.py` to recompute walk-forward trades with the **current** backtest logic while reusing the already-saved per-cycle models.

## What it does

- rebuilds the dataset from the original report config
- reloads each saved cycle model
- reruns inference on each cycle test slice
- recomputes directional PnL/trades with current execution logic
- writes an optional comparison JSON and replay trades CSV

## Execution semantics

Current training/backtest artifacts now record:

- signal reference: `signal_bar_close`
- entry execution: `next_bar_open`

That means:

- labels/features stay aligned to the signal bar
- simulated entries use the next bar's open price/time

## Example

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u training/replay_wf_report_no_retrain.py \
  --report-in training/_wfC_r0_flat200_300_s148_55_u58_62_d62_68.json \
  --out-json training/_wfC_r0_flat200_300_s148_55_u58_62_d62_68_replay.json \
  --out-trades training/_wfC_r0_flat200_300_s148_55_u58_62_d62_68_replay.csv
```

## Batch replay for future logic refinements

Use the batch wrapper to replay the main comparison set into a versioned folder:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u training/replay_wf_batch.py \
  --logic-label next_bar_open_v1
```

Or use the convenience script:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
bash training/_run_replay_batch_next_bar_open.sh
```

Outputs are written under:

- `training/replays/<logic-label>/batch_summary.csv`
- `training/replays/<logic-label>/batch_manifest.json`
- `training/replays/<logic-label>/<report-stem>.replay_summary.json`
- `training/replays/<logic-label>/<report-stem>.replay_trades.csv`

## Smaller smoke test

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 -u training/replay_wf_report_no_retrain.py \
  --report-in training/_wfC_r0_flat200_300_s148_55_u58_62_d62_68.json \
  --max-cycles 1
```

## Related tool

`training/sweep_wf_probs_no_retrain.py` can sweep directional stage-2 gates on a saved walk-forward run without retraining.

