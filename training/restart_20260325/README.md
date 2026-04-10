# Restart 2026-03-25

Fresh strategy research workspace.

## Current focus

Question 1:

> Does price drop more often at minute `0` of each 15-minute block?

This workspace starts by measuring that claim directly from `gold_prices` 1-minute data.

## Outputs

The first analysis writes:

- `reports/minute0_15m_slot_stats.csv`
- `reports/minute0_15m_summary.txt`

## Run

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/restart_20260325/scripts/analyze_15m_minute0_pattern.py
```

## WR + 3-down analysis with time filter and target optimization

The script `training/restart_20260325/scripts/analyze_wr3down_drop5.py` now supports:

- configurable WR lookback
- fixed hold time (`--hold-bars`)
- target sweep/optimization (`--target-values`)
- entry-time filtering (`--session-filter off|ny_broad|ny_config`)

Example:

```bash
cd /Users/alpha/Desktop/python/AlphaGold
python3 training/restart_20260325/scripts/analyze_wr3down_drop5.py \
  --session-filter ny_config \
  --hold-bars 15 \
  --target-values 3,5,7.5,10
```

Outputs are saved to:

- `training/restart_20260325/reports/wr3down_drop5_summary.csv`
- `training/restart_20260325/reports/wr3down_signals_detail.csv`
- `training/restart_20260325/reports/wr3down_drop5_summary.txt`

## Interpretation

The script checks multiple versions of the hypothesis:

1. Is slot `0` more likely to be a down minute (`close < open`)?
2. Is slot `0` more likely to close below the previous minute close?
3. Is the **lowest low of the 15-minute block** more likely to happen in slot `0`?
4. If slot `0` is down, does the block often recover by the end?

This is meant to validate the idea before building any strategy around it.

