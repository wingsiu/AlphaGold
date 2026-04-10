#!/usr/bin/env python3
"""Analyze empirical time-to-hit for uptrend thresholds on raw gold data.

Examples:
  python3 training/analyze_uptrend_timing.py \
    --start-date 2025-05-20 \
    --end-date 2026-02-04 \
    --thresholds 0.15,0.20,0.30,0.40,0.50,0.60,0.75 \
    --max-horizon 120 \
    --summary-out training/uptrend_timing_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import DataLoader
try:
    from training.training import prepare_gold_data
except ModuleNotFoundError:
    from training import prepare_gold_data


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(',') if x.strip()]


def compute_first_hit_bars(close: np.ndarray, high: np.ndarray, thresholds: list[float], max_horizon: int) -> dict[float, np.ndarray]:
    n = len(close)
    targets = {thr: close * (1.0 + thr / 100.0) for thr in thresholds}
    first_hit = {thr: np.full(n, np.nan) for thr in thresholds}
    remaining = {thr: np.ones(n, dtype=bool) for thr in thresholds}

    for h in range(1, max_horizon + 1):
        shifted_high = np.full(n, np.nan)
        shifted_high[:-h] = high[h:]
        valid = ~np.isnan(shifted_high)
        for thr in thresholds:
            mask = remaining[thr] & valid & (shifted_high >= targets[thr])
            first_hit[thr][mask] = h
            remaining[thr][mask] = False

    return first_hit


def summarize_hits(first_hit: dict[float, np.ndarray], n_rows: int) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    denom = max(1, n_rows - 1)
    for thr, arr in sorted(first_hit.items()):
        hit = arr[~np.isnan(arr)]
        row: dict[str, float | int] = {
            'threshold_pct': thr,
            'hits_within_horizon': int(len(hit)),
            'hit_rate_pct': float(len(hit) / denom * 100.0),
        }
        if len(hit):
            row.update(
                {
                    'median_bars': float(np.median(hit)),
                    'p75_bars': float(np.percentile(hit, 75)),
                    'p90_bars': float(np.percentile(hit, 90)),
                    'mean_bars': float(np.mean(hit)),
                }
            )
        else:
            row.update(
                {
                    'median_bars': np.nan,
                    'p75_bars': np.nan,
                    'p90_bars': np.nan,
                    'mean_bars': np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description='Analyze empirical time-to-hit for uptrend targets.')
    parser.add_argument('--table', default='gold_prices')
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--thresholds', default='0.15,0.20,0.30,0.40,0.50,0.60,0.75')
    parser.add_argument('--max-horizon', type=int, default=120)
    parser.add_argument('--summary-out', default=None)
    args = parser.parse_args()

    thresholds = parse_float_list(args.thresholds)
    loader = DataLoader()
    raw = loader.load_data(args.table, start_date=args.start_date, end_date=args.end_date)
    df = prepare_gold_data(raw)

    close = df['close'].to_numpy(dtype=float)
    high = df['high'].to_numpy(dtype=float)
    first_hit = compute_first_hit_bars(close, high, thresholds, args.max_horizon)
    summary = summarize_hits(first_hit, len(df))

    print(f'Rows analyzed: {len(df):,}')
    print(f'Max horizon: {args.max_horizon} bars')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f'Saved summary: {out_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

