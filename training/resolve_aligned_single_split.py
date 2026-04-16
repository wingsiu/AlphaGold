#!/usr/bin/env python3
"""Resolve an aligned single-split test start from the current walk-forward schedule.

This uses the same dataset build path as ``image_trend_ml.py`` and returns the first
walk-forward test-sample timestamp. That lets us do cheap ``single_split`` base-model
searches using the exact same sample universe / first evaluation boundary as the later
walk-forward runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from image_trend_ml import (
    _load_or_build_bars,
    _load_or_build_supervised_dataset,
    build_walkforward_windows,
)


@contextlib.contextmanager
def _redirect_noisy_stdout_to_stderr():
    """Keep resolver stdout machine-readable while preserving progress logs."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _make_prep_cfg(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        table=args.table,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        disable_time_filter=bool(args.disable_time_filter),
        window=int(args.window),
        window_15m=int(args.window_15m),
        use_15m_wick_features=False,
        wick_feature_min_range=40.0,
        wick_feature_min_pct=35.0,
        wick_feature_min_volume=3000.0,
        min_15m_drop=float(args.min_15m_drop),
        min_15m_rise=float(args.min_15m_rise),
        last_bar_wr90_high=args.last_bar_wr90_high,
        last_bar_wr90_low=args.last_bar_wr90_low,
        horizon=int(args.horizon),
        adverse_limit=float(args.adverse_limit),
        prep_cache_dir=args.prep_cache_dir,
        refresh_prep_cache=bool(args.refresh_prep_cache),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Resolve first walk-forward test start for aligned single-split runs")
    p.add_argument("--table", default="gold_prices")
    p.add_argument("--start-date", default="2025-05-20")
    p.add_argument("--end-date", default="2026-04-10")
    p.add_argument("--timeframe", default="1min")
    p.add_argument("--disable-time-filter", action="store_true")
    p.add_argument("--window", type=int, default=150)
    p.add_argument("--window-15m", type=int, default=0)
    p.add_argument("--min-window-range", type=float, default=40.0)
    p.add_argument("--min-15m-drop", type=float, default=15.0)
    p.add_argument("--min-15m-rise", type=float, default=0.0)
    p.add_argument("--last-bar-wr90-high", type=float, default=None,
                   help="Optional keep-sample filter: last-bar WR90 >= this value.")
    p.add_argument("--last-bar-wr90-low", type=float, default=None,
                   help="Optional keep-sample filter: last-bar WR90 <= this value.")
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--trend-threshold", type=float, default=0.008)
    p.add_argument("--adverse-limit", type=float, default=15.0)
    p.add_argument("--long-target-threshold", type=float, default=0.006)
    p.add_argument("--short-target-threshold", type=float, default=0.008)
    p.add_argument("--long-adverse-limit", type=float, default=15.0)
    p.add_argument("--short-adverse-limit", type=float, default=18.0)
    p.add_argument("--wf-init-train-months", type=int, default=6)
    p.add_argument("--wf-retrain-days", type=int, default=14)
    p.add_argument("--wf-max-train-days", type=int, default=365)
    p.add_argument("--wf-min-train-samples", type=int, default=300)
    p.add_argument("--wf-anchor-mode", choices=["elapsed_days", "weekend_fri_close"], default="weekend_fri_close")
    p.add_argument("--prep-cache-dir", default=None,
                   help="Optional shared prep cache directory matching image_trend_ml.py.")
    p.add_argument("--refresh-prep-cache", action="store_true",
                   help="Ignore any existing prep cache artifacts and rebuild them.")
    p.add_argument("--output-format", choices=["ts_only", "json"], default="ts_only")
    args = p.parse_args()

    with _redirect_noisy_stdout_to_stderr():
        prep_cfg = _make_prep_cfg(args)
        bars, bars_cache_info = _load_or_build_bars(prep_cfg)
        _, _, ts, _, _, _, _, _, kept, skipped, dataset_cache_info = _load_or_build_supervised_dataset(
            bars,
            prep_cfg,
            args.min_window_range,
            args.trend_threshold,
            long_target_threshold=args.long_target_threshold,
            short_target_threshold=args.short_target_threshold,
            long_adverse_limit=args.long_adverse_limit,
            short_adverse_limit=args.short_adverse_limit,
        )

    windows = build_walkforward_windows(
        ts,
        init_train_months=args.wf_init_train_months,
        retrain_days=args.wf_retrain_days,
        max_train_days=args.wf_max_train_days,
        min_train_samples=args.wf_min_train_samples,
        anchor_mode=args.wf_anchor_mode,
    )
    if not windows:
        raise ValueError("No walk-forward windows were produced for the current configuration")

    first = windows[0]
    payload = {
        "kept": int(kept),
        "skipped": int(skipped),
        "first_cycle_id": int(first.cycle_id),
        "train_end_day": first.train_end_day.date().isoformat() if first.train_end_day is not None else None,
        "test_start_day": first.test_start_day.date().isoformat() if first.test_start_day is not None else None,
        "test_end_day": first.test_end_day.date().isoformat() if first.test_end_day is not None else None,
        "test_start_sample_ts": ts[first.test_start].isoformat(),
        "test_end_sample_ts": ts[first.test_end - 1].isoformat(),
        "train_samples": int(first.train_end - first.train_start),
        "test_samples": int(first.test_end - first.test_start),
        "cache_info": {
            "bars": bars_cache_info,
            "dataset": dataset_cache_info,
        },
    }

    if args.output_format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(payload["test_start_sample_ts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

