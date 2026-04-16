#!/usr/bin/env python3
"""Regression tests for walk-forward weekend cycle boundaries."""

from __future__ import annotations

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from image_trend_ml import build_walkforward_windows
from sweep_wf_probs_no_retrain import _cycle_slice_bounds, _find_slice


class WalkForwardCycleBoundaryTest(unittest.TestCase):
    def test_weekend_cycles_report_monday_to_friday_trading_days(self) -> None:
        idx = pd.date_range("2025-05-20", "2026-01-10", freq="B", tz="UTC") + pd.Timedelta(hours=20)

        windows = build_walkforward_windows(
            idx,
            init_train_months=6,
            retrain_days=14,
            max_train_days=365,
            min_train_samples=50,
            anchor_mode="weekend_fri_close",
        )

        self.assertGreaterEqual(len(windows), 2)

        first = windows[0]
        second = windows[1]

        self.assertEqual(first.train_end_day.date().isoformat(), "2025-11-21")
        self.assertEqual(first.test_start_day.date().isoformat(), "2025-11-24")
        self.assertEqual(first.test_end_day.date().isoformat(), "2025-12-05")

        self.assertEqual(second.train_end_day.date().isoformat(), "2025-12-05")
        self.assertEqual(second.test_start_day.date().isoformat(), "2025-12-08")
        self.assertEqual(second.test_end_day.date().isoformat(), "2025-12-19")

    def test_find_slice_accepts_date_only_trading_day_bounds(self) -> None:
        idx = pd.DatetimeIndex([
            "2025-12-08T20:00:00Z",
            "2025-12-10T20:00:00Z",
            "2025-12-12T20:00:00Z",
            "2025-12-15T20:00:00Z",
            "2025-12-19T20:00:00Z",
            "2025-12-22T20:00:00Z",
        ])

        sl = _find_slice(idx, "2025-12-08", "2025-12-19")

        self.assertEqual(sl.start, 0)
        self.assertEqual(sl.stop, 5)
        self.assertEqual(list(idx[sl].strftime("%Y-%m-%d")), [
            "2025-12-08",
            "2025-12-10",
            "2025-12-12",
            "2025-12-15",
            "2025-12-19",
        ])

    def test_cycle_slice_bounds_prefer_reported_trading_day_window(self) -> None:
        cycle = {
            "test_start": "2025-12-08",
            "test_end": "2025-12-19",
            "test_start_sample_ts": "2025-12-10T20:00:00+00:00",
            "test_end_sample_ts": "2025-12-18T20:00:00+00:00",
        }

        start, end = _cycle_slice_bounds(cycle)

        self.assertEqual(start, "2025-12-08")
        self.assertEqual(end, "2025-12-19")


if __name__ == "__main__":
    unittest.main(verbosity=2)

