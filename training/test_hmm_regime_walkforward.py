#!/usr/bin/env python3
"""Smoke test for walk-forward HMM regime prediction."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from hmm_regime_walkforward import HMMConfig, run_walkforward


class HMMWalkForwardTest(unittest.TestCase):
    def test_walkforward_generates_predictions(self) -> None:
        rng = np.random.default_rng(42)

        n = 1800
        regime = np.zeros(n, dtype=int)
        regime[600:1200] = 1
        regime[1200:] = 2

        drift = np.select(
            [regime == 0, regime == 1, regime == 2],
            [-0.00015, 0.0, 0.00018],
            default=0.0,
        )
        noise = rng.normal(0.0, 0.0009, size=n)
        rets = drift + noise

        close = 2400 * np.exp(np.cumsum(rets))
        idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
        prices = pd.DataFrame({"close": close}, index=idx)

        cfg = HMMConfig(
            n_states=3,
            train_window=1200,
            min_train_rows=500,
            retrain_step=10,
            regime_threshold_pct=0.03,
            random_state=11,
            horizon_bars=15,
        )

        pred_df, summary = run_walkforward(prices, cfg)

        self.assertGreater(len(pred_df), 100)
        self.assertIn("predicted_regime", pred_df.columns)
        self.assertIn("actual_regime", pred_df.columns)
        self.assertIn("actual_ret_h_pct", pred_df.columns)
        self.assertTrue((pred_df["horizon_bars"] == 15).all())
        self.assertGreaterEqual(summary["accuracy"], 0.20)


if __name__ == "__main__":
    unittest.main()
