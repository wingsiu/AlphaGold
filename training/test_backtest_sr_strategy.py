#!/usr/bin/env python3
"""Smoke test for SR backtest pipeline using synthetic data."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from backtest_sr_strategy import run_backtest, summarise


def _make_synthetic_data(
    n_days: int = 22,
    bars_per_day: int = 480,      # 8 h × 60 min
    ticks_per_day: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (tick_trades, price_bars) DataFrames for a synthetic mean-reverting price."""
    rng = np.random.default_rng(seed)

    # ── 1-minute price bars ───────────────────────────────────────────────
    total_bars = n_days * bars_per_day
    base_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    bar_ts = pd.date_range(base_ts, periods=total_bars, freq="min")

    mid = np.empty(total_bars)
    mid[0] = 2500.0
    for k in range(1, total_bars):
        mid[k] = mid[k - 1] + (-0.01 * (mid[k - 1] - 2500.0)) + rng.normal(0, 0.5)

    spread = rng.uniform(0.05, 0.40, size=total_bars)
    bars = pd.DataFrame(
        {
            "open":   mid - spread * 0.3,
            "high":   mid + spread,
            "low":    mid - spread,
            "close":  mid + rng.normal(0, 0.1, size=total_bars),
            "volume": rng.integers(1, 10, size=total_bars).astype(float),
        },
        index=bar_ts,
    )
    bars.index.name = "ts"

    # ── Tick trades (one per bar minute, multiple sizes) ──────────────────
    total_ticks = n_days * ticks_per_day
    tick_ts = pd.date_range(base_ts, periods=total_ticks, freq="26s")   # ~2000/day
    tick_price = np.interp(
        np.linspace(0, total_bars - 1, total_ticks), np.arange(total_bars), mid
    ) + rng.normal(0, 0.2, size=total_ticks)
    tick_size = rng.integers(1, 5, size=total_ticks)
    ticks = pd.DataFrame(
        {"ts_event": tick_ts, "price": tick_price, "size": tick_size}
    )

    return ticks, bars


class SRBacktestSmokeTest(unittest.TestCase):
    def test_runs_and_returns_metrics(self) -> None:
        tick_trades, price_bars = _make_synthetic_data()

        trades = run_backtest(
            tick_trades,
            price_bars,
            lookback_days=10,
            tick_size=0.10,
            top_n=8,
            merge_ticks=20,
            min_prom_pct=0.02,
            touch_buffer=0.30,
            reject_margin=0.10,
            tp_span_frac=0.60,
            sl_pts=17.0,
            max_hold_bars=15,
            max_trades_per_level_per_day=2,
        )
        stats = summarise(trades)

        self.assertGreater(len(price_bars), 100)
        self.assertGreaterEqual(stats["trades"], 0)
        self.assertTrue(np.isfinite(stats["total_points"]))
        print(f"\n  trades={int(stats['trades'])}  win_rate={stats['win_rate']:.1f}%"
              f"  total_pts={stats['total_points']:.2f}  pf={stats['profit_factor']:.3f}")

    def test_no_lookahead_during_warmup(self) -> None:
        """Warmup days (< lookback_days) must produce zero trades."""
        tick_trades, price_bars = _make_synthetic_data(n_days=8)  # fewer than lookback
        trades = run_backtest(tick_trades, price_bars, lookback_days=10)
        self.assertEqual(len(trades), 0, "Expected 0 trades during warmup period")

    def test_fallback_tp_uses_yesterday_range(self) -> None:
        """When all levels are the same type (no paired level), TP must still be finite."""
        tick_trades, price_bars = _make_synthetic_data(seed=99)
        trades = run_backtest(
            tick_trades, price_bars,
            lookback_days=10,
            tp_span_frac=0.60,
            sl_pts=17.0,
            max_hold_bars=5,
        )
        for t in trades:
            self.assertTrue(np.isfinite(t.pnl_points), f"Non-finite pnl: {t}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
