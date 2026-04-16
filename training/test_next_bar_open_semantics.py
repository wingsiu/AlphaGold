#!/usr/bin/env python3
"""Tiny self-check for next-bar-open execution semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.image_trend_ml import LABEL_FLAT, LABEL_UP, _compute_state_features_from_pred, directional_pnl_report


def main() -> int:
    signal_ts = pd.DatetimeIndex([
        "2026-01-01T10:00:00Z",
        "2026-01-01T10:01:00Z",
        "2026-01-01T10:02:00Z",
    ])
    entry_ts = pd.DatetimeIndex([
        "2026-01-01T10:01:00Z",
        "2026-01-01T10:02:00Z",
        "2026-01-01T10:03:00Z",
    ])
    fut_ts = pd.DatetimeIndex([
        "2026-01-01T10:03:00Z",
        "2026-01-01T10:04:00Z",
        "2026-01-01T10:05:00Z",
    ])

    pred = np.array([LABEL_UP, LABEL_FLAT, LABEL_FLAT], dtype=np.int64)
    curr = np.array([100.0, 98.0, 101.0], dtype=np.float64)
    entry_px = np.array([105.0, 99.0, 102.0], dtype=np.float64)
    fut = np.array([110.0, 100.0, 103.0], dtype=np.float64)

    pnl, trades = directional_pnl_report(
        signal_ts,
        entry_ts,
        fut_ts,
        pred,
        curr,
        entry_px,
        fut,
        adverse_limit=50.0,
        allow_overlap=False,
        reverse_exit_prob=0.7,
    )

    assert len(trades) == 1, trades
    first = trades.iloc[0]
    assert first["ts"] == signal_ts[0].isoformat()
    assert first["entry_time"] == entry_ts[0].isoformat()
    assert float(first["entry_price"]) == 105.0
    assert abs(float(first["pnl"]) - 5.0) < 1e-9
    assert int(pnl["trades"]) == 1
    assert abs(float(pnl["total_pnl"]) - 5.0) < 1e-9
    all_stats = pnl["all"]
    time_dist = all_stats["time_distribution"]
    assert time_dist["timezone_labels"]["hkt"] == "Asia/Hong_Kong"
    assert time_dist["timezone_labels"]["ny"] == "America/New_York"
    assert len(time_dist["by_weekday_hkt"]) == 1
    assert time_dist["by_weekday_hkt"][0]["trades"] == 1
    assert len(time_dist["by_hour_hkt"]) == 1
    assert time_dist["by_hour_hkt"][0]["trades"] == 1
    assert len(time_dist["by_hour_ny"]) == 1
    assert time_dist["by_hour_ny"][0]["trades"] == 1
    assert len(time_dist["by_session"]) == 1
    assert list(time_dist["weekday_hour_hkt_heatmap"].values())[0]

    state = _compute_state_features_from_pred(
        pred=np.array([LABEL_UP, LABEL_FLAT], dtype=np.int64),
        curr=np.array([100.0, 101.0], dtype=np.float64),
        entry_px=np.array([105.0, 999.0], dtype=np.float64),
        adverse_limit=50.0,
        trend_threshold=0.01,
        pred_history_len=1,
    )
    unrealized_return = float(state[1, -3])
    assert abs(unrealized_return - ((101.0 - 105.0) / 105.0)) < 1e-9

    print("ok: next-bar-open semantics verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

