#!/usr/bin/env python3
"""Self-check for canonical session heatmaps in rebuilt directional_pnl."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from rebuild_directional_pnl_from_trades import _empty_bucket, _time_distribution_stats


def main() -> int:
    empty_time_dist = cast(dict[str, Any], _empty_bucket()["time_distribution"])
    assert empty_time_dist["timezone_labels"]["hkt"] == "Asia/Hong_Kong"
    assert empty_time_dist["timezone_labels"]["london"] == "Europe/London"
    assert empty_time_dist["timezone_labels"]["ny"] == "America/New_York"
    assert set(empty_time_dist["session_heatmaps"].keys()) == {"hkt", "london", "ny"}

    df = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                [
                    "2026-01-05T00:30:00Z",  # Monday 08:30 HKT
                    "2026-01-05T09:15:00Z",  # Monday 09:15 London
                    "2026-01-05T15:45:00Z",  # Monday 10:45 NY (also London overlap)
                ],
                utc=True,
            ),
            "pnl": [10.0, -5.0, 7.5],
        }
    )

    time_dist = _time_distribution_stats(df)
    session_heatmaps = cast(dict[str, Any], time_dist["session_heatmaps"])

    hkt = session_heatmaps["hkt"]
    assert hkt["trades"] == 1
    assert hkt["trade_count_heatmap"]["Monday"]["08:00"] == 1
    assert hkt["total_pnl_heatmap"]["Monday"]["08:00"] == 10.0

    london = session_heatmaps["london"]
    assert london["trades"] == 2
    assert london["trade_count_heatmap"]["Monday"]["09:00"] == 1
    assert london["trade_count_heatmap"]["Monday"]["15:00"] == 1
    assert london["avg_trade_heatmap"]["Monday"]["09:00"] == -5.0
    assert london["total_pnl_heatmap"]["Monday"]["15:00"] == 7.5

    ny = session_heatmaps["ny"]
    assert ny["trades"] == 1
    assert ny["trade_count_heatmap"]["Monday"]["10:00"] == 1
    assert ny["win_rate_pct_heatmap"]["Monday"]["10:00"] == 100.0
    assert "NY SESSION HEATMAP — TRADE COUNT" in ny["rendered_tables"]["trade_count"]

    print("ok: rebuilt directional_pnl session heatmaps verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

