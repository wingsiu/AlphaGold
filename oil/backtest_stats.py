"""Oil backtest reporting — monthly WF cycles and session heatmaps."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd


def normalize_oil_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize oil combined trades to rebuild_directional_pnl column names."""
    tdf = df.copy()
    tdf["pnl"] = tdf["pnl"].astype(float)

    if "entry_time" not in tdf.columns:
        tdf["entry_time"] = pd.to_datetime(tdf["entry"], utc=True)
    else:
        tdf["entry_time"] = pd.to_datetime(tdf["entry_time"], utc=True)

    if "exit_time" not in tdf.columns:
        tdf["exit_time"] = pd.to_datetime(tdf["exit"], utc=True)
    else:
        tdf["exit_time"] = pd.to_datetime(tdf["exit_time"], utc=True)

    if "exit_reason" not in tdf.columns:
        tdf["exit_reason"] = tdf.get("reason", "")

    if "side" in tdf.columns:
        side_num = pd.to_numeric(tdf["side"], errors="coerce")
        if side_num.notna().any():
            tdf.loc[side_num.notna(), "side"] = side_num.map({1: "up", -1: "down"})

    return tdf


def print_monthly_wf_cycle_stats(tdf: pd.DataFrame) -> None:
    """Group trades by calendar month (oil's monthly XGBoost WF retrain cycle)."""
    cdf = normalize_oil_trades(tdf)
    cdf["retrain_cycle"] = (
        cdf["entry_time"].dt.tz_convert("Asia/Hong_Kong").dt.strftime("%Y-%m")
    )
    cycle_stats = (
        cdf.groupby("retrain_cycle")["pnl"]
        .agg(
            trades="size",
            total_pnl="sum",
            avg_trade="mean",
            win_rate=lambda s: (s > 0).mean() * 100.0,
        )
        .reset_index()
        .sort_values("retrain_cycle")
    )
    cycle_stats["win_rate"] = cycle_stats["win_rate"].map(lambda v: f"{v:.1f}%")
    cycle_stats["cum_pnl"] = cycle_stats["total_pnl"].cumsum()

    print(f"\n{'=' * 60}")
    print("  MONTHLY WF RETRAIN CYCLE STATISTICS  (one XGBoost model per calendar month)")
    print(f"{'=' * 60}")
    print(cycle_stats.to_string(index=False))


def print_session_heatmaps(tdf: pd.DataFrame) -> None:
    """Session breakdown and HKT/London/NY heatmaps (gold-style)."""
    from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl

    norm = normalize_oil_trades(tdf)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = Path(f.name)
        norm.to_csv(path, index=False)
    try:
        stats = rebuild_directional_pnl(path)
    finally:
        path.unlink(missing_ok=True)

    all_stats = stats.get("all", {})
    td = all_stats.get("time_distribution", {})

    session_rows = td.get("by_session", [])
    if session_rows:
        session_df = pd.DataFrame(session_rows)
        print("\n  Session Breakdown:")
        print(
            session_df[["session", "trades", "total_pnl", "avg_trade", "win_rate_pct"]].to_string(
                index=False
            )
        )

    sh = td.get("session_heatmaps", {})
    if not sh:
        return

    print(f"\n{'=' * 60}")
    print("  SESSION HEATMAPS")
    print(f"{'=' * 60}")
    for sess in ("hkt", "london", "ny"):
        sess_block = sh.get(sess)
        if not sess_block:
            continue
        rendered = sess_block.get("rendered_tables", {})
        for metric_key in ("total_pnl", "win_rate_pct", "trade_count"):
            table = rendered.get(metric_key)
            if table:
                print(f"\n{table}")
