"""Full backtest statistics — v14/v15 hybrid-style report for v16 trade lists."""
from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from backtest.trade_display import print_trades_table_hkt

_OIL_LEG_TYPES = frozenset(
    {
        "wr90",
        "wr90_long",
        "ret",
        "oil_retrace",
        "ret_short",
        "oil_retrace_short",
        "long_ret",
        "long_retrace",
        "oil_long_retrace",
        "si",
        "short_impulse",
        "rip",
        "oil_rip_short",
    }
)


def _detect_asset(trades: list[dict]) -> Literal["gold", "oil"]:
    for t in trades:
        typ = str(t.get("type", t.get("_leg", "")))
        if typ in _OIL_LEG_TYPES:
            return "oil"
        if typ in ("v16_momentum", "v16_dip_short", "energetic") or t.get("matched_pattern"):
            return "gold"
    return "gold"


@contextmanager
def tee_stdout(path: Path):
    """Mirror stdout to a file (like shell `tee`)."""

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data: str) -> None:
            for stream in self.streams:
                stream.write(data)

        def flush(self) -> None:
            for stream in self.streams:
                stream.flush()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        old = sys.stdout
        sys.stdout = Tee(old, fh)
        try:
            yield
        finally:
            sys.stdout = old


def trades_to_dataframe(
    trades: list[dict],
    *,
    asset: Literal["gold", "oil", "auto"] = "auto",
) -> pd.DataFrame:
    """Normalize v16 gold or oil trade dicts to hybrid-backtest column names."""
    if not trades:
        return pd.DataFrame()
    if asset == "auto":
        asset = _detect_asset(trades)
    rows = []
    for t in trades:
        leg = str(t.get("_leg", t.get("type", "?")))
        typ = str(t.get("type", leg))
        if asset == "oil":
            src = typ
            matched = pd.NA
        else:
            src = typ if typ == "energetic" else (
                "pattern" if leg not in ("v16_momentum", "v16_dip_short", "energetic") else leg
            )
            if leg in ("v16_momentum", "v16_dip_short"):
                src = leg
            matched = typ if src == "pattern" else pd.NA
        side_raw = t.get("side", 1)
        side = int(side_raw) if side_raw in (1, -1) else (1 if str(side_raw).lower() in ("up", "long", "1") else -1)
        rows.append(
            {
                "entry_time": pd.Timestamp(t["entry"]).tz_convert("UTC"),
                "exit_time": pd.Timestamp(t["exit"]).tz_convert("UTC"),
                "pnl": float(t["pnl"]),
                "side": side,
                "source": src,
                "matched_pattern": matched,
                "pattern": leg,
                "exit_reason": t.get("reason", t.get("exit_reason", "")),
                "entry_price": t.get("entry_price"),
                "exit_price": t.get("exit_price"),
            }
        )
    return pd.DataFrame(rows)


def _as_tdf(
    trades: list[dict] | pd.DataFrame,
    *,
    asset: Literal["gold", "oil", "auto"] = "auto",
) -> pd.DataFrame:
    if isinstance(trades, pd.DataFrame):
        if "entry_time" in trades.columns:
            return trades.copy()
        return trades_to_dataframe(trades.to_dict("records"), asset=asset)
    return trades_to_dataframe(trades, asset=asset)


def _rebuild_stats(tdf: pd.DataFrame, csv_path: str | Path | None) -> dict:
    from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl

    if csv_path:
        path = Path(csv_path)
        tdf.to_csv(path, index=False)
        return rebuild_directional_pnl(path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = Path(f.name)
        tdf.to_csv(path, index=False)
    try:
        return rebuild_directional_pnl(path)
    finally:
        path.unlink(missing_ok=True)


def _print_wf_cycle_stats(tdf: pd.DataFrame) -> None:
    from config.hybrid_config import WF_CONFIG
    from xgboost_filter_model.pattern_training import iter_wf_cycles, wf_anchor_ts

    cdf = tdf.copy()
    cdf["entry_time"] = pd.to_datetime(cdf["entry_time"], utc=True)
    wf = wf_anchor_ts()
    bt_start_dt = cdf["entry_time"].min()
    end_dt = cdf["entry_time"].max() + pd.Timedelta(minutes=1)
    cycle_ranges: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for cycle, cur, ce in iter_wf_cycles(bt_start_dt, end_dt, wf):
        cycle_ranges.append((cur, ce, f"cycle_{cycle} ({cur.date()} to {ce.date()})"))

    def assign_cycle(ts: pd.Timestamp) -> str:
        ts = pd.to_datetime(ts, utc=True)
        for cur, ce, label in cycle_ranges:
            if cur <= ts < ce:
                return label
        return cycle_ranges[-1][2] if cycle_ranges else "unknown"

    cdf["retrain_cycle"] = cdf["entry_time"].apply(assign_cycle)
    cycle_stats = (
        cdf.groupby("retrain_cycle")["pnl"]
        .agg(
            trades="size",
            total_pnl="sum",
            avg_trade="mean",
            win_rate=lambda s: (s > 0).mean() * 100.0,
        )
        .reset_index()
    )
    cycle_stats["win_rate"] = cycle_stats["win_rate"].map(lambda v: f"{v:.1f}%")
    print(f"\n{'=' * 60}")
    print(f"  RETRAIN CYCLE STATISTICS  (every {WF_CONFIG['retrain_days']}d from WF anchor)")
    print(f"{'=' * 60}")
    print(cycle_stats.to_string(index=False))


def _print_session_heatmaps(stats: dict) -> None:
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


def print_full_stats(
    trades: list[dict] | pd.DataFrame,
    *,
    title: str,
    start: str,
    end: str,
    csv_path: str | Path | None = None,
    show_all_trades: bool = False,
    tail: int = 30,
    cycle_stats_mode: Literal["wf", "monthly", "none", "auto"] = "auto",
    asset: Literal["gold", "oil", "auto"] = "auto",
) -> dict:
    """Print v14/v15-style full statistics; return numeric stats dict."""
    if isinstance(trades, list):
        detected = asset if asset != "auto" else _detect_asset(trades)
    elif asset != "auto":
        detected = asset
    else:
        detected = "oil" if "oil" in title.lower() else "gold"

    if cycle_stats_mode == "auto":
        cycle_stats_mode = "monthly" if detected == "oil" else "wf"

    tdf = _as_tdf(trades, asset=detected)
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"  Period: {start} → {end}")
    print(f"{'=' * 60}")

    if tdf.empty:
        print("  No trades generated.")
        return {"trades": 0, "pnl": 0.0, "wr": 0.0, "max_dd": 0.0}

    tdf["pnl"] = tdf["pnl"].astype(float)
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"], utc=True)
    tdf["exit_time"] = pd.to_datetime(tdf["exit_time"], utc=True)

    wins = int((tdf["pnl"] > 0).sum())
    net_pnl = float(tdf["pnl"].sum())
    wr = float(wins / len(tdf) * 100.0) if len(tdf) else 0.0
    cum = tdf["pnl"].cumsum()
    max_dd = float((cum - cum.cummax()).min())

    print(f"  Trades       : {len(tdf)}  (W:{wins}  L:{len(tdf) - wins})")
    print(f"  Win Rate     : {wr:.1f}%")
    print(f"  Net PnL      : {net_pnl:+.1f}")
    print(f"  Avg Trade    : {net_pnl / len(tdf):+.2f}")
    print(f"  Max Drawdown : {max_dd:+.1f}")

    if "pattern" in tdf.columns:
        print("\n  By pattern:")
        for name, grp in tdf.groupby("pattern", dropna=False):
            label = name if pd.notna(name) else "unknown"
            print(
                f"    {str(label):20s}: {len(grp):4d} trades  "
                f"PnL={grp['pnl'].sum():+.1f}  WR={(grp['pnl'] > 0).mean() * 100:.0f}%  "
                f"avg={grp['pnl'].mean():+.2f}"
            )

    if "source" in tdf.columns:
        print("\n  By source:")
        for name, grp in tdf.groupby("source", dropna=False):
            label = name if pd.notna(name) else "unknown"
            print(
                f"    {str(label):20s}: {len(grp):4d} trades  "
                f"PnL={grp['pnl'].sum():+.1f}  WR={(grp['pnl'] > 0).mean() * 100:.0f}%  "
                f"avg={grp['pnl'].mean():+.2f}"
            )

    if "matched_pattern" in tdf.columns and tdf["matched_pattern"].notna().any():
        print("\n  By matched_pattern:")
        pat_df = tdf[tdf["matched_pattern"].notna()]
        for name, grp in pat_df.groupby("matched_pattern", dropna=False):
            label = name if pd.notna(name) else "unknown"
            print(
                f"    {str(label):20s}: {len(grp):4d} trades  "
                f"PnL={grp['pnl'].sum():+.1f}  WR={(grp['pnl'] > 0).mean() * 100:.0f}%  "
                f"avg={grp['pnl'].mean():+.2f}"
            )

    long_t = tdf[tdf["side"] == 1]
    short_t = tdf[tdf["side"] == -1]
    if len(long_t):
        print(
            f"\n  LONG : {len(long_t):4d} trades  PnL={long_t['pnl'].sum():+.1f}  "
            f"WR={(long_t['pnl'] > 0).mean() * 100:.0f}%  avg={long_t['pnl'].mean():+.2f}"
        )
    if len(short_t):
        print(
            f"  SHORT: {len(short_t):4d} trades  PnL={short_t['pnl'].sum():+.1f}  "
            f"WR={(short_t['pnl'] > 0).mean() * 100:.0f}%  avg={short_t['pnl'].mean():+.2f}"
        )

    if "exit_reason" in tdf.columns and tdf["exit_reason"].astype(str).str.strip().ne("").any():
        print("\n  Exit Breakdown:")
        for reason, grp in tdf.groupby("exit_reason", dropna=False):
            label = str(reason) if pd.notna(reason) and str(reason).strip() else "unknown"
            wr_r = (grp["pnl"] > 0).mean() * 100
            print(f"    {label:18s}: {len(grp):4d}  WR={wr_r:5.1f}%  avg={grp['pnl'].mean():7.2f}")

    stats = _rebuild_stats(tdf, csv_path)
    all_stats = stats.get("all", {})
    gross_win = float(all_stats.get("gross_profit") or 0.0)
    gross_loss = abs(float(all_stats.get("gross_loss") or 0.0))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    print(f"\n  Profit Factor   : {pf:.2f}")
    print(f"  Daily Drawdown  : {float(all_stats.get('daily_max_drawdown') or 0.0):.1f}")
    print(f"  Avg Day PnL     : {float(all_stats.get('avg_day') or 0.0):.1f}")
    print(f"  Positive Days   : {float(all_stats.get('positive_days_pct') or 0.0):.1f}%")
    print(f"  Trades/Day      : {float(all_stats.get('avg_trades_per_day') or 0.0):.1f}")
    print(f"  Avg Duration    : {float(all_stats.get('avg_duration_min') or 0.0):.1f} min")

    st = stats.get("streaks", {})
    print("\n  Streaks:")
    print(f"    Max Win Streak   : {int(st.get('max_win_streak', 0))}")
    print(f"    Max Loss Streak  : {int(st.get('max_loss_streak', 0))}")

    avg_win = float(all_stats.get("avg_win") or 0.0)
    avg_loss = float(all_stats.get("avg_loss") or 0.0)
    expectancy = (wr / 100.0) * avg_win + (1.0 - wr / 100.0) * avg_loss
    trades_per_day = float(all_stats.get("avg_trades_per_day") or 0.0)
    recovery_factor = (net_pnl / abs(max_dd)) if max_dd < 0 else float("inf")

    tdf_daily = tdf.copy()
    tdf_daily["trade_day"] = tdf_daily["entry_time"].dt.tz_convert("America/New_York").dt.floor("D")
    daily_pnl = tdf_daily.groupby("trade_day")["pnl"].sum().astype(float)
    mean_day = float(daily_pnl.mean()) if len(daily_pnl) else 0.0
    std_day = float(daily_pnl.std(ddof=1)) if len(daily_pnl) > 1 else 0.0
    downside = daily_pnl[daily_pnl < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sharpe = (mean_day / std_day) * np.sqrt(252.0) if std_day > 0 else 0.0
    sortino = (mean_day / downside_std) * np.sqrt(252.0) if downside_std > 0 else 0.0

    print("\n  Risk-Adjusted:")
    print(f"    Expectancy/Trade   : {expectancy:.2f}")
    print(f"    Expectancy/Day     : {expectancy * trades_per_day:.2f}")
    print(f"    Recovery Factor    : {recovery_factor:.3f}")
    print(f"    Sharpe  (annualized): {sharpe:.2f}")
    print(f"    Sortino (annualized): {sortino:.2f}")

    target_hit_stats = stats.get("target_hit", {})
    reverse_stats = stats.get("reverse_signal", {})
    timeout_stats = stats.get("timeout", {})
    print("\n  Exit Reason Details:")
    print(
        f"    Target Hit   : {int(target_hit_stats.get('trades', 0))} trades  "
        f"avg={float(target_hit_stats.get('avg_pnl') or 0.0):.2f}"
    )
    print(
        f"    Reverse Sig  : {int(reverse_stats.get('trades', 0))} trades  "
        f"WR={float(reverse_stats.get('win_rate_pct') or 0.0):.1f}%  "
        f"avg={float(reverse_stats.get('avg_pnl') or 0.0):.2f}"
    )
    print(
        f"    Timeout      : {int(timeout_stats.get('trades', 0))} trades  "
        f"WR={float(timeout_stats.get('win_rate_pct') or 0.0):.1f}%  "
        f"avg={float(timeout_stats.get('avg_pnl') or 0.0):.2f}"
    )

    print("\n  Target Updates:")
    print(
        f"    Mean={float(all_stats.get('target_updates_mean') or 0.0):.2f}  "
        f"Median={float(all_stats.get('target_updates_median') or 0.0):.2f}  "
        f"Max={int(all_stats.get('target_updates_max') or 0)}"
    )

    mdf = tdf.copy()
    mdf["month"] = (
        mdf["entry_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)
    )
    monthly = (
        mdf.groupby("month")["pnl"]
        .agg(
            trades="size",
            total_pnl="sum",
            avg_trade="mean",
            win_rate=lambda s: (s > 0).mean() * 100.0,
        )
        .reset_index()
    )
    monthly["win_rate"] = monthly["win_rate"].map(lambda v: f"{v:.1f}%")

    print(f"\n{'=' * 60}")
    print("  MONTHLY STATISTICS")
    print(f"{'=' * 60}")
    print(monthly.to_string(index=False))

    ydf = tdf.copy()
    ydf["year"] = ydf["entry_time"].dt.tz_convert("UTC").dt.year
    yearly = (
        ydf.groupby("year")["pnl"]
        .agg(
            trades="size",
            total_pnl="sum",
            avg_trade="mean",
            win_rate=lambda s: (s > 0).mean() * 100.0,
        )
        .reset_index()
    )
    yearly["win_rate"] = yearly["win_rate"].map(lambda v: f"{v:.1f}%")
    print(f"\n{'=' * 60}")
    print("  YEARLY STATISTICS")
    print(f"{'=' * 60}")
    print(yearly.to_string(index=False))

    if cycle_stats_mode == "wf":
        _print_wf_cycle_stats(tdf)
    elif cycle_stats_mode == "monthly":
        from oil.backtest_stats import print_monthly_wf_cycle_stats

        print_monthly_wf_cycle_stats(tdf)

    wdf = tdf.copy()
    wdf["weekday_utc2"] = (wdf["entry_time"] + pd.Timedelta(hours=2)).dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_utc2 = (
        wdf.groupby("weekday_utc2")["pnl"]
        .agg(
            trades="size",
            total_pnl="sum",
            avg_trade="mean",
            win_rate_pct=lambda s: (s > 0).mean() * 100.0,
        )
        .reindex([d for d in weekday_order if d in wdf["weekday_utc2"].unique()])
        .reset_index()
    )
    weekday_utc2["win_rate_pct"] = weekday_utc2["win_rate_pct"].map(lambda v: f"{v:.1f}%")
    print("\n  Weekday (UTC+2):")
    print(weekday_utc2.to_string(index=False))

    _print_session_heatmaps(stats)

    print_trades_table_hkt(tdf, show_all=show_all_trades, tail=tail)

    print(f"\n{'=' * 60}\n")
    return {
        "trades": len(tdf),
        "pnl": net_pnl,
        "wr": wr,
        "max_dd": max_dd,
        "profit_factor": pf,
        "sharpe": sharpe,
        "sortino": sortino,
    }


def print_extended_stats(
    trades: list[dict] | pd.DataFrame,
    *,
    title: str,
    start: str,
    end: str,
    csv_path: str | Path | None = None,
    show_all_trades: bool = False,
    tail: int = 30,
    cycle_stats_mode: Literal["wf", "monthly", "none", "auto"] = "auto",
    asset: Literal["gold", "oil", "auto"] = "auto",
) -> dict:
    """Alias for print_full_stats (v14/v15 complete report)."""
    return print_full_stats(
        trades,
        title=title,
        start=start,
        end=end,
        csv_path=csv_path,
        show_all_trades=show_all_trades,
        tail=tail,
        cycle_stats_mode=cycle_stats_mode,
        asset=asset,
    )
