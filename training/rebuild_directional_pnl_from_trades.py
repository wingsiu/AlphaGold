#!/usr/bin/env python3
"""Rebuild directional_pnl stats from an existing trades CSV.

This lets you refresh report format/stats without retraining the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypedDict, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY_TZ = ZoneInfo("America/New_York")
HK_TZ = ZoneInfo("Asia/Hong_Kong")
LONDON_TZ = ZoneInfo("Europe/London")
TRADING_DAY_CUTOFF_HOUR_NY = 17
ASIA_SESSION_START = 6
ASIA_SESSION_END = 19
NY_SESSION_START = 6
NY_SESSION_END = 17
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class SessionSpec(TypedDict):
    label: str
    timezone: str
    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int
    note: str


SESSION_SPECS: dict[str, SessionSpec] = {
    "hkt": {
        "label": "HKT",
        "timezone": "Asia/Hong_Kong",
        "start_hour": 8,
        "start_minute": 0,
        "end_hour": 16,
        "end_minute": 0,
        "note": "HKT session window 08:00 <= local time < 16:00",
    },
    "london": {
        "label": "London",
        "timezone": "Europe/London",
        "start_hour": 8,
        "start_minute": 0,
        "end_hour": 16,
        "end_minute": 30,
        "note": "London session window 08:00 <= local time < 16:30 (DST-aware)",
    },
    "ny": {
        "label": "NY",
        "timezone": "America/New_York",
        "start_hour": 9,
        "start_minute": 30,
        "end_hour": 16,
        "end_minute": 0,
        "note": "NY session window 09:30 <= local time < 16:00 (DST-aware)",
    },
}


def _spec_int(spec: SessionSpec, key: str) -> int:
    return int(cast(Any, spec[key]))


def _spec_str(spec: SessionSpec, key: str) -> str:
    return str(cast(Any, spec[key]))


def _streak_stats(pnl_vals: np.ndarray) -> dict[str, int]:
    max_win = 0
    max_loss = 0
    cur_win = 0
    cur_loss = 0
    for v in pnl_vals:
        if v > 0:
            cur_win += 1
            cur_loss = 0
            max_win = max(max_win, cur_win)
        elif v < 0:
            cur_loss += 1
            cur_win = 0
            max_loss = max(max_loss, cur_loss)
        else:
            cur_win = 0
            cur_loss = 0
    current_streak = cur_win if cur_win > 0 else cur_loss
    return {
        "max_win_streak": int(max_win),
        "max_loss_streak": int(max_loss),
        "current_win_streak": int(cur_win),
        "current_loss_streak": int(cur_loss),
        "current_streak": int(current_streak),
    }


def _empty_bucket() -> dict[str, object]:
    return {
        "trades": 0,
        "total_pnl": 0.0,
        "avg_trade": 0.0,
        "median_trade": 0.0,
        "win_rate_pct": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "profit_factor": None,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "avg_win": None,
        "avg_loss": None,
        "max_win_streak": 0,
        "max_loss_streak": 0,
        "current_streak": 0,
        "current_win_streak": 0,
        "current_loss_streak": 0,
        "avg_duration_min": 0.0,
        "median_duration_min": 0.0,
        "min_duration_min": 0.0,
        "max_duration_min": 0.0,
        "n_days": 0,
        "avg_trades_per_day": 0.0,
        "avg_day": None,
        "median_day": None,
        "best_day": None,
        "worst_day": None,
        "positive_days_pct": None,
        "trade_max_drawdown": 0.0,
        "daily_max_drawdown": 0.0,
        "exit_reason_counts": {},
        "reverse_signal_stats": {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "avg_pnl": None,
            "win_rate_pct": None,
            "loss_rate_pct": None,
        },
        "target_hit_stats": {
            "trades": 0,
            "avg_pnl": None,
        },
        "timeout_stats": {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "avg_pnl": None,
            "win_rate_pct": None,
            "loss_rate_pct": None,
        },
        "target_updates_mean": 0.0,
        "target_updates_median": 0.0,
        "target_updates_max": 0,
        "time_distribution": {
            "timezone_labels": {
                "hkt": "Asia/Hong_Kong",
                "london": "Europe/London",
                "ny": "America/New_York",
            },
            "by_weekday_hkt": [],
            "by_hour_hkt": [],
            "by_hour_ny": [],
            "by_session": [],
            "weekday_hour_hkt_heatmap": {},
            "session_heatmaps": {key: _empty_session_heatmap(key, spec) for key, spec in SESSION_SPECS.items()},
        },
    }


def _hour_values_for_session(spec: SessionSpec) -> list[int]:
    start_hour = _spec_int(spec, "start_hour")
    end_hour = _spec_int(spec, "end_hour")
    end_minute = _spec_int(spec, "end_minute")
    last_hour = end_hour if end_minute > 0 else end_hour - 1
    return list(range(start_hour, last_hour + 1))


def _hour_label(session_key: str, hour: int) -> str:
    if session_key == "ny" and hour == 9:
        return "09:30"
    return f"{hour:02d}:00"


def _in_session(local_ts: pd.Series, spec: SessionSpec) -> pd.Series:
    minute_of_day = local_ts.dt.hour * 60 + local_ts.dt.minute
    start_min = _spec_int(spec, "start_hour") * 60 + _spec_int(spec, "start_minute")
    end_min = _spec_int(spec, "end_hour") * 60 + _spec_int(spec, "end_minute")
    return (minute_of_day >= start_min) & (minute_of_day < end_min)


def _safe_pct_wins(pnl: pd.Series) -> float | None:
    if len(pnl) == 0:
        return None
    return float((pnl > 0).mean() * 100.0)


def _render_heatmap_table(title: str, day_rows: dict[str, dict[str, object]], hour_labels: list[str], formatter) -> str:
    col_width = max(8, max(len(h) for h in hour_labels) + 1)
    lines = [title]
    header = f"{'Day':<12}" + "".join(h.rjust(col_width) for h in hour_labels)
    lines.append(header)
    lines.append("-" * len(header))
    for day in DAY_ORDER:
        vals = day_rows[day]
        line = f"{day:<12}"
        for hour in hour_labels:
            line += formatter(vals[hour]).rjust(col_width)
        lines.append(line)
    return "\n".join(lines)


def _empty_session_heatmap(session_key: str, spec: SessionSpec) -> dict[str, object]:
    hour_labels = [_hour_label(session_key, hour) for hour in _hour_values_for_session(spec)]
    empty_rows: dict[str, dict[str, object]] = {day: {label: None for label in hour_labels} for day in DAY_ORDER}
    return {
        "label": _spec_str(spec, "label"),
        "timezone": _spec_str(spec, "timezone"),
        "session_window": {
            "start": f"{_spec_int(spec, 'start_hour'):02d}:{_spec_int(spec, 'start_minute'):02d}",
            "end_exclusive": f"{_spec_int(spec, 'end_hour'):02d}:{_spec_int(spec, 'end_minute'):02d}",
            "note": _spec_str(spec, "note"),
        },
        "trades": 0,
        "total_pnl": 0.0,
        "avg_trade": None,
        "win_rate_pct": None,
        "hour_labels": hour_labels,
        "day_labels": DAY_ORDER,
        "cell_stats": {day: dict(row) for day, row in empty_rows.items()},
        "trade_count_heatmap": {day: dict(row) for day, row in empty_rows.items()},
        "win_rate_pct_heatmap": {day: dict(row) for day, row in empty_rows.items()},
        "avg_trade_heatmap": {day: dict(row) for day, row in empty_rows.items()},
        "total_pnl_heatmap": {day: dict(row) for day, row in empty_rows.items()},
        "rendered_tables": {
            "trade_count": _render_heatmap_table(
                f"{_spec_str(spec, 'label')} SESSION HEATMAP — TRADE COUNT",
                {day: dict(row) for day, row in empty_rows.items()},
                hour_labels,
                lambda v: "--" if v is None else f"{int(cast(Any, v))}",
            ),
            "win_rate_pct": _render_heatmap_table(
                f"{_spec_str(spec, 'label')} SESSION HEATMAP — WIN RATE %",
                {day: dict(row) for day, row in empty_rows.items()},
                hour_labels,
                lambda v: "--" if v is None else f"{float(cast(Any, v)):.1f}% ",
            ),
            "avg_trade": _render_heatmap_table(
                f"{_spec_str(spec, 'label')} SESSION HEATMAP — AVG TRADE PNL",
                {day: dict(row) for day, row in empty_rows.items()},
                hour_labels,
                lambda v: "--" if v is None else f"{float(cast(Any, v)):.2f}",
            ),
            "total_pnl": _render_heatmap_table(
                f"{_spec_str(spec, 'label')} SESSION HEATMAP — TOTAL PNL",
                {day: dict(row) for day, row in empty_rows.items()},
                hour_labels,
                lambda v: "--" if v is None else f"{float(cast(Any, v)):.2f}",
            ),
        },
    }


def _build_session_heatmap(df: pd.DataFrame, session_key: str, spec: SessionSpec) -> dict[str, object]:
    tz = ZoneInfo(_spec_str(spec, "timezone"))
    entry_time = cast(pd.Series, df["entry_time"])
    local_ts = entry_time.dt.tz_convert(tz)
    sdf = df.loc[_in_session(local_ts, spec)].copy()
    if sdf.empty:
        return _empty_session_heatmap(session_key, spec)

    local_ts = local_ts.loc[sdf.index]
    sdf["local_day"] = local_ts.dt.day_name()
    sdf["local_hour"] = local_ts.dt.hour
    hour_values = _hour_values_for_session(spec)
    hour_labels = [_hour_label(session_key, hour) for hour in hour_values]
    hour_label_map = {hour: _hour_label(session_key, hour) for hour in hour_values}

    cell_stats: dict[str, dict[str, object]] = {day: {label: None for label in hour_labels} for day in DAY_ORDER}
    trade_count_heatmap: dict[str, dict[str, object]] = {day: {label: None for label in hour_labels} for day in DAY_ORDER}
    win_rate_heatmap: dict[str, dict[str, object]] = {day: {label: None for label in hour_labels} for day in DAY_ORDER}
    avg_trade_heatmap: dict[str, dict[str, object]] = {day: {label: None for label in hour_labels} for day in DAY_ORDER}
    total_pnl_heatmap: dict[str, dict[str, object]] = {day: {label: None for label in hour_labels} for day in DAY_ORDER}

    for day in DAY_ORDER:
        day_df = sdf[sdf["local_day"] == day]
        for hour in hour_values:
            label = hour_label_map[hour]
            hour_df = day_df[day_df["local_hour"] == hour]
            if hour_df.empty:
                continue
            pnl = hour_df["pnl"].astype(float)
            trades = int(len(hour_df))
            total_pnl = float(pnl.sum())
            avg_trade = float(pnl.mean())
            win_rate_pct = _safe_pct_wins(pnl)
            stats = {
                "trades": trades,
                "total_pnl": total_pnl,
                "avg_trade": avg_trade,
                "win_rate_pct": win_rate_pct,
            }
            cell_stats[day][label] = stats
            trade_count_heatmap[day][label] = trades
            win_rate_heatmap[day][label] = win_rate_pct
            avg_trade_heatmap[day][label] = avg_trade
            total_pnl_heatmap[day][label] = total_pnl

    session_pnl = sdf["pnl"].astype(float)
    return {
        "label": _spec_str(spec, "label"),
        "timezone": _spec_str(spec, "timezone"),
        "session_window": {
            "start": f"{_spec_int(spec, 'start_hour'):02d}:{_spec_int(spec, 'start_minute'):02d}",
            "end_exclusive": f"{_spec_int(spec, 'end_hour'):02d}:{_spec_int(spec, 'end_minute'):02d}",
            "note": _spec_str(spec, "note"),
        },
        "trades": int(len(sdf)),
        "total_pnl": float(session_pnl.sum()),
        "avg_trade": float(session_pnl.mean()),
        "win_rate_pct": _safe_pct_wins(session_pnl),
        "hour_labels": hour_labels,
        "day_labels": DAY_ORDER,
        "cell_stats": cell_stats,
        "trade_count_heatmap": trade_count_heatmap,
        "win_rate_pct_heatmap": win_rate_heatmap,
        "avg_trade_heatmap": avg_trade_heatmap,
        "total_pnl_heatmap": total_pnl_heatmap,
        "rendered_tables": {
            "trade_count": _render_heatmap_table(
                f"{_spec_str(spec, 'label')} SESSION HEATMAP — TRADE COUNT",
                trade_count_heatmap,
                hour_labels,
                lambda v: "--" if v is None else f"{int(cast(Any, v))}",
            ),
            "win_rate_pct": _render_heatmap_table(
                f"{_spec_str(spec, 'label')} SESSION HEATMAP — WIN RATE %",
                win_rate_heatmap,
                hour_labels,
                lambda v: "--" if v is None else f"{float(cast(Any, v)):.1f}% ",
            ),
            "avg_trade": _render_heatmap_table(
                f"{_spec_str(spec, 'label')} SESSION HEATMAP — AVG TRADE PNL",
                avg_trade_heatmap,
                hour_labels,
                lambda v: "--" if v is None else f"{float(cast(Any, v)):.2f}",
            ),
            "total_pnl": _render_heatmap_table(
                f"{_spec_str(spec, 'label')} SESSION HEATMAP — TOTAL PNL",
                total_pnl_heatmap,
                hour_labels,
                lambda v: "--" if v is None else f"{float(cast(Any, v)):.2f}",
            ),
        },
    }


def _time_distribution_stats(x: pd.DataFrame) -> dict[str, object]:
    if x.empty:
        return cast(dict[str, object], _empty_bucket()["time_distribution"])

    tdf = x.copy()
    entry_time = cast(pd.Series, tdf["entry_time"])
    hkt_time = entry_time.dt.tz_convert(HK_TZ)
    london_time = entry_time.dt.tz_convert(LONDON_TZ)
    ny_time = entry_time.dt.tz_convert(NY_TZ)
    tdf["weekday_hkt"] = hkt_time.dt.day_name()
    tdf["hour_hkt"] = hkt_time.dt.hour
    tdf["hour_london"] = london_time.dt.hour
    tdf["hour_ny"] = ny_time.dt.hour

    def _session_name(row: pd.Series) -> str:
        row_any = cast(Any, row)
        h_hkt = int(row_any["hour_hkt"])
        h_ny = int(row_any["hour_ny"])
        if ASIA_SESSION_START <= h_hkt <= ASIA_SESSION_END:
            return "Asia"
        if NY_SESSION_START <= h_ny <= NY_SESSION_END:
            return "NY"
        return "Other"

    tdf["session"] = tdf[["hour_hkt", "hour_ny"]].apply(_session_name, axis=1)

    def _summarize(group_cols: list[str], *, sort_key=None) -> list[dict[str, object]]:
        grouped = cast(pd.DataFrame, (
            tdf.groupby(group_cols, dropna=False)["pnl"]
            .agg(["size", "sum", "mean", lambda s: float((s > 0).mean() * 100.0)])
            .reset_index()
        ))
        grouped.columns = [*group_cols, "trades", "total_pnl", "avg_trade", "win_rate_pct"]
        rows: list[dict[str, object]] = []
        for _, row in grouped.iterrows():
            row_any = cast(Any, row)
            item: dict[str, object] = {}
            for col in group_cols:
                val = row_any[col]
                if pd.isna(val):
                    item[col] = None
                elif str(col).startswith("hour_"):
                    item[col] = int(val)
                else:
                    item[col] = str(val)
            item["trades"] = int(row_any["trades"])
            item["total_pnl"] = float(row_any["total_pnl"])
            item["avg_trade"] = float(row_any["avg_trade"])
            item["win_rate_pct"] = float(row_any["win_rate_pct"])
            rows.append(item)
        if sort_key is not None:
            rows.sort(key=sort_key)
        return rows

    by_weekday_hkt = _summarize(["weekday_hkt"], sort_key=lambda r: DAY_ORDER.index(cast(str, r["weekday_hkt"])) if cast(str, r["weekday_hkt"]) in DAY_ORDER else 99)
    by_hour_hkt = _summarize(["hour_hkt"], sort_key=lambda r: int(cast(Any, r["hour_hkt"])))
    by_hour_ny = _summarize(["hour_ny"], sort_key=lambda r: int(cast(Any, r["hour_ny"])))
    by_session = _summarize(["session"], sort_key=lambda r: {"Asia": 0, "NY": 1, "Other": 2}.get(str(r["session"]), 99))
    heatmap_rows = _summarize(
        ["weekday_hkt", "hour_hkt"],
        sort_key=lambda r: (
            DAY_ORDER.index(cast(str, r["weekday_hkt"])) if cast(str, r["weekday_hkt"]) in DAY_ORDER else 99,
            int(r["hour_hkt"]),
        ),
    )
    weekday_hour_hkt_heatmap: dict[str, dict[str, dict[str, object]]] = {}
    for row in heatmap_rows:
        day = str(row["weekday_hkt"])
        hour_key = f"{int(cast(Any, row['hour_hkt'])):02d}:00"
        weekday_hour_hkt_heatmap.setdefault(day, {})[hour_key] = {
            "trades": int(cast(Any, row["trades"])),
            "total_pnl": float(cast(Any, row["total_pnl"])),
            "avg_trade": float(cast(Any, row["avg_trade"])),
            "win_rate_pct": float(cast(Any, row["win_rate_pct"])),
        }

    return {
        "timezone_labels": {
            "hkt": "Asia/Hong_Kong",
            "london": "Europe/London",
            "ny": "America/New_York",
        },
        "by_weekday_hkt": by_weekday_hkt,
        "by_hour_hkt": by_hour_hkt,
        "by_hour_ny": by_hour_ny,
        "by_session": by_session,
        "weekday_hour_hkt_heatmap": weekday_hour_hkt_heatmap,
        "session_heatmaps": {key: _build_session_heatmap(tdf, key, spec) for key, spec in SESSION_SPECS.items()},
    }


def _bucket_stats(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return _empty_bucket()

    x = df.copy()
    x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True)
    x["exit_time"] = pd.to_datetime(x["exit_time"], utc=True)
    x["duration_min"] = (x["exit_time"] - x["entry_time"]).dt.total_seconds() / 60.0
    x["trading_day"] = (x["entry_time"].dt.tz_convert(NY_TZ) - pd.Timedelta(hours=TRADING_DAY_CUTOFF_HOUR_NY)).dt.floor("D")

    pnl = x["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    pf = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None

    daily = x.groupby("trading_day")["pnl"].sum()
    equity_trade = pnl.cumsum()
    trade_dd = float((equity_trade - equity_trade.cummax()).min())
    equity_day = daily.cumsum()
    daily_dd = float((equity_day - equity_day.cummax()).min()) if len(daily) > 0 else 0.0

    streak = _streak_stats(pnl.to_numpy(dtype=np.float64))
    if "target_updates" in x.columns:
        target_updates = pd.to_numeric(x["target_updates"], errors="coerce").fillna(0.0)
    else:
        target_updates = pd.Series(np.zeros(len(x), dtype=float))

    if "exit_reason" in x.columns:
        x["exit_reason_norm"] = x["exit_reason"].astype(str)
        legacy_sig = x["exit_reason_norm"] == "signal_target"
        x.loc[legacy_sig & (x["pnl"] > 0.0), "exit_reason_norm"] = "target_hit"
        x.loc[legacy_sig & (x["pnl"] <= 0.0), "exit_reason_norm"] = "timeout"
        x.loc[x["exit_reason_norm"] == "horizon", "exit_reason_norm"] = "timeout"
        x.loc[(x["exit_reason_norm"] == "target_hit") & (x["pnl"] <= 0.0), "exit_reason_norm"] = "timeout"

        reverse_df = x[x["exit_reason_norm"] == "reverse_signal"]
        target_hit_df = x[x["exit_reason_norm"] == "target_hit"]
        timeout_df = x[x["exit_reason_norm"] == "timeout"]
        exit_reason_counts = {str(k): int(v) for k, v in x["exit_reason_norm"].value_counts().items()}
    else:
        reverse_df = x.iloc[0:0]
        target_hit_df = x.iloc[0:0]
        timeout_df = x.iloc[0:0]
        exit_reason_counts = {}

    rev_n = int(len(reverse_df))
    rev_wins = int((reverse_df["pnl"] > 0).sum()) if rev_n else 0
    rev_losses = int((reverse_df["pnl"] < 0).sum()) if rev_n else 0
    rev_be = int((reverse_df["pnl"] == 0).sum()) if rev_n else 0
    th_n = int(len(target_hit_df))
    to_n = int(len(timeout_df))
    to_wins = int((timeout_df["pnl"] > 0).sum()) if to_n else 0
    to_losses = int((timeout_df["pnl"] < 0).sum()) if to_n else 0
    to_be = int((timeout_df["pnl"] == 0).sum()) if to_n else 0

    return {
        "trades": int(len(x)),
        "total_pnl": float(pnl.sum()),
        "avg_trade": float(pnl.mean()),
        "median_trade": float(pnl.median()),
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": float(pf) if pf is not None else None,
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "max_win_streak": int(streak["max_win_streak"]),
        "max_loss_streak": int(streak["max_loss_streak"]),
        "current_streak": int(streak["current_streak"]),
        "current_win_streak": int(streak["current_win_streak"]),
        "current_loss_streak": int(streak["current_loss_streak"]),
        "avg_duration_min": float(x["duration_min"].mean()),
        "median_duration_min": float(x["duration_min"].median()),
        "min_duration_min": float(x["duration_min"].min()),
        "max_duration_min": float(x["duration_min"].max()),
        "n_days": int(len(daily)),
        "avg_trades_per_day": float(len(x) / len(daily)) if len(daily) else 0.0,
        "avg_day": float(daily.mean()) if len(daily) else None,
        "median_day": float(daily.median()) if len(daily) else None,
        "best_day": float(daily.max()) if len(daily) else None,
        "worst_day": float(daily.min()) if len(daily) else None,
        "positive_days_pct": float((daily > 0).mean() * 100.0) if len(daily) else None,
        "trade_max_drawdown": trade_dd,
        "daily_max_drawdown": daily_dd,
        "exit_reason_counts": exit_reason_counts,
        "reverse_signal_stats": {
            "trades": rev_n,
            "wins": rev_wins,
            "losses": rev_losses,
            "breakeven": rev_be,
            "avg_pnl": float(reverse_df["pnl"].mean()) if rev_n else None,
            "win_rate_pct": float((rev_wins / rev_n) * 100.0) if rev_n else None,
            "loss_rate_pct": float((rev_losses / rev_n) * 100.0) if rev_n else None,
        },
        "target_hit_stats": {
            "trades": th_n,
            "avg_pnl": float(target_hit_df["pnl"].mean()) if th_n else None,
        },
        "timeout_stats": {
            "trades": to_n,
            "wins": to_wins,
            "losses": to_losses,
            "breakeven": to_be,
            "avg_pnl": float(timeout_df["pnl"].mean()) if to_n else None,
            "win_rate_pct": float((to_wins / to_n) * 100.0) if to_n else None,
            "loss_rate_pct": float((to_losses / to_n) * 100.0) if to_n else None,
        },
        "target_updates_mean": float(target_updates.mean()) if len(target_updates) else 0.0,
        "target_updates_median": float(target_updates.median()) if len(target_updates) else 0.0,
        "target_updates_max": int(target_updates.max()) if len(target_updates) else 0,
        "time_distribution": _time_distribution_stats(x),
    }


def rebuild_directional_pnl(trades_csv: Path) -> dict[str, object]:
    df = pd.read_csv(trades_csv)
    required = {"entry_time", "exit_time", "side", "pnl"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in trades CSV: {missing}")

    all_stats = cast(dict[str, Any], _bucket_stats(df))
    long_stats = cast(dict[str, Any], _bucket_stats(df[df["side"] == "up"]))
    short_stats = cast(dict[str, Any], _bucket_stats(df[df["side"] == "down"]))

    n_days = int(all_stats["n_days"])
    return {
        "trades": int(all_stats["trades"]),
        "total_pnl": float(all_stats["total_pnl"]),
        "avg_trade": float(all_stats["avg_trade"]),
        "n_days": n_days,
        "avg_trades_per_day": float(all_stats["avg_trades_per_day"]),
        "avg_day": float(all_stats["avg_day"]) if (n_days >= 5 and all_stats["avg_day"] is not None) else None,
        "positive_days_pct": float(all_stats["positive_days_pct"]) if (n_days >= 5 and all_stats["positive_days_pct"] is not None) else None,
        "max_drawdown": float(all_stats["trade_max_drawdown"]),
        "long": {
            "trades": int(long_stats["trades"]),
            "wins": int((df[(df["side"] == "up")]["pnl"] > 0).sum()) if long_stats["trades"] else 0,
            "losses": int((df[(df["side"] == "up")]["pnl"] < 0).sum()) if long_stats["trades"] else 0,
            "win_rate": float(long_stats["win_rate_pct"]) if long_stats["trades"] else None,
            "total_pnl": float(long_stats["total_pnl"]),
            "avg_trade": float(long_stats["avg_trade"]),
        },
        "short": {
            "trades": int(short_stats["trades"]),
            "wins": int((df[(df["side"] == "down")]["pnl"] > 0).sum()) if short_stats["trades"] else 0,
            "losses": int((df[(df["side"] == "down")]["pnl"] < 0).sum()) if short_stats["trades"] else 0,
            "win_rate": float(short_stats["win_rate_pct"]) if short_stats["trades"] else None,
            "total_pnl": float(short_stats["total_pnl"]),
            "avg_trade": float(short_stats["avg_trade"]),
        },
        "streaks": {
            "max_win_streak": int(all_stats["max_win_streak"]),
            "max_loss_streak": int(all_stats["max_loss_streak"]),
            "current_win_streak": int(all_stats["current_win_streak"]),
            "current_loss_streak": int(all_stats["current_loss_streak"]),
        },
        "reverse_signal": dict(all_stats.get("reverse_signal_stats", {})),
        "target_hit": dict(all_stats.get("target_hit_stats", {})),
        "timeout": dict(all_stats.get("timeout_stats", {})),
        "all": all_stats,
        "long_up": long_stats,
        "short_down": short_stats,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Rebuild directional_pnl from an existing trades CSV.")
    p.add_argument("--trades-csv", required=True, help="Path to trades CSV (e.g. ...backtest_trades_*.csv)")
    p.add_argument("--report-in", default=None, help="Optional existing report JSON to patch.")
    p.add_argument("--report-out", default=None, help="Output JSON path. If omitted and --report-in set, overwrite input report.")
    p.add_argument("--stats-out", default=None, help="Optional path to write directional_pnl JSON only.")
    args = p.parse_args()

    trades_csv = Path(args.trades_csv).expanduser().resolve()
    if not trades_csv.exists():
        raise FileNotFoundError(f"Trades CSV not found: {trades_csv}")

    pnl = rebuild_directional_pnl(trades_csv)

    if args.stats_out:
        stats_out = Path(args.stats_out).expanduser().resolve()
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_out.write_text(json.dumps(pnl, indent=2), encoding="utf-8")
        print(f"Saved directional_pnl stats: {stats_out}")

    if args.report_in:
        report_in = Path(args.report_in).expanduser().resolve()
        if not report_in.exists():
            raise FileNotFoundError(f"Report JSON not found: {report_in}")
        out_path = Path(args.report_out).expanduser().resolve() if args.report_out else report_in
        report = json.loads(report_in.read_text(encoding="utf-8"))
        report["directional_pnl"] = pnl
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved patched report: {out_path}")

    if not args.stats_out and not args.report_in:
        print(json.dumps(pnl, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

