"""Rebuild directional PnL report dict from a saved trades CSV.

Used by backtest_no_retrain.py to compute full stats after the engine
has already written trades to disk (avoids re-running the full backtest).
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HK_TZ = ZoneInfo("Asia/Hong_Kong")
LONDON_TZ = ZoneInfo("Europe/London")
NY_TZ = ZoneInfo("America/New_York")
TRADING_DAY_CUTOFF_HOUR_NY = 17
ASIA_SESSION_START = 6
ASIA_SESSION_END = 19
NY_SESSION_START = 6
NY_SESSION_END = 17

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Session definitions: (timezone, start_hour, end_hour inclusive)
_SESSION_SPECS = {
    "hkt":    (HK_TZ,     8, 15),
    "london": (LONDON_TZ, 8, 16),
    "ny":     (NY_TZ,     8, 15),
}


def _build_session_heatmaps(x: pd.DataFrame) -> dict:
    """Build session_heatmaps structure: {session: {cell_stats, rendered_tables}}."""
    result: dict = {}
    for session, (tz, start_h, end_h) in _SESSION_SPECS.items():
        sess_time = x["entry_time"].dt.tz_convert(tz)
        mask = (sess_time.dt.hour >= start_h) & (sess_time.dt.hour <= end_h)
        sf = x[mask].copy()
        sf["_weekday"] = sess_time[mask].dt.day_name()
        sf["_hour"] = sess_time[mask].dt.hour

        cell_stats: dict = {}
        if not sf.empty:
            for (day, hour), grp in sf.groupby(["_weekday", "_hour"], sort=False):
                pnl_s = grp["pnl"].astype(float)
                n = int(len(grp))
                tot = float(pnl_s.sum())
                avg = float(pnl_s.mean())
                wr = float((pnl_s > 0).mean() * 100.0)
                h_key = f"{int(hour):02d}:00"
                cell_stats.setdefault(str(day), {})[h_key] = {
                    "trades": n, "total_pnl": tot, "avg_trade": avg, "win_rate_pct": wr,
                }

        # Build simple ASCII rendered tables
        days_present = sorted(cell_stats.keys(), key=lambda d: WEEKDAY_ORDER.index(d) if d in WEEKDAY_ORDER else 99)
        all_hours: list[str] = sorted({h for d in cell_stats.values() for h in d}, key=lambda h: int(h.split(":")[0]))

        def _render_table(metric: str, fmt_fn) -> str:
            if not days_present or not all_hours:
                return f"[{session.upper()} {metric}: no data]"
            col_w = 8
            header = f"{'':12s}" + "".join(f"{h:>{col_w}s}" for h in all_hours)
            rows = [header, "-" * len(header)]
            for day in days_present:
                row = f"{day:<12s}"
                for h in all_hours:
                    st = cell_stats.get(day, {}).get(h)
                    row += f"{fmt_fn(st):>{col_w}s}"
                rows.append(row)
            return f"[{session.upper()} — {metric}]\n" + "\n".join(rows)

        rendered_tables = {
            "trade_count":  _render_table("Trade Count",  lambda s: str(s["trades"]) if s else "-"),
            "win_rate_pct": _render_table("Win Rate %",   lambda s: f"{s['win_rate_pct']:.0f}%" if s else "-"),
            "avg_trade":    _render_table("Avg Trade",    lambda s: f"{s['avg_trade']:.1f}" if s else "-"),
            "total_pnl":    _render_table("Total PnL",    lambda s: f"{s['total_pnl']:.1f}" if s else "-"),
        }
        result[session] = {"cell_stats": cell_stats, "rendered_tables": rendered_tables}
    return result


def rebuild_directional_pnl(trades_csv: Path | str) -> dict:
    """Load a trades CSV and return the same report dict as directional_pnl_report."""
    pdf = pd.read_csv(trades_csv)
    if pdf.empty:
        return _empty_report()

    def _streak_stats(pnl_vals: np.ndarray) -> dict:
        max_win = max_loss = cur_win = cur_loss = 0
        for v in pnl_vals:
            if v > 0:
                cur_win += 1; cur_loss = 0
                max_win = max(max_win, cur_win)
            elif v < 0:
                cur_loss += 1; cur_win = 0
                max_loss = max(max_loss, cur_loss)
            else:
                cur_win = cur_loss = 0
        return {
            "max_win_streak": int(max_win),
            "max_loss_streak": int(max_loss),
            "current_win_streak": int(cur_win),
            "current_loss_streak": int(cur_loss),
            "current_streak": int(cur_win if cur_win > 0 else cur_loss),
        }

    def _time_distribution_stats(x: pd.DataFrame) -> dict:
        if x.empty:
            return {
                "timezone_labels": {"hkt": "Asia/Hong_Kong", "ny": "America/New_York"},
                "by_weekday_hkt": [], "by_hour_hkt": [], "by_hour_ny": [],
                "by_session": [], "weekday_hour_hkt_heatmap": {}, "session_heatmaps": {},
            }
        tdf = x.copy()
        hkt_time = tdf["entry_time"].dt.tz_convert(HK_TZ)
        ny_time  = tdf["entry_time"].dt.tz_convert(NY_TZ)
        tdf["weekday_hkt"] = hkt_time.dt.day_name()
        tdf["hour_hkt"] = hkt_time.dt.hour
        tdf["hour_ny"]  = ny_time.dt.hour

        def _session(row: pd.Series) -> str:
            if ASIA_SESSION_START <= int(row["hour_hkt"]) <= ASIA_SESSION_END:
                return "Asia"
            if NY_SESSION_START <= int(row["hour_ny"]) <= NY_SESSION_END:
                return "NY"
            return "Other"

        tdf["session"] = tdf[["hour_hkt", "hour_ny"]].apply(_session, axis=1)

        def _summarize(group_cols, *, sort_key=None):
            grouped = (
                tdf.groupby(group_cols, dropna=False)["pnl"]
                .agg(["size", "sum", "mean", lambda s: float((s > 0).mean() * 100.0)])
                .reset_index()
            )
            grouped.columns = [*group_cols, "trades", "total_pnl", "avg_trade", "win_rate_pct"]
            rows = []
            for _, row in grouped.iterrows():
                item = {col: (None if pd.isna(row[col]) else (int(row[col]) if str(col).startswith("hour_") else str(row[col]))) for col in group_cols}
                item.update({"trades": int(row["trades"]), "total_pnl": float(row["total_pnl"]),
                              "avg_trade": float(row["avg_trade"]), "win_rate_pct": float(row["win_rate_pct"])})
                rows.append(item)
            if sort_key:
                rows.sort(key=sort_key)
            return rows

        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_weekday_hkt = _summarize(["weekday_hkt"], sort_key=lambda r: weekday_order.index(r["weekday_hkt"]) if r["weekday_hkt"] in weekday_order else 99)
        by_hour_hkt = _summarize(["hour_hkt"], sort_key=lambda r: int(r["hour_hkt"]))
        by_hour_ny  = _summarize(["hour_ny"],  sort_key=lambda r: int(r["hour_ny"]))
        by_session  = _summarize(["session"],  sort_key=lambda r: {"Asia": 0, "NY": 1, "Other": 2}.get(str(r["session"]), 99))
        heatmap_rows = _summarize(["weekday_hkt", "hour_hkt"],
            sort_key=lambda r: (weekday_order.index(r["weekday_hkt"]) if r["weekday_hkt"] in weekday_order else 99, int(r["hour_hkt"])))
        heatmap: dict = {}
        for row in heatmap_rows:
            day = str(row["weekday_hkt"])
            hk = f"{int(row['hour_hkt']):02d}:00"
            heatmap.setdefault(day, {})[hk] = {"trades": int(row["trades"]), "total_pnl": float(row["total_pnl"]),
                                                "avg_trade": float(row["avg_trade"]), "win_rate_pct": float(row["win_rate_pct"])}
        return {"timezone_labels": {"hkt": "Asia/Hong_Kong", "ny": "America/New_York"},
                "by_weekday_hkt": by_weekday_hkt, "by_hour_hkt": by_hour_hkt,
                "by_hour_ny": by_hour_ny, "by_session": by_session,
                "weekday_hour_hkt_heatmap": heatmap,
                "session_heatmaps": _build_session_heatmaps(x)}

    def _bucket_stats(df: pd.DataFrame) -> dict:
        if df.empty:
            return _empty_bucket()
        x = df.copy()
        x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True)
        x["exit_time"]  = pd.to_datetime(x["exit_time"],  utc=True)
        x["duration_min"] = (x["exit_time"] - x["entry_time"]).dt.total_seconds() / 60.0
        x["trading_day"] = (x["entry_time"].dt.tz_convert(NY_TZ) - pd.Timedelta(hours=TRADING_DAY_CUTOFF_HOUR_NY)).dt.floor("D")

        pnl = x["pnl"].astype(float)
        wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
        gross_profit = float(wins.sum()); gross_loss = float(losses.sum())
        pf = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None
        daily = x.groupby("trading_day")["pnl"].sum()
        equity_trade = pnl.cumsum()
        trade_dd = float((equity_trade - equity_trade.cummax()).min())
        equity_day = daily.cumsum()
        daily_dd = float((equity_day - equity_day.cummax()).min()) if len(daily) > 0 else 0.0
        streak = _streak_stats(pnl.to_numpy(dtype=np.float64))
        target_updates = pd.to_numeric(x.get("target_updates", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

        x["exit_reason_norm"] = x["exit_reason"].astype(str)
        legacy = x["exit_reason_norm"] == "signal_target"
        x.loc[legacy & (x["pnl"] > 0.0),  "exit_reason_norm"] = "target_hit"
        x.loc[legacy & (x["pnl"] <= 0.0), "exit_reason_norm"] = "timeout"
        x.loc[x["exit_reason_norm"] == "horizon", "exit_reason_norm"] = "timeout"
        x.loc[(x["exit_reason_norm"] == "target_hit") & (x["pnl"] <= 0.0), "exit_reason_norm"] = "timeout"

        reverse_df   = x[x["exit_reason_norm"] == "reverse_signal"]
        target_hit_df = x[x["exit_reason_norm"] == "target_hit"]
        timeout_df   = x[x["exit_reason_norm"] == "timeout"]
        rev_n = len(reverse_df); th_n = len(target_hit_df); to_n = len(timeout_df)
        rev_wins = int((reverse_df["pnl"] > 0).sum()) if rev_n else 0
        rev_losses = int((reverse_df["pnl"] < 0).sum()) if rev_n else 0
        rev_be = int((reverse_df["pnl"] == 0).sum()) if rev_n else 0
        to_wins = int((timeout_df["pnl"] > 0).sum()) if to_n else 0
        to_losses = int((timeout_df["pnl"] < 0).sum()) if to_n else 0
        to_be = int((timeout_df["pnl"] == 0).sum()) if to_n else 0

        return {
            "trades": int(len(x)), "total_pnl": float(pnl.sum()), "avg_trade": float(pnl.mean()),
            "median_trade": float(pnl.median()), "win_rate_pct": float((pnl > 0).mean() * 100.0),
            "gross_profit": gross_profit, "gross_loss": gross_loss,
            "profit_factor": float(pf) if pf is not None else None,
            "best_trade": float(pnl.max()), "worst_trade": float(pnl.min()),
            "avg_win": float(wins.mean()) if len(wins) else None,
            "avg_loss": float(losses.mean()) if len(losses) else None,
            **streak,
            "avg_duration_min": float(x["duration_min"].mean()),
            "median_duration_min": float(x["duration_min"].median()),
            "min_duration_min": float(x["duration_min"].min()),
            "max_duration_min": float(x["duration_min"].max()),
            "n_days": int(len(daily)),
            "avg_trades_per_day": float(len(x) / len(daily)) if len(daily) else 0.0,
            "avg_day":  float(daily.mean())   if len(daily) else None,
            "median_day": float(daily.median()) if len(daily) else None,
            "best_day":  float(daily.max())   if len(daily) else None,
            "worst_day": float(daily.min())   if len(daily) else None,
            "positive_days_pct": float((daily > 0).mean() * 100.0) if len(daily) else None,
            "trade_max_drawdown": trade_dd, "daily_max_drawdown": daily_dd,
            "exit_reason_counts": {str(k): int(v) for k, v in x["exit_reason_norm"].value_counts().items()},
            "reverse_signal_stats": {
                "trades": int(rev_n), "wins": rev_wins, "losses": rev_losses, "breakeven": rev_be,
                "avg_pnl": float(reverse_df["pnl"].mean()) if rev_n else None,
                "win_rate_pct": float(rev_wins / rev_n * 100.0) if rev_n else None,
                "loss_rate_pct": float(rev_losses / rev_n * 100.0) if rev_n else None,
            },
            "target_hit_stats": {
                "trades": int(th_n),
                "avg_pnl": float(target_hit_df["pnl"].mean()) if th_n else None,
            },
            "timeout_stats": {
                "trades": int(to_n), "wins": to_wins, "losses": to_losses, "breakeven": to_be,
                "avg_pnl": float(timeout_df["pnl"].mean()) if to_n else None,
                "win_rate_pct": float(to_wins / to_n * 100.0) if to_n else None,
                "loss_rate_pct": float(to_losses / to_n * 100.0) if to_n else None,
            },
            "target_updates_mean":   float(target_updates.mean())   if len(target_updates) else 0.0,
            "target_updates_median": float(target_updates.median()) if len(target_updates) else 0.0,
            "target_updates_max":    int(target_updates.max())      if len(target_updates) else 0,
            "time_distribution": _time_distribution_stats(x),
        }

    all_stats   = _bucket_stats(pdf)
    long_stats  = _bucket_stats(pdf[pdf["side"] == "up"])
    short_stats = _bucket_stats(pdf[pdf["side"] == "down"])
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
            "wins": int((pdf[pdf["side"] == "up"]["pnl"] > 0).sum()) if long_stats["trades"] else 0,
            "losses": int((pdf[pdf["side"] == "up"]["pnl"] < 0).sum()) if long_stats["trades"] else 0,
            "win_rate": float(long_stats["win_rate_pct"]) if long_stats["trades"] else None,
            "total_pnl": float(long_stats["total_pnl"]),
            "avg_trade": float(long_stats["avg_trade"]),
        },
        "short": {
            "trades": int(short_stats["trades"]),
            "wins": int((pdf[pdf["side"] == "down"]["pnl"] > 0).sum()) if short_stats["trades"] else 0,
            "losses": int((pdf[pdf["side"] == "down"]["pnl"] < 0).sum()) if short_stats["trades"] else 0,
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


def _empty_bucket() -> dict:
    return {
        "trades": 0, "total_pnl": 0.0, "avg_trade": 0.0, "median_trade": 0.0,
        "win_rate_pct": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
        "profit_factor": None, "best_trade": 0.0, "worst_trade": 0.0,
        "avg_win": None, "avg_loss": None,
        "max_win_streak": 0, "max_loss_streak": 0, "current_streak": 0,
        "current_win_streak": 0, "current_loss_streak": 0,
        "avg_duration_min": 0.0, "median_duration_min": 0.0,
        "min_duration_min": 0.0, "max_duration_min": 0.0,
        "n_days": 0, "avg_trades_per_day": 0.0,
        "avg_day": None, "median_day": None, "best_day": None, "worst_day": None,
        "positive_days_pct": None, "trade_max_drawdown": 0.0, "daily_max_drawdown": 0.0,
        "exit_reason_counts": {},
        "reverse_signal_stats": {"trades": 0, "wins": 0, "losses": 0, "breakeven": 0, "avg_pnl": None, "win_rate_pct": None, "loss_rate_pct": None},
        "target_hit_stats": {"trades": 0, "avg_pnl": None},
        "timeout_stats": {"trades": 0, "wins": 0, "losses": 0, "breakeven": 0, "avg_pnl": None, "win_rate_pct": None, "loss_rate_pct": None},
        "target_updates_mean": 0.0, "target_updates_median": 0.0, "target_updates_max": 0,
        "time_distribution": {"timezone_labels": {"hkt": "Asia/Hong_Kong", "ny": "America/New_York"},
                               "by_weekday_hkt": [], "by_hour_hkt": [], "by_hour_ny": [], "by_session": [],
                               "weekday_hour_hkt_heatmap": {}, "session_heatmaps": {}},
    }


def _empty_report() -> dict:
    b = _empty_bucket()
    return {
        "trades": 0, "total_pnl": 0.0, "avg_trade": 0.0, "n_days": 0,
        "avg_trades_per_day": 0.0, "avg_day": None, "positive_days_pct": None,
        "max_drawdown": 0.0,
        "long": {"trades": 0, "wins": 0, "losses": 0, "win_rate": None, "total_pnl": 0.0, "avg_trade": 0.0},
        "short": {"trades": 0, "wins": 0, "losses": 0, "win_rate": None, "total_pnl": 0.0, "avg_trade": 0.0},
        "streaks": {"max_win_streak": 0, "max_loss_streak": 0, "current_win_streak": 0, "current_loss_streak": 0},
        "reverse_signal": {}, "target_hit": {}, "timeout": {},
        "all": b, "long_up": b, "short_down": b,
    }

