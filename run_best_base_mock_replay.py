#!/usr/bin/env python3
"""Replay the promoted best-base model on a historical data slice.

This uses the same bot-side causal state-feature inference path as `trading_bot.py`,
but evaluates a fixed historical window and exports a compact JSON summary plus a
trade CSV so the results are easy to inspect.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from data import DataLoader
from trading_bot import (
    AlphaGoldTradingBot,
    BotConfig,
    DEFAULT_BEST_BASE_MODEL_PATH,
    prepare_raw_price_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parent
NY_TZ = ZoneInfo("America/New_York")
TRADING_DAY_CUTOFF_HOUR_NY = 17


def _safe_float(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _log_progress(step: str, started_at: float) -> None:
    elapsed = time.time() - started_at
    print(f"[{pd.Timestamp.utcnow().isoformat()}] {step} | elapsed={elapsed:.1f}s", flush=True)


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
        "reverse_signal_stats": {"trades": 0, "wins": 0, "losses": 0, "breakeven": 0, "avg_pnl": None, "win_rate_pct": None, "loss_rate_pct": None},
        "target_hit_stats": {"trades": 0, "avg_pnl": None},
        "timeout_stats": {"trades": 0, "wins": 0, "losses": 0, "breakeven": 0, "avg_pnl": None, "win_rate_pct": None, "loss_rate_pct": None},
        "target_updates_mean": 0.0,
        "target_updates_median": 0.0,
        "target_updates_max": 0,
    }


def _bucket_stats(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return _empty_bucket()

    x = df.copy()
    x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True)
    x["exit_time"] = pd.to_datetime(x["exit_time"], utc=True)
    x["duration_min"] = (x["exit_time"] - x["entry_time"]).dt.total_seconds() / 60.0
    x["trading_day"] = (x["entry_time"].dt.tz_convert(NY_TZ) - pd.Timedelta(hours=TRADING_DAY_CUTOFF_HOUR_NY)).dt.floor("D")

    pnl = pd.to_numeric(x["pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    pf = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None

    daily = x.groupby("trading_day")["pnl"].sum()
    equity_trade = pnl.cumsum()
    trade_dd = float((equity_trade - equity_trade.cummax()).min())
    equity_day = daily.cumsum()
    daily_dd = float((equity_day - equity_day.cummax()).min()) if len(daily) else 0.0

    streak = _streak_stats(pnl.to_numpy(dtype=np.float64))
    target_updates = pd.to_numeric(x.get("target_updates", 0.0), errors="coerce").fillna(0.0)
    x["exit_reason_norm"] = x["exit_reason"].astype(str)
    legacy_sig = x["exit_reason_norm"] == "signal_target"
    x.loc[legacy_sig & (x["pnl"] > 0.0), "exit_reason_norm"] = "target_hit"
    x.loc[legacy_sig & (x["pnl"] <= 0.0), "exit_reason_norm"] = "timeout"
    x.loc[x["exit_reason_norm"] == "horizon", "exit_reason_norm"] = "timeout"
    x.loc[(x["exit_reason_norm"] == "target_hit") & (x["pnl"] <= 0.0), "exit_reason_norm"] = "timeout"

    reverse_df = x[x["exit_reason_norm"] == "reverse_signal"]
    target_hit_df = x[x["exit_reason_norm"] == "target_hit"]
    timeout_df = x[x["exit_reason_norm"] == "timeout"]
    rev_n = int(len(reverse_df))
    th_n = int(len(target_hit_df))
    to_n = int(len(timeout_df))
    rev_wins = int((reverse_df["pnl"] > 0).sum()) if rev_n else 0
    rev_losses = int((reverse_df["pnl"] < 0).sum()) if rev_n else 0
    rev_be = int((reverse_df["pnl"] == 0).sum()) if rev_n else 0
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
        "exit_reason_counts": {str(k): int(v) for k, v in x["exit_reason_norm"].value_counts().items()},
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
    }


def _directional_pnl_from_trades(df: pd.DataFrame) -> dict[str, object]:
    all_stats = _bucket_stats(df)
    long_df = df[df["side"] == "up"].copy() if "side" in df.columns else df.iloc[0:0].copy()
    short_df = df[df["side"] == "down"].copy() if "side" in df.columns else df.iloc[0:0].copy()
    long_stats = _bucket_stats(long_df)
    short_stats = _bucket_stats(short_df)
    n_days = int(all_stats["n_days"])
    return {
        "trades": int(all_stats["trades"]),
        "total_pnl": float(all_stats["total_pnl"]),
        "avg_trade": float(all_stats["avg_trade"]),
        "n_days": n_days,
        "avg_trades_per_day": float(all_stats["avg_trades_per_day"]),
        "avg_day": float(all_stats["avg_day"]) if (n_days >= 1 and all_stats["avg_day"] is not None) else None,
        "positive_days_pct": float(all_stats["positive_days_pct"]) if (n_days >= 1 and all_stats["positive_days_pct"] is not None) else None,
        "max_drawdown": float(all_stats["trade_max_drawdown"]),
        "long": {
            "trades": int(long_stats["trades"]),
            "wins": int((pd.to_numeric(long_df.get("pnl"), errors="coerce").fillna(0.0) > 0).sum()) if len(long_df) else 0,
            "losses": int((pd.to_numeric(long_df.get("pnl"), errors="coerce").fillna(0.0) < 0).sum()) if len(long_df) else 0,
            "win_rate": float(long_stats["win_rate_pct"]) if len(long_df) else None,
            "total_pnl": float(long_stats["total_pnl"]),
            "avg_trade": float(long_stats["avg_trade"]),
        },
        "short": {
            "trades": int(short_stats["trades"]),
            "wins": int((pd.to_numeric(short_df.get("pnl"), errors="coerce").fillna(0.0) > 0).sum()) if len(short_df) else 0,
            "losses": int((pd.to_numeric(short_df.get("pnl"), errors="coerce").fillna(0.0) < 0).sum()) if len(short_df) else 0,
            "win_rate": float(short_stats["win_rate_pct"]) if len(short_df) else None,
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


def _parse_ts(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay the promoted best-base model on a historical slice.")
    p.add_argument("--table", default="gold_prices")
    p.add_argument("--start-date", default="2026-04-09")
    p.add_argument("--end-date", default="2026-04-15")
    p.add_argument("--signal-model-path", default=DEFAULT_BEST_BASE_MODEL_PATH)
    p.add_argument("--analysis-entry-start", default=None, help="Optional UTC timestamp filter applied after replay when comparing against a saved backtest window.")
    p.add_argument("--analysis-entry-end", default=None, help="Optional UTC timestamp upper bound applied after replay when comparing against a saved backtest window.")
    p.add_argument("--timeout-minutes", type=float, default=None, help="Optional hard backtest hold cap, independent of model horizon.")
    p.add_argument("--report-out", default="runtime/mock_best_base_after_2026-04-09_report.json")
    p.add_argument("--trades-out", default="runtime/mock_best_base_after_2026-04-09_trades.csv")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started_at = time.time()
    _log_progress("starting replay", started_at)

    _log_progress("loading raw data", started_at)
    raw = DataLoader().load_data(args.table, start_date=args.start_date, end_date=args.end_date)
    _log_progress(f"loaded raw data rows={len(raw)}", started_at)
    raw = prepare_raw_price_frame(raw)
    _log_progress("prepared raw price frame", started_at)

    cfg = BotConfig(
        table=args.table,
        signal_model_family="best_base_state",
        signal_model_path=args.signal_model_path,
        mode="signal_only",
        market_data_enabled=False,
        log_path="runtime/mock_best_base_replay.log",
        state_path="runtime/mock_best_base_replay_state.json",
        status_path="runtime/mock_best_base_replay_status.json",
        trade_log_path="runtime/mock_best_base_replay_unused.csv",
    )
    _log_progress("initializing trading bot/model bundle", started_at)
    bot = AlphaGoldTradingBot(cfg)
    _log_progress("building causal signal series", started_at)
    series = bot._build_best_base_signal_series(raw, require_future_horizon=True)
    _log_progress("built causal signal series", started_at)

    ts_idx = cast(pd.DatetimeIndex, series["ts"])
    entry_ts_idx = cast(pd.DatetimeIndex, series["entry_ts"])
    fut_ts_idx = cast(pd.DatetimeIndex, series["fut_ts"])
    pred = cast(np.ndarray, series["pred"])
    signal_prob = cast(np.ndarray, series["signal_prob"])
    curr = cast(np.ndarray, series["curr"])
    entry = cast(np.ndarray, series["entry"])
    fut = cast(np.ndarray, series["fut"])

    if len(ts_idx) == 0:
        raise ValueError("Replay produced no eligible best-base samples for the requested window.")

    image_trend = bot.image_trend
    model_cfg = dict(bot.model_bundle.get("config") or {})
    allow_overlap = bool(bot.model_bundle.get("allow_overlap_backtest", model_cfg.get("allow_overlap_backtest", False)))
    reverse_exit_prob_raw: Any = bot.model_bundle.get("reverse_exit_prob", model_cfg.get("reverse_exit_prob", 0.7))
    reverse_exit_prob = _safe_float(reverse_exit_prob_raw, 0.7)
    long_adverse_limit_raw: Any = model_cfg.get("long_adverse_limit")
    short_adverse_limit_raw: Any = model_cfg.get("short_adverse_limit")
    long_target_threshold_raw: Any = model_cfg.get("long_target_threshold", model_cfg.get("trend_threshold"))
    short_target_threshold_raw: Any = model_cfg.get("short_target_threshold", model_cfg.get("trend_threshold"))
    long_adverse_limit = None if long_adverse_limit_raw is None else _safe_float(long_adverse_limit_raw)
    short_adverse_limit = None if short_adverse_limit_raw is None else _safe_float(short_adverse_limit_raw)
    long_target_threshold = None if long_target_threshold_raw is None else _safe_float(long_target_threshold_raw)
    short_target_threshold = None if short_target_threshold_raw is None else _safe_float(short_target_threshold_raw)

    _log_progress("running directional pnl simulation", started_at)
    pnl, trades_df = image_trend.directional_pnl_report(
        ts_idx,
        entry_ts_idx,
        fut_ts_idx,
        pred,
        curr,
        entry,
        fut,
        signal_prob=signal_prob,
        adverse_limit=float(model_cfg.get("adverse_limit", 15.0)),
        long_target_threshold=long_target_threshold,
        short_target_threshold=short_target_threshold,
        long_adverse_limit=long_adverse_limit,
        short_adverse_limit=short_adverse_limit,
        allow_overlap=allow_overlap,
        reverse_exit_prob=reverse_exit_prob,
        max_hold_minutes=args.timeout_minutes,
    )
    _log_progress(f"simulation complete trades={len(trades_df)}", started_at)
    pre_filter_trades_df = trades_df.copy()
    analysis_entry_start = _parse_ts(args.analysis_entry_start)
    analysis_entry_end = _parse_ts(args.analysis_entry_end)
    filtered_trades_df = trades_df.copy()
    if analysis_entry_start is not None:
        filtered_trades_df = filtered_trades_df[pd.to_datetime(filtered_trades_df["entry_time"], utc=True) >= analysis_entry_start].copy()
    if analysis_entry_end is not None:
        filtered_trades_df = filtered_trades_df[pd.to_datetime(filtered_trades_df["entry_time"], utc=True) <= analysis_entry_end].copy()
    filtered_trades_df = filtered_trades_df.sort_values("entry_time").reset_index(drop=True)
    if analysis_entry_start is not None or analysis_entry_end is not None:
        _log_progress("applying analysis entry-time filter", started_at)
        pnl = _directional_pnl_from_trades(filtered_trades_df)
        trades_df = filtered_trades_df
    _log_progress(f"post-filter trades={len(trades_df)}", started_at)

    up_count = int((pred == image_trend.LABEL_UP).sum())
    down_count = int((pred == image_trend.LABEL_DOWN).sum())
    flat_count = int((pred == image_trend.LABEL_FLAT).sum())
    risky_count = int((pred == image_trend.LABEL_RISKY).sum())
    tradable_count = up_count + down_count
    best_base_runtime = cast(dict[str, Any], bot.last_best_base_payload_info)

    report = {
        "table": args.table,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "signal_model_path": args.signal_model_path,
        "model_horizon_minutes": int(model_cfg.get("horizon", 0) or 0),
        "timeout_minutes_override": None if args.timeout_minutes is None else float(args.timeout_minutes),
        "analysis_entry_start_utc": None if analysis_entry_start is None else analysis_entry_start.isoformat(),
        "analysis_entry_end_utc": None if analysis_entry_end is None else analysis_entry_end.isoformat(),
        "raw_rows": int(len(raw)),
        "raw_start_utc": raw.index[0].isoformat(),
        "raw_end_utc": raw.index[-1].isoformat(),
        "bars_1m": int(best_base_runtime.get("bars_rows", len(raw))),
        "candidate_samples": int(len(ts_idx)),
        "executed_trades_before_analysis_filter": int(len(pre_filter_trades_df)),
        "executed_trades_after_analysis_filter": int(len(trades_df)),
        "signal_start_utc": pd.Timestamp(ts_idx[0]).isoformat(),
        "signal_end_utc": pd.Timestamp(ts_idx[-1]).isoformat(),
        "prediction_counts": {
            "down": down_count,
            "flat": flat_count,
            "up": up_count,
            "risky": risky_count,
            "tradable": tradable_count,
        },
        "avg_signal_prob_tradable": float(np.mean(signal_prob[np.isin(pred, list(image_trend.TREND_LABELS))])) if tradable_count else 0.0,
        "directional_pnl": pnl,
    }

    report_path = PROJECT_ROOT / args.report_out
    trades_path = PROJECT_ROOT / args.trades_out
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    _log_progress(f"writing report -> {report_path}", started_at)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log_progress(f"writing trades -> {trades_path}", started_at)
    trades_df.to_csv(trades_path, index=False)
    _log_progress("replay finished", started_at)

    print(json.dumps(
        {
            "report_out": str(report_path),
            "trades_out": str(trades_path),
            "raw_rows": report["raw_rows"],
            "candidate_samples": report["candidate_samples"],
            "tradable_signals": tradable_count,
            "trades": int(pnl.get("trades", 0)),
            "total_pnl": _safe_float(pnl.get("total_pnl", 0.0)),
            "avg_day": _safe_float(pnl.get("avg_day", 0.0)),
            "timeout_minutes_override": None if args.timeout_minutes is None else float(args.timeout_minutes),
            "profit_factor": _safe_float((pnl.get("all") or {}).get("profit_factor", 0.0)),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

