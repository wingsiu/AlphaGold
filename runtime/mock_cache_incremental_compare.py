#!/usr/bin/env python3
"""Compare best-base predictions from incremental cache updates vs full-window backtest style.

Loads MySQL data once, then simulates live cache growth one row at a time using the
same merge/cap rules as `IGPredictionDataCache._merge_raw`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import DataLoader
from trading_bot import (
    AlphaGoldTradingBot,
    BotConfig,
    DEFAULT_BEST_BASE_MODEL_PATH,
    IGPredictionDataCache,
    prepare_raw_price_frame,
)



def _to_utc(text: str) -> pd.Timestamp:
    ts = pd.Timestamp(text)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _series_to_df(payload: dict[str, Any], *, tag: str) -> pd.DataFrame:
    ts_idx = pd.DatetimeIndex(payload.get("ts", []))
    if len(ts_idx) == 0:
        return pd.DataFrame(
            columns=[
                "signal_ts",
                "entry_ts",
                "pred",
                "signal_prob",
                "trend_prob",
                "up_prob",
                "tag",
            ]
        )
    return pd.DataFrame(
        {
            "signal_ts": ts_idx.astype("datetime64[ns, UTC]"),
            "entry_ts": pd.DatetimeIndex(payload["entry_ts"]).astype("datetime64[ns, UTC]"),
            "pred": np.asarray(payload["pred"], dtype=int),
            "signal_prob": np.asarray(payload["signal_prob"], dtype=float),
            "trend_prob": np.asarray(payload["trend_prob"], dtype=float),
            "up_prob": np.asarray(payload["up_prob"], dtype=float),
            "tag": tag,
        }
    )


def _safe_float(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(v)
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mock parity test: incremental cache vs backtest-style best-base predictions")
    p.add_argument("--table", default="gold_prices")
    p.add_argument("--start-date", default="2026-04-11")
    p.add_argument("--end-date", default="2026-04-17")
    p.add_argument("--signal-model-path", default=DEFAULT_BEST_BASE_MODEL_PATH)
    p.add_argument("--target-entry-utc", default="2026-04-17T17:30:00+00:00")
    p.add_argument("--focus-minutes", type=int, default=30)
    p.add_argument("--incremental-warmup-minutes", type=int, default=480)
    p.add_argument("--rows-out", default="runtime/mock_cache_vs_backtest_rows.csv")
    p.add_argument("--focus-out", default="runtime/mock_cache_vs_backtest_focus.csv")
    p.add_argument("--summary-out", default="runtime/mock_cache_vs_backtest_summary.json")
    return p


def main() -> int:
    args = build_parser().parse_args()
    target_entry = _to_utc(args.target_entry_utc)
    focus_start = target_entry - pd.Timedelta(minutes=int(args.focus_minutes))
    focus_end = target_entry + pd.Timedelta(minutes=int(args.focus_minutes))
    inc_start = target_entry - pd.Timedelta(minutes=int(args.focus_minutes) + int(args.incremental_warmup_minutes))
    inc_end = focus_end

    raw = DataLoader().load_data(args.table, start_date=args.start_date, end_date=args.end_date)
    raw = prepare_raw_price_frame(raw)
    if raw.empty:
        raise SystemExit("No rows loaded from MySQL")

    cfg = BotConfig(
        table=args.table,
        signal_model_family="best_base_state",
        signal_model_path=args.signal_model_path,
        mode="signal_only",
        market_data_enabled=False,
        log_path="runtime/mock_cache_incremental_compare.log",
        state_path="runtime/mock_cache_incremental_compare_state.json",
        status_path="runtime/mock_cache_incremental_compare_status.json",
        trade_log_path="runtime/mock_cache_incremental_compare_unused.csv",
    )
    bot = AlphaGoldTradingBot(cfg)

    # Baseline 1: backtest-style (future horizon required).
    baseline_future = bot._build_best_base_signal_series(raw, require_future_horizon=True)
    baseline_future_df = _series_to_df(baseline_future, tag="baseline_future")

    # Baseline 2: live-style on full window (no future-horizon requirement).
    baseline_live = bot._build_best_base_signal_series(raw, require_future_horizon=False)
    baseline_live_df = _series_to_df(baseline_live, tag="baseline_live")

    # Incremental cache simulation: add one MySQL row per step through _merge_raw logic.
    cache = IGPredictionDataCache(cfg, bot.logger)
    steps: list[dict[str, Any]] = []

    iter_raw = raw.loc[(raw.index >= inc_start) & (raw.index <= inc_end)].copy()
    for i in range(len(iter_raw)):
        incoming = iter_raw.iloc[i : i + 1].copy()
        now_utc = pd.Timestamp(incoming.index[-1])
        cache_raw = cache._merge_raw(incoming, now_utc)
        if cache_raw is None or cache_raw.empty:
            continue

        payload = bot._build_best_base_signal_series(cache_raw, require_future_horizon=False)
        ts_idx = pd.DatetimeIndex(payload.get("ts", []))
        if len(ts_idx) == 0:
            continue

        steps.append(
            {
                "raw_end_ts": pd.Timestamp(cache_raw.index[-1]),
                "cache_rows": int(len(cache_raw)),
                "signal_ts": pd.Timestamp(ts_idx[-1]),
                "entry_ts": pd.Timestamp(payload["entry_ts"][-1]),
                "pred": int(payload["pred"][-1]),
                "signal_prob": float(payload["signal_prob"][-1]),
                "trend_prob": float(payload["trend_prob"][-1]),
                "up_prob": _safe_float(payload["up_prob"][-1]),
            }
        )

    if not steps:
        raise SystemExit("No incremental samples were produced")

    inc_df = pd.DataFrame(steps).sort_values("raw_end_ts").reset_index(drop=True)

    # Join incremental rows to baseline predictions by signal timestamp.
    base_live_map = baseline_live_df.drop_duplicates("signal_ts", keep="last").set_index("signal_ts")
    base_future_map = baseline_future_df.drop_duplicates("signal_ts", keep="last").set_index("signal_ts")

    def _lookup(base_map: pd.DataFrame, ts: pd.Timestamp, col: str) -> Any:
        if ts in base_map.index:
            return base_map.loc[ts, col]
        return np.nan

    inc_df["base_live_pred"] = [
        _lookup(base_live_map, pd.Timestamp(ts), "pred") for ts in inc_df["signal_ts"]
    ]
    inc_df["base_live_signal_prob"] = [
        _lookup(base_live_map, pd.Timestamp(ts), "signal_prob") for ts in inc_df["signal_ts"]
    ]
    inc_df["base_future_pred"] = [
        _lookup(base_future_map, pd.Timestamp(ts), "pred") for ts in inc_df["signal_ts"]
    ]
    inc_df["base_future_signal_prob"] = [
        _lookup(base_future_map, pd.Timestamp(ts), "signal_prob") for ts in inc_df["signal_ts"]
    ]

    inc_df["pred_match_live"] = (inc_df["pred"].astype(float) == inc_df["base_live_pred"].astype(float))
    inc_df["pred_match_future"] = (inc_df["pred"].astype(float) == inc_df["base_future_pred"].astype(float))

    # Focus near the requested time; primary view is by cache raw-end timestamps.
    focus_df = inc_df[(inc_df["raw_end_ts"] >= focus_start) & (inc_df["raw_end_ts"] <= focus_end)].copy()

    target_mask = inc_df["entry_ts"] == target_entry
    target_rows = inc_df[target_mask].copy()

    out_rows = ROOT / args.rows_out
    out_focus = ROOT / args.focus_out
    out_summary = ROOT / args.summary_out
    out_rows.parent.mkdir(parents=True, exist_ok=True)

    inc_df.to_csv(out_rows, index=False)
    focus_df.to_csv(out_focus, index=False)

    summary = {
        "table": args.table,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "target_entry_utc": target_entry.isoformat(),
        "focus_minutes": int(args.focus_minutes),
        "incremental_warmup_minutes": int(args.incremental_warmup_minutes),
        "incremental_range_start_utc": inc_start.isoformat(),
        "incremental_range_end_utc": inc_end.isoformat(),
        "incremental_input_rows": int(len(iter_raw)),
        "raw_rows": int(len(raw)),
        "baseline_live_samples": int(len(baseline_live_df)),
        "baseline_future_samples": int(len(baseline_future_df)),
        "incremental_steps_with_signal": int(len(inc_df)),
        "focus_rows": int(len(focus_df)),
        "target_entry_rows": int(len(target_rows)),
        "target_entry_preview": target_rows.head(10).to_dict(orient="records"),
        "focus_preview": focus_df.head(20).to_dict(orient="records"),
    }
    out_summary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("saved rows:", out_rows)
    print("saved focus:", out_focus)
    print("saved summary:", out_summary)
    print("target_entry_rows:", len(target_rows))
    if len(target_rows):
        r = target_rows.iloc[0]
        print(
            "target first row:",
            {
                "raw_end_ts": str(r["raw_end_ts"]),
                "signal_ts": str(r["signal_ts"]),
                "entry_ts": str(r["entry_ts"]),
                "pred": int(r["pred"]),
                "signal_prob": float(r["signal_prob"]),
                "base_live_pred": _safe_float(r["base_live_pred"]),
                "base_future_pred": _safe_float(r["base_future_pred"]),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

