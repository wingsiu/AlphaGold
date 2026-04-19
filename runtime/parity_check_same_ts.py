#!/usr/bin/env python3
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import DataLoader
from trading_bot import AlphaGoldTradingBot, BotConfig, prepare_raw_price_frame


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "runtime/trading_bot_status.json"
OUT_PATH = ROOT / "runtime/_parity_same_ts_check.json"


def _to_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def main() -> int:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    summary = dict(status.get("prediction_cache_last_summary") or {})

    cache_end = summary.get("cache_end_utc") or status.get("prediction_cache_last_bucket_utc")
    if not cache_end:
        raise SystemExit("No cache_end_utc found in runtime/trading_bot_status.json")

    cache_start = summary.get("cache_start_utc")
    end_ts = _to_utc(str(cache_end))
    start_ts = _to_utc(str(cache_start)) if cache_start else end_ts - pd.Timedelta(days=2)

    table = (status.get("input_data") or {}).get("table", "gold_prices")
    model_path = status.get("signal_model_path", "runtime/backtest_model_best_base_weak_nostate.joblib")

    raw = DataLoader().load_data(
        table,
        start_date=start_ts.date().isoformat(),
        end_date=end_ts.date().isoformat(),
    )
    raw = prepare_raw_price_frame(raw)

    cfg = BotConfig(
        table=table,
        signal_model_family="best_base_state",
        signal_model_path=model_path,
        mode="signal_only",
        market_data_enabled=False,
        log_path="runtime/_parity_probe.log",
        state_path="runtime/_parity_probe_state.json",
        status_path="runtime/_parity_probe_status.json",
        trade_log_path="runtime/_parity_probe_trades.csv",
    )

    bot = AlphaGoldTradingBot(cfg)
    series = bot._build_best_base_signal_series(raw, require_future_horizon=False)
    ts_idx = pd.DatetimeIndex(series["ts"])

    match_idx = None
    if len(ts_idx):
        mask = ts_idx == end_ts
        if mask.any():
            match_idx = int(mask.argmax())

    raw_close = None
    if end_ts in raw.index:
        row = raw.loc[end_ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        raw_close = float(row.get("closePrice")) if row.get("closePrice") is not None else None

    result = {
        "target_ts_utc": end_ts.isoformat(),
        "data_load": {
            "table": table,
            "start_date": start_ts.date().isoformat(),
            "end_date": end_ts.date().isoformat(),
            "raw_rows": int(len(raw)),
            "raw_first_utc": raw.index[0].isoformat() if len(raw) else None,
            "raw_last_utc": raw.index[-1].isoformat() if len(raw) else None,
        },
        "live_status_snapshot": {
            "cache_end_utc": summary.get("cache_end_utc"),
            "cache_rows": summary.get("cache_rows"),
            "latest_close_from_status": (status.get("best_base_runtime") or {}).get("latest_close"),
            "latest_range150_ok": (status.get("best_base_runtime") or {}).get("latest_range150_ok"),
            "latest_15m_drop_ok": (status.get("best_base_runtime") or {}).get("latest_15m_drop_ok"),
        },
        "raw_row_at_target": {
            "exists": bool(end_ts in raw.index),
            "closePrice": raw_close,
        },
        "rebuild_series": {
            "candidate_samples": int(len(ts_idx)),
            "matched_target_ts": match_idx is not None,
        },
    }

    if match_idx is not None:
        pred = int(series["pred"][match_idx])
        signal_prob = float(series["signal_prob"][match_idx])
        trend_prob = float(series["trend_prob"][match_idx])
        up_prob = series["up_prob"][match_idx]
        side = "up" if pred == 2 else ("down" if pred == 0 else ("flat" if pred == 1 else "risk_off"))

        result["rebuild_series"]["sample_at_target"] = {
            "signal_ts": ts_idx[match_idx].isoformat(),
            "entry_ts": pd.Timestamp(series["entry_ts"][match_idx]).isoformat(),
            "pred": pred,
            "side": side,
            "signal_prob": signal_prob,
            "trend_prob": trend_prob,
            "up_prob": None if pd.isna(up_prob) else float(up_prob),
            "curr_close_like": float(series["curr"][match_idx]),
            "entry_price_like": float(series["entry"][match_idx]),
        }

        live_close = (status.get("best_base_runtime") or {}).get("latest_close")
        result["comparison"] = {
            "close_delta_live_status_vs_raw": None if live_close is None or raw_close is None else float(live_close) - float(raw_close),
            "close_delta_rebuild_curr_vs_raw": None if raw_close is None else float(series["curr"][match_idx]) - float(raw_close),
            "close_delta_rebuild_curr_vs_live_status": None if live_close is None else float(series["curr"][match_idx]) - float(live_close),
        }

    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(str(OUT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

