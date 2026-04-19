#!/usr/bin/env python3
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import DataLoader
from trading_bot import prepare_raw_price_frame


def _to_utc(ts_text: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts_text)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def main() -> int:
    status = json.loads((ROOT / "runtime/trading_bot_status.json").read_text(encoding="utf-8"))
    summary = dict(status.get("prediction_cache_last_summary") or {})

    start = summary.get("cache_start_utc")
    end = summary.get("cache_end_utc")
    if not start or not end:
        raise SystemExit("No cache_start_utc/cache_end_utc in status")

    start_ts = _to_utc(str(start))
    end_ts = _to_utc(str(end))

    table = (status.get("input_data") or {}).get("table", "gold_prices")
    max_rows = int((status.get("input_data") or {}).get("prediction_cache_max_rows") or 1200)

    raw = DataLoader().load_data(
        table,
        start_date=start_ts.date().isoformat(),
        end_date=end_ts.date().isoformat(),
    )
    df = prepare_raw_price_frame(raw)

    # Mimic cache window + cap behavior for inspection.
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    df = df.iloc[-max_rows:].copy()

    out_full = ROOT / "runtime/_cache_like_df.csv"
    out_tail = ROOT / "runtime/_cache_like_df_tail.csv"
    df.to_csv(out_full)
    df.tail(20).to_csv(out_tail)

    print("rows:", len(df))
    print("first:", df.index[0].isoformat() if len(df) else None)
    print("last:", df.index[-1].isoformat() if len(df) else None)
    print("saved:", out_full)
    print("saved:", out_tail)
    if len(df):
        print("last closePrice:", float(df.iloc[-1].get("closePrice")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
