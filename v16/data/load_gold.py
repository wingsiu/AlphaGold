"""v16 gold 1m loader — no v15 / v14 dependencies."""
from __future__ import annotations

import pandas as pd

from data.data_loader import DataLoader

SPREAD = 0.25


def load_gold_1m(start_date: str, end_date: str) -> pd.DataFrame:
    loader = DataLoader()
    raw = loader.load_data(table_name="gold_prices", start_date=start_date, end_date=end_date)
    raw.index = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)

    df = pd.DataFrame(index=raw.index)
    if "openPrice_ask" in raw.columns:
        df["open_ask"] = raw["openPrice_ask"].astype(float)
        df["open_bid"] = raw["openPrice_bid"].astype(float)
        df["high_ask"] = raw["highPrice_ask"].astype(float)
        df["high_bid"] = raw["highPrice_bid"].astype(float)
        df["low_ask"] = raw["lowPrice_ask"].astype(float)
        df["low_bid"] = raw["lowPrice_bid"].astype(float)
        df["close_ask"] = raw["closePrice_ask"].astype(float)
        df["close_bid"] = raw["closePrice_bid"].astype(float)
        df["volume"] = raw["lastTradedVolume"].astype(float)
    else:
        mid = raw["close"].astype(float)
        df["open_ask"] = raw["open"].astype(float) + SPREAD
        df["open_bid"] = raw["open"].astype(float) - SPREAD
        df["high_ask"] = raw["high"].astype(float) + SPREAD
        df["high_bid"] = raw["high"].astype(float) - SPREAD
        df["low_ask"] = raw["low"].astype(float) + SPREAD
        df["low_bid"] = raw["low"].astype(float) - SPREAD
        df["close_ask"] = mid + SPREAD
        df["close_bid"] = mid - SPREAD
        df["volume"] = raw.get("volume", pd.Series(0, index=raw.index)).astype(float)

    df["mid"] = (df["close_ask"] + df["close_bid"]) / 2.0
    df["spread"] = df["close_ask"] - df["close_bid"]
    return df.sort_index()
