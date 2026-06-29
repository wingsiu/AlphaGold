"""v16 oil 1m loader — prices table, v16 column naming."""
from __future__ import annotations

import pandas as pd

from data.data_loader import DataLoader


def load_oil_1m(start_date: str, end_date: str) -> pd.DataFrame:
    """Load crude 1m bars with bid/ask/mid columns (v16 gold-compatible schema)."""
    loader = DataLoader()
    raw = loader.load_data(table_name="prices", start_date=start_date, end_date=end_date)
    raw.index = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)

    df = pd.DataFrame(index=raw.index)
    df["open_ask"] = raw["openPrice_ask"].astype(float)
    df["open_bid"] = raw["openPrice_bid"].astype(float)
    df["high_ask"] = raw["highPrice_ask"].astype(float)
    df["high_bid"] = raw["highPrice_bid"].astype(float)
    df["low_ask"] = raw["lowPrice_ask"].astype(float)
    df["low_bid"] = raw["lowPrice_bid"].astype(float)
    df["close_ask"] = raw["closePrice_ask"].astype(float)
    df["close_bid"] = raw["closePrice_bid"].astype(float)
    df["volume"] = raw["lastTradedVolume"].astype(float)
    df["mid"] = (df["close_ask"] + df["close_bid"]) / 2.0
    df["spread"] = df["close_ask"] - df["close_bid"]

    # Legacy names used by oil.signal_engine / backtest_oil
    df["open"] = df["open_ask"]
    df["high"] = df["high_ask"]
    df["low"] = df["low_ask"]

    return df.sort_index()
