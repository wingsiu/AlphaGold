"""15-minute bar context for 1-minute gold bars."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_15m_context(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Attach previous *completed* 15m bar stats to each 1m row.

    At 10:02 the active slot started 10:00; prev_15m is 09:45–10:00 (closed at :00).
    """
    mid = df_1m["mid"]
    ohlc = pd.DataFrame(
        {
            "open": mid.resample("15min", label="right", closed="right").first(),
            "high": df_1m["high_ask"].resample("15min", label="right", closed="right").max(),
            "low": df_1m["low_bid"].resample("15min", label="right", closed="right").min(),
            "close": mid.resample("15min", label="right", closed="right").last(),
            "volume": df_1m["volume"].resample("15min", label="right", closed="right").sum(),
        }
    ).dropna(subset=["close"])

    ohlc["body"] = ohlc["close"] - ohlc["open"]
    ohlc["range"] = ohlc["high"] - ohlc["low"]
    ohlc["body_abs"] = ohlc["body"].abs()
    ohlc["dir"] = np.sign(ohlc["body"]).astype(int)

    prev = ohlc.shift(1)
    prev2 = ohlc.shift(2)
    prev.index.name = "ts_15"
    prev = prev.reset_index()
    prev = prev.rename(
        columns={
            "open": "prev_15m_open",
            "high": "prev_15m_high",
            "low": "prev_15m_low",
            "body": "prev_15m_body",
            "range": "prev_15m_range",
            "body_abs": "prev_15m_body_abs",
            "dir": "prev_15m_dir",
            "close": "prev_15m_close",
            "volume": "prev_15m_volume",
        }
    )
    prev2 = ohlc.shift(2)
    prev2.index.name = "ts_15_p2"
    prev2 = prev2.reset_index()
    prev2 = prev2.rename(
        columns={
            "body": "prev2_15m_body",
            "range": "prev2_15m_range",
            "body_abs": "prev2_15m_body_abs",
            "dir": "prev2_15m_dir",
            "close": "prev2_15m_close",
            "volume": "prev2_15m_volume",
        }
    )

    left = pd.DataFrame({"ts": df_1m.index})
    merged = pd.merge_asof(
        left.sort_values("ts"),
        prev.sort_values("ts_15"),
        left_on="ts",
        right_on="ts_15",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged.sort_values("ts"),
        prev2.sort_values("ts_15_p2"),
        left_on="ts",
        right_on="ts_15_p2",
        direction="backward",
    ).set_index("ts")

    out = merged[
        [
            "prev_15m_open",
            "prev_15m_high",
            "prev_15m_low",
            "prev_15m_body",
            "prev_15m_range",
            "prev_15m_body_abs",
            "prev_15m_dir",
            "prev_15m_close",
            "prev_15m_volume",
            "prev2_15m_body",
            "prev2_15m_range",
            "prev2_15m_body_abs",
            "prev2_15m_dir",
            "prev2_15m_close",
            "prev2_15m_volume",
        ]
    ].copy()

    out["minute_in_15m"] = df_1m.index.minute % 15

    # Running stats for the *active* 15m slot (as of each 1m bar)
    slot_id = df_1m.index.floor("15min")
    out["slot_open"] = mid.groupby(slot_id).transform("first")
    out["slot_low"] = df_1m["low_bid"].groupby(slot_id).cummin()
    out["slot_high"] = df_1m["high_ask"].groupby(slot_id).cummax()
    out["dip_from_slot_open"] = out["slot_open"] - mid
    out["slot_rip_pts"] = mid - out["slot_open"]
    out["slot_low_dip"] = out["slot_open"] - out["slot_low"]
    out["slot_down"] = (mid < out["slot_open"]).astype(float)
    out["slot_up"] = (mid > out["slot_open"]).astype(float)
    out["two_prev_15m_down"] = ((out["prev_15m_dir"] < 0) & (out["prev2_15m_dir"] < 0)).astype(float)
    out["two_prev_15m_up"] = ((out["prev_15m_dir"] > 0) & (out["prev2_15m_dir"] > 0)).astype(float)

    return out.replace([np.inf, -np.inf], np.nan)
