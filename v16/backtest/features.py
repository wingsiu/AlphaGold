"""v16 features — price/volume/time only."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LONDON = ZoneInfo("Europe/London")
NY = ZoneInfo("America/New_York")
HKT = ZoneInfo("Asia/Hong_Kong")


def _in_session(ts: pd.Timestamp, session: str) -> bool:
    if session == "london":
        t = ts.tz_convert(LONDON)
        m = t.hour * 60 + t.minute
        return 8 * 60 <= m < 16 * 60 + 30
    if session == "ny":
        t = ts.tz_convert(NY)
        m = t.hour * 60 + t.minute
        return 9 * 60 + 30 <= m < 16 * 60
    if session == "hkt":
        t = ts.tz_convert(HKT)
        m = t.hour * 60 + t.minute
        return 8 * 60 <= m < 16 * 60
    return False


def session_mask(index: pd.DatetimeIndex, sessions: tuple[str, ...]) -> pd.Series:
    return pd.Series(
        [any(_in_session(ts, s) for s in sessions) for ts in index],
        index=index,
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    from v16.backtest.bars_15m import build_15m_context
    from v16.config.v16_config import SIGNAL_CONFIG

    f = pd.DataFrame(index=df.index)
    mid = df["mid"]
    ctx = build_15m_context(df)
    for c in ctx.columns:
        f[c] = ctx[c]

    f["range_1"] = (df["high_ask"] - df["low_bid"]).astype(float)
    f["body"] = (df["close_ask"] - df["open_ask"]).astype(float)
    f["body_abs"] = f["body"].abs()
    f["close_loc"] = (mid - df["low_bid"]) / (df["high_ask"] - df["low_bid"] + 1e-9)

    for n in (3, 5, 10, 20):
        f[f"ret_{n}"] = mid.diff(n)
        f[f"range_{n}"] = f["range_1"].rolling(n).mean()
        f[f"vol_ma_{n}"] = df["volume"].rolling(n).mean()

    f["vol_ratio"] = df["volume"] / f["vol_ma_20"].replace(0, np.nan)
    f["range_expansion"] = f["range_1"] / f["range_20"].replace(0, np.nan)
    f["atr_14"] = f["range_1"].rolling(14).mean()

    # Trend / MA context (price EMAs — not used in v16 v1)
    for n in (20, 50, 100):
        ema = mid.ewm(span=n, adjust=False).mean()
        f[f"ema_{n}"] = ema
        f[f"dist_ema_{n}"] = mid - ema
        f[f"ema_{n}_slope"] = ema.diff(5)
    f["ema_cross_20_50"] = f["ema_20"] - f["ema_50"]
    f["ema_cross_50_100"] = f["ema_50"] - f["ema_100"]
    f["trend_strength"] = f["ret_20"] / f["atr_14"].replace(0, np.nan)
    f["above_ema_20"] = (mid > f["ema_20"]).astype(float)
    f["above_ema_50"] = (mid > f["ema_50"]).astype(float)

    lon = df.index.tz_convert(LONDON)
    ny = df.index.tz_convert(NY)
    f["lon_hour"] = lon.hour + lon.minute / 60.0
    f["ny_hour"] = ny.hour + ny.minute / 60.0
    f["dow"] = df.index.dayofweek.astype(float)

    # Interaction: large prior 15m up + early slot minute (fade setup)
    f["fade_short_setup"] = (
        (f["prev_15m_dir"] > 0)
        & (f["prev_15m_body_abs"] >= SIGNAL_CONFIG["fade_min_prev_body_pts"])
        & (f["minute_in_15m"].isin(SIGNAL_CONFIG.get("fade_open_minutes", (0, 1, 2))))
    ).astype(float)
    f["fade_long_setup"] = (
        (f["prev_15m_dir"] < 0)
        & (f["prev_15m_body_abs"] >= SIGNAL_CONFIG["fade_min_prev_body_pts"])
        & (f["minute_in_15m"].isin(SIGNAL_CONFIG.get("fade_open_minutes", (0, 1, 2))))
    ).astype(float)

    if "prev2_15m_body" in f.columns:
        f["prev_15m_body_sum2"] = f["prev_15m_body"] + f["prev2_15m_body"]
        f["prev_15m_range_sum2"] = f["prev_15m_range"] + f["prev2_15m_range"]

    return f.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def dip_ml_feature_columns(f: pd.DataFrame) -> list[str]:
    """15m dip bounce model — prev 1–2 bar change/range + slot dip context."""
    prefer = [
        "prev_15m_body",
        "prev_15m_range",
        "prev_15m_body_abs",
        "prev_15m_dir",
        "prev2_15m_body",
        "prev2_15m_range",
        "prev2_15m_body_abs",
        "prev2_15m_dir",
        "two_prev_15m_down",
        "two_prev_15m_up",
        "minute_in_15m",
        "dip_from_slot_open",
        "slot_rip_pts",
        "slot_low_dip",
        "slot_down",
        "slot_up",
        "range_1",
        "body",
        "vol_ratio",
        "ret_3",
        "ret_5",
        "ret_10",
        "dist_ema_20",
        "dist_ema_50",
        "ema_20_slope",
        "trend_strength",
        "lon_hour",
        "ny_hour",
    ]
    return [c for c in prefer if c in f.columns]


def feature_columns(f: pd.DataFrame) -> list[str]:
    return list(f.columns)
