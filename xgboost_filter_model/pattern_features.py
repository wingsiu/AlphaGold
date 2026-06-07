"""Extended Williams %R windows, 15m breakthrough flags, and cross flags for pattern routing."""

from __future__ import annotations

import pandas as pd
import pandas_ta as pta

EPS = 1e-9

# Break up: WR(90) not deeply oversold + sudden 3m rally.
BREAKOUT_WR90_MIN = -30.0
BREAKOUT_RET_3M_MIN = 4.0

# Break down: WR(90) oversold + sudden 3m drop.
BREAKDOWN_WR90_MAX = -70.0
BREAKDOWN_RET_3M_MAX = -10.0


def _add_breakthrough_up(df: pd.DataFrame) -> pd.Series:
    """WR(90) > -30 and ret_3m > 4 on the trigger bar."""
    if "wr_90" not in df.columns:
        raise ValueError("_add_breakthrough_up requires wr_90")

    ret_3m = df["ret_3m"] if "ret_3m" in df.columns else df["close"] - df["close"].shift(3)
    signal = (df["wr_90"] > BREAKOUT_WR90_MIN) & (ret_3m > BREAKOUT_RET_3M_MIN)
    return signal.fillna(False).astype(int)


def _add_breakthrough_down(df: pd.DataFrame) -> pd.Series:
    """WR(90) < -70 and ret_3m < -10 on the trigger bar."""
    if "wr_90" not in df.columns:
        raise ValueError("_add_breakthrough_down requires wr_90")

    ret_3m = df["ret_3m"] if "ret_3m" in df.columns else df["close"] - df["close"].shift(3)
    signal = (df["wr_90"] < BREAKDOWN_WR90_MAX) & (ret_3m < BREAKDOWN_RET_3M_MAX)
    return signal.fillna(False).astype(int)


def _add_breakthrough_15m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["c15_breakthrough_up"] = _add_breakthrough_up(df)
    df["c15_breakthrough_down"] = _add_breakthrough_down(df)
    return df


def add_pattern_features(df: pd.DataFrame, *, feature_set: str = "current") -> pd.DataFrame:
    """Add wr_120/wr_240, 240m swing structure, breakthrough flags, and cross/recovery flags.

    feature_set:
      current — full pattern feature set (96 model inputs)
      v2398   — pre-breakthrough-expansion set (93 inputs): omits rise_from_day_low,
                drop_from_day_high, wr_120_cross_up_10
    """
    df = df.copy()
    legacy = feature_set.strip().lower() == "v2398"

    low_240 = df["low"].rolling(240, min_periods=240).min()
    high_240 = df["high"].rolling(240, min_periods=240).max()
    df["rise_from_low_240"] = df["close"] - low_240
    df["drop_from_high_240"] = high_240 - df["close"]
    # Bearish 1-bar body in price units (open − close); 0 on green bars.
    df["bar_bear_drop"] = (df["open"] - df["close"]).clip(lower=0)

    if not legacy:
        if "day_low_rolling" in df.columns:
            df["rise_from_day_low"] = df["close"] - df["day_low_rolling"]
        if "day_high_rolling" in df.columns:
            df["drop_from_day_high"] = df["day_high_rolling"] - df["close"]

    for w in (120, 240):
        col = f"wr_{w}"
        if col not in df.columns:
            df[col] = pta.willr(df["high"], df["low"], df["close"], length=w)

    wr90 = df["wr_90"]
    prev = wr90.shift(1)
    wr120 = df["wr_120"]
    wr120_prev = wr120.shift(1)

    df["wr_90_cross_up_80"] = ((prev <= -80) & (wr90 > -80)).astype(int)
    df["wr_90_cross_down_20"] = ((prev >= -20) & (wr90 < -20)).astype(int)
    df["wr_30_cross_up_70"] = ((df["wr_30"].shift(1) <= -70) & (df["wr_30"] > -70)).astype(int)
    df["wr_30_cross_down_30"] = ((df["wr_30"].shift(1) >= -30) & (df["wr_30"] < -30)).astype(int)
    if not legacy:
        df["wr_120_cross_up_10"] = ((wr120_prev <= -10) & (wr120 > -10)).astype(int)

    roll = 15
    df["wr_90_recover_80"] = (
        (wr90 > -80) & (wr90.rolling(roll, min_periods=1).min() <= -80)
    ).astype(int)
    df["wr_90_recover_20"] = (
        (wr90 < -20) & (wr90.rolling(roll, min_periods=1).max() >= -20)
    ).astype(int)

    df = _add_breakthrough_15m(df)

    return df
