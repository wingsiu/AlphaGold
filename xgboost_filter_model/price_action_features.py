import os
import pandas as pd
import numpy as np

FILL_NA = 9999
ALL_GROUPS = ("fvg", "wick", "fake")


def _parse_groups(groups=None) -> set:
    if groups is not None:
        if isinstance(groups, str):
            raw = [g.strip() for g in groups.replace(",", " ").split() if g.strip()]
        else:
            raw = [str(g).strip() for g in groups if str(g).strip()]
    else:
        env = os.environ.get("V14_PA_GROUP", "").strip().lower()
        if not env:
            legacy = os.environ.get("V14_PA_GROUPS", "").strip().lower()
            env = "all" if legacy in ("all", "fvg,wick,fake") else legacy
        if not env or env == "none":
            return set()
        if env == "all":
            return set(ALL_GROUPS)
        raw = [g.strip() for g in env.replace(",", " ").split() if g.strip()]
    out = set()
    for g in raw:
        if g in ALL_GROUPS:
            out.add(g)
    return out


def add_price_action_features(df, groups=None):
    """
    Optional 15m price-action feature groups (enable via V14_PA_GROUP=fvg|wick|fake|all):
      fvg  — edge dist (bottom/top) + time from bull/bear FVG
      wick — time from long upper/lower wick (lower/upper wick > 35% of 15m range)
      fake — time from fake up/down (>=10 bars vs 15m open)
    """
    active = _parse_groups(groups)
    if not active:
        return df

    df = df.copy()

    df_15m = df.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )

    if "fvg" in active:
        min_gap = float(os.environ.get("V14_FVG_MIN_GAP", "0"))
        bull_gap = df_15m["low"] - df_15m["high"].shift(2)
        bear_gap = df_15m["low"].shift(2) - df_15m["high"]
        if min_gap <= 0:
            df_15m["fvg_bull"] = bull_gap > 0
            df_15m["fvg_bear"] = bear_gap > 0
        else:
            df_15m["fvg_bull"] = bull_gap >= min_gap
            df_15m["fvg_bear"] = bear_gap >= min_gap
        # Bull gap zone: bottom = high[-2], top = low[0]. Bear: top = low[-2], bottom = high[0].
        df_15m["fvg_bull_bottom"] = np.where(df_15m["fvg_bull"], df_15m["high"].shift(2), np.nan)
        df_15m["fvg_bull_top"] = np.where(df_15m["fvg_bull"], df_15m["low"], np.nan)
        df_15m["fvg_bear_top"] = np.where(df_15m["fvg_bear"], df_15m["low"].shift(2), np.nan)
        df_15m["fvg_bear_bottom"] = np.where(df_15m["fvg_bear"], df_15m["high"], np.nan)

    if "wick" in active:
        range_15m = df_15m["high"] - df_15m["low"]
        body_top = df_15m[["open", "close"]].max(axis=1)
        body_bottom = df_15m[["open", "close"]].min(axis=1)
        upper_wick = df_15m["high"] - body_top
        lower_wick = body_bottom - df_15m["low"]
        wick_min_pct = float(os.environ.get("V14_WICK_MIN_PCT", "0.35"))
        df_15m["long_upper_wick"] = (upper_wick > wick_min_pct * range_15m) & (range_15m > 0)
        df_15m["long_lower_wick"] = (lower_wick > wick_min_pct * range_15m) & (range_15m > 0)

    if "fake" in active:
        df["interval_15m"] = df.index.floor("15min")
        interval_opens = df.groupby("interval_15m")["open"].first()
        df["current_15m_open"] = df["interval_15m"].map(interval_opens)
        df["is_below_open"] = df["close"] < df["current_15m_open"]
        df["is_above_open"] = df["close"] > df["current_15m_open"]
        counts = df.groupby("interval_15m").agg(
            {"is_below_open": "sum", "is_above_open": "sum", "close": "last"}
        )
        fake_down = (counts["is_below_open"] >= 10) & (counts["close"] > interval_opens)
        fake_up = (counts["is_above_open"] >= 10) & (counts["close"] < interval_opens)
        df_15m["fake_down"] = fake_down
        df_15m["fake_up"] = fake_up

    df_15m_shifted = df_15m.shift(1)
    df_15m_ffill = df_15m_shifted.reindex(df.index, method="ffill")

    fill_cols = []

    if "fvg" in active:
        df["fvg_bull_bottom"] = df_15m_ffill["fvg_bull_bottom"].ffill()
        df["fvg_bull_top"] = df_15m_ffill["fvg_bull_top"].ffill()
        df["fvg_bear_top"] = df_15m_ffill["fvg_bear_top"].ffill()
        df["fvg_bear_bottom"] = df_15m_ffill["fvg_bear_bottom"].ffill()
        df["dist_fvg_bull_bottom"] = df["close"] - df["fvg_bull_bottom"]
        df["dist_fvg_bull_top"] = df["close"] - df["fvg_bull_top"]
        df["dist_fvg_bear_top"] = df["close"] - df["fvg_bear_top"]
        df["dist_fvg_bear_bottom"] = df["close"] - df["fvg_bear_bottom"]
        fvg_bull_times = pd.Series(
            np.where(df_15m_shifted["fvg_bull"], df_15m_shifted.index, np.nan),
            index=df_15m_shifted.index,
        ).ffill()
        fvg_bear_times = pd.Series(
            np.where(df_15m_shifted["fvg_bear"], df_15m_shifted.index, np.nan),
            index=df_15m_shifted.index,
        ).ffill()
        df["time_from_fvg_bull"] = (
            df.index - fvg_bull_times.reindex(df.index, method="ffill")
        ).dt.total_seconds() / 60.0
        df["time_from_fvg_bear"] = (
            df.index - fvg_bear_times.reindex(df.index, method="ffill")
        ).dt.total_seconds() / 60.0
        fill_cols.extend(
            [
                "dist_fvg_bull_bottom",
                "dist_fvg_bull_top",
                "dist_fvg_bear_top",
                "dist_fvg_bear_bottom",
                "time_from_fvg_bull",
                "time_from_fvg_bear",
            ]
        )

    if "wick" in active:
        upper_wick_times = pd.Series(
            np.where(df_15m_shifted["long_upper_wick"], df_15m_shifted.index, np.nan),
            index=df_15m_shifted.index,
        ).ffill()
        lower_wick_times = pd.Series(
            np.where(df_15m_shifted["long_lower_wick"], df_15m_shifted.index, np.nan),
            index=df_15m_shifted.index,
        ).ffill()
        df["time_from_long_upper_wick"] = (
            df.index - upper_wick_times.reindex(df.index, method="ffill")
        ).dt.total_seconds() / 60.0
        df["time_from_long_lower_wick"] = (
            df.index - lower_wick_times.reindex(df.index, method="ffill")
        ).dt.total_seconds() / 60.0
        fill_cols.extend(["time_from_long_upper_wick", "time_from_long_lower_wick"])

    if "fake" in active:
        fake_up_times = pd.Series(
            np.where(df_15m_shifted["fake_up"], df_15m_shifted.index, np.nan),
            index=df_15m_shifted.index,
        ).ffill()
        fake_down_times = pd.Series(
            np.where(df_15m_shifted["fake_down"], df_15m_shifted.index, np.nan),
            index=df_15m_shifted.index,
        ).ffill()
        df["time_from_fake_up"] = (
            df.index - fake_up_times.reindex(df.index, method="ffill")
        ).dt.total_seconds() / 60.0
        df["time_from_fake_down"] = (
            df.index - fake_down_times.reindex(df.index, method="ffill")
        ).dt.total_seconds() / 60.0
        fill_cols.extend(["time_from_fake_up", "time_from_fake_down"])

    df.drop(
        columns=["interval_15m", "current_15m_open", "is_below_open", "is_above_open"],
        inplace=True,
        errors="ignore",
    )
    if fill_cols:
        df[fill_cols] = df[fill_cols].fillna(FILL_NA)

    return df
