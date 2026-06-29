"""
ATR-scaled zigzag swings on resampled OHLC.

Produces a swing table (confirmed pivots) and per-bar structure context
for merge onto 1m signal timestamps (no lookahead: swings confirmed at bar close).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def resample_ohlc(
    df_1m: pd.DataFrame,
    rule: str = "15min",
) -> pd.DataFrame:
    """Resample 1m gold bars to OHLCV (mid + bid/ask extremes)."""
    mid = df_1m["mid"]
    out = pd.DataFrame(
        {
            "open": mid.resample(rule, label="right", closed="right").first(),
            "high": df_1m["high_ask"].resample(rule, label="right", closed="right").max(),
            "low": df_1m["low_bid"].resample(rule, label="right", closed="right").min(),
            "close": mid.resample(rule, label="right", closed="right").last(),
            "volume": df_1m["volume"].resample(rule, label="right", closed="right").sum(),
        }
    ).dropna(subset=["close"])
    out["range"] = out["high"] - out["low"]
    return out


def build_15m_ohlc(df_1m: pd.DataFrame) -> pd.DataFrame:
    return resample_ohlc(df_1m, "15min")


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy(dtype=float)
    return atr


def zigzag_swings(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    *,
    atr_mult: float = 3.0,
    min_bars: int = 2,
) -> list[dict]:
    """
    Classic ATR reversal zigzag.

    Returns swings in chronological order:
      kind 'H' | 'L', bar index, timestamp (filled by caller), price, leg_pts, leg_bars.
    """
    n = len(close)
    if n < 2:
        return []

    swings: list[dict] = []
    # 0 = seeking first leg, 1 = up leg (track high), -1 = down leg (track low)
    mode = 0
    pivot_idx = 0
    pivot_price = float(close[0])
    extreme_idx = 0
    extreme_price = float(close[0])

    def _threshold(i: int) -> float:
        a = float(atr[i]) if np.isfinite(atr[i]) and atr[i] > 0 else float(np.nanmean(atr))
        return max(atr_mult * a, 0.5)

    for i in range(1, n):
        if mode == 0:
            up = float(high[i]) - pivot_price
            dn = pivot_price - float(low[i])
            th = _threshold(i)
            if up >= th and up >= dn:
                mode = 1
                extreme_idx = i
                extreme_price = float(high[i])
            elif dn >= th:
                mode = -1
                extreme_idx = i
                extreme_price = float(low[i])
            continue

        th = _threshold(i)

        if mode == 1:
            if float(high[i]) >= extreme_price:
                extreme_price = float(high[i])
                extreme_idx = i
            elif extreme_price - float(low[i]) >= th and extreme_idx - pivot_idx >= min_bars:
                leg_pts = extreme_price - pivot_price
                swings.append(
                    {
                        "bar_idx": extreme_idx,
                        "kind": "H",
                        "price": extreme_price,
                        "leg_pts": leg_pts,
                        "leg_bars": extreme_idx - pivot_idx,
                        "from_kind": "L" if swings and swings[-1]["kind"] == "L" else ("H" if swings else None),
                    }
                )
                pivot_idx = extreme_idx
                pivot_price = extreme_price
                mode = -1
                extreme_idx = i
                extreme_price = float(low[i])
        else:
            if float(low[i]) <= extreme_price:
                extreme_price = float(low[i])
                extreme_idx = i
            elif float(high[i]) - extreme_price >= th and extreme_idx - pivot_idx >= min_bars:
                leg_pts = pivot_price - extreme_price
                swings.append(
                    {
                        "bar_idx": extreme_idx,
                        "kind": "L",
                        "price": extreme_price,
                        "leg_pts": leg_pts,
                        "leg_bars": extreme_idx - pivot_idx,
                        "from_kind": "H" if swings and swings[-1]["kind"] == "H" else ("L" if swings else None),
                    }
                )
                pivot_idx = extreme_idx
                pivot_price = extreme_price
                mode = 1
                extreme_idx = i
                extreme_price = float(high[i])

    return swings


def _trend_from_last_swings(highs: list[float], lows: list[float]) -> int:
    """1 = HH+HL uptrend, -1 = LH+LL downtrend, 0 = mixed/range."""
    if len(highs) < 2 or len(lows) < 2:
        return 0
    hh = highs[-1] > highs[-2]
    hl = lows[-1] > lows[-2]
    lh = highs[-1] < highs[-2]
    ll = lows[-1] < lows[-2]
    if hh and hl:
        return 1
    if lh and ll:
        return -1
    return 0


def build_swing_table(
    ohlc: pd.DataFrame,
    *,
    atr_mult: float = 3.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Confirmed swings indexed by bar close timestamp."""
    if ohlc.empty:
        return pd.DataFrame()

    high = ohlc["high"].to_numpy(dtype=float)
    low = ohlc["low"].to_numpy(dtype=float)
    close = ohlc["close"].to_numpy(dtype=float)
    atr = _atr(high, low, close, atr_period)

    raw = zigzag_swings(high, low, close, atr, atr_mult=atr_mult)
    if not raw:
        return pd.DataFrame()

    ts = ohlc.index
    rows: list[dict] = []
    highs: list[float] = []
    lows: list[float] = []

    for s in raw:
        idx = int(s["bar_idx"])
        kind = s["kind"]
        if kind == "H":
            highs.append(float(s["price"]))
        else:
            lows.append(float(s["price"]))
        trend = _trend_from_last_swings(highs, lows)
        rows.append(
            {
                "ts": ts[idx],
                "bar_idx": idx,
                "kind": kind,
                "price": float(s["price"]),
                "leg_pts": float(s["leg_pts"]),
                "leg_bars": int(s["leg_bars"]),
                "trend": trend,
                "n_highs": len(highs),
                "n_lows": len(lows),
            }
        )

    return pd.DataFrame(rows).set_index("ts").sort_index()


def _structure_state_at(
    swings: pd.DataFrame,
    price: float,
    asof_ts: pd.Timestamp,
    *,
    bar_minutes: int = 15,
) -> dict:
    """Structure snapshot using only swings confirmed at or before asof_ts."""
    past = swings.loc[:asof_ts]
    if past.empty:
        return {
            "struct_trend": 0,
            "struct_last_kind": 0,
            "struct_last_swing_price": price,
            "struct_dist_pts": 0.0,
            "struct_leg_pts": 0.0,
            "struct_prior_leg_pts": 0.0,
            "struct_pullback_pct": 0.0,
            "struct_leg_age_15m": 0,
            "struct_hh": 0.0,
            "struct_hl": 0.0,
            "struct_lh": 0.0,
            "struct_ll": 0.0,
        }

    last = past.iloc[-1]
    last_kind = 1 if last["kind"] == "H" else -1
    last_price = float(last["price"])
    prior_leg = float(last["leg_pts"])
    trend = int(last["trend"])

    highs = past[past["kind"] == "H"]["price"].tolist()
    lows = past[past["kind"] == "L"]["price"].tolist()
    hh = float(highs[-1] > highs[-2]) if len(highs) >= 2 else 0.0
    hl = float(lows[-1] > lows[-2]) if len(lows) >= 2 else 0.0
    lh = float(highs[-1] < highs[-2]) if len(highs) >= 2 else 0.0
    ll = float(lows[-1] < lows[-2]) if len(lows) >= 2 else 0.0

    dist = price - last_price
    leg_pts = abs(dist)

    prior_leg_pts = 0.0
    if len(past) >= 2:
        prior_leg_pts = float(past.iloc[-2]["leg_pts"])

    pullback_pct = 0.0
    if prior_leg_pts > 0:
        if last_kind == 1 and trend <= 0:
            # Last pivot was high — measure drop from high
            pullback_pct = max(0.0, (last_price - price) / prior_leg_pts)
        elif last_kind == -1 and trend >= 0:
            pullback_pct = max(0.0, (price - last_price) / prior_leg_pts)

    leg_age = int((asof_ts - past.index[-1]) / pd.Timedelta(minutes=bar_minutes))

    return {
        "struct_trend": trend,
        "struct_last_kind": last_kind,
        "struct_last_swing_price": last_price,
        "struct_dist_pts": dist,
        "struct_leg_pts": leg_pts,
        "struct_prior_leg_pts": prior_leg_pts,
        "struct_pullback_pct": pullback_pct,
        "struct_leg_age_15m": leg_age,
        "struct_hh": hh,
        "struct_hl": hl,
        "struct_lh": lh,
        "struct_ll": ll,
    }


def build_structure_context(
    df_1m: pd.DataFrame,
    *,
    rule: str = "15min",
    atr_mult: float = 3.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Per-1m structure columns from HTF zigzag (default 15m).

    Uses merge_asof on swing confirmation times — no future swings leak in.
    """
    ohlc = resample_ohlc(df_1m, rule)
    swings = build_swing_table(ohlc, atr_mult=atr_mult, atr_period=atr_period)
    if swings.empty:
        return pd.DataFrame(index=df_1m.index)

    bar_minutes = int(pd.Timedelta(rule).total_seconds() // 60) if rule != "15min" else 15
    if rule == "15min":
        bar_minutes = 15

    # Snapshot at each swing confirmation
    snap_rows: list[dict] = []
    for ts, row in swings.iterrows():
        st = _structure_state_at(swings, float(row["price"]), ts, bar_minutes=bar_minutes)
        st["ts"] = ts
        snap_rows.append(st)
    snap = pd.DataFrame(snap_rows).set_index("ts").sort_index()

    left = pd.DataFrame({"ts": df_1m.index, "mid": df_1m["mid"].values})
    merged = pd.merge_asof(
        left.sort_values("ts"),
        snap.reset_index().rename(columns={"ts": "swing_ts"}).sort_values("swing_ts"),
        left_on="ts",
        right_on="swing_ts",
        direction="backward",
    ).set_index("ts")

    # Refresh distance / leg / pullback vs live mid (still using last confirmed swing only)
    last_price = merged["struct_last_swing_price"].astype(float)
    last_kind = merged["struct_last_kind"].astype(float)
    mid = merged["mid"].astype(float)
    dist = mid - last_price
    merged["struct_dist_pts"] = dist
    merged["struct_leg_pts"] = dist.abs()

    prior = merged["struct_prior_leg_pts"].replace(0, np.nan).astype(float)
    pb = np.where(
        last_kind > 0,
        np.maximum(0.0, (last_price - mid).values) / prior.values,
        np.maximum(0.0, (mid - last_price).values) / prior.values,
    )
    merged["struct_pullback_pct"] = np.nan_to_num(pb, nan=0.0)

    if "swing_ts" in merged.columns:
        delta = merged.index - merged["swing_ts"]
        age = (delta / pd.Timedelta(minutes=bar_minutes)).fillna(0)
        merged["struct_leg_age_15m"] = age.clip(lower=0).astype(int)

    out_cols = [
        "struct_trend",
        "struct_last_kind",
        "struct_last_swing_price",
        "struct_dist_pts",
        "struct_leg_pts",
        "struct_prior_leg_pts",
        "struct_pullback_pct",
        "struct_leg_age_15m",
        "struct_hh",
        "struct_hl",
        "struct_lh",
        "struct_ll",
    ]
    return merged[[c for c in out_cols if c in merged.columns]].astype(float)
