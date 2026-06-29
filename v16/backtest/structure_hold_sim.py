"""Hold with-trend until 15m structure breaks or protective swing is taken."""
from __future__ import annotations

import pandas as pd

from v16.structure.swing_levels import protective_stop, swing_table_for_cfg
from v16.structure.swing_zigzag import build_structure_context


def simulate_structure_trend_hold(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Single position. Exit on:
      - swing break (long: low <= last L; short: high >= last H)
      - struct_trend flip vs entry trend
      - horizon safety timeout
    No fixed TP — ride trend until structure fails.
    """
    if signals.empty:
        return pd.DataFrame()

    from v16.config import v16_config

    c = cfg or v16_config.STRUCTURE_TREND_HOLD
    ex = c.get("exit", {})
    horizon = int(ex.get("horizon_minutes", 480))
    on_struct = bool(ex.get("on_structure_change", True))
    on_swing = bool(ex.get("on_swing_break", True))
    struct_min_pnl = float(ex.get("min_pnl_on_structure_exit", -1e9))

    sc = c.get("structure", {})
    struct = build_structure_context(
        df,
        rule=str(sc.get("rule", "15min")),
        atr_mult=float(sc.get("atr_mult", 3.0)),
        atr_period=int(sc.get("atr_period", 14)),
    )
    struct_trend = struct["struct_trend"].reindex(df.index).ffill().fillna(0).astype(int).to_numpy()
    swings = swing_table_for_cfg(df, c)

    entry_by_idx: dict[int, pd.Series] = {}
    for ts, row in signals.iterrows():
        entry_by_idx[int(row["entry_idx"])] = row

    low_bid = df["low_bid"].to_numpy(dtype=float)
    high_ask = df["high_ask"].to_numpy(dtype=float)
    close_bid = df["close_bid"].to_numpy(dtype=float)
    close_ask = df["close_ask"].to_numpy(dtype=float)
    index = df.index
    n = len(df) - 1

    trades: list[dict] = []
    active: dict | None = None
    i = 0

    while i < n:
        now_ts = index[i]

        if active:
            s = active["side"]
            exit_info = None

            if on_swing:
                stop = active.get("swing_stop")
                if stop is not None:
                    if s == 1 and low_bid[i] <= stop:
                        exit_info = (float(stop), "swing_break")
                    elif s == -1 and high_ask[i] >= stop:
                        exit_info = (float(stop), "swing_break")

            if exit_info is None and on_struct:
                cur = int(struct_trend[i])
                if cur != int(active["entry_struct_trend"]):
                    px = float(close_bid[i]) if s == 1 else float(close_ask[i])
                    pnl = (px - active["entry_price"]) * s
                    if pnl > struct_min_pnl:
                        exit_info = (px, "structure_change")

            if exit_info is None and now_ts >= active["timeout"]:
                px = float(close_bid[i]) if s == 1 else float(close_ask[i])
                exit_info = (px, "timeout")

            if exit_info:
                px, reason = exit_info
                pnl = (px - active["entry_price"]) * s
                trades.append(
                    {
                        **active,
                        "exit_time": now_ts,
                        "exit_price": px,
                        "exit_reason": reason,
                        "pnl": pnl,
                        "win": pnl > 0,
                        "hold_min": (now_ts - active["entry_time"]).total_seconds() / 60.0,
                    }
                )
                active = None

            # trail protective swing on new confirmed pivots (same trend)
            if active and on_swing:
                sp = protective_stop(s, swings, now_ts)
                if sp is not None:
                    if s == 1:
                        active["swing_stop"] = max(active.get("swing_stop", sp), sp)
                    else:
                        active["swing_stop"] = min(active.get("swing_stop", sp), sp)

        if active is None and i in entry_by_idx:
            row = entry_by_idx[i]
            s = int(row["side"])
            ep = float(row["entry_price"])
            entry_ts = index[i]
            sp = protective_stop(s, swings, entry_ts)
            active = {
                "side": s,
                "signal_ts": row.name,
                "entry_time": entry_ts,
                "entry_price": ep,
                "entry_struct_trend": int(row.get("struct_trend", struct_trend[i])),
                "pullback_pct": float(row.get("pullback_pct", 0.0)),
                "leg_age_15m": int(row.get("leg_age_15m", 0)),
                "swing_stop": sp,
                "horizon": horizon,
                "timeout": entry_ts + pd.Timedelta(minutes=horizon),
            }

        i += 1

    return pd.DataFrame(trades)
