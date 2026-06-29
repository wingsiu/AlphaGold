"""WR90 struct-hold exit on 15m bars (v16-style, no fixed TP)."""
from __future__ import annotations

import pandas as pd

from v16.config.oil_config import WR90_STRUCT_HOLD
from v16.structure.swing_levels import protective_stop, swing_table_for_cfg
from v16.structure.swing_zigzag import build_structure_context


def sim_wr90_struct_hold(
    d15: pd.DataFrame,
    d1m_v16: pd.DataFrame,
    entry_indices: list[int],
    *,
    cfg: dict | None = None,
) -> list[dict]:
    """Simulate WR90 long entries with structure/swing exit."""
    c = cfg or WR90_STRUCT_HOLD
    ex = c
    horizon_bars = int(ex["horizon_minutes"] // 15)
    sl_pts = float(ex["sl"])
    ny_h, ny_m = ex.get("ny_close_h", 14), ex.get("ny_close_m", 28)

    sc = ex.get("structure", {})
    struct = build_structure_context(
        d1m_v16,
        rule=str(sc.get("rule", "15min")),
        atr_mult=float(sc.get("atr_mult", 3.0)),
        atr_period=int(sc.get("atr_period", 14)),
    )
    struct_trend = (
        struct["struct_trend"].reindex(d1m_v16.index).ffill().fillna(0).astype(int)
        if not struct.empty
        else pd.Series(0, index=d1m_v16.index)
    )
    swings = swing_table_for_cfg(d1m_v16, {"structure": sc})

    trades: list[dict] = []
    for idx in entry_indices:
        if idx >= len(d15) - 1:
            continue
        ep = float(d15.iloc[idx]["close_ask"])
        eb_ts = d15.index[idx]
        entry_struct = 0
        slot_end = eb_ts + pd.Timedelta(minutes=14)
        sub = struct_trend.loc[:slot_end]
        if len(sub):
            entry_struct = int(sub.iloc[-1])

        swing_stop = protective_stop(1, swings, eb_ts)
        hard_stop = ep - sl_pts
        stop = max(hard_stop, swing_stop) if swing_stop is not None else hard_stop

        ex_p, ex_ts, reason = None, None, None
        for j in range(idx + 1, min(idx + 1 + horizon_bars, len(d15))):
            b = d15.iloc[j]
            ts = d15.index[j]
            ny = ts.tz_convert("America/New_York")
            if ny.hour > ny_h or (ny.hour == ny_h and ny.minute >= ny_m):
                ex_p = float(b["close_bid"])
                ex_ts, reason = ts, "ny_close"
                break
            if float(b["low"]) <= stop:
                ex_p = stop
                ex_ts, reason = ts, "swing_break" if swing_stop and stop >= hard_stop else "sl"
                break
            cur_struct = 0
            se = ts + pd.Timedelta(minutes=14)
            ss = struct_trend.loc[:se]
            if len(ss):
                cur_struct = int(ss.iloc[-1])
            if ex.get("exit_on_structure_change") and cur_struct != entry_struct and cur_struct <= 0:
                ex_p = float(b["close_bid"])
                ex_ts, reason = ts, "structure_change"
                break

        if ex_p is None:
            j = min(idx + horizon_bars, len(d15) - 1)
            raw = float(d15.iloc[j]["close_bid"]) - ep
            ex_p = ep + max(raw, -sl_pts)
            ex_ts, reason = d15.index[j], "timeout"

        trades.append(
            {
                "entry": eb_ts,
                "exit": ex_ts,
                "pnl": ex_p - ep,
                "reason": reason,
                "type": "wr90",
                "side": 1,
                "entry_price": ep,
                "exit_price": ex_p,
                "_leg": "wr90",
            }
        )
    return trades
