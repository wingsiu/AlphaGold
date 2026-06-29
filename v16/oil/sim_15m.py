"""15m leg simulation — one trade per signal, no intra-leg cascade (v16 realistic)."""
from __future__ import annotations

import pandas as pd

NY_FC_H, NY_FC_M = 14, 28
LONG_MAX_B = 60


def _ny_force_close(b) -> bool:
    return (b["ny_h"] > NY_FC_H) or (b["ny_h"] == NY_FC_H and b["ny_m"] >= NY_FC_M)


def sim_15m_long(
    d: pd.DataFrame,
    sigs: list[dict],
    tp: float,
    sl: float,
    stype: str = "ret",
) -> tuple[list[float], list[dict], list[int]]:
    """Long: enter close_ask; one independent trade per signal."""
    pnls: list[float] = []
    trades: list[dict] = []
    mapped: list[int] = []

    for si, sig in enumerate(sigs):
        idx = sig["idx"]
        if idx >= len(d) - 1:
            continue
        ep = float(d.iloc[idx]["close_ask"])
        eb = d.index[idx]
        ex_p, ex_ts, reason = None, None, None

        for j in range(idx + 1, min(idx + 1 + LONG_MAX_B, len(d))):
            b = d.iloc[j]
            if _ny_force_close(b):
                ex_p = float(b["close_bid"])
                ex_ts, reason = d.index[j], "ny_close"
                break
            if float(b["high"]) >= ep + tp:
                ex_p, ex_ts, reason = ep + tp, d.index[j], "tp"
                break
            if float(b["low"]) <= ep - sl:
                ex_p, ex_ts, reason = ep - sl, d.index[j], "sl"
                break

        if ex_p is None:
            j = min(idx + LONG_MAX_B, len(d) - 1)
            raw = float(d.iloc[j]["close_bid"]) - ep
            ex_p = ep + max(raw, -sl)
            ex_ts, reason = d.index[j], "timeout"

        pnl = ex_p - ep
        pnls.append(pnl)
        mapped.append(si)
        trades.append(
            {
                "entry": eb,
                "exit": ex_ts,
                "pnl": pnl,
                "reason": reason,
                "type": stype,
                "side": 1,
                "entry_price": ep,
                "exit_price": ex_p,
            }
        )
    return pnls, trades, mapped


def sim_15m_short(
    d: pd.DataFrame,
    sigs: list[dict],
    tp: float,
    sl: float,
    stype: str = "ret_short",
) -> tuple[list[float], list[dict], list[int]]:
    """Short: enter close_bid; one independent trade per signal."""
    pnls: list[float] = []
    trades: list[dict] = []
    mapped: list[int] = []

    for si, sig in enumerate(sigs):
        idx = sig["idx"]
        if idx >= len(d) - 1:
            continue
        ep = float(d.iloc[idx]["close_bid"])
        eb = d.index[idx]
        ex_p, ex_ts, reason = None, None, None

        for j in range(idx + 1, min(idx + 1 + LONG_MAX_B, len(d))):
            b = d.iloc[j]
            if _ny_force_close(b):
                ex_p = float(b["close_ask"])
                ex_ts, reason = d.index[j], "ny_close"
                break
            if float(b["low"]) <= ep - tp:
                ex_p, ex_ts, reason = ep - tp, d.index[j], "tp"
                break
            if float(b["high"]) >= ep + sl:
                ex_p, ex_ts, reason = ep + sl, d.index[j], "sl"
                break

        if ex_p is None:
            j = min(idx + LONG_MAX_B, len(d) - 1)
            raw = ep - float(d.iloc[j]["close_ask"])
            pnl_raw = raw
            pnl = max(pnl_raw, -sl)
            ex_p = ep - pnl
            ex_ts, reason = d.index[j], "timeout"

        pnl = ep - ex_p
        pnls.append(pnl)
        mapped.append(si)
        trades.append(
            {
                "entry": eb,
                "exit": ex_ts,
                "pnl": pnl,
                "reason": reason,
                "type": stype,
                "side": -1,
                "entry_price": ep,
                "exit_price": ex_p,
            }
        )
    return pnls, trades, mapped
