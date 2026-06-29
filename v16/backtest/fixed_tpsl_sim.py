"""Fixed TP/SL horizon exit (v15-style) for v16 research."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FixedTpSlResult:
    pnl: float
    exit_reason: str
    bars_held: int
    side: int


def simulate_fixed_tpsl(
    df: pd.DataFrame,
    entry_idx: int,
    side: int,
    entry_price: float,
    *,
    tp: float = 30.0,
    sl: float = 25.0,
    horizon: int = 30,
) -> FixedTpSlResult:
    start = entry_idx + 1
    end = min(entry_idx + 1 + horizon, len(df))
    if start >= end:
        return FixedTpSlResult(0.0, "no_bars", 0, side)

    for j in range(start, end):
        row = df.iloc[j]
        bars = j - entry_idx
        if side == 1:
            hi = float(row["high_bid"]) if "high_bid" in row else float(row["high_ask"]) - 0.25
            lo = float(row["low_bid"])
            if hi - entry_price >= tp:
                return FixedTpSlResult(tp, "target_hit", bars, side)
            if entry_price - lo >= sl:
                return FixedTpSlResult(-sl, "stop_loss", bars, side)
        else:
            hi = float(row["high_ask"])
            lo = float(row["low_bid"])
            if entry_price - lo >= tp:
                return FixedTpSlResult(tp, "target_hit", bars, side)
            if hi - entry_price >= sl:
                return FixedTpSlResult(-sl, "stop_loss", bars, side)

        if j == end - 1:
            if side == 1:
                px = float(row["close_bid"])
                pnl = px - entry_price
            else:
                px = float(row["close_ask"])
                pnl = entry_price - px
            return FixedTpSlResult(pnl, "timeout", bars, side)

    return FixedTpSlResult(0.0, "no_bars", 0, side)
