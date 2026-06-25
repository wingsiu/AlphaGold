"""Print backtest trade rows with entry/exit times and prices (v13-style, HKT)."""

from __future__ import annotations

import pandas as pd


def print_trades_table_hkt(
    tdf: pd.DataFrame,
    *,
    tail: int = 30,
    show_all: bool = False,
) -> None:
    if tdf.empty:
        print("\n  (no trades)")
        return

    view = tdf.copy()
    view["entry_hkt"] = (
        pd.to_datetime(view["entry_time"], utc=True)
        .dt.tz_convert("Asia/Hong_Kong")
        .dt.strftime("%m-%d %H:%M")
    )
    view["exit_hkt"] = (
        pd.to_datetime(view["exit_time"], utc=True)
        .dt.tz_convert("Asia/Hong_Kong")
        .dt.strftime("%H:%M")
    )
    view["dir"] = view["side"].map({1: "up", -1: "down"})
    show = view if show_all else view.tail(tail)
    header = "ALL TRADES (HKT)" if show_all else f"LAST {min(tail, len(view))} TRADES (HKT)"
    print(f"\n{'─'*60}\n  {header}\n{'─'*60}")
    cols = ["entry_hkt", "exit_hkt", "dir", "entry_price", "exit_price", "pnl", "exit_reason"]
    if "pattern" in show.columns:
        cols.append("pattern")
    missing = [c for c in cols if c not in show.columns]
    if missing:
        print(f"  (missing columns: {', '.join(missing)})")
        cols = [c for c in cols if c in show.columns]
    print(show[cols].to_string(index=False))
