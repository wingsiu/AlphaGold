"""v16 gold replay — combined portfolio (hybrid + momentum + dip)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from v16.gold.combined_signal_engine import (
    load_replay_data,
    replay_portfolio,
    replay_portfolio_from_df,
)


def replay_portfolio_legacy(
    start: str,
    end: str,
    *,
    data_start: Optional[str] = None,
    verbose: bool = False,
) -> list[dict]:
    """Replay max-PnL portfolio — same path as combined_run backtest."""
    return replay_portfolio(start, end, data_start=data_start, verbose=verbose)


__all__ = [
    "replay_portfolio",
    "replay_portfolio_from_df",
    "replay_portfolio_legacy",
    "load_replay_data",
]
