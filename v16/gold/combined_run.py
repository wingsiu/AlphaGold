"""v16 gold production combined runner — max-PnL portfolio."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.config.gold_config import BACKTEST, GOLD_TRAIN_START
from v16.data.load_gold import load_gold_1m
from v16.gold.hybrid_legs import run_hybrid_legs
from v16.gold.merge import merge_gold_trades
from v16.gold.v16_legs import run_v16_legs


def _utc_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _leg_stats(trades: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in trades:
        leg = str(t.get("_leg", t.get("type", "?")))
        out.setdefault(leg, {"trades": 0, "pnl": 0.0})
        out[leg]["trades"] += 1
        out[leg]["pnl"] += float(t["pnl"])
    return out


def run_gold_v16_combined(
    data_start: str,
    end: str,
    *,
    oos_start: Optional[str] = None,
    verbose: bool = True,
    use_hybrid_cache: bool = False,
) -> tuple[list[dict], dict[str, dict]]:
    """Run hybrid + momentum + dip; merge single-slot. Returns (merged, leg_stats)."""
    oos = oos_start or data_start
    oos_ts = _utc_ts(oos)
    end_ts = _utc_ts(end) + pd.Timedelta(hours=23, minutes=59)

    hybrid_cache = PROJECT_ROOT / BACKTEST["hybrid_cache"]
    hybrid_trades: list[dict] = []

    if use_hybrid_cache and hybrid_cache.exists():
        tdf = pd.read_csv(hybrid_cache)
        tdf["entry"] = pd.to_datetime(tdf["entry"], utc=True)
        tdf["exit"] = pd.to_datetime(tdf["exit"], utc=True)
        hybrid_trades = [
            r for _, r in tdf.iterrows()
            if oos_ts <= pd.Timestamp(r["entry"]).tz_convert("UTC") <= end_ts
        ]
        hybrid_trades = [dict(r) for r in hybrid_trades]
        if verbose:
            print(f"  (hybrid from cache {hybrid_cache}, {len(hybrid_trades)} OOS trades)")

    if not hybrid_trades:
        if verbose:
            print("Running hybrid patterns + energetic…")
        hybrid_trades = run_hybrid_legs(oos, end, verbose=verbose)
        hybrid_trades = [
            t for t in hybrid_trades
            if oos_ts <= _utc_ts(t["entry"]) <= end_ts
        ]
        if verbose:
            pd.DataFrame(hybrid_trades).to_csv(hybrid_cache, index=False)

    if verbose:
        print("Loading gold 1m for v16 legs…")
    df = load_gold_1m(data_start, end)

    if verbose:
        print("Running v16 momentum + dip short…")
    mom_trades, dip_trades = run_v16_legs(df, oos)

    raw = hybrid_trades + mom_trades + dip_trades
    merged = merge_gold_trades(raw)

    stats = {
        "hybrid": _leg_stats(hybrid_trades),
        "v16_momentum": {"trades": len(mom_trades), "pnl": sum(t["pnl"] for t in mom_trades)},
        "v16_dip_short": {"trades": len(dip_trades), "pnl": sum(t["pnl"] for t in dip_trades)},
        "merged": _leg_stats(merged),
        "raw_n": len(raw),
        "merged_n": len(merged),
    }
    return merged, stats


def save_combined_trades(trades: list[dict], path: Optional[Path] = None) -> Path:
    out = path or (PROJECT_ROOT / BACKTEST["trades_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out, index=False)
    return out
