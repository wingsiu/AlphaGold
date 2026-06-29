#!/usr/bin/env python3
"""Gold v16 combined backtest — max-PnL production stack.

  PYTHONPATH=. python3 v16/research/gold_v16_combined_backtest.py [start] [end]

Stack: hybrid patterns + energetic + v16 momentum + v16 dip short (single slot).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v16.config.gold_config import BACKTEST, GOLD_TRAIN_START
from v16.gold.combined_run import run_gold_v16_combined, save_combined_trades
from v16.gold.merge import merge_gold_trades
from v16.gold.hybrid_legs import run_hybrid_legs
from v16.gold.v16_legs import run_v16_legs
from v16.data.load_gold import load_gold_1m


def _stats(trades: list[dict], label: str) -> dict:
    if not trades:
        print(f"  {label:28s}   0 trades  PnL=+0.0")
        return {"label": label, "trades": 0, "pnl": 0.0}
    pnls = [t["pnl"] for t in trades]
    wr = 100 * sum(1 for p in pnls if p > 0) / len(pnls)
    pnl = sum(pnls)
    print(f"  {label:28s} {len(trades):4d} trades  PnL={pnl:+8.1f}  WR={wr:.1f}%  avg={pnl/len(pnls):+.2f}")
    return {"label": label, "trades": len(trades), "pnl": pnl, "wr": wr}


def _by_leg(trades: list[dict]) -> None:
    legs: dict[str, list] = {}
    for t in trades:
        legs.setdefault(str(t.get("_leg", t.get("type", "?"))), []).append(t["pnl"])
    print("\n  In portfolio:")
    for leg, pnls in sorted(legs.items()):
        print(f"    {leg:22s} {len(pnls):4d}t  PnL={sum(pnls):+.1f}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    start = args[0] if args else BACKTEST["default_start"]
    end = args[1] if len(args) > 1 else BACKTEST["default_end"]

    print("=" * 72)
    print("  GOLD v16 COMBINED BACKTEST (production)")
    print(f"  OOS: {start} → {end}")
    print("=" * 72)

    merged, stats = run_gold_v16_combined(GOLD_TRAIN_START, end, oos_start=start, verbose=True)

    print("\n--- Pre-merge legs ---")
    hybrid = run_hybrid_legs(start, end, verbose=False)
    df = load_gold_1m(GOLD_TRAIN_START, end)
    mom, dip = run_v16_legs(df, start)
    _stats(hybrid, "hybrid patterns+energetic")
    _stats(mom, "v16 momentum")
    _stats(dip, "v16 dip short")
    raw_n = len(hybrid) + len(mom) + len(dip)
    print(f"  {'raw total':28s} {raw_n:4d} trades  PnL={sum(t['pnl'] for t in hybrid+mom+dip):+.1f}")

    print("\n--- MERGED (single slot) ---")
    comb = _stats(merged, "v16 combined")
    _by_leg(merged)
    print(f"\n  Dropped by merge: {raw_n - len(merged)}")

    out = save_combined_trades(merged)
    print(f"\n  CSV: {out}")
    print("DONE.")


if __name__ == "__main__":
    main()
