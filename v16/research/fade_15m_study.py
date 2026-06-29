#!/usr/bin/env python3
"""
15m → :00 open study

Tests: after a large UP prior 15m bar, does price dip from the slot open
at minute :00/:15/:30/:45 over the next 1–5 minutes?

Usage:
  PYTHONPATH=. python3 v16/research/fade_15m_study.py 2025-06-01 2026-06-25
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16.backtest.bars_15m import build_15m_context
from v16.backtest.features import session_mask
from v16.config.v16_config import SIGNAL_CONFIG
from v16.data.load_gold import load_gold_1m


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    start = args[0] if args else "2025-06-01"
    end = args[1] if len(args) > 1 else pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    min_body = float(SIGNAL_CONFIG["fade_min_prev_body_pts"])
    open_mins = tuple(SIGNAL_CONFIG["fade_open_minutes"])
    strict_mins = tuple(SIGNAL_CONFIG.get("fade_minutes_strict", (0,)))

    df = load_gold_1m(start, end)
    ctx = build_15m_context(df)
    in_sess = session_mask(df.index, SIGNAL_CONFIG["sessions"])
    mid = df["mid"]

    slot_open = mid.groupby(mid.index.floor("15min")).transform("first")
    ctx["chg_from_slot_open"] = mid - slot_open
    for h in (1, 2, 3, 4, 5):
        ctx[f"fwd_{h}m_from_here"] = mid.shift(-h) - mid

    m = in_sess & ctx["minute_in_15m"].astype(int).isin(open_mins)
    sub = ctx.loc[m].copy()

    print("=" * 70)
    print(f"  15m → early-minute fade  |  {start} → {end}")
    print(f"  Large prev 15m: |body| >= {min_body} pts")
    print("=" * 70)

    def block(label: str, mask: pd.Series) -> None:
        if mask.sum() < 40:
            print(f"\n{label}: n={mask.sum()} (too few)")
            return
        g = sub.loc[mask]
        print(f"\n{label}  (n={len(g):,}, prev_body={g['prev_15m_body'].mean():+.1f})")
        print(f"  At this 1m bar (chg from slot open): mean={g['chg_from_slot_open'].mean():+.2f}")
        for h in (1, 2, 3, 5):
            ch = g[f"fwd_{h}m_from_here"]
            print(
                f"  Next +{h}m from bar: mean={ch.mean():+.2f}  "
                f"median={ch.median():+.2f}  %down={(ch<0).mean()*100:.1f}%"
            )

    large_up = sub["prev_15m_body"] >= min_body
    large_dn = sub["prev_15m_body"] <= -min_body

    block("LARGE prior 15m UP — all open mins 0-2", large_up)
    block("LARGE prior 15m DOWN — all open mins 0-2", large_dn)

    print("\n--- LARGE UP: by minute_in_15m (your :00 idea) ---")
    for minute in list(open_mins) + [3, 4, 5]:
        g = sub.loc[large_up & (sub["minute_in_15m"] == minute)]
        if len(g) < 40:
            continue
        ch1 = g["fwd_1m_from_here"]
        ch3 = g["fwd_3m_from_here"]
        bar0 = g.loc[g["minute_in_15m"] == 0, "chg_from_slot_open"] if minute == 0 else None
        extra = ""
        if minute == 0:
            extra = f"  slot-open chg@0={g['chg_from_slot_open'].mean():+.2f}"
        print(
            f"  :{minute:02d}  n={len(g):4d}  "
            f"+1m={(ch1<0).mean()*100:.0f}% down  mean={ch1.mean():+.2f}  "
            f"+3m={(ch3<0).mean()*100:.0f}% down  mean={ch3.mean():+.2f}{extra}"
        )

    print("\n--- STRICT :00 only (minute 0 of each 15m slot) ---")
    z = sub["minute_in_15m"].astype(int).isin(strict_mins)
    block(":00 after LARGE UP  → expect dip / short", large_up & z)
    block(":00 after LARGE DOWN → expect bounce / long", large_dn & z)

    # 1m bar at :00 itself (the opening minute candle)
    print("\n--- The :00 1m bar itself (open minute of slot) ---")
    z0 = sub["minute_in_15m"].astype(int) == 0
    for label, mask in [
        ("Large UP prev", large_up & z0),
        ("Large DOWN prev", large_dn & z0),
        ("Small prev", ~(large_up | large_dn) & z0),
    ]:
        g = sub.loc[mask]
        if len(g) < 40:
            continue
        body = df.loc[g.index, "close_ask"] - df.loc[g.index, "open_ask"]
        print(
            f"  {label:16s} n={len(g):4d}  "
            f":00 1m body mean={body.mean():+.2f}  %red={(body<0).mean()*100:.0f}%"
        )


if __name__ == "__main__":
    main()
