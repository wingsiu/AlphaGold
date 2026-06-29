#!/usr/bin/env python3
"""
Compare impulse entry modes (impulse-bar stop exit, 3R, H=120).

Modes:
  open      — next-bar market at slot open (baseline)
  pullback  — limit at 50% of impulse bar (5m timeout)
  breakout  — stop entry at impulse H/L (10m timeout)

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_entry_modes.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_entry_modes.py 2025-06-01 2026-06-25 --sweep-pullback
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.impulse_entry import build_resolved_entry_table
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_signal_table

MODES = ("open", "pullback", "breakout")
PULLBACK_FRACS = (0.38, 0.5, 0.62)
HORIZON = 120
TP_R = 3.0


def _cfg(mode: str, *, pullback_fraction: float | None = None) -> dict:
    c = copy.deepcopy(v16_config.MOMENTUM_15M_HOLD)
    c["entry_mode"] = mode
    if mode == "pullback" and pullback_fraction is not None:
        c["entry_pullback"] = {**c.get("entry_pullback", {}), "fraction": pullback_fraction}
    return c


def _report(
    tdf: pd.DataFrame,
    *,
    label: str,
    signals: int,
    filled: int,
) -> dict:
    if tdf.empty:
        print(f"\n{label}: no trades  (signals={signals} filled={filled})")
        return {"label": label, "signals": signals, "filled": filled, "trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0}
    net = float(tdf["pnl"].sum())
    row = {
        "label": label,
        "signals": signals,
        "filled": filled,
        "trades": len(tdf),
        "fill_pct": round(100.0 * filled / max(signals, 1), 1),
        "wr": round(float(tdf["win"].mean() * 100), 1),
        "net": round(net, 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
    }
    print(
        f"\n{label}: {row['trades']} tr  fill={row['fill_pct']:.0f}%  "
        f"WR={row['wr']:.1f}%  net={row['net']:+.1f}  avg={row['avg']:+.2f}"
    )
    if "entry_style" in tdf.columns:
        for style, g in tdf.groupby("entry_style"):
            print(f"  {style}: {len(g)} tr  net={g['pnl'].sum():+.1f}")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"    {reason:12s}: {len(g):4d}  PnL={g['pnl'].sum():+.1f}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--sweep-pullback", action="store_true")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--tp-r", type=float, default=TP_R)
    args = parser.parse_args()

    print("=" * 80)
    print(f"  impulse entry modes  |  {args.start} → {args.end}")
    print(f"  exit: impulse SL  TP={args.tp_r}R  H={args.horizon}")
    print("=" * 80)

    df = load_gold_1m(args.start, args.end)
    base_cfg = v16_config.MOMENTUM_15M_HOLD
    signals = build_signal_table(df, cfg=base_cfg)
    print(f"\nImpulse signals: {len(signals)}")

    rows: list[dict] = []

    if args.sweep_pullback:
        for frac in PULLBACK_FRACS:
            cfg = _cfg("pullback", pullback_fraction=frac)
            entries = build_resolved_entry_table(df, signals, cfg=cfg)
            tdf = simulate_position_impulse_stop(
                df,
                signals,
                tp_multiple=args.tp_r,
                horizon=args.horizon,
                same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
                cfg=cfg,
            )
            rows.append(
                _report(
                    tdf,
                    label=f"pullback {frac:.0%}",
                    signals=len(signals),
                    filled=len(entries),
                )
            )
        sweep = pd.DataFrame(rows)
        out = PROJECT_ROOT / "runtime/v16_momentum_entry_pullback_sweep.csv"
        sweep.to_csv(out, index=False)
        print(f"\nSaved -> {out}")
        return

    for mode in MODES:
        cfg = _cfg(mode)
        entries = build_resolved_entry_table(df, signals, cfg=cfg)
        is_cfg = cfg.get("impulse_stop", {})
        tdf = simulate_position_impulse_stop(
            df,
            signals,
            tp_multiple=args.tp_r,
            horizon=args.horizon,
            min_sl_pts=float(is_cfg.get("min_sl_pts", 1.0)),
            max_sl_pts=float(is_cfg.get("max_sl_pts", 80.0)),
            same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
            cfg=cfg,
        )
        rows.append(_report(tdf, label=mode, signals=len(signals), filled=len(entries)))

    summary = pd.DataFrame(rows)
    out = PROJECT_ROOT / "runtime/v16_momentum_entry_modes.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
