#!/usr/bin/env python3
"""
v16 dip_short_rip backtest — v15-style single position, same-dir refresh.

Router: prev 15m up, slot up, rip ≥5 pts above open, minute < 10
No overlapping trades: new same-direction signals extend horizon and trail target.

Usage:
  PYTHONPATH=. python3 v16/research/dip_short_rip_backtest.py 2025-06-01 2026-06-25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v16._paths import PROJECT_ROOT
from v16.backtest.dip_short_rip_run import run_dip_short_rip
from v16.backtest.features import build_features
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.dip_short_rip import resolve_execution, router_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--ml-prob", type=float, default=None)
    args = parser.parse_args()

    cfg = v16_config.DIP_SHORT_RIP.copy()
    ml_p = float(args.ml_prob if args.ml_prob is not None else cfg["ml_prob"])
    cfg["ml_prob"] = ml_p
    ex = resolve_execution(cfg, mechanical=False)
    ex_m = resolve_execution(cfg, mechanical=True)

    print("=" * 70)
    print(f"  v16 dip_short_rip  |  {args.start} → {args.end}")
    print(f"  single position | same_dir_refresh={cfg.get('same_dir_refresh', 'entry')}")
    print(f"  ML p>={ml_p} | labels={cfg.get('ml_label_source')}")
    print(
        f"  exit ML: TP{ex['tp']:.0f}/SL{ex['sl']:.0f}/H{ex['horizon']}  "
        f"| mech: TP{ex_m['tp']:.0f}/SL{ex_m['sl']:.0f}/H{ex_m['horizon']}"
    )
    print("=" * 70)

    df = load_gold_1m(args.start, args.end)
    feats = build_features(df)
    print(f"\nRouter pool (raw signals): {int(router_mask(feats, df.index, cfg=cfg).sum())}")

    mech = run_dip_short_rip(df, feats, cfg, mechanical=True)
    if not mech.empty:
        print(f"\nMechanical: {len(mech)} trades, PnL {mech['pnl'].sum():+.1f}, avg {mech['pnl'].mean():+.2f}")
        if "target_updates" in mech.columns:
            print(f"  target refreshes: {int(mech['target_updates'].sum())} total")

    tdf = run_dip_short_rip(df, feats, cfg, mechanical=False, ml_prob=ml_p)
    if tdf.empty:
        print("\nNo ML trades.")
        return

    print(f"\nML filtered:")
    print(f"  Trades     : {len(tdf)}")
    print(f"  Win rate   : {tdf['win'].mean()*100:.1f}%")
    print(f"  Net PnL    : {tdf['pnl'].sum():+.1f}")
    print(f"  Avg/trade  : {tdf['pnl'].mean():+.2f}")
    if "target_updates" in tdf.columns:
        print(f"  Target refresh avg: {tdf['target_updates'].mean():.1f}/trade")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"    {reason:12s}: {len(g):4d}  WR={g['win'].mean()*100:.0f}%  PnL={g['pnl'].sum():+.1f}")

    out = PROJECT_ROOT / "runtime" / "v16_dip_short_rip_trades.csv"
    tdf.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
