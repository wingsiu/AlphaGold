#!/usr/bin/env python3
"""
Sweep horizon + initial_sl for impulse_1m_15m (v16 scale-out +5/+10).

Fixed: first_scale_pnl=5, final_scale_pnl=10, first_scale_frac=0.5, runner_lock_pnl=5
One signal build + one WF-free run per combo.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_sweep.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_sweep.py 2025-06-01 2026-06-25 --quick
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.position_sim import simulate_position_sided_scaleout
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_side_signals

GRIDS = {
    "quick": {
        "sl": [15.0, 20.0, 25.0, 30.0],
        "horizon": [10, 15, 20, 30, 45],
    },
    "full": {
        "sl": [12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0, 35.0],
        "horizon": [8, 10, 12, 15, 20, 25, 30, 45, 60],
    },
}


def _stats(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {"trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0, "max_dd": 0.0, "scaled_pct": 0.0}
    eq = tdf["pnl"].cumsum()
    max_dd = float((eq - eq.cummax()).min())
    scaled = float(tdf["scaled_half"].mean() * 100) if "scaled_half" in tdf.columns else 0.0
    return {
        "trades": len(tdf),
        "wr": round(float((tdf["pnl"] > 0).mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
        "max_dd": round(max_dd, 1),
        "scaled_pct": round(scaled, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    grid = GRIDS["quick" if args.quick else "full"]
    cfg = v16_config.MOMENTUM_15M_HOLD
    base_so = copy.deepcopy(cfg.get("scaleout", v16_config.EXIT_CONFIG))
    combos = list(itertools.product(grid["sl"], grid["horizon"]))

    print("=" * 88)
    print(f"  impulse_1m_15m SL/H sweep  |  {args.start} → {args.end}")
    print(f"  scale-out +{base_so['first_scale_pnl']:.0f}/+{base_so['final_scale_pnl']:.0f}  "
          f"combos={len(combos)}")
    print("=" * 88)

    df = load_gold_1m(args.start, args.end)
    sides = build_side_signals(df, cfg=cfg)
    print(f"Signals: {int((sides != 0).sum())}")

    rows = []
    t0 = time.time()
    for sl, h in combos:
        so = copy.deepcopy(base_so)
        so["initial_sl"] = sl
        so["horizon_minutes"] = h
        tdf = simulate_position_sided_scaleout(
            df,
            sides,
            scaleout_kw=so,
            same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
        )
        st = _stats(tdf)
        rows.append({"sl": sl, "horizon": h, **st})

    elapsed = time.time() - t0
    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "runtime" / "v16_momentum_15m_hold_sl_h_sweep.csv"
    out.to_csv(out_path, index=False)
    print(f"\nDone in {elapsed:.1f}s  ->  {out_path}")

    print("\nTop by net PnL:")
    print(f"{'SL':>5} {'H':>4} {'trades':>6} {'WR%':>6} {'net':>9} {'avg':>6} {'maxDD':>8} {'+5%':>5}")
    print("-" * 58)
    for _, r in out.sort_values("net", ascending=False).head(15).iterrows():
        print(
            f"{r['sl']:5.0f} {int(r['horizon']):4d} {int(r['trades']):6d} "
            f"{r['wr']:6.1f} {r['net']:+9.1f} {r['avg']:+6.2f} {r['max_dd']:+8.1f} {r['scaled_pct']:5.1f}"
        )

    base_sl = float(base_so.get("initial_sl", 20))
    base_h = int(base_so.get("horizon_minutes", 10))
    base_row = out[(out["sl"] == base_sl) & (out["horizon"] == base_h)]
    if not base_row.empty:
        r = base_row.iloc[0]
        print(f"\nBaseline SL{base_sl:.0f}/H{base_h}: {int(r['trades'])} tr  net={r['net']:+.1f}  avg={r['avg']:+.2f}")


if __name__ == "__main__":
    main()
