#!/usr/bin/env python3
"""Print ranked summary table for sweep CSVs by horizon."""
import csv
import sys
from pathlib import Path

res = Path(__file__).resolve().parents[2] / "training" / "testing" / "results"
horizons = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [25, 45]

for h in horizons:
    files = sorted(res.glob(f"sweep_holdcap_h{h}_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"h{h}: NO CSV FOUND")
        continue
    p = files[-1]
    rows = list(csv.DictReader(p.open()))
    good = [r for r in rows if not r.get("error")]
    errs = [r for r in rows if r.get("error")]
    print(f"=== horizon={h}  file={p.name}  ({len(good)} ok, {len(errs)} err) ===")
    header = f"{'cap':>4}  {'wk':>2}  {'p1_pnl':>7}  {'p1_wr':>6}  {'p1_pf':>5}  {'p2_pnl':>7}  {'p2_wr':>6}  {'p2_pf':>5}  {'p2_atr':>6}  {'ntpd':>5}  {'dd_tr':>7}  {'dd_dy':>7}  {'delta':>7}"
    print(header)
    print("-" * len(header))
    for r in sorted(good, key=lambda r: float(r["p2_total_pnl"]), reverse=True):
        cap   = int(r["max_hold_minutes"])
        wk    = int(r["weak_cells"])
        p1p   = float(r["p1_total_pnl"])
        p1wr  = float(r["p1_win_rate_pct"])
        p1pf  = float(r["p1_profit_factor"])
        p2p   = float(r["p2_total_pnl"])
        p2wr  = float(r["p2_win_rate_pct"])
        p2pf  = float(r["p2_profit_factor"])
        atr   = float(r["p2_avg_trade"])
        ntpd  = float(r["p2_avg_trades_per_day"])
        ddtr  = float(r["p2_trade_max_drawdown"])
        dddy  = float(r["p2_daily_max_drawdown"])
        delta = float(r["delta_total_pnl"])
        print(f"{cap:>4}  {wk:>2}  {p1p:>7.0f}  {p1wr:>5.1f}%  {p1pf:>5.3f}  {p2p:>7.0f}  {p2wr:>5.1f}%  {p2pf:>5.3f}  {atr:>6.2f}  {ntpd:>5.1f}  {ddtr:>7.0f}  {dddy:>7.0f}  {delta:>7.0f}")
    if errs:
        print(f"  ERRORS: {[(e['max_hold_minutes'], e['error']) for e in errs]}")
    print()

