#!/usr/bin/env python3
"""Print ranked summary tables for sweep CSVs.

Supports:
  sweep_holdcap_h<H>_*.csv   – holdcap sweeps
  sweep_stop_h<H>_c<C>_*.csv – stop-limit sweeps
  sweep_s2dir_*.csv           – stage-2 directional sweeps

Usage:
  python3 summarise_sweeps.py           # holdcap h25, h45
  python3 summarise_sweeps.py 25 45     # explicit horizons (holdcap only)
  python3 summarise_sweeps.py --all     # every sweep type
  python3 summarise_sweeps.py --stops   # stop sweeps only
  python3 summarise_sweeps.py --s2dir   # stage-2 directional sweeps only
"""
import csv
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "training" / "testing" / "results"


def _best_file(files: list[Path]) -> Path:
    """Return the file with the most valid (non-error) rows; break ties by mtime."""
    def score(p: Path):
        rows = list(csv.DictReader(p.open()))
        good = sum(1 for r in rows if not r.get("error"))
        return (good, p.stat().st_mtime)
    return max(files, key=score)


# ── Holdcap ────────────────────────────────────────────────────────────────────

def show_holdcap(h: int) -> None:
    files = sorted(RES.glob(f"sweep_holdcap_h{h}_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"h{h}: NO holdcap CSV found")
        return
    p    = _best_file(files)
    rows = list(csv.DictReader(p.open()))
    new_fmt = "p2_total_pnl" in (rows[0].keys() if rows else [])
    good = [r for r in rows if not r.get("error")]
    errs = [r for r in rows if r.get("error")]
    fmt_label = "2-pass (new)" if new_fmt else "legacy"
    print(f"=== holdcap h={h}  [{fmt_label}]  {p.name}  ({len(good)} ok, {len(errs)} err) ===")

    if new_fmt:
        hdr = f"{'cap':>4}  {'wk':>2}  {'p1_pnl':>7}  {'p1_wr':>6}  {'p1_pf':>5}  {'p2_pnl':>7}  {'p2_wr':>6}  {'p2_pf':>5}  {'p2_atr':>6}  {'ntpd':>5}  {'dd_tr':>7}  {'dd_dy':>7}  {'delta':>7}"
        print(hdr); print("-" * len(hdr))
        for r in sorted(good, key=lambda r: float(r["p2_total_pnl"]), reverse=True):
            print(
                f"{int(r['max_hold_minutes']):>4}  {int(r['weak_cells']):>2}"
                f"  {float(r['p1_total_pnl']):>7.0f}  {float(r['p1_win_rate_pct']):>5.1f}%  {float(r['p1_profit_factor']):>5.3f}"
                f"  {float(r['p2_total_pnl']):>7.0f}  {float(r['p2_win_rate_pct']):>5.1f}%  {float(r['p2_profit_factor']):>5.3f}"
                f"  {float(r['p2_avg_trade']):>6.2f}  {float(r['p2_avg_trades_per_day']):>5.1f}"
                f"  {float(r['p2_trade_max_drawdown']):>7.0f}  {float(r['p2_daily_max_drawdown']):>7.0f}"
                f"  {float(r['delta_total_pnl']):>7.0f}"
            )
    else:
        hdr = f"{'cap':>4}  {'pnl':>7}  {'wr':>6}  {'pf':>5}  {'atr':>6}  {'ntpd':>5}  {'dd_tr':>7}  {'dd_dy':>7}"
        print(hdr); print("-" * len(hdr))
        for r in sorted(good, key=lambda r: float(r["total_pnl"]), reverse=True):
            print(
                f"{int(r['max_hold_minutes']):>4}"
                f"  {float(r['total_pnl']):>7.0f}  {float(r['win_rate_pct']):>5.1f}%  {float(r['profit_factor']):>5.3f}"
                f"  {float(r['avg_trade']):>6.2f}  {float(r['avg_trades_per_day']):>5.1f}"
                f"  {float(r['trade_max_drawdown']):>7.0f}  {float(r['daily_max_drawdown']):>7.0f}"
            )
    if errs:
        print(f"  ERRORS: {[(e['max_hold_minutes'], e['error']) for e in errs]}")
    print()


# ── Stop sweeps ────────────────────────────────────────────────────────────────

def show_stops() -> None:
    files = sorted(RES.glob("sweep_stop_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        print("No stop sweep CSVs found")
        return
    # Group by (horizon, cap) if multiple
    groups: dict[str, list[Path]] = {}
    for f in files:
        key = "_".join(f.stem.split("_")[:5])   # sweep_stop_h<H>_c<C>
        groups.setdefault(key, []).append(f)

    for key, grp in sorted(groups.items()):
        p    = _best_file(grp)
        rows = list(csv.DictReader(p.open()))
        good = [r for r in rows if not r.get("error")]
        errs = [r for r in rows if r.get("error")]
        print(f"=== {key}  {p.name}  ({len(good)} ok, {len(errs)} err) ===")
        hdr = f"{'la':>3}  {'sa':>3}  {'wk':>2}  {'p1_pnl':>7}  {'p1_wr':>6}  {'p1_pf':>5}  {'p2_pnl':>7}  {'p2_wr':>6}  {'p2_pf':>5}  {'atr':>6}  {'ntpd':>5}  {'dd_tr':>7}  {'dd_dy':>7}  {'delta':>7}"
        print(hdr); print("-" * len(hdr))
        for r in sorted(good, key=lambda r: float(r["p2_total_pnl"]), reverse=True):
            print(
                f"{int(r['long_adverse']):>3}  {int(r['short_adverse']):>3}  {int(r['weak_cells']):>2}"
                f"  {float(r['p1_total_pnl']):>7.0f}  {float(r['p1_win_rate_pct']):>5.1f}%  {float(r['p1_profit_factor']):>5.3f}"
                f"  {float(r['p2_total_pnl']):>7.0f}  {float(r['p2_win_rate_pct']):>5.1f}%  {float(r['p2_profit_factor']):>5.3f}"
                f"  {float(r['p2_avg_trade']):>6.2f}  {float(r['p2_avg_trades_per_day']):>5.1f}"
                f"  {float(r['p2_trade_max_drawdown']):>7.0f}  {float(r['p2_daily_max_drawdown']):>7.0f}"
                f"  {float(r['delta_total_pnl']):>7.0f}"
            )
        if errs:
            print(f"  ERRORS: {[(e['long_adverse'], e['short_adverse'], e['error']) for e in errs]}")
        print()


# ── Stage-2 directional sweeps ─────────────────────────────────────────────────

def show_s2dir() -> None:
    files = sorted(RES.glob("sweep_s2dir_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        print("No s2dir sweep CSVs found")
        return
    p    = _best_file(files)
    rows = list(csv.DictReader(p.open()))
    good = [r for r in rows if not r.get("error")]
    errs = [r for r in rows if r.get("error")]
    print(f"=== sweep_s2dir  {p.name}  ({len(good)} ok, {len(errs)} err) ===")
    hdr = f"{'up':>5}  {'dn':>5}  {'pnl':>8}  {'avg':>7}  {'wr%':>6}  {'pf':>5}  {'trades':>7}  {'dd_tr':>7}  {'dd_dy':>7}  {'l_wr':>6}  {'s_wr':>6}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(good, key=lambda r: float(r["total_pnl"]), reverse=True):
        print(
            f"{r['stage2_up']:>5}  {r['stage2_down']:>5}"
            f"  {float(r['total_pnl']):>8.0f}  {float(r['avg_trade']):>7.4f}"
            f"  {float(r['win_rate_pct']):>6.1f}  {float(r['profit_factor']):>5.3f}"
            f"  {int(r['trades']):>7}"
            f"  {float(r['trade_max_drawdown']):>7.0f}  {float(r['daily_max_drawdown']):>7.0f}"
            f"  {float(r['long_wr']):>6.1f}  {float(r['short_wr']):>6.1f}"
        )
    if errs:
        print(f"  ERRORS: {len(errs)} rows")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    show_all   = "--all"   in args
    show_stop  = "--stops" in args or show_all
    show_s2    = "--s2dir" in args or show_all
    show_hcap  = not (show_stop or show_s2) or show_all

    if show_hcap:
        horizons = [int(a) for a in args if a.isdigit()] or [25, 45]
        for h in horizons:
            show_holdcap(h)
    if show_stop:
        show_stops()
    if show_s2:
        show_s2dir()


if __name__ == "__main__":
    main()
