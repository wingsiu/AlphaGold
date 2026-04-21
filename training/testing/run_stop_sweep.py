#!/usr/bin/env python3
"""
Backtest-stop sweep: varies --backtest-long-adverse-limit × --backtest-short-adverse-limit.

Label generation stays FIXED (la=15, sa=18 matching the saved model), so the
entire dataset is built once and cached — all combos share the same prep-cache.

Usage:
  python3 training/testing/run_stop_sweep.py
  python3 training/testing/run_stop_sweep.py --long-stops 10,12,14,15,16,18,20 --short-stops 14,15,16,18,20,22
  python3 training/testing/run_stop_sweep.py --dry-run
"""
import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.testing.sweep_utils import (
    fmt_sec, generate_weak_filter, parse_report, print_progress, print_top,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT   = PROJECT_ROOT / "training" / "image_trend_ml.py"
MODEL_IN = PROJECT_ROOT / "runtime" / "bot_assets" / "backtest_model_best_base_weak_nostate.joblib"
OUTDIR   = PROJECT_ROOT / "training" / "testing" / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)


def _parse_list(raw: str) -> list[int]:
    return [int(t.strip()) for t in raw.split(",") if t.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2-pass sweep over long/short adverse stop limits.")
    p.add_argument("--horizon",      type=int, default=45)
    p.add_argument("--cap",          type=int, default=50)
    p.add_argument("--long-stops",   default="10,12,14,15,16,18,20",
                   help="Comma-separated long-adverse-limit values")
    p.add_argument("--short-stops",  default="14,15,16,18,20,22",
                   help="Comma-separated short-adverse-limit values")
    p.add_argument("--prep-cache-dir", default=None)
    p.add_argument("--refresh-prep-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned combos without executing.")
    return p.parse_args()


def _build_cmd(*, fixed: list[str], long_stop: int, short_stop: int,
               out_stem: Path, weak_filter: "Path | None",
               prep_cache_dir: Path, refresh_prep_cache: bool) -> list[str]:
    cmd = [
        sys.executable, str(SCRIPT),
        *fixed,
        "--long-adverse-limit",  str(long_stop),
        "--short-adverse-limit", str(short_stop),
        "--prep-cache-dir",      str(prep_cache_dir),
        "--model-out",   str(out_stem) + "_model.joblib",
        "--report-out",  str(out_stem) + "_report.json",
        "--trades-out",  str(out_stem) + "_trades.csv",
    ]
    if refresh_prep_cache:
        cmd.append("--refresh-prep-cache")
    if weak_filter is not None:
        cmd.extend(["--weak-periods-json", str(weak_filter)])
    return cmd


def main() -> None:
    args = _parse_args()
    long_stops  = _parse_list(args.long_stops)
    short_stops = _parse_list(args.short_stops)
    combos = [(la, sa) for la in long_stops for sa in short_stops]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = OUTDIR / f"sweep_stop_h{args.horizon}_c{args.cap}_{ts}.csv"
    prep_cache_dir = (Path(args.prep_cache_dir) if args.prep_cache_dir
                      else OUTDIR / f"prep_cache_h{args.horizon}")

    fixed = [
        "--start-date",             "2025-05-20",
        "--end-date",               "2026-04-10",
        "--test-start-date",        "2025-11-25T17:02:00+00:00",
        "--timeframe",              "1min",
        "--eval-mode",              "single_split",
        "--disable-time-filter",
        "--window",                 "150",
        "--window-15m",             "0",
        "--min-window-range",       "30",
        "--min-15m-drop",           "15",
        "--min-15m-rise",           "0",
        "--horizon",                str(args.horizon),
        "--trend-threshold",        "0.008",
        # adverse-limit fixed = saved model value → dataset cache key is
        # identical across all combos, so data loads ONCE then hits every time.
        "--adverse-limit",          "15",
        "--long-target-threshold",  "0.008",
        "--short-target-threshold", "0.008",
        "--classifier",             "gradient_boosting",
        "--max-flat-ratio",         "2.5",
        "--stage1-min-prob",        "0.55",
        "--stage2-min-prob",        "0.58",
        "--stage2-min-prob-up",     "0.65",
        "--stage2-min-prob-down",   "0.62",
        "--max-hold-minutes",       str(args.cap),
        "--model-in",               str(MODEL_IN),
    ]

    if args.dry_run:
        print(f"stop sweep: horizon={args.horizon} cap={args.cap}  "
              f"{len(long_stops)} long × {len(short_stops)} short = {len(combos)} combos (×2 passes)")
        for i, (la, sa) in enumerate(combos, 1):
            print(f"  [{i:3d}] la={la} sa={sa}")
        return

    prep_cache_dir.mkdir(parents=True, exist_ok=True)
    total = len(combos)
    print(f"stop sweep: horizon={args.horizon} cap={args.cap}  "
          f"{len(long_stops)} long × {len(short_stops)} short = {total} combos  "
          f"(×2 passes = {total*2} runs)")
    print(f"model   : {MODEL_IN}")
    print(f"cache   : {prep_cache_dir}")
    print(f"fixed   : adverse_limit=15  horizon={args.horizon}  cap={args.cap}  "
          f"s1=0.55  s2=0.58  s2up=0.65  s2dn=0.62")
    print(f"sweeping: long_stops={long_stops}  short_stops={short_stops}")
    print()

    rows:      list[dict] = []
    sweep_start            = time.time()
    durations: list[float] = []

    fieldnames = [
        "horizon", "cap", "long_adverse", "short_adverse", "weak_cells",
        "p1_trades", "p1_total_pnl", "p1_win_rate_pct", "p1_profit_factor",
        "p2_trades", "p2_total_pnl", "p2_avg_trade", "p2_win_rate_pct", "p2_profit_factor",
        "p2_avg_trades_per_day", "p2_trade_max_drawdown", "p2_daily_max_drawdown",
        "delta_total_pnl", "error",
    ]

    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (la, sa) in enumerate(combos, 1):
            tag = f"h{args.horizon}_c{args.cap}_{ts}_la{la}_sa{sa}"
            print_progress(i, total, sweep_start, durations)
            print(f"  la={la} sa={sa}  pass1 (no weak filter)")
            t0 = time.time()

            # ── pass 1 ────────────────────────────────────────────────────────
            p1_stem = OUTDIR / f"stop_{tag}_p1"
            rc1 = subprocess.run(
                _build_cmd(fixed=fixed, long_stop=la, short_stop=sa,
                           out_stem=p1_stem, weak_filter=None,
                           prep_cache_dir=prep_cache_dir,
                           refresh_prep_cache=bool(args.refresh_prep_cache and i == 1)),
                cwd=str(PROJECT_ROOT), check=False,
            ).returncode
            if rc1 != 0:
                row = {"long_adverse": la, "short_adverse": sa, "error": f"pass1 rc={rc1}"}
                rows.append(row)
                for fn in fieldnames: row.setdefault(fn, "")
                writer.writerow(row); f.flush()
                durations.append(time.time() - t0)
                print(f"  ⚠ pass1 failed (rc={rc1})")
                continue

            p1 = parse_report(Path(str(p1_stem) + "_report.json"))

            # ── build weak filter ──────────────────────────────────────────────
            wf_json    = OUTDIR / f"stop_{tag}_wf.json"
            weak_cells = generate_weak_filter(Path(str(p1_stem) + "_trades.csv"), wf_json)

            # ── pass 2 ────────────────────────────────────────────────────────
            print(f"  pass2 (weak_cells={len(weak_cells)})")
            p2_stem = OUTDIR / f"stop_{tag}_p2"
            rc2 = subprocess.run(
                _build_cmd(fixed=fixed, long_stop=la, short_stop=sa,
                           out_stem=p2_stem,
                           weak_filter=wf_json if weak_cells else None,
                           prep_cache_dir=prep_cache_dir,
                           refresh_prep_cache=False),
                cwd=str(PROJECT_ROOT), check=False,
            ).returncode
            if rc2 != 0:
                row = {"long_adverse": la, "short_adverse": sa, "error": f"pass2 rc={rc2}"}
                rows.append(row)
                for fn in fieldnames: row.setdefault(fn, "")
                writer.writerow(row); f.flush()
                durations.append(time.time() - t0)
                print(f"  ⚠ pass2 failed (rc={rc2})")
                continue

            p2 = parse_report(Path(str(p2_stem) + "_report.json"))
            run_sec = time.time() - t0
            durations.append(run_sec)

            row = {
                "horizon": args.horizon, "cap": args.cap,
                "long_adverse": la, "short_adverse": sa,
                "weak_cells": len(weak_cells),
                "p1_trades": p1["trades"],       "p1_total_pnl": p1["total_pnl"],
                "p1_win_rate_pct": p1["win_rate_pct"], "p1_profit_factor": p1["profit_factor"],
                "p2_trades": p2["trades"],       "p2_total_pnl": p2["total_pnl"],
                "p2_avg_trade": p2["avg_trade"], "p2_win_rate_pct": p2["win_rate_pct"],
                "p2_profit_factor": p2["profit_factor"],
                "p2_avg_trades_per_day": p2["avg_trades_per_day"],
                "p2_trade_max_drawdown": p2["trade_max_drawdown"],
                "p2_daily_max_drawdown": p2["daily_max_drawdown"],
                "delta_total_pnl": round(p2["total_pnl"] - p1["total_pnl"], 2),
                "error": "",
            }
            rows.append(row)
            for fn in fieldnames: row.setdefault(fn, "")
            writer.writerow(row); f.flush()

            valid    = [r for r in rows if not r.get("error")]
            cur_best = max(valid, key=lambda r: float(r["p2_total_pnl"]))
            print(f"  → {fmt_sec(run_sec)}  p2_pnl={p2['total_pnl']:.0f}  "
                  f"pf={p2['profit_factor']:.3f}  dd_dy={p2['daily_max_drawdown']:.0f}  "
                  f"best_so_far: la={cur_best['long_adverse']} sa={cur_best['short_adverse']} "
                  f"pnl={float(cur_best['p2_total_pnl']):.0f}")

    total_sec = time.time() - sweep_start
    print(f"\n✅ done in {fmt_sec(total_sec)}  →  {summary}")
    print_top(rows, sort_key="p2_total_pnl", param_keys=["long_adverse", "short_adverse", "weak_cells"])


if __name__ == "__main__":
    main()

