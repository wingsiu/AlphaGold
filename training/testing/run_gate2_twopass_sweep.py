#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.testing.sweep_utils import generate_weak_filter, parse_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_TREND_ML = PROJECT_ROOT / "training" / "image_trend_ml.py"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _frange(start: float, end: float, step: float) -> list[float]:
    vals: list[float] = []
    x = start
    while x <= end + 1e-9:
        vals.append(round(x, 4))
        x += step
    return vals


def _build_cmd(
    model_in: Path,
    out_base: Path,
    gate2: float,
    weak_filter: Path | None,
    prep_cache_dir: Path | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(IMAGE_TREND_ML),
        "--start-date",
        "2025-05-20",
        "--end-date",
        "2026-04-10",
        "--test-start-date",
        "2025-11-25T17:02:00+00:00",
        "--timeframe",
        "1min",
        "--eval-mode",
        "single_split",
        "--disable-time-filter",
        "--window",
        "150",
        "--window-15m",
        "0",
        "--min-window-range",
        "30",
        "--min-15m-drop",
        "15",
        "--min-15m-rise",
        "0",
        "--horizon",
        "25",
        "--trend-threshold",
        "0.008",
        "--adverse-limit",
        "15",
        "--long-target-threshold",
        "0.008",
        "--short-target-threshold",
        "0.008",
        "--long-adverse-limit",
        "12",
        "--short-adverse-limit",
        "18",
        "--classifier",
        "gradient_boosting",
        "--max-flat-ratio",
        "2.5",
        "--stage1-min-prob",
        "0.55",
        "--stage2-min-prob",
        "0.58",
        "--stage2-min-prob-up",
        f"{gate2:.2f}",
        "--stage2-min-prob-down",
        f"{gate2:.2f}",
        "--model-in",
        str(model_in),
        "--model-out",
        str(out_base) + "_model.joblib",
        "--report-out",
        str(out_base) + "_report.json",
        "--trades-out",
        str(out_base) + "_trades.csv",
    ]
    if prep_cache_dir is not None:
        cmd += ["--prep-cache-dir", str(prep_cache_dir)]
    if weak_filter is not None:
        cmd += ["--weak-periods-json", str(weak_filter)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-pass sweep for gate2 probability.")
    ap.add_argument("--model-in", default="runtime/_tmp_backtest_no_retrain/H25_full_predefined_pass2_model.joblib")
    ap.add_argument("--start", type=float, default=0.60)
    ap.add_argument("--end", type=float, default=0.90)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument(
        "--prep-cache-dir",
        default="runtime/_tmp_backtest_no_retrain/prep_cache_gate2_twopass",
        help="Shared prep cache folder so bars/dataset are built once then reused across all runs.",
    )
    ap.add_argument("--refresh-prep-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_in = Path(args.model_in)
    if not model_in.is_absolute():
        model_in = PROJECT_ROOT / model_in
    if not model_in.exists():
        raise FileNotFoundError(f"model not found: {model_in}")

    prep_cache_dir = Path(args.prep_cache_dir)
    if not prep_cache_dir.is_absolute():
        prep_cache_dir = PROJECT_ROOT / prep_cache_dir
    prep_cache_dir.mkdir(parents=True, exist_ok=True)

    gates = _frange(args.start, args.end, args.step)
    print(f"Gate values: {gates}")
    print(f"Two-pass runs: {len(gates) * 2}")
    print(f"Prep cache dir: {prep_cache_dir}")
    if args.dry_run:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = RESULTS_DIR / f"sweep_gate2_{args.start:.2f}_{args.end:.2f}_twopass_{ts}.csv"
    fields = [
        "gate2",
        "trades",
        "total_pnl",
        "avg_trade",
        "win_rate_pct",
        "profit_factor",
        "trade_max_drawdown",
        "error",
    ]

    rows: list[dict[str, object]] = []
    t0 = time.time()
    for i, gate in enumerate(gates, 1):
        print(f"[{i}/{len(gates)}] gate2={gate:.2f} pass1(no weak filter)")
        prefix = RESULTS_DIR / f"gate2_{gate:.2f}_{ts}"

        p1 = Path(str(prefix) + "_p1")
        cmd1 = _build_cmd(model_in, p1, gate, None, prep_cache_dir)
        if args.refresh_prep_cache and i == 1:
            cmd1 += ["--refresh-prep-cache"]
        rc1 = subprocess.run(cmd1, cwd=str(PROJECT_ROOT), check=False).returncode
        if rc1 != 0:
            rows.append({"gate2": gate, "error": f"pass1 rc={rc1}"})
            continue

        wf = Path(str(prefix) + "_wf.json")
        weak_cells = generate_weak_filter(Path(str(p1) + "_trades.csv"), wf)
        print(f"    generated weak_cells={len(weak_cells)}")

        print(f"[{i}/{len(gates)}] gate2={gate:.2f} pass2(with generated filter)")
        p2 = Path(str(prefix) + "_p2")
        rc2 = subprocess.run(
            _build_cmd(model_in, p2, gate, wf if weak_cells else None, prep_cache_dir),
            cwd=str(PROJECT_ROOT),
            check=False,
        ).returncode
        if rc2 != 0:
            rows.append({"gate2": gate, "error": f"pass2 rc={rc2}"})
            continue

        rpt = parse_report(Path(str(p2) + "_report.json"))
        row = {
            "gate2": gate,
            "trades": rpt.get("trades", ""),
            "total_pnl": rpt.get("total_pnl", ""),
            "avg_trade": rpt.get("avg_trade", ""),
            "win_rate_pct": rpt.get("win_rate_pct", ""),
            "profit_factor": rpt.get("profit_factor", ""),
            "trade_max_drawdown": rpt.get("trade_max_drawdown", ""),
            "error": rpt.get("error", ""),
        }
        rows.append(row)
        print(
            f"    pass2 pnl={row['total_pnl']} trades={row['trades']} "
            f"wr={row['win_rate_pct']} pf={row['profit_factor']}"
        )

    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok_rows = [r for r in rows if not r.get("error") and r.get("total_pnl") not in ("", None)]
    ok_rows.sort(key=lambda r: float(r["total_pnl"]), reverse=True)

    print("\n=== Sweep Completed ===")
    print(f"summary: {summary_csv}")
    print(f"elapsed: {int(time.time() - t0)}s")
    if ok_rows:
        b = ok_rows[0]
        print(
            f"best gate2={b['gate2']} pnl={b['total_pnl']} trades={b['trades']} "
            f"wr={b['win_rate_pct']} pf={b['profit_factor']}"
        )


if __name__ == "__main__":
    main()

