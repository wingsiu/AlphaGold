#!/usr/bin/env python3
"""Run and/or summarize backtest parameter sweeps.

This script wraps `training/backtest.py` so you can avoid fragile shell heredocs.

Examples:
  # 1) Report from existing logs (no new backtests)
  python3 optimise.py report \
    --log-dir training/refine_pct_logs \
    --top-n 10

  # 2) Run a percent-stop sweep and print a report
  python3 optimise.py run \
    --start-date 2025-05-01 \
    --end-date 2026-02-04 \
    --model-type gradient_boosting \
    --signals-file training/test_period_signals.csv \
    --stop-mode pct \
    --stop-values 0.15,0.20,0.25,0.30 \
    --tp-mode usd \
    --tp-values 0.40,0.50,0.60 \
    --hold-values 40,50,60 \
    --optimize-cutoff \
    --cutoff-min 0.40 \
    --cutoff-max 0.80 \
    --cutoff-step 0.05 \
    --log-dir training/refine_pct_logs \
    --csv-dir training/refine_pct_csv \
    --top-n 10
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parent
BACKTEST_PATH = ROOT_DIR / "training" / "backtest.py"


@dataclass
class ResultRow:
    model: str
    stop_mode: str
    stop_value: float
    tp_mode: str
    tp: float
    hold: int
    session_filter: bool
    cutoff: float
    trades: int
    win_rate: float
    total_profit: float
    log_file: str


def parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def stop_token(stop_mode: str, stop_value: float) -> str:
    if stop_mode == "pct":
        return f"stoppct{stop_value:.2f}"
    return f"stopusd{int(stop_value)}"


def tp_token(tp_mode: str, tp_value: float) -> str:
    if tp_mode == "usd":
        return f"tpusd{tp_value:.2f}"
    return f"tppct{tp_value:.2f}"


def build_run_name(
    model: str,
    stop_mode: str,
    stop_value: float,
    tp_mode: str,
    tp: float,
    hold: int,
    session_filter: bool = False,
) -> str:
    base = f"{model}_{stop_token(stop_mode, stop_value)}_{tp_token(tp_mode, tp)}_hold{hold}"
    return f"{base}_sessf" if session_filter else base


def run_one(
    *,
    model: str,
    start_date: str,
    end_date: str,
    signals_file: str,
    tp_mode: str,
    take_profit_pct: float,
    hold: int,
    stop_mode: str,
    stop_value: float,
    optimize_cutoff: bool,
    cutoff_min: float,
    cutoff_max: float,
    cutoff_step: float,
    session_filter: bool,
    config_file: str,
    log_path: Path,
    csv_path: Path,
) -> int:
    cmd = [
        sys.executable,
        "-u",
        str(BACKTEST_PATH),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--model-type",
        model,
        "--signals-file",
        signals_file,
        "--max-hold-bars",
        str(hold),
        "--out",
        str(csv_path),
    ]

    if tp_mode == "usd":
        cmd.extend(["--take-profit-usd", f"{take_profit_pct:.2f}"])
    else:
        cmd.extend(["--take-profit-pct", f"{take_profit_pct:.2f}"])

    if stop_mode == "pct":
        cmd.extend(["--stop-loss-pct", f"{stop_value:.2f}"])
    else:
        cmd.extend(["--stop-loss-usd", str(int(stop_value))])

    if optimize_cutoff:
        cmd.extend(
            [
                "--optimize-cutoff",
                "--cutoff-min",
                f"{cutoff_min:.2f}",
                "--cutoff-max",
                f"{cutoff_max:.2f}",
                "--cutoff-step",
                f"{cutoff_step:.2f}",
            ]
        )

    if session_filter:
        cmd.append("--session-filter")
        cmd.extend(["--config-file", config_file])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), stdout=f, stderr=subprocess.STDOUT)
        start_ts = time.time()
        last_log_size = 0
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc

            time.sleep(15)
            elapsed = int(time.time() - start_ts)
            try:
                log_size = log_path.stat().st_size
            except OSError:
                log_size = 0

            if log_size > last_log_size:
                print(f"    ...running {elapsed}s (log {log_size} bytes)")
                last_log_size = log_size
            else:
                print(f"    ...running {elapsed}s (waiting for next progress line)")


def parse_log_file(path: Path) -> ResultRow | None:
    name = path.name
    m = re.match(
        r"(random_forest|gradient_boosting)_(stoppct|stopusd)([0-9.]+)_(tppct|tpusd)([0-9.]+)_hold([0-9]+)(_sessf)?\.log$",
        name,
    )
    if m:
        model = m.group(1)
        stop_mode = "pct" if m.group(2) == "stoppct" else "usd"
        stop_value = float(m.group(3))
        tp_mode = "pct" if m.group(4) == "tppct" else "usd"
        tp = float(m.group(5))
        hold = int(m.group(6))
        session_filter = bool(m.group(7))
    else:
        # Backward compatibility for older naming: *_tpX.YY_holdZ.log (percent TP).
        m = re.match(
            r"(random_forest|gradient_boosting)_(stoppct|stopusd)([0-9.]+)_tp([0-9.]+)_hold([0-9]+)(_sessf)?\.log$",
            name,
        )
        if not m:
            return None
        model = m.group(1)
        stop_mode = "pct" if m.group(2) == "stoppct" else "usd"
        stop_value = float(m.group(3))
        tp_mode = "pct"
        tp = float(m.group(4))
        hold = int(m.group(5))
        session_filter = bool(m.group(6))

    txt = path.read_text(encoding="utf-8", errors="ignore")

    def pick(pattern: str, cast=float):
        mm = re.search(pattern, txt)
        return cast(mm.group(1)) if mm else None

    cutoff = pick(r"Using cutoff: ([0-9.]+)")
    trades = pick(r"Trades: (\d+)", int)
    win = pick(r"Win rate: ([0-9.]+)%")
    total = pick(r"Total profit \(USD, 1 unit\): \$([-0-9.]+)")

    if None in (cutoff, trades, win, total):
        return None

    return ResultRow(
        model=model,
        stop_mode=stop_mode,
        stop_value=stop_value,
        tp_mode=tp_mode,
        tp=tp,
        hold=hold,
        session_filter=session_filter,
        cutoff=float(cutoff),
        trades=int(trades),
        win_rate=float(win),
        total_profit=float(total),
        log_file=name,
    )


def parse_results(log_dir: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for path in sorted(log_dir.glob("*.log")):
        row = parse_log_file(path)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (r.total_profit, r.win_rate, r.trades), reverse=True)
    return rows


def print_report(rows: Iterable[ResultRow], top_n: int) -> None:
    rows = list(rows)
    if not rows:
        print("No valid logs parsed.")
        return

    print("Top results by total USD profit")
    for i, r in enumerate(rows[:top_n], 1):
        stop_label = f"{r.stop_value:.2f}%" if r.stop_mode == "pct" else f"${int(r.stop_value)}"
        tp_label = f"{r.tp:.2f}%" if r.tp_mode == "pct" else f"${r.tp:.2f}"
        print(
            f"{i:2d}. model={r.model} stop={stop_label} hold={r.hold} tp={tp_label} "
            f"sessf={'on' if r.session_filter else 'off'} cutoff={r.cutoff:.2f} "
            f"trades={r.trades} win={r.win_rate:.2f}% profit=${r.total_profit:.2f}"
        )

    best = rows[0]
    stop_label = f"{best.stop_value:.2f}%" if best.stop_mode == "pct" else f"${int(best.stop_value)}"
    tp_label = f"{best.tp:.2f}%" if best.tp_mode == "pct" else f"${best.tp:.2f}"
    print("\nBEST:")
    print(
        f"model={best.model} stop={stop_label} hold={best.hold} tp={tp_label} "
        f"sessf={'on' if best.session_filter else 'off'} cutoff={best.cutoff:.2f} "
        f"trades={best.trades} win={best.win_rate:.2f}% "
        f"profit=${best.total_profit:.2f} ({best.log_file})"
    )


def write_csv(rows: Iterable[ResultRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "stop_mode",
                "stop_value",
                "take_profit_mode",
                "max_hold_bars",
                "session_filter",
                "take_profit_value",
                "cutoff",
                "trades",
                "win_rate_pct",
                "total_profit_usd",
                "log_file",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.model,
                    r.stop_mode,
                    r.stop_value,
                    r.tp_mode,
                    r.hold,
                    r.session_filter,
                    r.tp,
                    r.cutoff,
                    r.trades,
                    r.win_rate,
                    r.total_profit,
                    r.log_file,
                ]
            )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-dir", default="training/refine_pct_logs")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--summary-csv", default=None, help="Optional CSV output path for parsed results")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run and report backtest optimization sweeps.")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Execute backtest sweep then print report")
    run.add_argument("--start-date", required=True)
    run.add_argument("--end-date", required=True)
    run.add_argument("--model-type", default="gradient_boosting", choices=["random_forest", "gradient_boosting"])
    run.add_argument("--signals-file", default="training/test_period_signals.csv")
    run.add_argument("--stop-mode", required=True, choices=["pct", "usd"])
    run.add_argument("--stop-values", required=True, help="Comma list, e.g. 0.15,0.20 or 10,15")
    run.add_argument("--tp-mode", default="pct", choices=["pct", "usd"], help="Interpret --tp-values as percent or USD distance")
    run.add_argument("--tp-values", required=True, help="Comma list, e.g. 0.40,0.50")
    run.add_argument("--hold-values", required=True, help="Comma list, e.g. 40,50,60")
    run.add_argument("--optimize-cutoff", action="store_true")
    run.add_argument("--cutoff-min", type=float, default=0.40)
    run.add_argument("--cutoff-max", type=float, default=0.80)
    run.add_argument("--cutoff-step", type=float, default=0.05)
    run.add_argument("--session-filter", action="store_true", help="Apply time slot filters from ml_config.json")
    run.add_argument("--config-file", default="ml_config.json", help="Path passed through to backtest --config-file")
    run.add_argument("--csv-dir", default="training/refine_pct_csv")
    run.add_argument("--skip-existing", action="store_true", help="Skip runs with an existing output CSV")
    add_common_args(run)

    report = sub.add_parser("report", help="Parse existing logs and print report")
    add_common_args(report)

    return p


def cmd_run(args: argparse.Namespace) -> int:
    stop_values = parse_float_list(args.stop_values)
    tp_values = parse_float_list(args.tp_values)
    hold_values = parse_int_list(args.hold_values)


    if args.stop_mode == "usd":
        stop_values = [int(v) for v in stop_values]

    log_dir = ROOT_DIR / args.log_dir
    csv_dir = ROOT_DIR / args.csv_dir

    total_runs = len(stop_values) * len(tp_values) * len(hold_values)
    done = 0
    failed = 0

    for stop in stop_values:
        for hold in hold_values:
            for tp in tp_values:
                run_name = build_run_name(
                    args.model_type,
                    args.stop_mode,
                    float(stop),
                    args.tp_mode,
                    tp,
                    hold,
                    session_filter=args.session_filter,
                )
                log_path = log_dir / f"{run_name}.log"
                csv_path = csv_dir / f"{run_name}.csv"

                if args.skip_existing and csv_path.exists():
                    print(f"Skip existing: {run_name}")
                    done += 1
                    continue

                print(f"[{done + 1}/{total_runs}] Running {run_name}")
                rc = run_one(
                    model=args.model_type,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    signals_file=args.signals_file,
                    tp_mode=args.tp_mode,
                    take_profit_pct=tp,
                    hold=hold,
                    stop_mode=args.stop_mode,
                    stop_value=float(stop),
                    optimize_cutoff=args.optimize_cutoff,
                    cutoff_min=args.cutoff_min,
                    cutoff_max=args.cutoff_max,
                    cutoff_step=args.cutoff_step,
                    session_filter=args.session_filter,
                    config_file=args.config_file,
                    log_path=log_path,
                    csv_path=csv_path,
                )

                if rc != 0:
                    failed += 1
                    print(f"  FAILED ({rc}): {run_name}")
                done += 1

    if failed:
        print(f"\nCompleted with failures: {failed}/{total_runs}")
    else:
        print(f"\nCompleted successfully: {total_runs} runs")

    rows = parse_results(log_dir)
    print_report(rows, args.top_n)

    if args.summary_csv:
        write_csv(rows, ROOT_DIR / args.summary_csv)
        print(f"Saved summary CSV: {args.summary_csv}")

    return 1 if failed else 0


def cmd_report(args: argparse.Namespace) -> int:
    log_dir = ROOT_DIR / args.log_dir
    rows = parse_results(log_dir)
    print_report(rows, args.top_n)

    if args.summary_csv:
        write_csv(rows, ROOT_DIR / args.summary_csv)
        print(f"Saved summary CSV: {args.summary_csv}")

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())

