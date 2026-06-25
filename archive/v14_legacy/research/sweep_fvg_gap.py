#!/usr/bin/env python3
"""Sweep FVG-only price action with minimum gap size (points). Baseline = no PA."""
import os
import re
import subprocess
import sys
from pathlib import Path

from v14._paths import PROJECT_ROOT
BT = ["2026-01-01", "2026-05-21"]
BASELINE_PNL = 1612.4

TESTS = [
    ("NONE", None, None, "runtime/bot_assets/wf_models_v14"),
    ("FVG_any", "fvg", "0", "runtime/bot_assets/wf_models_v14_fvg_g0"),
    ("FVG_g3", "fvg", "3", "runtime/bot_assets/wf_models_v14_fvg_g3"),
    ("FVG_g5", "fvg", "5", "runtime/bot_assets/wf_models_v14_fvg_g5"),
]


def parse_backtest(stdout: str) -> tuple[int, float, float]:
    trades, wr, pnl = 0, 0.0, 0.0
    for line in stdout.split("\n"):
        if "Trades       :" in line:
            m = re.search(r"Trades\s*:\s*(\d+)", line)
            if m:
                trades = int(m.group(1))
        elif "Win Rate     :" in line:
            m = re.search(r"Win Rate\s*:\s*([0-9.]+)", line)
            if m:
                wr = float(m.group(1))
        elif "Net PnL      :" in line:
            m = re.search(r"Net PnL\s*:\s*([0-9.-]+)", line)
            if m:
                pnl = float(m.group(1))
    return trades, wr, pnl


def run_one(pa_group: str, min_gap: str, model_dir: str) -> tuple[int, float, float]:
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
    env.pop("V14_USE_PRICE_ACTION", None)
    env["V14_PA_GROUP"] = pa_group
    env["V14_FVG_MIN_GAP"] = min_gap
    env["V14_MODEL_OUTPUT_DIR"] = model_dir

    subprocess.run(
        [sys.executable, "xgboost_filter_model/train_filter_v14.py"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [sys.executable, "xgboost_filter_model/train_stage2_v14_directional.py"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    r = subprocess.run(
        [sys.executable, "backtest_v14.py"] + BT,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr[-2000:] if r.stderr else r.stdout[-2000:])
        r.check_returncode()
    return parse_backtest(r.stdout)


def main():
    print("FVG min-gap sweep (30/30/25, S1=0.50, S2=0.55)")
    print(f"{'Label':<8} | {'min_gap':>7} | {'Trades':>6} | {'Win%':>6} | {'Net PnL':>8} | {'vs base':>8}")
    print("-" * 62)
    sys.stdout.flush()

    out = PROJECT_ROOT / "fvg_gap_sweep_results.csv"
    with open(out, "w") as f:
        f.write("label,min_gap,trades,win_rate,net_pnl,delta_vs_none\n")

    for label, pa_group, min_gap, model_dir in TESTS:
        if label == "NONE":
            print("Baseline (no PA, existing models)...", flush=True)
            env = os.environ.copy()
            env.pop("V14_PA_GROUP", None)
            env.pop("V14_FVG_MIN_GAP", None)
            env["V14_MODEL_OUTPUT_DIR"] = model_dir
            r = subprocess.run(
                [sys.executable, "backtest_v14.py"] + BT,
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            trades, wr, pnl = parse_backtest(r.stdout)
            gap_s = "-"
        else:
            print(f"Train+BT {label} (min_gap={min_gap})...", flush=True)
            trades, wr, pnl = run_one(pa_group, min_gap, model_dir)
            gap_s = min_gap

        delta = pnl - BASELINE_PNL
        print(f"{label:<8} | {gap_s:>7} | {trades:6d} | {wr:5.1f}% | {pnl:8.1f} | {delta:+8.1f}")
        sys.stdout.flush()
        with open(out, "a") as f:
            f.write(f"{label},{gap_s},{trades},{wr},{pnl},{delta}\n")

    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
