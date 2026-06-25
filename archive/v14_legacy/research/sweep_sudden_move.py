#!/usr/bin/env python3
"""Sweep sudden 3m move thresholds (ret_3m > a or < -b). Baseline = thresholds unset."""
import os
import re
import subprocess
import sys
from pathlib import Path

from v14._paths import PROJECT_ROOT
BT = ["2026-01-01", "2026-05-21"]
BASELINE_PNL = 1612.4

# (label, rise_a, drop_b) — None,None = disabled
TESTS = [
    ("NONE", None, None, "runtime/bot_assets/wf_models_v14"),
    ("a5_b5", 5, 5, "runtime/bot_assets/wf_models_v14_sudden_5_5"),
    ("a8_b8", 8, 8, "runtime/bot_assets/wf_models_v14_sudden_8_8"),
    ("a10_b10", 10, 10, "runtime/bot_assets/wf_models_v14_sudden_10_10"),
    ("a12_b12", 12, 12, "runtime/bot_assets/wf_models_v14_sudden_12_12"),
    ("a15_b15", 15, 15, "runtime/bot_assets/wf_models_v14_sudden_15_15"),
    ("a10_b15", 10, 15, "runtime/bot_assets/wf_models_v14_sudden_10_15"),
    ("a15_b10", 15, 10, "runtime/bot_assets/wf_models_v14_sudden_15_10"),
    ("a20_b20", 20, 20, "runtime/bot_assets/wf_models_v14_sudden_20_20"),
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


def run_one(rise_a: float, drop_b: float, model_dir: str) -> tuple[int, float, float]:
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
    env["V14_SUDDEN_RISE_A"] = str(rise_a)
    env["V14_SUDDEN_DROP_B"] = str(drop_b)
    env["V14_MODEL_OUTPUT_DIR"] = model_dir
    env.pop("V14_PA_GROUP", None)

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
    r.check_returncode()
    return parse_backtest(r.stdout)


def main():
    print("Sudden 3m move sweep (ret_3m > a or < -b)")
    print(f"{'Label':<10} | {'a':>4} | {'b':>4} | {'Trades':>6} | {'Win%':>6} | {'Net PnL':>8} | {'vs base':>8}")
    print("-" * 68)
    sys.stdout.flush()

    out = PROJECT_ROOT / "sudden_move_sweep_results.csv"
    with open(out, "w") as f:
        f.write("label,rise_a,drop_b,trades,win_rate,net_pnl,delta_vs_none\n")

    for label, a, b, model_dir in TESTS:
        if label == "NONE":
            print("Baseline (no sudden features)...", flush=True)
            env = os.environ.copy()
            env.pop("V14_SUDDEN_RISE_A", None)
            env.pop("V14_SUDDEN_DROP_B", None)
            env["V14_MODEL_OUTPUT_DIR"] = model_dir
            r = subprocess.run(
                [sys.executable, "backtest_v14.py"] + BT,
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            trades, wr, pnl = parse_backtest(r.stdout)
        else:
            print(f"Train+BT {label} (a={a}, b={b})...", flush=True)
            trades, wr, pnl = run_one(a, b, model_dir)

        delta = pnl - BASELINE_PNL
        a_s = "-" if a is None else str(a)
        b_s = "-" if b is None else str(b)
        print(f"{label:<10} | {a_s:>4} | {b_s:>4} | {trades:6d} | {wr:5.1f}% | {pnl:8.1f} | {delta:+8.1f}")
        sys.stdout.flush()
        with open(out, "a") as f:
            f.write(f"{label},{a_s},{b_s},{trades},{wr},{pnl},{delta}\n")

    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
