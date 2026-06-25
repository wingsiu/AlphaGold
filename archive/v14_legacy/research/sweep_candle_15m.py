#!/usr/bin/env python3
"""A/B: 15m candle shape/pattern features vs baseline (2026 window)."""
import os
import re
import subprocess
import sys
from pathlib import Path

from v14._paths import PROJECT_ROOT
BT = ["2026-01-01", "2026-05-21"]
BASELINE_PNL = 1612.4

TESTS = [
    ("NONE", False, "runtime/bot_assets/wf_models_v14"),
    ("CANDLE15", True, "runtime/bot_assets/wf_models_v14_candle15"),
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


def run_one(use_candle: bool, model_dir: str) -> tuple[int, float, float]:
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
    env["V14_MODEL_OUTPUT_DIR"] = model_dir
    env.pop("V14_PA_GROUP", None)
    env.pop("V14_SUDDEN_RISE_A", None)
    env.pop("V14_SUDDEN_DROP_B", None)
    if use_candle:
        env["V14_CANDLE_15M"] = "1"
    else:
        env.pop("V14_CANDLE_15M", None)

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
    print("15m candle pattern A/B (30/30/25)")
    print(f"{'Label':<10} | {'Trades':>6} | {'Win%':>6} | {'Net PnL':>8} | {'vs base':>8}")
    print("-" * 52)
    out = PROJECT_ROOT / "candle_15m_sweep_results.csv"
    with open(out, "w") as f:
        f.write("label,trades,win_rate,net_pnl,delta_vs_none\n")

    for label, use_candle, model_dir in TESTS:
        if label == "NONE":
            env = os.environ.copy()
            env.pop("V14_CANDLE_15M", None)
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
            print(f"Training + backtest {label}...", flush=True)
            trades, wr, pnl = run_one(use_candle, model_dir)

        print(f"{label:<10} | {trades:6d} | {wr:5.1f}% | {pnl:8.1f} | {pnl - BASELINE_PNL:+8.1f}")
        with open(out, "a") as f:
            f.write(f"{label},{trades},{wr},{pnl},{pnl - BASELINE_PNL}\n")

    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
