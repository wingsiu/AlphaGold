#!/usr/bin/env python3
"""Test adding one price-action group at a time (30/30/25, S1=0.50, S2=0.55)."""
import os
import re
import subprocess
import sys
from pathlib import Path

from v14._paths import PROJECT_ROOT
BT = ["2026-01-01", "2026-05-21"]
BASELINE_PNL = 1612.4

TESTS = [
    ("NONE", "", "runtime/bot_assets/wf_models_v14"),
    ("FVG", "fvg", "runtime/bot_assets/wf_models_v14_pa_fvg"),
    ("WICK", "wick", "runtime/bot_assets/wf_models_v14_pa_wick"),
    ("FAKE", "fake", "runtime/bot_assets/wf_models_v14_pa_fake"),
]


def run_one(label: str, pa_group: str, model_dir: str) -> tuple[int, float, float]:
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
    env.pop("V14_USE_PRICE_ACTION", None)
    if pa_group:
        env["V14_PA_GROUP"] = pa_group
    else:
        env.pop("V14_PA_GROUP", None)
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
    trades, wr, pnl = 0, 0.0, 0.0
    for line in r.stdout.split("\n"):
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


def main():
    print("Single price-action group test (H=30, TP=30, SL=25)")
    print(f"{'Group':<6} | {'Trades':>6} | {'Win%':>6} | {'Net PnL':>8} | {'vs base':>8}")
    print("-" * 52)
    sys.stdout.flush()

    out = PROJECT_ROOT / "single_pa_results.csv"
    with open(out, "w") as f:
        f.write("group,trades,win_rate,net_pnl,delta_vs_none\n")

    for label, pa_group, model_dir in TESTS:
        if label == "NONE":
            print("Skipping NONE retrain (using existing no-PA models)...", flush=True)
            env = os.environ.copy()
            env.pop("V14_PA_GROUP", None)
            env["V14_MODEL_OUTPUT_DIR"] = model_dir
            r = subprocess.run(
                [sys.executable, "backtest_v14.py"] + BT,
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            trades, wr, pnl = 0, 0.0, 0.0
            for line in r.stdout.split("\n"):
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
        else:
            print(f"Training + backtest with +{label}...", flush=True)
            trades, wr, pnl = run_one(label, pa_group, model_dir)

        delta = pnl - BASELINE_PNL
        print(f"{label:<6} | {trades:6d} | {wr:5.1f}% | {pnl:8.1f} | {delta:+8.1f}")
        sys.stdout.flush()
        with open(out, "a") as f:
            f.write(f"{label},{trades},{wr},{pnl},{delta}\n")

    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
