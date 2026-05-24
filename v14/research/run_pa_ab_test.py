#!/usr/bin/env python3
"""A/B: v14 with vs without 15m price-action features (FVG, wicks, fake up/down)."""
import os
import re
import subprocess
import sys
from pathlib import Path

from sweep_v14 import update_config

from v14._paths import PROJECT_ROOT
BT = ["2026-01-01", "2026-05-21"]
CONFIGS = [(30, 30, 15), (30, 30, 25)]
MODES = [
    ("WITH_PA", "1", "runtime/bot_assets/wf_models_v14"),
    ("NO_PA", "0", "runtime/bot_assets/wf_models_v14_no_pa"),
]


def run_one(horizon: int, tp: float, sl: float, use_pa: str, model_dir: str) -> tuple[int, float, float]:
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
    env["V14_USE_PRICE_ACTION"] = use_pa
    env["V14_MODEL_OUTPUT_DIR"] = model_dir

    update_config(horizon, tp, sl)
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
    print("Price-action A/B (S1=0.50, S2=0.55, 2026-01-01 → 2026-05-21)")
    print(f"{'Mode':<8} | {'H':>2} | {'TP':>2} | {'SL':>2} | {'Trades':>6} | {'Win%':>6} | {'Net PnL':>8}")
    print("-" * 58)
    sys.stdout.flush()

    results_path = PROJECT_ROOT / "pa_ab_results.csv"
    with open(results_path, "w") as f:
        f.write("mode,horizon,tp,sl,trades,win_rate,net_pnl\n")

    for h, tp, sl in CONFIGS:
        for label, use_pa, model_dir in MODES:
            print(f"Running {label} H={h} TP={tp} SL={sl}...", flush=True)
            trades, wr, pnl = run_one(h, tp, sl, use_pa, model_dir)
            print(f"{label:<8} | {h:2d} | {tp:2.0f} | {sl:2.0f} | {trades:6d} | {wr:5.1f}% | {pnl:8.1f}")
            sys.stdout.flush()
            with open(results_path, "a") as f:
                f.write(f"{label},{h},{tp},{sl},{trades},{wr},{pnl}\n")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
