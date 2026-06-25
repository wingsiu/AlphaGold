#!/usr/bin/env python3
"""Run adaptive vs. static prob backtest comparison for energetic-only S1/S2.

Usage:
  .venv/bin/python3 run_adaptive_comparison.py              # full default (wf_start → today)
  .venv/bin/python3 run_adaptive_comparison.py 2025-06-01 2026-05-30  # custom window

Time filter (runtime/v14_weak_time_slots.json) is ON by default.
Disable: V14_NO_TIME_FILTER=1
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd: list[str], label: str, extra_env: dict | None = None) -> int:
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    if extra_env:
        env.update(extra_env)
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  ENV: { {k: v for k, v in env.items() if k.startswith('V14_')} }")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode


def main() -> None:
    argv = sys.argv[1:]
    bt_cmd = [sys.executable, "v14/tools/backtest_v14.py", *argv]

    wtf_path = PROJECT_ROOT / "runtime" / "v14_weak_time_slots.json"
    has_wtf = wtf_path.exists()
    print(f"Time filter: {'ON' if has_wtf else 'OFF'} ({wtf_path})")

    # ── STATIC ──
    ret = run_cmd(
        bt_cmd,
        "ENERGETIC STATIC — original fixed thresholds",
        extra_env={
            "V14_ADAPTIVE_ENERGETIC": "0",
            "V14_NO_TIME_FILTER": "0" if has_wtf else "1",
        },
    )
    if ret == 0:
        src = PROJECT_ROOT / "runtime" / "v14_backtest_trades.csv"
        dst = PROJECT_ROOT / "runtime" / "v14_backtest_static.csv"
        if src.exists():
            src.rename(dst)
            print(f"  → Saved: {dst}")
        else:
            print("  ⚠️  No trades CSV produced.")
    else:
        print("\n⚠️  Static backtest failed.")

    # ── ADAPTIVE ──
    ret = run_cmd(
        bt_cmd,
        "ENERGETIC ADAPTIVE — volatility-scaled thresholds",
        extra_env={
            "V14_ADAPTIVE_ENERGETIC": "1",
            "V14_NO_TIME_FILTER": "0" if has_wtf else "1",
        },
    )
    if ret == 0:
        src = PROJECT_ROOT / "runtime" / "v14_backtest_trades.csv"
        dst = PROJECT_ROOT / "runtime" / "v14_backtest_adaptive.csv"
        if src.exists():
            src.rename(dst)
            print(f"  → Saved: {dst}")
        else:
            print("  ⚠️  No trades CSV produced.")
    else:
        print("\n⚠️  Adaptive backtest failed.")

    # ── SUMMARY ──
    print("\n" + "="*70)
    print("  ENERGETIC S1/S2 — STATIC vs ADAPTIVE SUMMARY")
    print("="*70)
    _print_summary(PROJECT_ROOT / "runtime" / "v14_backtest_static.csv", "Static")
    _print_summary(PROJECT_ROOT / "runtime" / "v14_backtest_adaptive.csv", "Adaptive")


def _print_summary(path: Path, label: str) -> None:
    if not path.exists():
        print(f"  {label}: NO TRADES FILE")
        return
    import pandas as pd

    try:
        tdf = pd.read_csv(path)
    except Exception:
        print(f"  {label}: ERROR reading CSV")
        return
    if tdf.empty:
        print(f"  {label}: 0 trades, 0.0 PnL")
        return
    wins = int((tdf["pnl"] > 0).sum())
    net_pnl = float(tdf["pnl"].sum())
    wr = wins / len(tdf) * 100
    avg = net_pnl / len(tdf)
    cum = tdf["pnl"].cumsum()
    max_dd = float((cum - cum.cummax()).min())
    gross_win = tdf[tdf["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(tdf[tdf["pnl"] <= 0]["pnl"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    print(f"  {label:10s}: {len(tdf):5d} trades  PnL={net_pnl:+.1f}  WR={wr:.1f}%  "
          f"avg={avg:+.2f}  DD={max_dd:+.1f}  PF={pf:.2f}")


if __name__ == "__main__":
    main()
