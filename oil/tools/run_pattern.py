#!/usr/bin/env python3
"""
Train (prod-only) + backtest one oil pattern — same workflow as gold try_add_pattern.

Usage:
  PYTHONPATH=. .venv/bin/python3 oil/tools/run_pattern.py oil_downtrend_retrace
  PYTHONPATH=. .venv/bin/python3 oil/tools/run_pattern.py 2025-06-01 2026-05-23 oil_downtrend_retrace
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> None:
    args = sys.argv[1:]
    train_cmd = [PY, str(ROOT / "oil" / "tools" / "train.py"), *args]
    bt_cmd = [PY, str(ROOT / "oil" / "tools" / "backtest.py"), *args]
    print("=== Train (prod-only) ===")
    subprocess.run(train_cmd, cwd=ROOT, check=True)
    print("\n=== Backtest (no gold time filter) ===")
    subprocess.run(bt_cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
