#!/usr/bin/env python3
"""Tiny smoke run for the image trend ML pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "training" / "image_trend_ml.py"),
        "--timeframe", "1min",
        "--start-date", "2025-11-01",
        "--end-date", "2026-02-04",
        "--window", "120",
        "--window-15m", "0",        # 1m-only baseline (15m branch disabled)
        "--min-window-range", "40",
        "--min-15m-drop", "20",
        "--horizon", "25",
        "--trend-threshold", "0.004",
        "--max-samples", "3000",
        "--model-out", "training/image_trend_model_smoke.joblib",
        "--report-out", "training/image_trend_report_smoke.json",
    ]
    completed = subprocess.run(cmd, cwd=str(root), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

