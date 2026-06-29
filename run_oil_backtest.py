#!/usr/bin/env python3
"""Oil v16 combined backtest — one command, full stats (legacy-style launcher).

  python3 run_oil_backtest.py
  python3 run_oil_backtest.py 2024-07-01 2026-06-30
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_oil_v16_backtest import main

if __name__ == "__main__":
    main()
