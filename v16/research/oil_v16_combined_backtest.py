#!/usr/bin/env python3
"""v16 oil combined backtest — delegates to run_oil_v16_backtest (full stats).

  PYTHONPATH=. python3 v16/research/oil_v16_combined_backtest.py [start] [end]
  PYTHONPATH=. python3 v16/research/oil_v16_combined_backtest.py --struct-hold
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_oil_v16_backtest import main

if __name__ == "__main__":
    main()
