#!/usr/bin/env python3
"""Gold v16 combined backtest — one command, full stats (drop-in for legacy launcher).

  python3 run_hybrid_backtest.py
  python3 run_hybrid_backtest.py 2025-06-01 2026-06-25
  python3 run_hybrid_backtest.py --last30
  python3 run_hybrid_backtest.py --quick
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_gold_v16_backtest import main

if __name__ == "__main__":
    main()
