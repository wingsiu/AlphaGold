#!/usr/bin/env python3
"""Gold v16 parity check wrapper."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v16.gold.parity_check import run_parity

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    start = args[0] if args else None
    end = args[1] if len(args) > 1 else start
    print(run_parity(start, end))
