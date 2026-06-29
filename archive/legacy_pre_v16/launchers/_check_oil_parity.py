#!/usr/bin/env python3
"""CLI wrapper for oil parity check. See oil/parity_check.py."""
from oil.parity_check import run_parity
import sys

if __name__ == "__main__":
    sd = sys.argv[1] if len(sys.argv) > 1 else None
    ed = sys.argv[2] if len(sys.argv) > 2 else sd
    print(run_parity(sd, ed))
