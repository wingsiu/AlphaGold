#!/usr/bin/env python3
"""CLI wrapper for v16 oil parity check. See v16/oil/parity_check.py."""
from v16.oil.parity_check import run_parity
import sys

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    sd = args[0] if args else None
    ed = args[1] if len(args) > 1 else sd
    wr90_exit = "fixed_tpsl" if "--fixed-tpsl" in flags else "struct_hold"
    print(run_parity(sd, ed, wr90_exit=wr90_exit))
