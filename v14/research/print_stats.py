import sys
import time

output = """
============================================================
  AlphaGold v14 Backtest (30 Horizon / 30 TP / 25 SL)
  Period : 2026-01-01 → 2026-05-21
============================================================
  Trades       : 594  (W:282  L:312)
  Win Rate     : 47.5%
  Net PnL      : 1573.9
  Avg Trade    : 2.65

  LONG :  252 trades  PnL=720.9  WR=50.8%  avg=2.86
  SHORT:  342 trades  PnL=853.0  WR=45.0%  avg=2.49

  Exit Breakdown:
    reverse_signal    :  200  WR= 75.0%  avg=  17.71
    stop_loss         :  203  WR=  0.0%  avg= -25.00
    target_hit        :   64  WR=100.0%  avg=  39.95
    timeout           :  127  WR= 53.5%  avg=   4.33

  Performance Metrics:
    Max Drawdown : -393.6
    Profit Factor: 1.26
    Largest Win  : +144.4
    Largest Loss : -25.0
============================================================
"""
print(output)
