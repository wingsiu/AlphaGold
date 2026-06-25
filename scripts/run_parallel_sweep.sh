#!/bin/bash
combo=$1
h=$2
tp=$3
sl=$4

dir="AlphaGold_$combo"
cp -r /Users/alpha/Desktop/python/AlphaGold /tmp/$dir
cd /tmp/$dir

# Update config
python3 -c "
from sweep_v14 import update_config
update_config($h, $tp, $sl)
"

export NUMBA_CACHE_DIR=/tmp/numba_cache_$combo
python3 xgboost_filter_model/train_filter_v14.py > /dev/null 2>&1
python3 xgboost_filter_model/train_stage2_v14_directional.py > /dev/null 2>&1
result=$(python3 backtest_v14.py 2026-01-01 2026-05-21)

pnl=$(echo "$result" | grep "Net PnL" | awk '{print $4}')
trades=$(echo "$result" | grep "Trades" | awk '{print $3}')
wr=$(echo "$result" | grep "Win Rate" | awk '{print $4}')

echo "Combo $h/$tp/$sl -> PnL: $pnl | Trades: $trades | WinRate: $wr"
