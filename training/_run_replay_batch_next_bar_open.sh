#!/usr/bin/env bash
set -euo pipefail

cd /Users/alpha/Desktop/python/AlphaGold

python3 -u training/replay_wf_batch.py \
  --logic-label next_bar_open_v1

