#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

P = Path('/Users/alpha/Desktop/python/AlphaGold/runtime/_tmp_weak_zone_gate_probs_v2.json')
obj = json.loads(P.read_text())

for sweep_name in ['entry_signal_prob_sweep', 'last_signal_prob_sweep']:
    print('\n', sweep_name)
    for thr in [0.7, 0.8, 0.9]:
        better = 0
        worse = 0
        flat = 0
        total_delta = 0.0
        for item in obj['per_cell']:
            row = next(r for r in item[sweep_name] if abs(r['threshold'] - thr) < 1e-9)
            delta = float(row['delta_total_pnl_vs_base'])
            total_delta += delta
            if delta > 1e-9:
                better += 1
            elif delta < -1e-9:
                worse += 1
            else:
                flat += 1
        print({'threshold': thr, 'better_cells': better, 'worse_cells': worse, 'flat_cells': flat, 'sum_delta': total_delta})

