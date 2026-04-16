#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

P = Path('/Users/alpha/Desktop/python/AlphaGold/runtime/_tmp_weak_zone_gate_probs_v2.json')
obj = json.loads(P.read_text())

print('baseline_gates', obj['baseline_gates'])
print('weak_cells_count', len(obj['weak_cells']))
print('aggregate_weak_union', obj['aggregate_union']['weak_cells_union'])
print('aggregate_non_weak_union', obj['aggregate_union']['non_weak_union'])

print('\nentry_sweep_weak_union')
for row in obj['aggregate_union']['weak_entry_signal_prob_sweep']:
    print(row)

print('\nlast_sweep_weak_union')
for row in obj['aggregate_union']['weak_last_signal_prob_sweep']:
    print(row)

print('\nfirst_4_cells')
for item in obj['per_cell'][:4]:
    print(item['cell'])
    print(' summary=', item['summary'])
    entry70 = next(r for r in item['entry_signal_prob_sweep'] if abs(r['threshold'] - 0.70) < 1e-9)
    entry80 = next(r for r in item['entry_signal_prob_sweep'] if abs(r['threshold'] - 0.80) < 1e-9)
    last70 = next(r for r in item['last_signal_prob_sweep'] if abs(r['threshold'] - 0.70) < 1e-9)
    last80 = next(r for r in item['last_signal_prob_sweep'] if abs(r['threshold'] - 0.80) < 1e-9)
    print(' entry70=', entry70)
    print(' entry80=', entry80)
    print(' last70=', last70)
    print(' last80=', last80)
    print()

