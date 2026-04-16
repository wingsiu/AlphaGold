# Saved test-period exact shared comparison

## Comparison basis
- Type: exact shared `entry_time` timestamps across all three models
- Shared entry count: `170`
- Shared window start (HKT): `2025-11-28 13:25:00 HKT`
- Shared window end (HKT): `2026-04-08 17:45:00 HKT`

## Ranking by exact-shared total pnl
1. `promoted_single_split` pnl=1249.98 pf=2.1503377445657197 trades=170
2. `walk_forward_fixed` pnl=996.83 pf=1.9926805951124351 trades=170
3. `walk_forward_prob_sweep` pnl=855.67 pf=1.838883932510475 trades=170

## Per-model exact-shared results
- **promoted_single_split**
  - exact shared trades: 170
  - total pnl: 1249.98
  - profit factor: 2.1503377445657197
  - avg day: 34.721666666666756
  - positive days %: 58.333333333333336
  - long/up: trades=131 pnl=1213.24
  - short/down: trades=39 pnl=36.74
  - exits: {'stop_loss': 75, 'reverse_signal': 53, 'timeout': 27, 'target_hit': 15}
- **walk_forward_fixed**
  - exact shared trades: 170
  - total pnl: 996.83
  - profit factor: 1.9926805951124351
  - avg day: 27.689722222222372
  - positive days %: 61.111111111111114
  - long/up: trades=120 pnl=822.18
  - short/down: trades=50 pnl=174.65
  - exits: {'reverse_signal': 77, 'stop_loss': 67, 'target_hit': 18, 'timeout': 8}
- **walk_forward_prob_sweep**
  - exact shared trades: 170
  - total pnl: 855.67
  - profit factor: 1.838883932510475
  - avg day: 23.76861111111134
  - positive days %: 66.66666666666666
  - long/up: trades=123 pnl=781.53
  - short/down: trades=47 pnl=74.14
  - exits: {'reverse_signal': 78, 'stop_loss': 68, 'target_hit': 16, 'timeout': 8}

## Practical suggestion
- Best exact-shared pnl: `promoted_single_split`
- Suggested stance: `promoted_single_split_still_leads_on_exact_shared_subset`

## Interpretation
- This is stricter than the common-window report because it compares only identical saved `entry_time` timestamps.
- If the promoted single-split still leads here, that strengthens the case that its test-period edge is not just coming from extra windows outside the shared subset.
- If fixed-gate remains closer on stability but not pnl, keep using it as the main robustness benchmark rather than replacing the promoted artifact outright.
