# Saved test-period overlap comparison

## Comparison basis
- Type: common saved `entry_time` window across all three models
- Window start (HKT): `2025-11-28 13:25:00 HKT`
- Window end (HKT): `2026-04-09 02:27:00 HKT`
- Exact shared entry timestamps across all three: `170`

## Ranking by overlap-window total pnl
1. `promoted_single_split` pnl=6504.07 pf=1.8391114081283533 trades=1415
2. `walk_forward_fixed` pnl=3739.63 pf=1.4848633755793972 trades=1327
3. `walk_forward_prob_sweep` pnl=3212.78 pf=1.4264471059208934 trades=1293

## Per-model overlap results
- **promoted_single_split**
  - overlap trades: 1415
  - overlap total pnl: 6504.07
  - overlap profit factor: 1.8391114081283533
  - overlap avg day: 91.60661971830991
  - overlap positive days %: 67.6056338028169
  - long/up: trades=954 pnl=4384.30
  - short/down: trades=461 pnl=2119.77
  - exits: {'reverse_signal': 578, 'stop_loss': 498, 'timeout': 266, 'target_hit': 73}
- **walk_forward_fixed**
  - overlap trades: 1327
  - overlap total pnl: 3739.63
  - overlap profit factor: 1.4848633755793972
  - overlap avg day: 59.359206349206325
  - overlap positive days %: 76.19047619047619
  - long/up: trades=755 pnl=2729.67
  - short/down: trades=572 pnl=1009.96
  - exits: {'reverse_signal': 584, 'stop_loss': 433, 'target_hit': 228, 'timeout': 82}
- **walk_forward_prob_sweep**
  - overlap trades: 1293
  - overlap total pnl: 3212.78
  - overlap profit factor: 1.4264471059208934
  - overlap avg day: 49.42738461538465
  - overlap positive days %: 72.3076923076923
  - long/up: trades=757 pnl=2315.12
  - short/down: trades=536 pnl=897.66
  - exits: {'reverse_signal': 563, 'stop_loss': 431, 'target_hit': 221, 'timeout': 78}

## Practical suggestion
- Best overlap-window pnl: `promoted_single_split`
- Suggested stance: `keep_current_promoted_artifact_as_single_split_reference_but_use_fixed_gate_walk_forward_as_main_robustness_benchmark`

## Interpretation
- Use this report to compare the saved **test period**, not the short post-period replay.
- If single-split wins on overlap pnl but fixed-gate WF remains close, treat single-split as the stronger upside reference and fixed-gate WF as the stronger robustness benchmark.
- The probability-sweep WF variant should only be preferred if it clearly improves both overlap pnl and robustness, not just selected cycles.
