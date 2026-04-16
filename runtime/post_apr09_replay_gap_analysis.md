# Post-Apr-09 replay gap analysis

## Current status confirmed
- Bot mode: `signal_only`
- Signal family: `best_base_state`
- Promoted model: `training/_tmp_feature_single_split_aligned_state_stop_sweep/w150_h25_thr0.008_d15_lt0.006_st0.008_ls12_ss18_r0_f2.5_p10.48_p20.5_sf.joblib`
- Open position: `None`
- Latest prediction-cache bucket (HKT): `2026-04-15 17:41:00 HKT`
- Cached rows in saved status: `10351`
- Candidate samples in saved status: `3624`

## Key finding: saved training backtests do not cover the weak replay window
- Corrected single-split trades end at (HKT): `2026-04-09 02:47:00 HKT`
- Post-Apr-09 replay window starts at (HKT): `2026-04-09 08:00:00 HKT`
- Direct overlap exists: `False`

So the negative replay after `2026-04-09` is **not** contradicted by the saved corrected backtest; it is simply outside that saved trade file's coverage.

## Saved whole-period health
- Corrected single-split total pnl: `6504.070000000004`
- Fixed-gate walk-forward total pnl: `3743.6399999999985`
- Fixed-gate walk-forward profit factor: `1.4853832938964695`
- Probability-sweep walk-forward total pnl: `3213.640000000002`
- Probability-sweep walk-forward profit factor: `1.426561257686994`

## Immediate pre-replay window from corrected backtest (`2026-04-02` .. `2026-04-09`)
- Trades: `67`
- Total pnl: `127.68`
- Profit factor: `1.316462598522769`
- Side mix: `{'up': 37, 'down': 30}`
- Exit mix: `{'timeout': 23, 'stop_loss': 22, 'reverse_signal': 18, 'target_hit': 4}`
- Coverage end (HKT): `2026-04-09 02:47:00 HKT`

## Aligned replay-window comparison (`2026-04-09 08:00:00 HKT` .. `2026-04-16 07:59:59 HKT`)
- **promoted_single_split**
  - model: `training/_tmp_feature_single_split_aligned_state_stop_sweep/w150_h25_thr0.008_d15_lt0.006_st0.008_ls12_ss18_r0_f2.5_p10.48_p20.5_sf.joblib`
  - signals: tradable=28 up=28 down=0 flat=1184
  - avg tradable prob: 0.9955
  - trades: 8
  - total pnl: -15.51
  - profit factor: 0.6365127724397048
  - exits: {'timeout': 5, 'stop_loss': 3}
- **walk_forward_fixed_cycle10**
  - model: `training/backtest_model_best_base_wf_10cycles_cycles/cycle_10.joblib`
  - signals: tradable=30 up=22 down=8 flat=1182
  - avg tradable prob: 0.7470
  - trades: 9
  - total pnl: -13.27
  - profit factor: 0.7218029350104911
  - exits: {'timeout': 5, 'stop_loss': 2, 'reverse_signal': 2}
- **walk_forward_prob_cycle10**
  - model: `training/backtest_model_best_base_wf_10cycles_prob_sweep_cycles/cycle_10.joblib`
  - signals: tradable=26 up=20 down=6 flat=1186
  - avg tradable prob: 0.7429
  - trades: 7
  - total pnl: -8.72
  - profit factor: 0.7114493712773152
  - exits: {'timeout': 4, 'stop_loss': 2, 'reverse_signal': 1}

## Conclusion
- The promoted artifact is still the strongest **saved single-split benchmark**, but it is **not fully validated for live deployment**.
- The recent weakness looks **real enough to respect**, because the latest walk-forward cycle alternatives were also negative on the same slice.
- If forced to choose a robustness framework, prefer **fixed-gate walk-forward** over the probability-sweep variant.
- But there is **not yet a clearly superior replacement artifact** from this aligned recent-slice check.

## Most sensible next step
1. Keep the current promoted artifact as the bot's default **research/signal-only** model.
2. Do **not** treat it as live-ready.
3. Extend aligned comparison coverage through the same recent dates with training-side artifacts / overlap windows before any live-execution build-out becomes primary.
