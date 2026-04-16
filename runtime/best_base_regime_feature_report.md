# Best-base regime feature report

## Coverage
- Trades analyzed: `1415`
- Start (HKT): `2025-11-28 13:25:00 HKT`
- End (HKT): `2026-04-09 02:27:00 HKT`

## Baseline
- Avg trade: `4.60`
- Profit factor: `1.8391114081283533`
- Total pnl: `6504.07`
- Win rate %: `47.35`

## Top positive regime-feature buckets
- `hour_bucket_hkt = 18:00` trades=28 avg_trade=12.71 uplift=8.11 total_pnl=355.92 pf=4.006588950836195
- `hour_bucket_hkt = 11:00` trades=30 avg_trade=12.36 uplift=7.76 total_pnl=370.74 pf=4.778819692182264
- `swing_15_bucket = (52.948, 293.73]` trades=283 avg_trade=10.77 uplift=6.18 total_pnl=3049.25 pf=2.6728935558554685
- `hour_bucket_hkt = 15:00` trades=25 avg_trade=10.71 uplift=6.11 total_pnl=267.70 pf=3.0727835849787253
- `hour_bucket_hkt = 19:00` trades=38 avg_trade=10.14 uplift=5.54 total_pnl=385.27 pf=3.645540067293782
- `range_150_bucket = (143.8, 434.62]` trades=275 avg_trade=9.89 uplift=5.30 total_pnl=2720.21 pf=2.453251131257979
- `hour_bucket_hkt = 17:00` trades=36 avg_trade=9.64 uplift=5.04 total_pnl=346.89 pf=3.177452765049284
- `hour_bucket_hkt = 07:00` trades=60 avg_trade=9.45 uplift=4.85 total_pnl=567.00 pf=2.8018304309139537
- `volatility_30_bucket = (0.00156, 0.00542]` trades=283 avg_trade=9.36 uplift=4.76 total_pnl=2648.90 pf=2.351307231220508
- `session_hkt = europe_overlap` trades=149 avg_trade=9.11 uplift=4.52 total_pnl=1358.11 pf=3.0185939357907134
- `wr90_bucket = near_low` trades=337 avg_trade=8.47 uplift=3.87 total_pnl=2852.81 pf=2.4683937183769893
- `hour_bucket_hkt = 13:00` trades=54 avg_trade=7.90 uplift=3.30 total_pnl=426.56 pf=2.6491146679038122

## Top negative regime-feature buckets
- `weekday_hkt = Saturday` trades=71 avg_trade=-0.23 uplift=-4.82 total_pnl=-16.00 pf=0.9659146588270459
- `hour_bucket_hkt = 06:00` trades=28 avg_trade=0.16 uplift=-4.44 total_pnl=4.34 pf=1.021229760798318
- `hour_bucket_hkt = 23:00` trades=138 avg_trade=0.25 uplift=-4.35 total_pnl=34.26 pf=1.0370466489327175
- `hour_bucket_hkt = 01:00` trades=52 avg_trade=0.63 uplift=-3.96 total_pnl=32.96 pf=1.1056037935343284
- `bar_return_bucket = [2.0,6.0)` trades=271 avg_trade=0.73 uplift=-3.87 total_pnl=198.22 pf=1.1261045760781743
- `swing_15_bucket = (3.8890000000000002, 23.208]` trades=283 avg_trade=1.02 uplift=-3.58 total_pnl=288.62 pf=1.1940994102100213
- `move_15_bucket = [-15.0,-5.0]` trades=249 avg_trade=1.13 uplift=-3.47 total_pnl=281.42 pf=1.2020998506262217
- `wr90_bucket = near_high` trades=131 avg_trade=1.15 uplift=-3.45 total_pnl=150.07 pf=1.1762088157245902
- `move_30_bucket = [-20.0,-8.0]` trades=219 avg_trade=1.28 uplift=-3.32 total_pnl=279.84 pf=1.2256901599283845
- `move_60_bucket = [12.0,30.0)` trades=179 avg_trade=1.63 uplift=-2.97 total_pnl=292.02 pf=1.2870596099402354
- `hour_bucket_hkt = 22:00` trades=158 avg_trade=1.99 uplift=-2.60 total_pnl=315.15 pf=1.3418038654259141
- `range_150_bucket = (78.634, 100.78]` trades=284 avg_trade=2.11 uplift=-2.49 total_pnl=599.60 pf=1.3912663299531536

## Outcome metadata buckets (not usable as forward filters)
- `exit_reason_bucket = target_hit` trades=73 avg_trade=69.96 uplift=65.36 total_pnl=5106.82 pf=nan
- `exit_reason_bucket = reverse_signal` trades=578 avg_trade=10.03 uplift=5.43 total_pnl=5795.95 pf=7.786270446216393
- `exit_reason_bucket = timeout` trades=266 avg_trade=7.71 uplift=3.12 total_pnl=2051.30 pf=5.588319502538685
- `exit_reason_bucket = stop_loss` trades=498 avg_trade=-12.95 uplift=-17.55 total_pnl=-6450.00 pf=0.0

## Interpretation
- These are reconstructable regime/context proxies, not raw model feature importances.
- The ranked positive/negative lists above are restricted to **entry-available** features.
- Positive buckets indicate conditions where the promoted best-base trades materially outperform the overall average trade.
- Negative buckets indicate conditions that may be worth filtering or deprioritizing in future regime gating experiments.
