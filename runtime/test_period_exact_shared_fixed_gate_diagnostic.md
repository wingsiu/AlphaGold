# Exact shared trade diagnostic: promoted vs fixed WF

## Shared basis
- Shared trades: `170`
- Start (HKT): `2025-11-28 13:25:00 HKT`
- End (HKT): `2026-04-08 17:45:00 HKT`

## Net edge
- Promoted total pnl: `1249.98`
- Fixed WF total pnl: `996.83`
- Promoted minus fixed: `253.15`
- Avg trade delta: `1.49`

## Agreement
- Same side count: `139`
- Same side %: `81.76470588235294`
- Same exit reason count: `109`
- Same exit reason %: `64.11764705882354`

## Entry probability
- Promoted mean entry prob: `0.9261780862530943`
- Fixed WF mean entry prob: `0.8570665418579555`
- Mean prob delta (promoted - fixed): `0.06911154439513871`

## Top side-pair contributors
- `up -> up` count=110 delta_sum=515.14 delta_mean=4.68
- `down -> down` count=29 delta_sum=22.20 delta_mean=0.77
- `down -> up` count=10 delta_sum=-113.07 delta_mean=-11.31
- `up -> down` count=21 delta_sum=-171.12 delta_mean=-8.15

## Top exit-reason contributors
- `target_hit -> reverse_signal` count=10 delta_sum=472.41 delta_mean=47.24
- `reverse_signal -> stop_loss` count=7 delta_sum=182.19 delta_mean=26.03
- `timeout -> reverse_signal` count=11 delta_sum=101.33 delta_mean=9.21
- `target_hit -> stop_loss` count=1 delta_sum=60.08 delta_mean=60.08
- `timeout -> stop_loss` count=1 delta_sum=7.91 delta_mean=7.91
- `timeout -> timeout` count=5 delta_sum=7.27 delta_mean=1.45
- `stop_loss -> stop_loss` count=58 delta_sum=6.00 delta_mean=0.10
- `reverse_signal -> timeout` count=2 delta_sum=1.59 delta_mean=0.79

## Interpretation
- Positive `delta_sum` means the promoted model outperformed fixed WF in that bucket.
- The strongest explanatory buckets are usually where side choice differs or the exit path differs materially.
- Use the JSON report for the top positive/negative exact-shared trades if you want to inspect the specific timestamps driving the edge.
