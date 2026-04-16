from pathlib import Path
import joblib
import pandas as pd


cache_path = Path('/Users/alpha/Desktop/python/AlphaGold/training/_prep_cache_single_split_aligned/bars/db0303fb00f0ce9735b7a14c.joblib')
out_path = Path('/Users/alpha/Desktop/python/AlphaGold/training/_tmp_check_continuity_report.txt')
payload = joblib.load(cache_path)
bars = payload['bars'].copy().sort_index()
idx = pd.DatetimeIndex(bars.index)

delta = idx.to_series().diff().dropna()
value_counts = delta.value_counts().sort_index()
lines: list[str] = []

lines.append(f'bars_len={len(bars)}')
lines.append(f'start={idx[0].isoformat()}')
lines.append(f'end={idx[-1].isoformat()}')
lines.append('diff_counts:')
for k, v in value_counts.items():
    lines.append(f'  {k}: {int(v)}')

non_1m = delta[delta != pd.Timedelta(minutes=1)]
lines.append(f'non_1m_gap_count={len(non_1m)}')
lines.append('first_20_non_1m_gaps:')
for ts, d in non_1m.head(20).items():
    prev = idx[idx.get_loc(ts) - 1]
    lines.append(f'  prev={prev.isoformat()} curr={ts.isoformat()} diff={d}')

continuous_15 = 0
broken_15 = 0
for i in range(14, len(idx)):
    w = idx[i - 14:i + 1]
    if ((w[1:] - w[:-1]) == pd.Timedelta(minutes=1)).all():
        continuous_15 += 1
    else:
        broken_15 += 1
lines.append(f'continuous_15_windows={continuous_15}')
lines.append(f'broken_15_windows={broken_15}')

continuous_150 = 0
broken_150 = 0
for i in range(149, len(idx)):
    w = idx[i - 149:i + 1]
    if ((w[1:] - w[:-1]) == pd.Timedelta(minutes=1)).all():
        continuous_150 += 1
    else:
        broken_150 += 1
lines.append(f'continuous_150_windows={continuous_150}')
lines.append(f'broken_150_windows={broken_150}')

out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

