#!/usr/bin/env python3
"""Oil Retrace Fast Sweep — fixed TP=60/SL=20, sweep entry params only."""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_S, NY_E, NY_FC_H, NY_FC_M = 3, 12, 14, 28
LONG_MAX_B = 60

def load():
    loader = DataLoader()
    raw = loader.load_data('prices', '2024-01-01', '2026-06-30')
    raw.index = pd.to_datetime(raw['timestamp'], unit='ms')
    df = pd.DataFrame(index=raw.index)
    for c, src in [('open', 'openPrice_ask'), ('high', 'highPrice_ask'), ('low', 'lowPrice_ask'),
                   ('close_ask', 'closePrice_ask'), ('close_bid', 'closePrice_bid'), ('volume', 'lastTradedVolume')]:
        df[c] = raw[src].astype(float)
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    return df

def build_15m(df_1m):
    d = df_1m.resample('15min', label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close_ask': 'last', 'close_bid': 'last', 'volume': 'sum'}).dropna()
    ny = d.index.tz_convert('America/New_York')
    d['ny_h'] = ny.hour; d['ny_m'] = ny.minute
    d['in_sess'] = (d['ny_h'] >= NY_S) & (d['ny_h'] <= NY_E)
    d['Dlow'] = d['low'].groupby(ny.date).transform('min')
    d['range'] = d['high'] - d['low']
    d['avg_range3'] = d['range'].rolling(3, min_periods=3).mean()
    d['bar_up'] = (d['close_ask'] > d['open']).astype(int)
    d['bar_down'] = (d['close_ask'] < d['open']).astype(int)
    d['pat_up_up_down'] = ((d['bar_up'].shift(2) == 1) & (d['bar_up'].shift(1) == 1) & (d['bar_down'] == 1))
    d['wick_below'] = np.minimum(d['open'], d['close_ask']) - d['low']
    d['bar_change'] = d['close_ask'] - d['open']
    d['close_above_dlow'] = d['close_ask'] - d['Dlow']
    return d

def find_signals(d, dlow_min, avg_range_min, bar_chg_max, wick_max, require_pattern=True):
    in_s = d['in_sess']
    o = ((d['close_above_dlow'] > dlow_min) & (d['avg_range3'] > avg_range_min) &
         (d['bar_change'] < bar_chg_max) & (d['wick_below'] < wick_max) & in_s)
    if require_pattern: o = o & d['pat_up_up_down']
    return [{'idx': i} for i in range(len(d)) if o.iloc[i]]

def sim(d, sigs, tp, sl):
    pnls = []; it = False; ct = cs = ep = ei = bh = 0; si = 0
    while si < len(sigs):
        si_i = sigs[si]['idx']
        if not it:
            it = True; ep = d.iloc[si_i]['close_ask']; ct = ep + tp; cs = ep - sl
            ei = si_i; bh = 0; si += 1; continue
        if si_i - ei > LONG_MAX_B:
            pnls.append(d.iloc[ei + LONG_MAX_B]['close_bid'] - ep); it = False; continue
        ex_si = False
        for j in range(ei + bh + 1, si_i + 1):
            b = d.iloc[j]; post = (b['ny_h'] > NY_FC_H) or (b['ny_h'] == NY_FC_H and b['ny_m'] >= NY_FC_M)
            if post:
                pnls.append(b['close_bid'] - ep); it = False
                if j == si_i: ex_si = True
                break
            if b['high'] >= ct:
                pnls.append(tp); it = False
                if j == si_i: ex_si = True
                break
            if b['low'] <= cs:
                pnls.append(-sl); it = False
                if j == si_i: ex_si = True
                break
        bh = si_i - ei
        if not it:
            if ex_si: si += 1; continue
            it = True; ep = d.iloc[si_i]['close_ask']; ct = ep + tp; cs = ep - sl
            ei = si_i; bh = 0; si += 1; continue
        ne = d.iloc[si_i]['close_ask']
        ct = max(ct, ne + tp)
        cs = cs if cs < ne - sl else max(cs, ne - sl)
        ei = si_i; bh = 0; si += 1
    if it: pnls.append(d.iloc[min(ei + LONG_MAX_B, len(d) - 1)]['close_bid'] - ep)
    return pnls

def stats(pnls):
    if not pnls: return {'t': 0, 'pnl': 0, 'wr': 0, 'pf': 0}
    n = len(pnls); t = sum(pnls); wr = sum(1 for p in pnls if p > 0) / n * 100
    ps = sum(p for p in pnls if p > 0); ns = abs(sum(p for p in pnls if p < 0))
    return {'t': n, 'pnl': t, 'wr': wr, 'pf': ps / ns if ns > 0 else 99}

print('=' * 60)
print('  OIL RETRACE — Fast Sweep (TP=60, SL=20)')
print('=' * 60)
d1m = load(); d15 = build_15m(d1m)

# With pattern
print(f'\n  {"Dlow":>5s} {"Rng":>5s} {"Chg":>5s} {"Wick":>5s} {"Sigs":>5s} {"T":>5s} {"PnL":>9s} {"WR":>6s} {"PF":>6s}')
results = []
for dlow in [40, 50, 60, 70, 80, 100]:
    for rng_min in [20, 30, 35, 40, 50]:
        for chg in [-5, -8, -10, -15, -20]:
            for wick in [4, 6, 8, 10, 15]:
                sigs = find_signals(d15, dlow, rng_min, chg, wick, True)
                if len(sigs) == 0: continue
                pnls = sim(d15, sigs, 60, 20)
                s = stats(pnls)
                results.append({'dlow': dlow, 'rng': rng_min, 'chg': chg, 'wick': wick,
                                'sigs': len(sigs), 't': s['t'], 'pnl': s['pnl'], 'wr': s['wr'], 'pf': s['pf']})

rdf = pd.DataFrame(results).sort_values('pnl', ascending=False)
for _, r in rdf.head(20).iterrows():
    print(f'  {int(r["dlow"]):>5d} {int(r["rng"]):>5d} {int(r["chg"]):>5d} {int(r["wick"]):>5d} '
          f'{int(r["sigs"]):>5d} {int(r["t"]):>5d} {int(r["pnl"]):>+9.0f} {r["wr"]:>5.1f}% {r["pf"]:>5.2f}')

# Without pattern
print(f'\nTop 10 WITHOUT up-up-down:')
results_np = []
for dlow in [40, 60, 80, 100]:
    for rng_min in [20, 35, 50]:
        for chg in [-5, -10, -15, -20]:
            for wick in [4, 8, 12, 16]:
                sigs = find_signals(d15, dlow, rng_min, chg, wick, False)
                if len(sigs) == 0: continue
                pnls = sim(d15, sigs, 60, 20)
                s = stats(pnls)
                results_np.append({'dlow': dlow, 'rng': rng_min, 'chg': chg, 'wick': wick,
                                   'sigs': len(sigs), 't': s['t'], 'pnl': s['pnl'], 'wr': s['wr'], 'pf': s['pf']})
rdf2 = pd.DataFrame(results_np).sort_values('pnl', ascending=False)
for _, r in rdf2.head(10).iterrows():
    print(f'  {int(r["dlow"]):>5d} {int(r["rng"]):>5d} {int(r["chg"]):>5d} {int(r["wick"]):>5d} '
          f'{int(r["sigs"]):>5d} {int(r["t"]):>5d} {int(r["pnl"]):>+9.0f} {r["wr"]:>5.1f}% {r["pf"]:>5.2f}')

best = rdf.iloc[0] if len(rdf) > 0 else None
if best is not None and best['pnl'] > 0:
    print(f'\nBEST: Dlow>{int(best["dlow"])} Rng>{int(best["rng"])} Chg<{int(best["chg"])} Wick<{int(best["wick"])} '
          f'PnL={int(best["pnl"]):+d} WR={best["wr"]:.1f}% PF={best["pf"]:.2f}')
elif best is not None:
    print(f'\nBest still negative ({int(best["pnl"]):+d}) — retrace not viable for oil.')
print('DONE.')
