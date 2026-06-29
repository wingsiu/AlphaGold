#!/usr/bin/env python3
"""Oil Retrace 1-Min — 1m bar index with rolling 15-min window features.
==================================================================
Uses 1-min bars as the primary index. For each 1-min bar, computes
rolling 15-min window OHLC. All 15-min features recalculated from
these rolling values. Signals fire on 1-min bars.

Config:
  Dlow>60, avgRange3>50, cl-op<-4 (tighter for 1m), wick<8
  TP=50/SL=50, no pattern
"""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_S,NY_E,NY_FC_H,NY_FC_M = 3,12,14,28
MAX_BARS_1M = 900  # 60 15-min bars * 15
RET_TP,RET_SL = 50,50

# Tighter thresholds for 1-min granularity
RET_DLOW,RET_RNG,RET_CHG,RET_WICK = 60,50,-4,8

def load():
    loader=DataLoader();raw=loader.load_data('prices','2024-01-01','2026-06-30')
    raw.index=pd.to_datetime(raw['timestamp'],unit='ms')
    df=pd.DataFrame(index=raw.index)
    for c,src in [('open','openPrice_ask'),('high','highPrice_ask'),('low','lowPrice_ask'),
                   ('close_ask','closePrice_ask'),('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c]=raw[src].astype(float)
    if df.index.tz is None:df.index=df.index.tz_localize('UTC')
    return df

def add_15m_window_features(d1m):
    d = d1m.copy()
    ny = d.index.tz_convert('America/New_York')
    d['ny_h'] = ny.hour; d['ny_m'] = ny.minute
    d['in_sess'] = (d['ny_h'] >= NY_S) & (d['ny_h'] <= NY_E)
    d['t15'] = d.index.floor('15min')
    
    # Rolling 15-min window (partial bar)
    d['o15'] = d.groupby('t15')['open'].transform('first')
    d['h15'] = d.groupby('t15')['high'].cummax()
    d['l15'] = d.groupby('t15')['low'].cummin()
    d['c15'] = d['close_ask']
    d['v15'] = d.groupby('t15')['volume'].cumsum()
    d['wb15'] = np.minimum(d['o15'], d['c15']) - d['l15']
    d['bc15'] = d['c15'] - d['o15']
    
    # Dlow: NY-day low
    d['ny_date'] = ny.date
    d['Dlow'] = d.groupby('ny_date')['low'].transform('min')
    d['ca_dlow'] = d['c15'] - d['Dlow']
    
    # Completed 15-min bars (lagged, for avg_range3)
    c15 = d1m.resample('15min', label='right', closed='right').agg(
        {'high': 'max', 'low': 'min'}).dropna()
    c15['range'] = c15['high'] - c15['low']
    c15['ar3'] = c15['range'].rolling(3, 3).mean()
    c15 = c15.shift(1)  # lag so current 1m sees only completed periods
    d['avg_range3'] = d['t15'].map(c15['ar3'])
    
    # Series that need to be forward-filled after mapping
    d['avg_range3'] = d['avg_range3'].ffill()
    
    return d

def find_signals(d):
    mask = (
        (d['ca_dlow'] > RET_DLOW) &
        (d['avg_range3'] > RET_RNG) &
        (d['bc15'] < RET_CHG) &
        (d['wb15'] < RET_WICK) &
        d['in_sess']
    )
    return [{'idx': i, 'ts': d.index[i]} for i in range(len(d)) if mask.iloc[i]]

def sim(d, sigs):
    """Simulate trades on 1-min bars."""
    pnls = []; trades = []
    in_trade = False; ct = cs = ep = ei = bh = 0; entry_bar = 0
    sig_idx = 0
    
    while sig_idx < len(sigs):
        si = sigs[sig_idx]['idx']
        
        if not in_trade:
            in_trade = True
            ep = d.iloc[si]['close_ask']
            ct = ep + RET_TP; cs = ep - RET_SL
            ei = si; bh = 0; entry_bar = si
            sig_idx += 1; continue
        
        if si - ei > MAX_BARS_1M:
            px = d.iloc[ei + MAX_BARS_1M]['close_bid']
            pnls.append(px - ep); in_trade = False; continue
        
        ex = False
        for j in range(ei + bh + 1, si + 1):
            b = d.iloc[j]
            bp = (b['ny_h'] > NY_FC_H) or (b['ny_h'] == NY_FC_H and b['ny_m'] >= NY_FC_M)
            if bp: pnls.append(b['close_bid'] - ep); in_trade = False; ex = (j == si); break
            if b['high'] >= ct: pnls.append(RET_TP); in_trade = False; ex = (j == si); break
            if b['low'] <= cs: pnls.append(-RET_SL); in_trade = False; ex = (j == si); break
        
        bh = si - ei
        
        if not in_trade:
            if ex: sig_idx += 1; continue
            in_trade = True; ep = d.iloc[si]['close_ask']
            ct = ep + RET_TP; cs = ep - RET_SL
            ei = si; bh = 0; entry_bar = si
            sig_idx += 1; continue
        
        ne = d.iloc[si]['close_ask']
        ct = max(ct, ne + RET_TP)
        cs = cs if cs < ne - RET_SL else max(cs, ne - RET_SL)
        ei = si; bh = 0
        sig_idx += 1
    
    if in_trade:
        last = min(ei + MAX_BARS_1M, len(d) - 1)
        pnls.append(d.iloc[last]['close_bid'] - ep)
    
    return pnls, trades

def stats(pnls):
    if not pnls: return {'t': 0, 'pnl': 0, 'wr': 0, 'pf': 0}
    n = len(pnls); t = sum(pnls); wr = sum(1 for p in pnls if p > 0) / n * 100
    ps = sum(p for p in pnls if p > 0); ns = abs(sum(p for p in pnls if p < 0))
    return {'t': n, 'pnl': t, 'wr': wr, 'pf': ps / ns if ns > 0 else 99}

print('='*72)
print('  OIL RETRACE — 1-Min Bar Index')
print(f'  Config: Dlow>{RET_DLOW} Rng>{RET_RNG} Chg<{RET_CHG} Wick<{RET_WICK} TP={RET_TP}/SL={RET_SL}')
print('='*72)

d1m = load()
print(f'Loaded {len(d1m):,} 1m bars')
d = add_15m_window_features(d1m)
print('Features computed')

# Quick sweep of thresholds
import itertools
print('\nThreshold sweep (1m index):')
print(f'  {"dlow":>5s} {"rng3":>5s} {"bchg":>5s} {"wick":>5s} {"sigs":>6s} {"t":>5s} {"pnl":>8s} {"wr":>5s}')
best_pnl = -999999
for dlow, rng3, bchg, wick in itertools.product([40,60,80,100],[35,50,70],[-3,-4,-5,-8],[4,6,8,12]):
    mask = (d['ca_dlow']>dlow) & (d['avg_range3']>rng3) & (d['bc15']<bchg) & (d['wb15']<wick) & d['in_sess']
    sigs = [i for i in range(len(d)) if mask.iloc[i]]
    if len(sigs) < 30: continue
    pnls, _ = sim(d, [{'idx': i, 'ts': d.index[i]} for i in sigs])
    n = len(pnls); t = sum(pnls); wr = sum(1 for x in pnls if x>0)/n*100
    if t > best_pnl: best_pnl = t
    print(f'  {dlow:>5d} {rng3:>5d} {bchg:>5d} {wick:>5d} {len(sigs):>6d} {n:>5d} {t:>+8.0f} {wr:>4.0f}%')

print(f'\nDONE.')
