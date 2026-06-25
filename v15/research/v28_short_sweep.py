#!/usr/bin/env python3
"""v28 Short Sweep — CumVol, WR Entry, TP/SL
===============================================
Sweep CumVol [5k,10k,15k,20k], WR entry [-10,-15,-20,-25], TP/SL combos.
Long model best: WR<-80, CumVol≥15k, TP=80/SL=30, advance target.
Goal: find best short config that adds to long, or confirm short doesn't work.
"""

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_SESSION_START = 3; NY_SESSION_END = 12
NY_FORCE_CLOSE_H = 14; NY_FORCE_CLOSE_M = 28
MAX_BARS = 60; EP_BARS_MIN = 3
SHORT_RECOVERY = -80; SHORT_WEAK = -20; WEAKNESS_TIMEOUT = 12

def load(s='2024-01-01', e='2026-06-30'):
    loader=DataLoader();raw=loader.load_data(table_name='prices',start_date=s,end_date=e)
    raw.index=pd.to_datetime(raw['timestamp'],unit='ms')
    df=pd.DataFrame(index=raw.index)
    for c,src in [('open','openPrice_ask'),('high','highPrice_ask'),('low','lowPrice_ask'),
                   ('close_ask','closePrice_ask'),('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c]=raw[src].astype(float)
    if df.index.tz is None: df.index=df.index.tz_localize('UTC')
    return df

def build(df):
    d=df.resample('15min',label='right',closed='right').agg({'open':'first','high':'max','low':'min','close_ask':'last','close_bid':'last','volume':'sum'}).dropna()
    n=14;hh=d['high'].rolling(n).max();ll=d['low'].rolling(n).min()
    d['wr']=((hh-d['close_ask'])/(hh-ll+0.01))*-100
    ny=d.index.tz_convert('America/New_York'); d['ny_h']=ny.hour; d['ny_m']=ny.minute
    d['in_sess']=(d['ny_h']>=NY_SESSION_START)&(d['ny_h']<=NY_SESSION_END)
    return d

def find_short_signals(d, wr_entry, min_cv):
    in_s=d['in_sess']; signal_bars=(d['wr']>wr_entry)&in_s
    sigs=[]; in_ep=False; cv=0.0; bc=0
    for i in range(len(d)):
        if signal_bars.iloc[i]:
            if not in_ep: ep_start=i; cv=0.0; bc=0
            in_ep=True; cv+=d['volume'].iloc[i]; bc+=1
        else:
            if in_ep:
                ebi=i
                if ebi<len(d)-1 and in_s.iloc[ebi] and cv>=min_cv and bc>=EP_BARS_MIN:
                    sigs.append({'idx':ebi,'cv':cv,'bc':bc})
                in_ep=False; cv=0.0; bc=0
    return sigs

def sim_short(ei, d, tp, sl):
    ep=d.iloc[ei]['close_ask']; h=min(MAX_BARS,len(d)-ei-1)
    reached=False; wc=0
    for i in range(1,h+1):
        b=d.iloc[ei+i]
        post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
        if post: return ep-b['close_bid']
        if b['low']<=ep-tp: return tp
        if b['high']>=ep+sl: return -sl
        if b['wr']<=SHORT_RECOVERY: reached=True
        if b['wr']>SHORT_WEAK: wc+=1
        else: wc=0
        if reached and post: return ep-b['close_bid']
        if not reached and wc>=WEAKNESS_TIMEOUT: return ep-b['close_bid']
    return ep-d.iloc[ei+h]['close_bid']

def stats(pnls):
    if not pnls: return {'t':0,'pnl':0,'wr':0,'pf':0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'t':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99}

d1=load(); d15=build(d1)
print(f'Data: {len(d1):,} 1m → {len(d15):,} 15m')

# Store best combos
best_configs=[]

for cv in [5000,10000,15000,20000,25000]:
    for wr in [-10,-15,-20,-25,-30]:
        sigs=find_short_signals(d15,wr,cv)
        if len(sigs)<20: continue
        for tp in [60,70,80,90,100,120]:
            for sl in [25,30,35,40]:
                pnls=[sim_short(s['idx'],d15,tp,sl) for s in sigs]
                s=stats(pnls)
                if len(best_configs)<30 or s['pnl']>(min(c['s']['pnl'] for c in best_configs) if best_configs else -9999):
                    best_configs.append({'wr':wr,'cv':cv,'tp':tp,'sl':sl,'trades':s['t'],'s':s})
                    best_configs.sort(key=lambda x:-x['s']['pnl'])
                    if len(best_configs)>30: best_configs=best_configs[:30]

# Short + Long combined sweep on top 5 short configs
# Long baseline
LONG_ENTRY=-80; LONG_RECOVERY=-20; LONG_WEAK=-50
def find_long_signals(d, min_cv=15000):
    in_s=d['in_sess']; o=(d['wr']<LONG_ENTRY)&in_s
    sigs=[]; ie=False; cv=0.0; bc=0
    for i in range(len(d)):
        if o.iloc[i]:
            if not ie: ep_start=i; cv=0.0; bc=0
            ie=True; cv+=d['volume'].iloc[i]; bc+=1
        else:
            if ie:
                ebi=i
                if ebi<len(d)-1 and in_s.iloc[ebi] and cv>=min_cv and bc>=EP_BARS_MIN:
                    sigs.append({'idx':ebi,'cv':cv,'bc':bc})
                ie=False; cv=0.0; bc=0
    return sigs

def sim_long_with_advance(sigs, d, tp=80, sl=30):
    pnls=[]; in_trade=False; ct=0; cs=0; ep=0; ei=0; bh=0; reached=False; wc=0; sig_idx=0
    while sig_idx<len(sigs):
        si=sigs[sig_idx]['idx']
        if not in_trade:
            in_trade=True; ep=d.iloc[si]['close_ask']; ct=ep+tp; cs=ep-sl
            ei=si; bh=0; reached=False; wc=0; sig_idx+=1; continue
        if si-ei>MAX_BARS: pnls.append(d.iloc[ei+MAX_BARS]['close_bid']-ep); in_trade=False; continue
        for j in range(ei+bh+1, si+1):
            b=d.iloc[j]
            post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
            if post: pnls.append(b['close_bid']-ep); in_trade=False; break
            if b['high']>=ct: pnls.append(ct-ep); in_trade=False; break
            if b['low']<=cs: pnls.append(cs-ep); in_trade=False; break
            if b['wr']>=LONG_RECOVERY: reached=True
            if b['wr']<LONG_WEAK: wc+=1
            else: wc=0
            if reached and post: pnls.append(b['close_bid']-ep); in_trade=False; break
            if not reached and wc>=WEAKNESS_TIMEOUT: pnls.append(b['close_bid']-ep); in_trade=False; break
        bh=si-ei
        if not in_trade:
            in_trade=True; ep=d.iloc[si]['close_ask']; ct=ep+tp; cs=ep-sl
            ei=si; bh=0; reached=False; wc=0; sig_idx+=1; continue
        ne=d.iloc[si]['close_ask']; ct=ne+tp; cs=min(cs,ne-sl)
        ei=si; bh=0; reached=False; wc=0; sig_idx+=1
    if in_trade:
        last=min(ei+MAX_BARS,len(d)-1); pnls.append(d.iloc[last]['close_bid']-ep)
    return pnls

long_sigs=find_long_signals(d15)
pnls_l_adv=sim_long_with_advance(long_sigs,d15)
sl_adv=stats(pnls_l_adv)

def sim_short_with_advance(sigs, d, tp, sl):
    pnls=[]; in_trade=False; ct=0; cs=0; ep=0; ei=0; bh=0; reached=False; wc=0; sig_idx=0
    while sig_idx<len(sigs):
        si=sigs[sig_idx]['idx']
        if not in_trade:
            in_trade=True; ep=d.iloc[si]['close_ask']; ct=ep-tp; cs=ep+sl
            ei=si; bh=0; reached=False; wc=0; sig_idx+=1; continue
        if si-ei>MAX_BARS: pnls.append(ep-d.iloc[ei+MAX_BARS]['close_bid']); in_trade=False; continue
        for j in range(ei+bh+1, si+1):
            b=d.iloc[j]
            post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
            if post: pnls.append(ep-b['close_bid']); in_trade=False; break
            if b['low']<=ct: pnls.append(ep-ct); in_trade=False; break
            if b['high']>=cs: pnls.append(ep-cs); in_trade=False; break
            if b['wr']<=SHORT_RECOVERY: reached=True
            if b['wr']>SHORT_WEAK: wc+=1
            else: wc=0
            if reached and post: pnls.append(ep-b['close_bid']); in_trade=False; break
            if not reached and wc>=WEAKNESS_TIMEOUT: pnls.append(ep-b['close_bid']); in_trade=False; break
        bh=si-ei
        if not in_trade:
            in_trade=True; ep=d.iloc[si]['close_ask']; ct=ep-tp; cs=ep+sl
            ei=si; bh=0; reached=False; wc=0; sig_idx+=1; continue
        ne=d.iloc[si]['close_ask']; ct=ne-tp; cs=min(cs,ne+sl)
        ei=si; bh=0; reached=False; wc=0; sig_idx+=1
    if in_trade:
        last=min(ei+MAX_BARS,len(d)-1); pnls.append(ep-d.iloc[last]['close_bid'])
    return pnls

# ===== SHORT SWEEP (top 30) =====
print(f'\n{"="*100}')
print(f'  SHORT SWEEP — Top 30 by PnL (single trade, no advance)')
print(f'  {"WR":>5s} {"CumV":>7s} {"TP":>4s} {"SL":>4s} {"T":>5s} {"PnL":>8s} {"WR%":>7s} {"PF":>6s}')
print(f'  {"-"*55}')
for c in best_configs[:30]:
    print(f'  {c["wr"]:>+4d} {c["cv"]:>7d} {c["tp"]:>4d} {c["sl"]:>4d} {c["trades"]:>5d} {c["s"]["pnl"]:>+8.0f} {c["s"]["wr"]:>6.1f}% {c["s"]["pf"]:>5.2f}')

# ===== TOP 5 SHORT + LONG COMBINED =====
print(f'\n{"="*100}')
print(f'  TOP 5 SHORT + LONG COMBINED (both with advance target)')
print(f'  {"WR":>5s} {"CumV":>7s} {"TP":>4s} {"SL":>4s} {"ShortT":>7s} {"Short":>8s} {"Long":>8s} {"Combined":>10s} {"WR%":>7s} {"PF":>6s}')
print(f'  {"-"*80}')
baseline_long=sl_adv['pnl']; baseline_long_t=sl_adv['t']
print(f'  Baseline Long: {baseline_long_t}t, {baseline_long:+.0f}pts, WR={sl_adv["wr"]:.1f}%, PF={sl_adv["pf"]:.2f}')
for c in best_configs[:5]:
    ss=find_short_signals(d15,c['wr'],c['cv'])
    s_pnls=sim_short_with_advance(ss,d15,c['tp'],c['sl'])
    ss_stats=stats(s_pnls)
    combined=pnls_l_adv+s_pnls
    sc=stats(combined)
    print(f'  {c["wr"]:>+4d} {c["cv"]:>7d} {c["tp"]:>4d} {c["sl"]:>4d} {ss_stats["t"]:>7d} {ss_stats["pnl"]:>+8.0f} {baseline_long:>+8.0f} {sc["pnl"]:>+10.0f} {sc["wr"]:>6.1f}% {sc["pf"]:>5.2f}')

# ===== ALSO TRY SHORT WITH WR ENTRY -80 (mirror long exactly) =====
print(f'\n{"="*100}')
print(f'  SHORT: WR < -80 (mirror of long, oversold → short continuation?)')
print(f'  TP/SL sweep on WR<-80 short signals')
for cv in [5000,10000,15000]:
    sigs_c=find_short_signals(d15,-80,cv)  # using long entry as short signal
    if len(sigs_c)<10: continue
    for tp in [60,80,100]:
        for sl in [25,30,40]:
            pnls=[sim_short(s['idx'],d15,tp,sl) for s in sigs_c]
            s=stats(pnls)
            print(f'  WR<-80 CV≥{cv:>5d} TP={tp}/SL={sl}: {s["t"]}t, {s["pnl"]:+.0f}pts, WR={s["wr"]:.0f}%, PF={s["pf"]:.2f}')

print('\nDONE.')
