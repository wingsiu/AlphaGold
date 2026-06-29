#!/usr/bin/env python3
"""v28 Short Sweep V2 — WR Entry + CumVol sweep with correct bid/ask pricing
=============================================================================
Short: sell at bid (close_bid), buy back at ask (close_ask) → pays spread
Single trade only, no advance. TP=80/SL=30, TP=100/SL=40 options.
WR entries: -15, -20, -25, -30, -35, -40, -50, -60, -70, -80
CumVol: 5000, 10000, 15000, 20000
"""

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_S=3;NY_E=12;NY_FC_H=14;NY_FC_M=28;MAX_B=60;EP_MIN=3

def load(s='2024-01-01',e='2026-06-30'):
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
    ny=d.index.tz_convert('America/New_York');d['ny_h']=ny.hour;d['ny_m']=ny.minute
    d['in_sess']=(d['ny_h']>=NY_S)&(d['ny_h']<=NY_E)
    return d

def find_short_sigs(d,wr_entry,min_cv):
    in_s=d['in_sess'];sb=(d['wr']>wr_entry)&in_s
    sigs=[];ie=False;cv=0.0;bc=0
    for i in range(len(d)):
        if sb.iloc[i]:
            if not ie:ep_s=i;cv=0.0;bc=0
            ie=True;cv+=d['volume'].iloc[i];bc+=1
        else:
            if ie:
                ebi=i
                if ebi<len(d)-1 and in_s.iloc[ebi] and cv>=min_cv and bc>=EP_MIN:
                    sigs.append({'idx':ebi,'cv':cv,'bc':bc})
                ie=False;cv=0.0;bc=0
    return sigs

def sim_short(ei,d,tp=80,sl=30):
    ep=d.iloc[ei]['close_bid']
    h=min(MAX_B,len(d)-ei-1)
    for i in range(1,h+1):
        b=d.iloc[ei+i]
        post=(b['ny_h']>NY_FC_H)or(b['ny_h']==NY_FC_H and b['ny_m']>=NY_FC_M)
        if post: return ep-b['close_ask']
        if b['low']<=ep-tp: return tp
        if b['high']>=ep+sl: return -sl
    return ep-d.iloc[ei+h]['close_ask']

def stats(pnls):
    if not pnls: return {'t':0,'pnl':0,'wr':0,'pf':0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'t':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99}

d1=load();d15=build(d1)

print('='*80)
print('  V28 SHORT SWEEP V2 — WR Entry + CumVol + TP/SL')
print('  Short: sell at BID, buy back at ASK (pays spread)')
print('  NY 03-12 session, force-close 14:28, EpBars>=3')
print('='*80)

wr_entries=[-15,-20,-25,-30,-35,-40,-50,-60,-70,-80]
cumvols=[5000,10000,15000,20000]
tp_sl_pairs=[(80,30),(80,40),(100,30),(100,40),(120,40)]

# Header
print(f'\n  {"WR>":>5s} {"CV>=":>6s} {"TP/SL":>7s} {"T":>5s} {"PnL":>8s} {"WR":>6s} {"PF":>5s} {"Avg":>7s}')
print(f'  {"-"*51}')

best_pnl=-9999;best_cfg=''

for wr in wr_entries:
    for cv in cumvols:
        sigs=find_short_sigs(d15,wr,cv)
        if len(sigs)<15: continue
        for tp,sl in tp_sl_pairs:
            pnls=[sim_short(s['idx'],d15,tp,sl) for s in sigs]
            s=stats(pnls)
            if s['pnl']>best_pnl:
                best_pnl=s['pnl'];best_cfg=f'WR>{wr} CV>={cv} TP={tp}/SL={sl}'
            label = str(wr)
            tp_sl_str = str(tp) + "/" + str(sl)
            print(f'  {label:>5s} {cv:>6d} {tp_sl_str:>7s} {s["t"]:>5d} {s["pnl"]:>+8.0f} {s["wr"]:>5.0f}% {s["pf"]:>4.2f} {s["pnl"]/max(1,s["t"]):>+7.1f}')

print(f'\n  {"="*60}')
print(f'  BEST SHORT: {best_cfg}')
print(f'  Best PnL:   {best_pnl:+.0f} pts')
print()
print(f'  For reference — LONG baseline (with advance): 279t, +2985pts, WR=50.2%, PF=1.26')
print(f'  {"="*60}')
print('\nDONE.')
