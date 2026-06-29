#!/usr/bin/env python3
"""Oil Retrace Pattern — similar to gold retrace logic.
Conditions (all on 15m bars):
  1. close - Dlow > 60 (close well above day low)
  2. Pattern: up, up, down on last 3 bars (bar=up if close>open)
  3. avgRange(3 bars) > 35 (average high-low range)
  4. close - open < -10 (current bar is a strong down bar)
  5. min(bar open, close) - low < 8 (bar barely wicked below)
  
  Entry: long at close_ask, TP/SL with advance target, NY close at 14:28.
"""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_S=3; NY_E=12; NY_FC_H=14; NY_FC_M=28
LONG_MAX_B=60
LONG_RECOVERY=-20; LONG_WEAK=-50; LONG_WT=12

def load():
    loader=DataLoader();raw=loader.load_data('prices','2024-01-01','2026-06-30')
    raw.index=pd.to_datetime(raw['timestamp'],unit='ms')
    df=pd.DataFrame(index=raw.index)
    for c,src in [('open','openPrice_ask'),('high','highPrice_ask'),('low','lowPrice_ask'),
                   ('close_ask','closePrice_ask'),('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c]=raw[src].astype(float)
    if df.index.tz is None: df.index=df.index.tz_localize('UTC')
    return df

def build_15m(df_1m):
    d=df_1m.resample('15min',label='right',closed='right').agg(
        {'open':'first','high':'max','low':'min','close_ask':'last','close_bid':'last','volume':'sum'}).dropna()
    ny=d.index.tz_convert('America/New_York');d['ny_h']=ny.hour;d['ny_m']=ny.minute
    d['in_sess']=(d['ny_h']>=NY_S)&(d['ny_h']<=NY_E)
    ny=d.index.tz_convert('America/New_York')
    d['Dlow']=d['low'].groupby(ny.date).transform('min')
    d['range']=d['high']-d['low']
    d['avg_range3']=d['range'].rolling(3,min_periods=3).mean()
    d['bar_up']=(d['close_ask']>d['open']).astype(int)
    d['bar_down']=(d['close_ask']<d['open']).astype(int)
    d['pat_up_up_down']=((d['bar_up'].shift(2)==1)&(d['bar_up'].shift(1)==1)&(d['bar_down']==1))
    return d

def find_signals(d):
    in_s=d['in_sess']
    o=((d['close_ask']-d['Dlow'])>60)& \
      (d['pat_up_up_down'])& \
      (d['avg_range3']>35)& \
      ((d['close_ask']-d['open'])<-10)& \
      (np.minimum(d['open'],d['close_ask'])-d['low']<8)& \
      in_s
    sigs=[]
    for i in range(len(d)):
        if o.iloc[i]:sigs.append({'idx':i})
    return sigs

def sim_long_advance(d15,sigs,tp_val,sl_val):
    pnls=[]
    in_trade=False;ct=0;cs=0;ep=0;ei=0;bh=0
    reached=False;wc=0;sig_idx=0
    while sig_idx<len(sigs):
        si=sigs[sig_idx]['idx']
        if not in_trade:
            in_trade=True;ep=d15.iloc[si]['close_ask'];ct=ep+tp_val;cs=ep-sl_val
            ei=si;bh=0;reached=False;wc=0
            sig_idx+=1;continue
        if si-ei>LONG_MAX_B:
            px=d15.iloc[ei+LONG_MAX_B]['close_bid'];pnls.append(px-ep);in_trade=False;continue
        exit_at_si=False
        for j in range(ei+bh+1,si+1):
            b=d15.iloc[j];post=(b['ny_h']>NY_FC_H)or(b['ny_h']==NY_FC_H and b['ny_m']>=NY_FC_M)
            if post:
                px=b['close_bid'];pnls.append(px-ep);in_trade=False
                if j==si:exit_at_si=True
                break
            if b['high']>=ct:
                pnls.append(tp_val);in_trade=False
                if j==si:exit_at_si=True
                break
            if b['low']<=cs:
                pnls.append(-sl_val);in_trade=False
                if j==si:exit_at_si=True
                break
            if b.get('wr',0)>=LONG_RECOVERY:reached=True
            if b.get('wr',0)<LONG_WEAK:wc+=1
            else:wc=0
            if reached and post:
                px=b['close_bid'];pnls.append(px-ep);in_trade=False
                if j==si:exit_at_si=True
                break
            if not reached and wc>=LONG_WT:
                px=b['close_bid'];pnls.append(px-ep);in_trade=False
                if j==si:exit_at_si=True
                break
        bh=si-ei
        if not in_trade:
            if exit_at_si:sig_idx+=1;continue
            in_trade=True;ep=d15.iloc[si]['close_ask'];ct=ep+tp_val;cs=ep-sl_val
            ei=si;bh=0;reached=False;wc=0;sig_idx+=1;continue
        ne=d15.iloc[si]['close_ask']
        ct=max(ct,ne+tp_val)
        cs=cs if cs<ne-sl_val else max(cs,ne-sl_val)
        ei=si;bh=0;reached=False;wc=0;sig_idx+=1
    if in_trade:
        last=min(ei+LONG_MAX_B,len(d15)-1)
        px=d15.iloc[last]['close_bid'];pnls.append(px-ep)
    return pnls

def stats(pnls):
    if not pnls:return(0,0,0,0)
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return(n,int(t),round(wr,1),round(ps/ns if ns>0 else 99,2))

print('='*72)
print('  OIL RETRACE PATTERN — Quick Test')
print('  up-up-down, close>Dlow+60, avgRange3>35, cl-op<-10, wick<8')
print('='*72)
d1m=load();d15=build_15m(d1m)
sigs=find_signals(d15)
print(f'\nSignals: {len(sigs)}')

tp_range=[40,50,60,70,80,90,100]
sl_range=[15,20,25,30,40,50,60]
results=[]
for tp in tp_range:
    for sl in sl_range:
        pnls=sim_long_advance(d15,sigs,tp,sl)
        s=stats(pnls)
        results.append({'tp':tp,'sl':sl,'t':s[0],'pnl':s[1],'wr':s[2],'pf':s[3]})

rdf=pd.DataFrame(results)
print(f'\n  {"TP":>5s} {"SL":>5s} {"T":>5s} {"PnL":>9s} {"WR":>7s} {"PF":>7s} {"Avg":>7s}')
print(f'  {"-"*5} {"-"*5} {"-"*5} {"-"*9} {"-"*7} {"-"*7} {"-"*7}')
top=rdf.sort_values('pnl',ascending=False).head(15)
for _,r in top.iterrows():
    avg=r['pnl']/r['t'] if r['t']>0 else 0
    print(f'  {int(r["tp"]):>5d} {int(r["sl"]):>5d} {int(r["t"]):>5d} {int(r["pnl"]):>+9d} {r["wr"]:>6.1f}% {r["pf"]:>6.2f} {avg:>+7.1f}')

best=rdf.sort_values('pnl',ascending=False).iloc[0]
print(f'\n  Best PnL: TP={int(best["tp"])} SL={int(best["sl"])} → {int(best["pnl"]):+d}pts  {best["wr"]:.1f}%  PF={best["pf"]:.2f}  {best["t"]}t')
print(f'DONE.')
