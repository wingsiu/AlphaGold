#!/usr/bin/env python3
"""Verify short trade exit prices are correct"""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader; import warnings; warnings.filterwarnings('ignore')

NY_SESSION_START=3;NY_SESSION_END=12;NY_FORCE_CLOSE_H=14;NY_FORCE_CLOSE_M=28;MAX_BARS=60;EP_BARS_MIN=3

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
    d['in_sess']=(d['ny_h']>=NY_SESSION_START)&(d['ny_h']<=NY_SESSION_END)
    return d

def find_oversold(d,min_cv=15000):
    in_s=d['in_sess'];o=(d['wr']<-80)&in_s;
    sigs=[];ie=False;cv=0.0;bc=0
    for i in range(len(d)):
        if o.iloc[i]:
            if not ie:ep_start=i;cv=0.0;bc=0
            ie=True;cv+=d['volume'].iloc[i];bc+=1
        else:
            if ie:
                ebi=i
                if ebi<len(d)-1 and in_s.iloc[ebi] and cv>=min_cv and bc>=EP_BARS_MIN:
                    sigs.append({'idx':ebi,'cv':cv,'bc':bc})
                ie=False;cv=0.0;bc=0
    return sigs

def find_overbought(d,min_cv=5000):
    in_s=d['in_sess'];o=(d['wr']>-20)&in_s;
    sigs=[];ie=False;cv=0.0;bc=0
    for i in range(len(d)):
        if o.iloc[i]:
            if not ie:ep_start=i;cv=0.0;bc=0
            ie=True;cv+=d['volume'].iloc[i];bc+=1
        else:
            if ie:
                ebi=i
                if ebi<len(d)-1 and in_s.iloc[ebi] and cv>=min_cv and bc>=EP_BARS_MIN:
                    sigs.append({'idx':ebi,'cv':cv,'bc':bc})
                ie=False;cv=0.0;bc=0
    return sigs

d1=load();d15=build(d1)
oversold=find_oversold(d15)
overbought=find_overbought(d15)
ov_set={s['idx'] for s in oversold}
ob_set={s['idx'] for s in overbought}
overlap=ov_set&ob_set
print(f'Oversold signals: {len(oversold)}')
print(f'Overbought signals: {len(overbought)}')
print(f'Overlapping (same bar, BOTH long AND short entry): {len(overlap)}')
if overlap:
    print(f'First 10 overlapping: {sorted(overlap)[:10]}')
    for oi in list(overlap)[:3]:
        wr=d15.iloc[oi]['wr']
        print(f'  bar@{oi} WR={wr:.0f} — WR>{-20}? {wr>-20}  WR<{-80}? {wr<-80}')

# Verify short: entry at close_ask, exit at close_bid
print(f'\n{"="*60}')
print(f'SHORT TRADE VERIFICATION (first 5 overbought)')
for s in overbought[:5]:
    si=s['idx'];ep=d15.iloc[si]['close_ask']
    for i in range(1,min(MAX_BARS,len(d15)-si-1)+1):
        b=d15.iloc[si+i]
        post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
        if post:
            exit_px=b['close_bid']
            pnl=ep-exit_px
            print(f'WR={d15.iloc[si]["wr"]:+.0f} entry_ask={ep:.1f} exit_bid={exit_px:.1f} pnl={pnl:+.1f} | price change from entry={exit_px-ep:+.1f}')
            break

print(f'\n{"="*60}')
print(f'LONG TRADE VERIFICATION (first 3 oversold)')
for s in oversold[:3]:
    si=s['idx'];ep=d15.iloc[si]['close_ask']
    for i in range(1,min(MAX_BARS,len(d15)-si-1)+1):
        b=d15.iloc[si+i]
        post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
        if post:
            exit_px=b['close_bid']
            pnl=exit_px-ep
            print(f'WR={d15.iloc[si]["wr"]:+.0f} entry_ask={ep:.1f} exit_bid={exit_px:.1f} pnl={pnl:+.1f} | price change from entry={exit_px-ep:+.1f}')
            break
