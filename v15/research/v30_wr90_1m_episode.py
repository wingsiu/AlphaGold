#!/usr/bin/env python3
'''v30 WR(90) on 1-Minute Bars — Episode-based entry.
Williams %R with period=90 on 1-minute data.
Episode: WR drops below entry threshold, accumulates volume, then exits oversold.
Entry on first bar after oversold episode ends (WR crosses above threshold).
Exit: ride to session end if WR reaches -20, or TP/SL.
'''

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

def load_oil(s='2024-01-01',e='2026-05-22'):
    loader=DataLoader();raw=loader.load_data(table_name='prices',start_date=s,end_date=e)
    raw.index=pd.to_datetime(raw['timestamp'],unit='ms')
    df=pd.DataFrame(index=raw.index)
    for c,src in [('open','openPrice_ask'),('high','highPrice_ask'),('low','lowPrice_ask'),
                   ('close_ask','closePrice_ask'),('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c]=raw[src].astype(float)
    if df.index.tz is None: df.index=df.index.tz_localize('UTC')
    return df

def build_1m_wr(df1m, period=90):
    df=df1m.copy()
    hh=df['high'].rolling(period).max(); ll=df['low'].rolling(period).min()
    df['wr90']=((hh-df['close_ask'])/(hh-ll+0.01))*-100
    df['hour']=df.index.hour
    df['is_uk']=df['hour'].isin([7,8,9,10,11,12,13,14,15,16])
    return df

def find_wr90_episodes(df, entry_th=-80, session='uk'):
    in_s=df['is_uk'] if session=='uk' else True
    oversold=(df['wr90']<entry_th)&in_s
    eps=[]; in_ep=False; cv=0.0; bc=0; wr_min=-999
    for i in range(len(df)):
        if oversold.iloc[i]:
            if not in_ep: cv=0.0; bc=0; wr_min=-999
            in_ep=True; cv+=df['volume'].iloc[i]; bc+=1
            wr_min=max(wr_min,df['wr90'].iloc[i])
        else:
            if in_ep:
                ebi=i
                if ebi<len(df)-1 and in_s.iloc[ebi]:
                    eps.append({'entry':ebi,'cum_vol':cv,'bars':bc,'wr_min':wr_min})
                in_ep=False; cv=0.0; bc=0
    return eps

def sim_trade(ei, df, tp, sl, max_bars=90, recovery=-20, session_end=16):
    ep=df.iloc[ei]['close_ask']; h=min(max_bars,len(df)-ei-1)
    reached=False
    for i in range(1,h+1):
        b=df.iloc[ei+i]
        if b['high']>=ep+tp: return ep+tp,i,'tp'
        if b['low']<=ep-sl: return ep-sl,i,'sl'
        if b['wr90']>=-20: reached=True
        if reached and b.name.hour==session_end: return b['close_bid'],i,'ride_end'
    return df.iloc[ei+h]['close_bid'],h,'timeout'

def stats(pnls):
    if not pnls: return {'trades':0,'pnl':0.0,'wr':0.0,'pf':0.0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'trades':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99}

def main():
    print('='*72); print('v30 WR(90) on 1-Minute Bars — Episode Entry'); print('='*72)
    print(); print('[1] Loading + WR(90)...')
    df1m=load_oil(); df=build_1m_wr(df1m,90)
    df=df.dropna(subset=['wr90'])
    print(f'  {len(df):,} 1m bars')

    print(); print('[2] Entry threshold sweep (no CumVol filter)...')
    print(f"  {'Entry<':>8s} {'Episodes':>10s} {'Trades':>8s} {'PnL':>10s} {'WR':>7s} {'PF':>6s}")
    print(f"  {'-'*60}")
    for eth in [-80,-75,-70,-65,-60,-55,-50]:
        eps=find_wr90_episodes(df,eth,'uk')
        trades=[]
        for ep in eps:
            ex,bars,reason=sim_trade(ep['entry'],df,80,40,90)
            trades.append({'pnl':ex-df.iloc[ep['entry']]['close_ask']})
        pnls=[t['pnl'] for t in trades]; s=stats(pnls)
        print(f"  {eth:>+8d} {len(eps):>10d} {s['trades']:>8d} {s['pnl']:>+10.1f} "
              f"{s['wr']:>6.1f}% {s['pf']:>5.2f}")

    # Sweep CumVol for best entry
    print(); print('[3] CumVol sweep (entry<-80, TP=80/SL=40)...')
    eps=find_wr90_episodes(df,-80,'uk')
    print(f'  {len(eps)} episodes total')
    cvs=[e['cum_vol'] for e in eps]
    print(f'  CumVol: p25={np.percentile(cvs,25):.0f} p50={np.percentile(cvs,50):.0f} '
          f'p75={np.percentile(cvs,75):.0f} p90={np.percentile(cvs,90):.0f}')
    print(f"  {'CumVol>':>10s} {'Episodes':>10s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s}")
    for cmin in [0,50,200,500,1000,2000]:
        filtered=[e for e in eps if e['cum_vol']>=cmin]
        trades=[]
        for ep in filtered:
            ex,bars,reason=sim_trade(ep['entry'],df,80,40,90)
            trades.append({'pnl':ex-df.iloc[ep['entry']]['close_ask']})
        pnls=[t['pnl'] for t in trades]; s=stats(pnls)
        print(f"  {cmin:>10d} {len(filtered):>10d} {s['trades']:>7d} {s['pnl']:>+10.1f} "
              f"{s['wr']:>6.1f}% {s['pf']:>5.2f}")

    # TP/SL sweep
    print(); print('[4] TP/SL sweep (entry<-80, no CumVol filter)...')
    eps80=find_wr90_episodes(df,-80,'uk')
    print(f"  {'TP/SL':<10s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'Ride%':>7s}")
    print(f"  {'-'*55}")
    for tp,sl in [(40,30),(60,40),(80,40),(80,60),(100,50),(100,60)]:
        if sl>=tp: continue
        trades=[]; rides=0
        for ep in eps80:
            ex,bars,reason=sim_trade(ep['entry'],df,tp,sl,90)
            trades.append({'pnl':ex-df.iloc[ep['entry']]['close_ask'],'reason':reason})
            if reason=='ride_end': rides+=1
        pnls=[t['pnl'] for t in trades]; s=stats(pnls)
        avg=s['pnl']/s['trades'] if s['trades']>0 else 0
        print(f"  {tp:>4d}/{sl:<4d} {s['trades']:>8d} {s['pnl']:>+10.1f} "
              f"{s['wr']:>6.1f}% {s['pf']:>5.2f} {rides/(s['trades']+0.01)*100:>6.1f}%")

    print(); print('DONE.')

if __name__=='__main__': main()
