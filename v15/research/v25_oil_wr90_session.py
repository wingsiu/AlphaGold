#!/usr/bin/env python3
'''v25 Oil WR90 Session Trend Pattern — Williams %R(14) on 15m bars.'''
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader

def load_oil_data(start_date='2024-01-01', end_date='2026-05-22'):
    loader = DataLoader(); raw = loader.load_data(table_name='prices', start_date=start_date, end_date=end_date)
    raw.index = pd.to_datetime(raw['timestamp'], unit='ms')
    df = pd.DataFrame(index=raw.index)
    for c, src in [('open','openPrice_ask'),('high','highPrice_ask'),('low','lowPrice_ask'),
                    ('close_ask','closePrice_ask'),('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c] = raw[src].astype(float)
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    return df

def build_15m(df_1m):
    df15 = df_1m.resample('15min',label='right',closed='right').agg(
        {'open':'first','high':'max','low':'min','close_ask':'last','close_bid':'last','volume':'sum'}).dropna()
    n=14; hh=df15['high'].rolling(n).max(); ll=df15['low'].rolling(n).min()
    df15['wr'] = ((hh-df15['close_ask'])/(hh-ll+0.01))*-100
    df15['wr_prev'] = df15['wr'].shift(1)
    df15['hour'] = df15.index.hour
    df15['is_uk'] = df15['hour'].isin([7,8,9,10,11,12,13,14,15,16])
    df15['is_us'] = df15['hour'].isin([12,13,14,15,16,17,18,19,20])
    return df15

def sim_long(ei, ep, df15, tp, sl, max_bars, exit_hour=None):
    horizon = min(max_bars, len(df15)-ei-1)
    for i in range(1, horizon+1):
        b = df15.iloc[ei+i]
        if b['low'] <= ep-sl: return ep-sl, i, 'sl'
        if b['high'] >= ep+tp: return ep+tp, i, 'tp'
    return df15.iloc[ei+horizon]['close_bid'], horizon, 'timeout'

def find_entries(df15, entry_cross=-80, session='uk', first_of_day=True):
    in_s = df15['is_uk'] if session=='uk' else (df15['is_us'] if session=='us' else df15['is_uk']|df15['is_us'])
    cross_up = (df15['wr_prev']<=entry_cross)&(df15['wr']>entry_cross)&in_s
    if not first_of_day:
        return cross_up
    # Keep only first cross per trading day
    df15['day'] = df15.index.date
    df15['cross_day_rank'] = cross_up.groupby(df15['day']).cumsum() * cross_up
    return cross_up & (df15['cross_day_rank'] == 1)

def evaluate(df15, entry, tp, sl, max_bars, session):
    mask = find_entries(df15, entry, session)
    trades=[]
    for idx in df15.index[mask]:
        ei=df15.index.get_loc(idx); ep=df15.iloc[ei]['close_ask']
        ex,bars,reason=sim_long(ei,ep,df15,tp,sl,max_bars)
        trades.append({'entry_idx':idx,'pnl':ex-ep,'reason':reason,'bars':bars})
    return trades

def stats(pnls):
    if not pnls: return {'trades':0,'pnl':0.0,'wr':0.0,'pf':0.0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'trades':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99}

def main():
    print('='*72); print('v25 Oil WR90 Session Trend Pattern'); print('='*72)
    print(); print('[1] Loading...'); df1m=load_oil_data(); df15=build_15m(df1m)
    print(f'  {len(df15):,} 15m bars')
    print(); print('[2] WR90 entry sweep (TP=80/SL=40)...')
    print(f"  {'Entry':>8s} {'Sess':>6s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s}")
    print(f"  {'-'*55}")
    for entry in [-90,-85,-80,-75,-70]:
        for sess in ['uk','us']:
            trades=evaluate(df15,entry,80,40,80,sess); pnls=[t['pnl'] for t in trades]; s=stats(pnls)
            if s['trades']<5: continue
            print(f"  {entry:>+8d} {sess:>6s} {s['trades']:>7d} {s['pnl']:>+10.1f} {s['wr']:>6.1f}% {s['pf']:>5.2f}")
    print(); print('[3] TP/SL sweep for WR>-80 UK...')
    print(f"  {'TP/SL':<12s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s}"); print(f"  {'-'*55}")
    for tp in [40,60,80,100,120]:
        for sl_ in [30,40,50,60]:
            if sl_>=tp: continue
            trades=evaluate(df15,-80,tp,sl_,80,'uk'); pnls=[t['pnl'] for t in trades]; s=stats(pnls)
            if s['trades']<5: continue
            print(f"  TP={tp}/SL={sl_:<6} {s['trades']:>7d} {s['pnl']:>+10.1f} {s['wr']:>6.1f}% {s['pf']:>5.2f}")
    print(); print('[4] Ride-to-timeout (80 bars, no TP/SL)...')
    print(f"  {'Entry':>8s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s}")
    for entry in [-90,-85,-80,-75]:
        trades=evaluate(df15,entry,999,999,80,'uk'); pnls=[t['pnl'] for t in trades]; s=stats(pnls)
        print(f"  {entry:>+8d} {s['trades']:>7d} {s['pnl']:>+10.1f} {s['wr']:>6.1f}% {s['pf']:>5.2f}")
    print(); print('[5] Directional edge: +100pts before -50pts (80 bars)...')
    for entry in [-90,-85,-80]:
        mask=find_entries(df15,entry,'uk'); n=mask.sum(); wins=0
        for idx in df15.index[mask]:
            ei=df15.index.get_loc(idx);ep=df15.iloc[ei]['close_ask'];h=min(80,len(df15)-ei-1)
            for i in range(1,h+1):
                b=df15.iloc[ei+i]
                if b['high']>=ep+100: wins+=1; break
                if b['low']<=ep-50: break
        print(f"  WR>{entry}: {wins}/{n} ({wins/n*100:.1f}%) reach +100 before -50")
    print(); print('DONE.')

if __name__=='__main__': main()
