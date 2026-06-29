#!/usr/bin/env python3
"""v28 WR90 TP/SL + Dynamic Target Sweep
=========================================
Config: EpBars≥3, CumVol≥15k, NY 03-12 entry, force-close NY 14:28
Sweeps: TP, SL, ATR-based dynamic TP, WR-depth-based dynamic exit.
"""

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_SESSION_START = 3; NY_SESSION_END = 12
NY_FORCE_CLOSE_H = 14; NY_FORCE_CLOSE_M = 28
CUMVOL_MIN = 15000; EP_BARS_MIN = 3; MAX_BARS = 60
RECOVERY = -20; WEAK = -50; WEAKNESS_TIMEOUT = 12

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
    d['atr14']=(d['high']-d['low']).rolling(14).mean()
    return d

def episodes(d, entry_wr):
    in_s=d['in_sess']; o=(d['wr']<entry_wr)&in_s
    eps=[]; ie=False; cv=0.0; bc=0
    for i in range(len(d)):
        if o.iloc[i]:
            if not ie: ep_start=i; cv=0.0; bc=0
            ie=True; cv+=d['volume'].iloc[i]; bc+=1
        else:
            if ie:
                ebi=i
                if ebi<len(d)-1 and in_s.iloc[ebi]:
                    eps.append({'s':ep_start,'e':ebi,'cv':cv,'bc':bc})
                ie=False; cv=0.0; bc=0
    return eps

def sim_trade(ei, d, tp, sl, max_bars=60, recovery=-20):
    ep=d.iloc[ei]['close_ask']; h=min(max_bars,len(d)-ei-1)
    reached_r=-99; wc=0
    for i in range(1,h+1):
        b=d.iloc[ei+i]
        post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
        if post: return b['close_bid'],i,'ny_close'
        if b['high']>=ep+tp: return ep+tp,i,'tp'
        if b['low']<=ep-sl: return ep-sl,i,'sl'
        if b['wr']>=recovery: reached_r=recovery
        if b['wr']<WEAK: wc+=1
        else: wc=0
        if reached_r==recovery and post: return b['close_bid'],i,'ride_end'
        if reached_r!=recovery and wc>=WEAKNESS_TIMEOUT: return b['close_bid'],i,'weak'
    return d.iloc[ei+h]['close_bid'],h,'timeout'

def stats(pnls):
    if not pnls: return {'t':0,'pnl':0,'wr':0,'pf':0,'avg':0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'t':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99,'avg':t/n}

# ===== MAIN =====
print('='*80)
print('  V28 WR90 TP/SL + DYNAMIC TARGET SWEEP')
print(f'  Session: NY {NY_SESSION_START:02d}:00-{NY_SESSION_END:02d}:00, Force-close {NY_FORCE_CLOSE_H}:{NY_FORCE_CLOSE_M:02d}')
print(f'  CumVol>{CUMVOL_MIN:,}  EpBars>{EP_BARS_MIN}')
print('='*80)

d1=load(); d15=build(d1)
eps=episodes(d15, -80)
eps=[e for e in eps if e['cv']>=CUMVOL_MIN and e['bc']>=EP_BARS_MIN]
print(f'\nEpisodes: {len(eps)}')

# ===== 1. FIXED TP/SL SWEEP =====
print(f'\n{"="*80}')
print(f'  [1] FIXED TP/SL SWEEP')
print(f'  {"TP/SL":>10s} {"Trades":>7s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s}')
print(f'  {"-"*48}')

best_fixed=None
for tp in [60,70,80,90,100,120]:
    for sl in [30,35,40,45,50]:
        if sl>=tp: continue
        pnls=[]
        for ep in eps:
            ex,bars,reason=sim_trade(ep['e'],d15,tp,sl)
            pnls.append(ex-d15.iloc[ep['e']]['close_ask'])
        s=stats(pnls)
        if s['t']<10: continue
        if best_fixed is None or s['pf']>best_fixed['pf']:
            best_fixed={**s,'tp':tp,'sl':sl,'trades':s['t'],'pnls':pnls}
        if s['pf']>=1.40:
            print(f'  {tp:>4d}/{sl:<4d} {s["t"]:>7d} {s["pnl"]:>+10.0f} {s["wr"]:>6.1f}% {s["pf"]:>5.2f} {s["avg"]:>+8.1f}')

if best_fixed:
    print(f'\n  Best fixed: TP={best_fixed["tp"]}/SL={best_fixed["sl"]} → {best_fixed["t"]}t, {best_fixed["pnl"]:+.0f}pts, WR={best_fixed["wr"]:.1f}%, PF={best_fixed["pf"]:.2f}')

# ===== 2. DYNAMIC TP BY WR DEPTH =====
print(f'\n{"="*80}')
print(f'  [2] DYNAMIC TP BY WR DEPTH (fixed SL=40)')
print(f'  {"WR<-":>9s} {"TP":>6s} {"Trades":>7s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s}')
print(f'  {"-"*52}')

for entry_wr in [-80,-85,-90]:
    eps_d=episodes(d15, entry_wr)
    eps_d=[e for e in eps_d if e['cv']>=CUMVOL_MIN and e['bc']>=EP_BARS_MIN]
    if len(eps_d)<10: continue
    for tp in [60,80,100,120]:
        pnls=[]
        for ep in eps_d:
            ex,bars,reason=sim_trade(ep['e'],d15,tp,40)
            pnls.append(ex-d15.iloc[ep['e']]['close_ask'])
        s=stats(pnls)
        if s['t']>=10:
            print(f'  {entry_wr:>+8d} {tp:>4d} {s["t"]:>7d} {s["pnl"]:>+10.0f} {s["wr"]:>6.1f}% {s["pf"]:>5.2f} {s["avg"]:>+8.1f}')

# ===== 3. TP BY ATR MULTIPLIER =====
print(f'\n{"="*80}')
print(f'  [3] DYNAMIC TP = N × ATR(14) at entry (SL = TP/2)')
print(f'  {"ATR×":>9s} {"Trades":>7s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s} {"TP_range":>12s}')
print(f'  {"-"*58}')

atr_med=d15['atr14'].median()
for mult in [1.5,2.0,2.5,3.0,3.5,4.0]:
    pnls=[]; tps=[]
    for ep in eps:
        atr=max(d15.iloc[ep['e']]['atr14'], atr_med*0.5)  # floor
        tp=atr*mult; sl=tp/2
        ex,bars,reason=sim_trade(ep['e'],d15,tp,sl)
        pnls.append(ex-d15.iloc[ep['e']]['close_ask'])
        tps.append(tp)
    s=stats(pnls)
    if s['t']>=10:
        print(f'  {mult:>7.1f}x {s["t"]:>7d} {s["pnl"]:>+10.0f} {s["wr"]:>6.1f}% {s["pf"]:>5.2f} {s["avg"]:>+8.1f} TP={np.mean(tps):.0f}-{np.max(tps):.0f}')

# ===== 4. SWING TP (WR-based dynamic exit) =====
print(f'\n{"="*80}')
print(f'  [4] SWING EXIT: Varying "ride to -20" recovery levels')
print(f'  {"Recov@":>9s} {"Trades":>7s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s} {"Ride%":>8s}')
print(f'  {"-"*53}')

for recov in [-15,-20,-25,-30]:
    pnls=[]; rides=0
    for ep in eps:
        ex,bars,reason=sim_trade(ep['e'],d15,80,40,recovery=recov)
        pnls.append(ex-d15.iloc[ep['e']]['close_ask'])
        if reason=='ride_end':rides+=1
    s=stats(pnls)
    if s['t']>=10:
        print(f'  {recov:>+8d} {s["t"]:>7d} {s["pnl"]:>+10.0f} {s["wr"]:>6.1f}% {s["pf"]:>5.2f} {s["avg"]:>+8.1f} {rides/s["t"]*100:>7.1f}%')

# ===== 5. SCALE-OUT: TP1/TP2 partial exit =====
print(f'\n{"="*80}')
print(f'  [5] SCALE-OUT: TP1 + TP2 (exit half at each)')
print(f'  {"TP1/TP2":>10s} {"Trades":>7s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s}')
print(f'  {"-"*48}')

for tp1,tp2 in [(40,80),(50,100),(60,80),(60,100)]:
    pnls=[]
    for ep in eps:
        ei=ep['e']; ep_p=d15.iloc[ei]['close_ask']; h=min(60,len(d15)-ei-1)
        t1_hit=False; ex_p=0.0
        for i in range(1,h+1):
            b=d15.iloc[ei+i]
            post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
            if post: ex_p=b['close_bid']; break
            if not t1_hit and b['high']>=ep_p+tp1:
                t1_hit=True; ex_p=ep_p+tp1  # partial
                # move SL to breakeven for remaining half
            if t1_hit and b['high']>=ep_p+tp2: ex_p=ep_p+tp2; break
            if b['low']<=ep_p-40: ex_p=ep_p-40; break
            if b['wr']>=RECOVERY:
                if b['ny_h']>=NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M: ex_p=b['close_bid']; break
        pnls.append(ex_p-ep_p)
    s=stats(pnls)
    if s['t']>=10:
        print(f'  {tp1:>4d}/{tp2:<4d} {s["t"]:>7d} {s["pnl"]:>+10.0f} {s["wr"]:>6.1f}% {s["pf"]:>5.2f} {s["avg"]:>+8.1f}')

# ===== 6. TRAILING STOP =====
print(f'\n{"="*80}')
print(f'  [6] TRAILING STOP (activate after profit, trail by X pts)')
print(f'  {"Act@/Trail":>12s} {"Trades":>7s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s}')
print(f'  {"-"*52}')

for activate,trail in [(20,15),(30,20),(40,25),(40,30),(50,30)]:
    pnls=[]
    for ep in eps:
        ei=ep['e']; ep_p=d15.iloc[ei]['close_ask']; h=min(60,len(d15)-ei-1)
        reached=-99; wc=0; exit_p=0.0; trailing_sl=None
        for i in range(1,h+1):
            b=d15.iloc[ei+i]
            post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
            if post: exit_p=b['close_bid']; break
            profit=b['high']-ep_p
            if trailing_sl is not None and b['low']<=trailing_sl: exit_p=trailing_sl; break
            if b['low']<=ep_p-40: exit_p=ep_p-40; break
            if profit>=activate and trailing_sl is None: trailing_sl=ep_p+profit-trail
            if trailing_sl is not None: trailing_sl=max(trailing_sl,ep_p+profit-trail)
            if b['wr']>=RECOVERY: reached=RECOVERY
            if b['wr']<WEAK: wc+=1
            else: wc=0
            if reached!=RECOVERY and wc>=WEAKNESS_TIMEOUT: exit_p=b['close_bid']; break
        pnls.append(exit_p-ep_p)
    s=stats(pnls)
    if s['t']>=10:
        print(f'  {activate:>4d}/{trail:<4d} {s["t"]:>7d} {s["pnl"]:>+10.0f} {s["wr"]:>6.1f}% {s["pf"]:>5.2f} {s["avg"]:>+8.1f}')

# ===== BASELINE =====
pnls_base=[]
for ep in eps:
    ex,bars,reason=sim_trade(ep['e'],d15,80,40)
    pnls_base.append(ex-d15.iloc[ep['e']]['close_ask'])
sb=stats(pnls_base)
print(f'\n{"="*80}')
print(f'  BASELINE (TP=80/SL=40): {sb["t"]}t, {sb["pnl"]:+.0f}pts, WR={sb["wr"]:.1f}%, PF={sb["pf"]:.2f}, Avg={sb["avg"]:+.1f}')
print('  DONE.')
