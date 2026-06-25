#!/usr/bin/env python3
"""v28 Short — Overbought vs Oversold Entry Comparison
========================================================
Two short entry logic trees:
1. OVERBOUGHT (rev): WR > -20 → price at high → short expecting reversal down
   Recovery: WR < -80 (price falls to oversold)
   This is the mirror of: long WR < -80 expecting reversal UP
   
2. OVERSOLD (cont): WR < -80 → price at low → short expecting continuation down
   Recovery: none (just TP/SL)
   This is momentum continuation — oversold episodes keep falling
   
Both with advance target, force-close NY 14:28, NY 03-12 session.
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
WEAKNESS_TIMEOUT = 12

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

def find_signals(d, cmp_op, entry_thresh, min_cv, min_bars=EP_BARS_MIN):
    """cmp_op: 'lt' for WR<thresh, 'gt' for WR>thresh"""
    in_s=d['in_sess']
    if cmp_op=='lt': signal_bars=(d['wr']<entry_thresh)&in_s
    else: signal_bars=(d['wr']>entry_thresh)&in_s
    sigs=[]; in_ep=False; cv=0.0; bc=0
    for i in range(len(d)):
        if signal_bars.iloc[i]:
            if not in_ep: ep_start=i; cv=0.0; bc=0
            in_ep=True; cv+=d['volume'].iloc[i]; bc+=1
        else:
            if in_ep:
                ebi=i
                if ebi<len(d)-1 and in_s.iloc[ebi] and cv>=min_cv and bc>=min_bars:
                    sigs.append({'idx':ebi,'cv':cv,'bc':bc,'start':ep_start})
                in_ep=False; cv=0.0; bc=0
    return sigs

def sim_trade(d, ei, tp, sl, direction, recovery=None, weakness=None):
    """
    direction: 'long' or 'short'
    recovery: WR level that triggers 'ride to close' (None = skip)
    weakness: WR level below/above which counts as weakness bars
    """
    ep=d.iloc[ei]['close_ask']; h=min(MAX_BARS,len(d)-ei-1)
    reached=False; wc=0
    for i in range(1,h+1):
        b=d.iloc[ei+i]
        post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
        
        if direction=='long':
            if post: return b['close_bid']-ep, 'ny_close'
            if b['high']>=ep+tp: return tp, 'tp'
            if b['low']<=ep-sl: return -sl, 'sl'
            if recovery is not None:
                if b['wr']>=recovery: reached=True
                if b['wr']<weakness: wc+=1
                else: wc=0
                if reached and post: return b['close_bid']-ep, 'ride_end'
                if not reached and wc>=WEAKNESS_TIMEOUT: return b['close_bid']-ep, 'weak'
        else:  # short
            if post: return ep-b['close_bid'], 'ny_close'
            if b['low']<=ep-tp: return tp, 'tp'
            if b['high']>=ep+sl: return -sl, 'sl'
            if recovery is not None:
                # Overbought recovery: WR falls below recovery level (= oversold)
                if b['wr']<=recovery: reached=True
                if b['wr']>weakness: wc+=1  # still overbought
                else: wc=0
                if reached and post: return ep-b['close_bid'], 'ride_end'
                if not reached and wc>=WEAKNESS_TIMEOUT: return ep-b['close_bid'], 'weak'
    if direction=='long': return d.iloc[ei+h]['close_bid']-ep, 'timeout'
    else: return ep-d.iloc[ei+h]['close_bid'], 'timeout'

def sim_with_advance(d, sigs, tp, sl, direction, recovery=None, weakness=None):
    """Sequential simulation with advance target on new signal."""
    pnls=[]; in_trade=False; ct=0; cs=0; ep=0; ei=0; bh=0
    reached=False; wc=0; sig_idx=0
    while sig_idx<len(sigs):
        si=sigs[sig_idx]['idx']
        if not in_trade:
            in_trade=True; ep=d.iloc[si]['close_ask']
            if direction=='long': ct=ep+tp; cs=ep-sl
            else: ct=ep-tp; cs=ep+sl
            ei=si; bh=0; reached=False; wc=0; sig_idx+=1; continue
        if si-ei>MAX_BARS:
            if direction=='long': pnls.append(d.iloc[ei+MAX_BARS]['close_bid']-ep)
            else: pnls.append(ep-d.iloc[ei+MAX_BARS]['close_bid'])
            in_trade=False; continue
        for j in range(ei+bh+1, si+1):
            b=d.iloc[j]
            post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
            if direction=='long':
                if post: pnls.append(b['close_bid']-ep); in_trade=False; break
                if b['high']>=ct: pnls.append(ct-ep); in_trade=False; break
                if b['low']<=cs: pnls.append(cs-ep); in_trade=False; break
                if recovery is not None:
                    if b['wr']>=recovery: reached=True
                    if b['wr']<weakness: wc+=1
                    else: wc=0
                    if reached and post: pnls.append(b['close_bid']-ep); in_trade=False; break
                    if not reached and wc>=WEAKNESS_TIMEOUT: pnls.append(b['close_bid']-ep); in_trade=False; break
            else:
                if post: pnls.append(ep-b['close_bid']); in_trade=False; break
                if b['low']<=ct: pnls.append(ep-ct); in_trade=False; break
                if b['high']>=cs: pnls.append(ep-cs); in_trade=False; break
                if recovery is not None:
                    if b['wr']<=recovery: reached=True
                    if b['wr']>weakness: wc+=1
                    else: wc=0
                    if reached and post: pnls.append(ep-b['close_bid']); in_trade=False; break
                    if not reached and wc>=WEAKNESS_TIMEOUT: pnls.append(ep-b['close_bid']); in_trade=False; break
        bh=si-ei
        if not in_trade:
            in_trade=True; ep=d.iloc[si]['close_ask']
            if direction=='long': ct=ep+tp; cs=ep-sl
            else: ct=ep-tp; cs=ep+sl
            ei=si; bh=0; reached=False; wc=0; sig_idx+=1; continue
        ne=d.iloc[si]['close_ask']
        if direction=='long': ct=ne+tp; cs=min(cs,ne-sl)
        else: ct=ne-tp; cs=min(cs,ne+sl)
        ei=si; bh=0; reached=False; wc=0; sig_idx+=1
    if in_trade:
        last=min(ei+MAX_BARS,len(d)-1)
        if direction=='long': pnls.append(d.iloc[last]['close_bid']-ep)
        else: pnls.append(ep-d.iloc[last]['close_bid'])
    return pnls

def stats(pnls):
    if not pnls: return {'t':0,'pnl':0,'wr':0,'pf':0,'avg':0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'t':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99,'avg':t/n}

def print_row(label, t, pnl, wr, pf, avg, tag=''):
    print(f'  {label:<25s} {t:>6d} {pnl:>+10.0f} {wr:>6.1f}% {pf:>6.2f} {avg:>+8.1f} {tag}')

# ===== MAIN =====
d1=load(); d15=build(d1)
print('='*90)
print('  SHORT ENTRY COMPARISON: Overbought vs Oversold Continuation')
print('  NY 03-12 session, force-close 14:28, EpBars≥3')
print('='*90)

# ----- Long baseline -----
long_sigs=find_signals(d15,'lt',-80,15000)
l_pnls=sim_with_advance(d15,long_sigs,80,30,'long',recovery=-20,weakness=-50)
ls=stats(l_pnls)

# ----- Overbought Short (WR > -20, recovery at WR < -80) -----
print(f'\n{"="*90}')
print(f'  [1] OVERBOUGHT SHORT (WR > -20): price near high, expect reversal DOWN')
print(f'      Recovery: WR < -80 (overbought→oversold = trend reversal)')
print(f'      Weakness: WR > -20 (staying overbought = weakness for short)')
print(f'{"="*90}')
print(f'  {"Config":<25s} {"Trades":>6s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s}')
print(f'  {"-"*62}')

for cv in [5000,10000,15000,20000]:
    for short_wr in [-10,-15,-20,-25,-30]:
        sigs=find_signals(d15,'gt',short_wr,cv)
        if len(sigs)<20: continue
        for tp,sl in [(80,30),(80,40),(100,30),(100,40)]:
            pnls=sim_with_advance(d15,sigs,tp,sl,'short',recovery=-80,weakness=-20)
            s=stats(pnls)
            combined=l_pnls+pnls; cs2=stats(combined)
            if cs2['pnl']>2800:
                print_row(f'WR>{short_wr:+d} CV≥{cv:>5d} TP={tp}/SL={sl}', s['t'], s['pnl'], s['wr'], s['pf'], s['avg'], f'Comb={cs2["pnl"]:+.0f}')

# ----- Oversold Continuation Short (WR < -80, momentum) -----
print(f'\n{"="*90}')
print(f'  [2] OVERSOLD CONTINUATION SHORT (WR < -80): price near low, expect continuation DOWN')
print(f'      No recovery logic — pure momentum, exit by TP/SL/ny_close only')
print(f'{"="*90}')
print(f'  {"Config":<25s} {"Trades":>6s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s}')
print(f'  {"-"*62}')

best_cont=None
for cv in [5000,10000,15000,20000]:
    sigs=find_signals(d15,'lt',-80,cv)  # same signals as long!
    if len(sigs)<20: continue
    for tp,sl in [(80,30),(80,40),(100,30),(100,40),(120,40)]:
        pnls=sim_with_advance(d15,sigs,tp,sl,'short',recovery=None,weakness=None)
        s=stats(pnls)
        combined=l_pnls+pnls; cs2=stats(combined)
        if best_cont is None or cs2['pnl']>(best_cont['comb'] if best_cont else -9999):
            best_cont={'cv':cv,'tp':tp,'sl':sl,'t':s['t'],'pnl':s['pnl'],'wr':s['wr'],'pf':s['pf'],'avg':s['avg'],'comb':cs2['pnl'],'comb_wr':cs2['wr'],'comb_pf':cs2['pf']}
        if cs2['pnl']>2800:
            print_row(f'WR<-80 CV≥{cv:>5d} TP={tp}/SL={sl}', s['t'], s['pnl'], s['wr'], s['pf'], s['avg'], f'Comb={cs2["pnl"]:+.0f}')

# ----- SUMMARY -----
print(f'\n{"="*90}')
print(f'  SUMMARY')
print(f'{"="*90}')
print(f'  {"Strategy":<25s} {"Trades":>6s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s} {"Combined":>12s}')
print(f'  {"-"*75}')
print_row('Long only (advance)', ls['t'], ls['pnl'], ls['wr'], ls['pf'], ls['avg'], '')
print()

for cv in [10000,15000,20000]:
    for wr in [-15,-20,-25]:
        oversig=find_signals(d15,'gt',wr,cv)
        if len(oversig)<20: continue
        op=sim_with_advance(d15,oversig,80,40,'short',recovery=-80,weakness=-20)
        os=stats(op)
        if os['pnl']>0:
            combined=l_pnls+op; cs3=stats(combined)
            print_row(f'Overbought WR>{wr} CV≥{cv}', os['t'], os['pnl'], os['wr'], os['pf'], os['avg'], f'+{cs3["pnl"]:+.0f}')

if best_cont:
    print_row(f'Oversold cont CV≥{best_cont["cv"]}', best_cont['t'], best_cont['pnl'], best_cont['wr'], best_cont['pf'], best_cont['avg'], f'+{best_cont["comb"]:+.0f}')

print(f'\n  NOTE: "Combined" = Long + Short simultaneously (different signals,')
print(f'         not the same trade. Long and short positions can coexist.')

print('\nDONE.')
