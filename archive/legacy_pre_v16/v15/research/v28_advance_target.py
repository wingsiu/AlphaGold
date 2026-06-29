#!/usr/bin/env python3
"""v28 WR90 — Advance Target on New Signal
=============================================
When a new WR90 oversold episode starts while a trade is open,
advance the target: reset TP=100/SL=30 from the new entry price.
Logic: fresh exhaustion confirms the downtrend, let the profit run.
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
ENTRY_THRESH = -80; RECOVERY = -20; WEAK = -50; WEAKNESS_TIMEOUT = 12

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
    in_s=d['in_sess']; d['oversold']=(d['wr']<ENTRY_THRESH)&in_s
    return d

def find_signals(d, min_cv=CUMVOL_MIN, min_bars=EP_BARS_MIN):
    """Find valid signal bars (episode ends, CumVol>=min, EpBars>=min)."""
    in_s=d['in_sess']; oversold=d['oversold']
    sigs=[]; in_ep=False; cv=0.0; bc=0
    for i in range(len(d)):
        if oversold.iloc[i]:
            if not in_ep: ep_start=i; cv=0.0; bc=0
            in_ep=True; cv+=d['volume'].iloc[i]; bc+=1
        else:
            if in_ep:
                ebi=i
                if ebi<len(d)-1 and in_s.iloc[ebi] and cv>=min_cv and bc>=min_bars:
                    sigs.append({'idx':ebi,'cv':cv,'bc':bc})
                in_ep=False; cv=0.0; bc=0
    return sigs

def sim_trade_single(ei, d, tp, sl, max_bars=60):
    """Simulate one trade without advance (baseline)."""
    ep_p=d.iloc[ei]['close_ask']; h=min(max_bars,len(d)-ei-1)
    reached=-99; wc=0
    for i in range(1,h+1):
        b=d.iloc[ei+i]
        post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
        if post: return b['close_bid']-ep_p, 'ny_close'
        if b['high']>=ep_p+tp: return tp, 'tp'
        if b['low']<=ep_p-sl: return -sl, 'sl'
        if b['wr']>=RECOVERY: reached=RECOVERY
        if b['wr']<WEAK: wc+=1
        else: wc=0
        if reached==RECOVERY and post: return b['close_bid']-ep_p, 'ride_end'
        if reached!=RECOVERY and wc>=WEAKNESS_TIMEOUT: return b['close_bid']-ep_p, 'weak'
    return d.iloc[ei+h]['close_bid']-ep_p, 'timeout'

def sim_with_advance(sigs, d, tp, sl, max_bars=60):
    """Simulate sequentially: when new signal fires during active trade, advance TP/SL."""
    pnls=[]
    in_trade=False; current_tp=0; current_sl=0; entry_price=0; entry_idx=0; exit_idx=0; bars_held=0
    reached=-99; wc=0
    sig_idx=0
    while sig_idx<len(sigs):
        signal=sigs[sig_idx]
        si=signal['idx']
        if not in_trade:
            # Enter new trade
            in_trade=True; entry_price=d.iloc[si]['close_ask']
            current_tp=entry_price+tp; current_sl=entry_price-sl
            entry_idx=si; bars_held=0; reached=-99; wc=0
            sig_idx+=1; continue

        # Check if trade exceeds max bars
        if si-entry_idx>max_bars:
            exit_price=d.iloc[entry_idx+max_bars]['close_bid']
            pnls.append(exit_price-entry_price)
            in_trade=False; continue

        # Trade is active — walk forward from last check to this signal bar
        for j in range(entry_idx+bars_held+1, si+1):
            b=d.iloc[j]
            post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
            if post:
                pnls.append(b['close_bid']-entry_price)
                in_trade=False; break
            if b['high']>=current_tp:
                pnls.append(current_tp-entry_price)
                in_trade=False; break
            if b['low']<=current_sl:
                pnls.append(current_sl-entry_price)
                in_trade=False; break
            if b['wr']>=RECOVERY: reached=RECOVERY
            if b['wr']<WEAK: wc+=1
            else: wc=0
            if reached==RECOVERY and post: pnls.append(b['close_bid']-entry_price); in_trade=False; break
            if reached!=RECOVERY and wc>=WEAKNESS_TIMEOUT: pnls.append(b['close_bid']-entry_price); in_trade=False; break
        bars_held=si-entry_idx

        if not in_trade:
            # Trade closed before new signal — re-enter at this signal
            in_trade=True; entry_price=d.iloc[si]['close_ask']
            current_tp=entry_price+tp; current_sl=entry_price-sl
            entry_idx=si; bars_held=0; reached=-99; wc=0
            sig_idx+=1; continue

        # Trade survived to the new signal — ADVANCE TARGET
        new_entry=d.iloc[si]['close_ask']
        current_tp=new_entry+tp
        current_sl=min(current_sl, new_entry-sl)  # tighter SL wins
        entry_idx=si; bars_held=0; reached=-99; wc=0
        sig_idx+=1

    # Close any remaining open trade at last bar
    if in_trade:
        last=min(entry_idx+max_bars, len(d)-1)
        last_b=d.iloc[last]
        pnls.append(last_b['close_bid']-entry_price)
    return pnls

def stats(pnls):
    if not pnls: return {'t':0,'pnl':0,'wr':0,'pf':0,'avg':0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'t':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99,'avg':t/n}

# ===== MAIN =====
print('='*72)
print('  V28 WR90 — ADVANCE TARGET ON NEW SIGNAL')
print(f'  Entry: WR<{ENTRY_THRESH}, CumVol>{CUMVOL_MIN:,}, EpBars>{EP_BARS_MIN}')
print(f'  NY 03-12 entry, force-close NY 14:28')
print('='*72)

d1=load(); d15=build(d1)
sigs=find_signals(d15)
print(f'\nSignals: {len(sigs)}')

for tp,sl in [(80,40),(100,30),(80,30),(100,35)]:
    # Baseline (no advance)
    pnls_base=[sim_trade_single(s['idx'],d15,tp,sl)[0] for s in sigs]
    sb=stats(pnls_base)

    # With advance target
    pnls_adv=sim_with_advance(sigs,d15,tp,sl)
    sa=stats(pnls_adv)

    print(f'\n{"="*72}')
    print(f'  TP={tp}/SL={sl}')
    print(f'  {"Metric":>20s} {"Baseline":>12s} {"Advance":>12s} {"Delta":>10s}')
    print(f'  {"-"*58}')
    print(f'  {"Trades":>20s} {sb["t"]:>12d} {sa["t"]:>12d} {sa["t"]-sb["t"]:>+10d}')
    print(f'  {"PnL":>20s} {sb["pnl"]:>+12.0f} {sa["pnl"]:>+12.0f} {sa["pnl"]-sb["pnl"]:>+10.0f}')
    print(f'  {"WR":>20s} {sb["wr"]:>11.1f}% {sa["wr"]:>11.1f}% {sa["wr"]-sb["wr"]:>+9.1f}%')
    print(f'  {"PF":>20s} {sb["pf"]:>11.2f} {sa["pf"]:>11.2f} {sa["pf"]-sb["pf"]:>+9.2f}')
    print(f'  {"Avg Trade":>20s} {sb["avg"]:>+11.2f} {sa["avg"]:>+11.2f} {sa["avg"]-sb["avg"]:>+9.2f}')

print('\nDONE.')
