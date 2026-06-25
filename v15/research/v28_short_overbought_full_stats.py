#!/usr/bin/env python3
"""v28 WR90 Short — Overbought Pattern (Mirror of Long Model)
================================================================
Long model: WR<-80 (oversold=price near low), TP=80/SL=30
Short model: WR>-20 (overbought=price near high), TP=80/SL=30
Same filters: CumVol≥15k, EpBars≥3, NY 03-12, force-close 14:28
Advance target on new signal.
"""

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_SESSION_START = 3; NY_SESSION_END = 12
NY_FORCE_CLOSE_H = 14; NY_FORCE_CLOSE_M = 28
MAX_BARS = 60
CUMVOL_MIN = 15000; EP_BARS_MIN = 3
TP = 80; SL = 30

# Long: WR<-80 entry, recovery at WR>-20, weakness if WR<-50
# Short: WR>-20 entry, recovery at WR<-80, weakness if WR>-50 
LONG_ENTRY = -80; LONG_RECOVERY = -20; LONG_WEAK = -50
SHORT_ENTRY = -20; SHORT_RECOVERY = -80; SHORT_WEAK = -20  # >-20 is weak for short
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
    d['wr']=((hh-d['close_ask'])/(hh-ll+0.01))*-100  # range -100 to 0
    ny=d.index.tz_convert('America/New_York'); d['ny_h']=ny.hour; d['ny_m']=ny.minute
    d['in_sess']=(d['ny_h']>=NY_SESSION_START)&(d['ny_h']<=NY_SESSION_END)
    d['atr14']=(d['high']-d['low']).rolling(14).mean()
    d['hour']=d.index.hour; d['dayofweek']=d.index.dayofweek
    return d

def find_signals(d, entry_thresh, mode='lt', min_cv=CUMVOL_MIN, min_bars=EP_BARS_MIN):
    """mode='lt' for WR < thresh, 'gt' for WR > thresh"""
    in_s=d['in_sess']
    signal_bars=((d['wr']<entry_thresh)&in_s) if mode=='lt' else ((d['wr']>entry_thresh)&in_s)
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

def sim_trade_long(ei, d, tp=TP, sl=SL):
    """Long: entered at close_ask. TP up, SL down."""
    ep=d.iloc[ei]['close_ask']; h=min(MAX_BARS,len(d)-ei-1)
    reached=False; wc=0
    for i in range(1,h+1):
        b=d.iloc[ei+i]
        post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
        if post: return b['close_bid']-ep, 'ny_close'
        if b['high']>=ep+tp: return tp, 'tp'
        if b['low']<=ep-sl: return -sl, 'sl'
        if b['wr']>=LONG_RECOVERY: reached=True  # recovering from oversold
        if b['wr']<LONG_WEAK: wc+=1  # staying oversold
        else: wc=0
        if reached and post: return b['close_bid']-ep, 'ride_end'
        if not reached and wc>=WEAKNESS_TIMEOUT: return b['close_bid']-ep, 'weak'
    return d.iloc[ei+h]['close_bid']-ep, 'timeout'

def sim_trade_short(ei, d, tp=TP, sl=SL):
    """Short: entered at close_ask. TP down (price drop), SL up (price rise)."""
    ep=d.iloc[ei]['close_ask']; h=min(MAX_BARS,len(d)-ei-1)
    reached=False; wc=0
    for i in range(1,h+1):
        b=d.iloc[ei+i]
        post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
        if post: return ep-b['close_bid'], 'ny_close'
        if b['low']<=ep-tp: return tp, 'tp'      # price drops = short wins
        if b['high']>=ep+sl: return -sl, 'sl'     # price rises = short loses
        if b['wr']<=SHORT_RECOVERY: reached=True  # overbought→oversold = recovery
        if b['wr']>SHORT_WEAK: wc+=1              # still overbought = weakness for short
        else: wc=0
        if reached and post: return ep-b['close_bid'], 'ride_end'
        if not reached and wc>=WEAKNESS_TIMEOUT: return ep-b['close_bid'], 'weak'
    return ep-d.iloc[ei+h]['close_bid'], 'timeout'

def sim_with_advance_long(sigs, d, tp=TP, sl=SL):
    pnls=[]
    in_trade=False; ct=0; cs=0; ep=0; ei=0; bh=0; reached=False; wc=0; sig_idx=0
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

def sim_with_advance_short(sigs, d, tp=TP, sl=SL):
    pnls=[]
    in_trade=False; ct=0; cs=0; ep=0; ei=0; bh=0; reached=False; wc=0; sig_idx=0
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

def stats(pnls):
    if not pnls: return {'t':0,'pnl':0,'wr':0,'pf':0,'avg':0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'t':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99,'avg':t/n}

def report(trades, label):
    if not trades: print(f'\n  {label}: NO TRADES'); return {'t':0,'pnl':0,'wr':0,'pf':0}
    pnls=[t['pnl'] for t in trades]
    cum=np.cumsum(pnls); peak=np.maximum.accumulate(cum); dd=cum-peak
    max_dd=dd.min()
    in_dd=False; ld=0; cd=0
    for i in range(len(cum)):
        if dd[i]<0: cd+=1; in_dd=True
        elif in_dd: ld=max(ld,cd); cd=0; in_dd=False
    if in_dd: ld=max(ld,cd)
    s=stats(pnls)
    print(f'\n{"="*72}')
    print(f'  {label} — SUMMARY')
    print(f'  Trades={s["t"]}  PnL={s["pnl"]:+.0f}  WR={s["wr"]:.1f}%  PF={s["pf"]:.2f}  Avg={s["avg"]:+.1f}  MaxDD={max_dd:+.0f}  LongDD={ld}')
    reasons={}
    for t in trades: r=t.get('reason',''); reasons[r]=reasons.get(r,0)+1
    if reasons:
        print(f'  Exit: ',end='')
        for r,c in sorted(reasons.items(),key=lambda x:-x[1]):
            rpnls=[t['pnl'] for t in trades if t.get('reason')==r]
            rs=stats(rpnls)
            print(f'{r}={c}({rs["pnl"]:+.0f}/{rs["wr"]:.0f}%) ',end='')
        print()
    # Monthly
    months={}
    for t in trades:
        m=t['entry_time'].tz_convert('Asia/Hong_Kong').strftime('%Y-%m')
        if m not in months: months[m]=[]
        months[m].append(t['pnl'])
    print(f'\n  MONTHLY (HKT):')
    print(f'  {"Month":>8s} {"T":>4s} {"PnL":>9s} {"WR":>6s} {"PF":>5s}')
    run=0.0
    for m in sorted(months.keys())[-12:]:
        p=months[m];n=len(p);s2=sum(p);wr=sum(1 for x in p if x>0)/n*100
        ps=sum(x for x in p if x>0);ns=abs(sum(x for x in p if x<0))
        pf=ps/ns if ns>0 else 99; run+=s2
        print(f'  {m:>8s} {n:>4d} {s2:>+9.0f} {wr:>5.0f}% {pf:>4.2f}')
    # Yearly
    yearly={}
    for t in trades:
        y=t['entry_time'].tz_convert('Asia/Hong_Kong').year
        if y not in yearly: yearly[y]=[]
        yearly[y].append(t['pnl'])
    for y in sorted(yearly.keys()):
        p=yearly[y];n=len(p);yt=sum(p);wr=sum(1 for x in p if x>0)/max(n,1)*100
        ps=sum(x for x in p if x>0);ns=abs(sum(x for x in p if x<0))
        print(f'  {y}: {n}t PnL={yt:+.0f} WR={wr:.0f}% PF={ps/ns if ns>0 else 99:.2f}')
    return s

# ===== MAIN =====
print('='*72)
print(f'  V28 WR90 — LONG (WR<{LONG_ENTRY}) + SHORT (WR>{SHORT_ENTRY})')
print(f'  TP={TP}/SL={SL}, Adv target, NY {NY_SESSION_START:02d}-{NY_SESSION_END:02d}')
print(f'  CumVol≥{CUMVOL_MIN:,}, EpBars≥{EP_BARS_MIN}')
print('='*72)

d1=load(); d15=build(d1)

# LONG
long_sigs=find_signals(d15, LONG_ENTRY, 'lt')
print(f'\nLong signals (WR<{LONG_ENTRY}): {len(long_sigs)}')
pnls_l_adv=sim_with_advance_long(long_sigs,d15)
sl_adv=stats(pnls_l_adv)
print(f'  LONG + advance: {sl_adv["t"]}t, {sl_adv["pnl"]:+.0f}pts, WR={sl_adv["wr"]:.1f}%, PF={sl_adv["pf"]:.2f}')

long_trades=[]
for s in long_sigs:
    pnl,reason=sim_trade_long(s['idx'],d15)
    row=d15.iloc[s['idx']]
    long_trades.append({
        'entry_time':d15.index[s['idx']], 'entry_price':row['close_ask'],
        'pnl':pnl, 'reason':reason, 'cum_vol':s['cv'], 'ep_bars':s['bc'],
        'entry_wr':row['wr'], 'atr':row['atr14'], 'dow':row['dayofweek'], 'hour_utc':row['hour'],
    })
report(long_trades, 'LONG (single trade)')

# SHORT
short_sigs=find_signals(d15, SHORT_ENTRY, 'gt')
print(f'\n\nShort signals (WR>{SHORT_ENTRY}): {len(short_sigs)}')
if short_sigs:
    pnls_s_adv=sim_with_advance_short(short_sigs,d15)
    ss_adv=stats(pnls_s_adv)
    print(f'  SHORT + advance: {ss_adv["t"]}t, {ss_adv["pnl"]:+.0f}pts, WR={ss_adv["wr"]:.1f}%, PF={ss_adv["pf"]:.2f}')

    short_trades=[]
    for s in short_sigs:
        pnl,reason=sim_trade_short(s['idx'],d15)
        row=d15.iloc[s['idx']]
        short_trades.append({
            'entry_time':d15.index[s['idx']], 'entry_price':row['close_ask'],
            'pnl':pnl, 'reason':reason, 'cum_vol':s['cv'], 'ep_bars':s['bc'],
            'entry_wr':row['wr'], 'atr':row['atr14'], 'dow':row['dayofweek'], 'hour_utc':row['hour'],
        })
    report(short_trades, 'SHORT (single trade)')

    # Combined
    combined=pnls_l_adv+pnls_s_adv
    sc=stats(combined)
    print(f'\n{"="*72}')
    print(f'  COMBINED: {sc["t"]}t, {sc["pnl"]:+.0f}pts, WR={sc["wr"]:.1f}%, PF={sc["pf"]:.2f}')
    print(f'  Long={sl_adv["pnl"]:+.0f} ({sl_adv["t"]}t) + Short={ss_adv["pnl"]:+.0f} ({ss_adv["t"]}t)')
else:
    print('  SHORT: No signals found at this threshold.')

print('\nDONE.')
