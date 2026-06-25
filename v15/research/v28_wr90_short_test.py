#!/usr/bin/env python3
"""WR90 Short version — overbought WR>80, short with advance target."""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_S=3;NY_E=12;NY_FC_H=14;NY_FC_M=28
LONG_MAX_B=60;LONG_EP_MIN=3;LONG_CV=15000
LONG_RECOVERY=-20;LONG_WEAK=-50;LONG_WT=12

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
    n=14;hh=d['high'].rolling(n).max();ll=d['low'].rolling(n).min()
    d['wr']=((hh-d['close_ask'])/(hh-ll+0.01))*-100
    ny=d.index.tz_convert('America/New_York');d['ny_h']=ny.hour;d['ny_m']=ny.minute
    d['in_sess']=(d['ny_h']>=NY_S)&(d['ny_h']<=NY_E)
    return d

def find_short_signals(d, entry_thresh=80):
    """Overbought WR > -20 (close near 14-bar high), high volume."""
    in_s=d['in_sess'];o=(d['wr']>-20)&in_s
    sigs=[];ie=False;cv=0.0;bc=0
    for i in range(len(d)):
        if o.iloc[i]:
            if not ie: ep_s=i;cv=0.0;bc=0
            ie=True;cv+=d['volume'].iloc[i];bc+=1
        else:
            if ie:
                ebi=i
                if ebi<len(d)-1 and in_s.iloc[ebi] and cv>=LONG_CV and bc>=LONG_EP_MIN:
                    sigs.append({'idx':ebi,'cv':cv,'bc':bc})
                ie=False;cv=0.0;bc=0
    return sigs

def sim_short_advance(d15,sigs,tp_val,sl_val):
    """Short with advance target. Entry at bid, exit at ask. TP/SL capped."""
    pnls=[]
    in_trade=False;ct=0;cs=0;ep=0;ei=0;bh=0
    reached=False;wc=0;sig_idx=0
    while sig_idx<len(sigs):
        si=sigs[sig_idx]['idx']
        if not in_trade:
            # Short entry at bid
            in_trade=True;ep=d15.iloc[si]['close_bid'];ct=ep-tp_val;cs=ep+sl_val
            ei=si;bh=0;reached=False;wc=0
            sig_idx+=1;continue
        # Timeout
        if si-ei>LONG_MAX_B:
            px=d15.iloc[ei+LONG_MAX_B]['close_ask'];pnls.append(ep-px)
            in_trade=False;continue
        exit_at_si=False
        for j in range(ei+bh+1,si+1):
            b=d15.iloc[j]
            post=(b['ny_h']>NY_FC_H)or(b['ny_h']==NY_FC_H and b['ny_m']>=NY_FC_M)
            if post:
                px=b['close_ask'];pnls.append(ep-px);in_trade=False
                if j==si:exit_at_si=True;break
            if b['low']<=ct:  # TP hit (price dropped to target)
                pnls.append(tp_val);in_trade=False
                if j==si:exit_at_si=True;break
            if b['high']>=cs:  # SL hit (price rose to stop)
                pnls.append(-sl_val);in_trade=False
                if j==si:exit_at_si=True;break
            # For shorts: wr recovery = wr falls below threshold (not above)
            if b['wr']<=LONG_RECOVERY:reached=True
            if b['wr']>LONG_WEAK:wc+=1
            else:wc=0
            if reached and post:
                px=b['close_ask'];pnls.append(ep-px);in_trade=False
                if j==si:exit_at_si=True;break
            if not reached and wc>=LONG_WT:
                px=b['close_ask'];pnls.append(ep-px);in_trade=False
                if j==si:exit_at_si=True;break
        bh=si-ei
        if not in_trade:
            if exit_at_si:sig_idx+=1;continue
            in_trade=True;ep=d15.iloc[si]['close_bid'];ct=ep-tp_val;cs=ep+sl_val
            ei=si;bh=0;reached=False;wc=0;sig_idx+=1;continue
        # Advance targets for short: ratchet TP lower, SL lower
        ne=d15.iloc[si]['close_bid'];ct=ne-tp_val;cs=min(cs,ne+sl_val)
        ei=si;bh=0;reached=False;wc=0;sig_idx+=1
    if in_trade:
        last=min(ei+LONG_MAX_B,len(d15)-1)
        px=d15.iloc[last]['close_ask'];pnls.append(ep-px)
    return pnls

def stats(pnls):
    if not pnls:return dict(t=0,pnl=0,wr=0,pf=0)
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return dict(t=n,pnl=t,wr=wr,pf=ps/ns if ns>0 else 99)

print('='*72)
print('  WR90 SHORT (Overbought) Quick Test')
print(f'  WR>80  CV≥{LONG_CV}  EpB≥{LONG_EP_MIN}  max_bars={LONG_MAX_B}')
print('='*72)

d1m=load();d15=build_15m(d1m)
sigs=find_short_signals(d15,80)
print(f'\nWR>80 signals: {len(sigs)}  (cf. WR<-80 long: 285)')
if len(sigs)==0:
    print('No signals found — WR rarely exceeds +80!')
else:
    # Quick sweep
    for tp in [60,70,80,90]:
        for sl in [30,40,50,60]:
            pnls=sim_short_advance(d15,sigs,tp,sl)
            s=stats(pnls)
            print(f'  TP={tp} SL={sl}: {s["t"]}t PnL={s["pnl"]:+.0f} WR={s["wr"]:.1f}% PF={s["pf"]:.2f} Avg={s["pnl"]/s["t"]:+.1f}')

print(f'\nDone.')
