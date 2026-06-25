#!/usr/bin/env python3
"""WR90 Long TP/SL sweep — find best TP/SL combo with advance target."""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_S=3;NY_E=12;NY_FC_H=14;NY_FC_M=28
LONG_MAX_B=60;LONG_EP_MIN=3;LONG_ENTRY=-80;LONG_CV=15000
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

def find_signals(d):
    in_s=d['in_sess'];o=(d['wr']<LONG_ENTRY)&in_s
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

def sim_long_advance(d15,sigs,tp_val,sl_val):
    """WR90 Long with advance target, capped SL. Entry at ask, exit at bid.
    
    Mirror of sim_long_with_advance() from v28_wr90_long_plus_short_impulse.py,
    with TP/SL parameterized. Only difference: no re-entry on same signal
    after exit (the re-entry path was causing double-trades in original).
    """
    pnls=[]
    in_trade=False;ct=0;cs=0;ep=0;ei=0;bh=0
    reached=False;wc=0;sig_idx=0
    while sig_idx<len(sigs):
        si=sigs[sig_idx]['idx']
        if not in_trade:
            in_trade=True;ep=d15.iloc[si]['close_ask'];ct=ep+tp_val;cs=ep-sl_val
            ei=si;bh=0;reached=False;wc=0
            sig_idx+=1;continue
        # Timeout: trade exceeded max bars
        if si-ei>LONG_MAX_B:
            px=d15.iloc[ei+LONG_MAX_B]['close_bid'];pnls.append(px-ep)
            in_trade=False;continue
        # Bar-by-bar exit check
        exit_at_si=False
        for j in range(ei+bh+1,si+1):
            b=d15.iloc[j]
            post=(b['ny_h']>NY_FC_H)or(b['ny_h']==NY_FC_H and b['ny_m']>=NY_FC_M)
            if post:
                px=b['close_bid'];pnls.append(px-ep);in_trade=False
                if j==si:exit_at_si=True;break
            if b['high']>=ct:
                pnls.append(tp_val);in_trade=False
                if j==si:exit_at_si=True;break
            if b['low']<=cs:
                pnls.append(-sl_val);in_trade=False
                if j==si:exit_at_si=True;break
            if b['wr']>=LONG_RECOVERY:reached=True
            if b['wr']<LONG_WEAK:wc+=1
            else:wc=0
            if reached and post:
                px=b['close_bid'];pnls.append(px-ep);in_trade=False
                if j==si:exit_at_si=True;break
            if not reached and wc>=LONG_WT:
                px=b['close_bid'];pnls.append(px-ep);in_trade=False
                if j==si:exit_at_si=True;break
        bh=si-ei
        if not in_trade:
            if exit_at_si:sig_idx+=1;continue
            # Exit happened earlier — this signal starts a NEW trade
            in_trade=True;ep=d15.iloc[si]['close_ask'];ct=ep+tp_val;cs=ep-sl_val
            ei=si;bh=0;reached=False;wc=0;sig_idx+=1;continue
        # Still in trade — advance targets
        ne=d15.iloc[si]['close_ask'];ct=ne+tp_val;cs=max(cs,ne-sl_val)
        ei=si;bh=0;reached=False;wc=0;sig_idx+=1
    # Final cleanup
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
print('  WR90 LONG TP/SL SWEEP (WR<-80, ask/bid, advance target)')
print(f'  CV≥{LONG_CV}  EpB≥{LONG_EP_MIN}  max_bars={LONG_MAX_B}')
print('='*72)

d1m=load();d15=build_15m(d1m)
sigs=find_signals(d15)
print(f'\nSignals: {len(sigs)}  |  Period: {d15.index[0]}→{d15.index[-1]}')

tp_range=[40,50,60,70,80,90,100,110,120]
sl_range=[20,30,40,50,60,70,80]
results=[]

for tp in tp_range:
    for sl in sl_range:
        pnls=sim_long_advance(d15,sigs,tp,sl)
        s=stats(pnls)
        results.append({'tp':tp,'sl':sl,'t':s[0],'pnl':s[1],'wr':s[2],'pf':s[3]})

rdf=pd.DataFrame(results)
print(f'\n  {"TP":>5s} {"SL":>5s} {"T":>5s} {"PnL":>9s} {"WR":>7s} {"PF":>7s} {"Avg":>7s}')
print(f'  {"-"*5} {"-"*5} {"-"*5} {"-"*9} {"-"*7} {"-"*7} {"-"*7}')
top=rdf.sort_values('pnl',ascending=False).head(20)
for _,r in top.iterrows():
    avg=r['pnl']/r['t'] if r['t']>0 else 0
    print(f'  {int(r["tp"]):>5d} {int(r["sl"]):>5d} {int(r["t"]):>5d} {int(r["pnl"]):>+9d} {r["wr"]:>6.1f}% {r["pf"]:>6.2f} {avg:>+7.1f}')

best=rdf.sort_values('pnl',ascending=False).iloc[0]
print(f'\n  Best PnL: TP={int(best["tp"])} SL={int(best["sl"])} → {int(best["pnl"]):+d}pts  {best["wr"]:.1f}%  PF={best["pf"]:.2f}  {best["t"]}t')
best_pf=rdf.sort_values('pf',ascending=False).iloc[0]
print(f'  Best PF:  TP={int(best_pf["tp"])} SL={int(best_pf["sl"])} → PF={best_pf["pf"]:.2f}  PnL={int(best_pf["pnl"]):+d}pts  {best_pf["t"]}t')
print(f'\nDONE.')
