#!/usr/bin/env python3
"""Oil Retrace TP/SL Sweep + Overlap Check with WR90 Long.
- Sweeps TP/SL for retrace entries (no pattern, Dlow>40,Rng>50,Chg<-10,Wick<16)
- Proper SL capping (losses capped at SL value)
- Checks for overlapping signals with WR90 Long
- Reports standalone retrace stats AND combined overlap-free stats
"""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd; from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_S,NY_E,NY_FC_H,NY_FC_M=3,12,14,28
LONG_MAX_B=60;LONG_EP_MIN=3;LONG_ENTRY=-80;LONG_CV=15000
LONG_RECOVERY,LONG_WEAK,LONG_WT=-20,-50,12
LONG_TP,LONG_SL=60,20
RET_DLOW,RET_RNG,RET_CHG,RET_WICK=40,50,-10,16

def load():
    loader=DataLoader();raw=loader.load_data('prices','2024-01-01','2026-06-30')
    raw.index=pd.to_datetime(raw['timestamp'],unit='ms')
    df=pd.DataFrame(index=raw.index)
    for c,src in [('open','openPrice_ask'),('high','highPrice_ask'),('low','lowPrice_ask'),
                   ('close_ask','closePrice_ask'),('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c]=raw[src].astype(float)
    if df.index.tz is None:df.index=df.index.tz_localize('UTC')
    return df

def build_15m(df_1m):
    d=df_1m.resample('15min',label='right',closed='right').agg(
        {'open':'first','high':'max','low':'min','close_ask':'last','close_bid':'last','volume':'sum'}).dropna()
    n=14;hh=d['high'].rolling(n).max();ll=d['low'].rolling(n).min()
    d['wr']=((hh-d['close_ask'])/(hh-ll+0.01))*-100
    ny=d.index.tz_convert('America/New_York')
    d['Dlow']=d['low'].groupby(ny.date).transform('min')
    d['range']=d['high']-d['low'];d['avg_range3']=d['range'].rolling(3,min_periods=3).mean()
    d['wick_below']=np.minimum(d['open'],d['close_ask'])-d['low']
    d['bar_change']=d['close_ask']-d['open'];d['close_above_dlow']=d['close_ask']-d['Dlow']
    ny=d.index.tz_convert('America/New_York');d['ny_h']=ny.hour;d['ny_m']=ny.minute
    d['in_sess']=(d['ny_h']>=NY_S)&(d['ny_h']<=NY_E)
    return d

def find_wr90_signals(d):
    in_s=d['in_sess'];o=(d['wr']<LONG_ENTRY)&in_s
    sigs=[];ie=False;cv=0.0;bc=0
    for i_d in range(len(d)):
        if o.iloc[i_d]:
            if not ie:cv=0.0;bc=0
            ie=True;cv+=d['volume'].iloc[i_d];bc+=1
        elif ie:
            ebi=i_d
            if ebi<len(d)-1 and in_s.iloc[ebi] and cv>=LONG_CV and bc>=LONG_EP_MIN:
                sigs.append({'idx':ebi})
            ie=False;cv=0.0;bc=0
    return sigs

def find_ret_signals(d):
    in_s=d['in_sess']
    mask=((d['close_above_dlow']>RET_DLOW)&(d['avg_range3']>RET_RNG)&
          (d['bar_change']<RET_CHG)&(d['wick_below']<RET_WICK)&in_s)
    return [{'idx':i} for i in range(len(d)) if mask.iloc[i]]

def sim_long_advance(d,sigs,tp,sl):
    """WR90 Long — advance target, proper SL capping."""
    pnls=[];it=False;ct=cs=ep=ei=bh=0;rec=False;wc=0;si=0
    while si<len(sigs):
        si_i=sigs[si]['idx']
        if not it:
            it=True;ep=d.iloc[si_i]['close_ask'];ct=ep+tp;cs=ep-sl
            ei=si_i;bh=0;rec=False;wc=0;si+=1;continue
        if si_i-ei>LONG_MAX_B:
            pnls.append(d.iloc[ei+LONG_MAX_B]['close_bid']-ep);it=False;continue
        ex_si=False
        for j in range(ei+bh+1,si_i+1):
            b=d.iloc[j];post=(b['ny_h']>NY_FC_H)or(b['ny_h']==NY_FC_H and b['ny_m']>=NY_FC_M)
            if post:
                pnls.append(b['close_bid']-ep);it=False
                if j==si_i:ex_si=True
                break
            if b['high']>=ct:
                pnls.append(tp);it=False
                if j==si_i:ex_si=True
                break
            if b['low']<=cs:
                pnls.append(-sl);it=False
                if j==si_i:ex_si=True
                break
            if b['wr']>=LONG_RECOVERY:rec=True
            if b['wr']<LONG_WEAK:wc+=1
            else:wc=0
            if rec and post:
                pnls.append(b['close_bid']-ep);it=False
                if j==si_i:ex_si=True
                break
            if not rec and wc>=LONG_WT:
                pnls.append(b['close_bid']-ep);it=False
                if j==si_i:ex_si=True
                break
        bh=si_i-ei
        if not it:
            if ex_si:si+=1;continue
            it=True;ep=d.iloc[si_i]['close_ask'];ct=ep+tp;cs=ep-sl
            ei=si_i;bh=0;rec=False;wc=0;si+=1;continue
        ne=d.iloc[si_i]['close_ask']
        ct=max(ct,ne+tp)
        cs=cs if cs<ne-sl else max(cs,ne-sl)
        ei=si_i;bh=0;rec=False;wc=0;si+=1
    if it:pnls.append(d.iloc[min(ei+LONG_MAX_B,len(d)-1)]['close_bid']-ep)
    return pnls

def sim_retrace_advance(d,sigs,tp,sl):
    """Oil Retrace — advance target, proper SL capping (same logic as WR90)."""
    pnls=[];it=False;ct=cs=ep=ei=bh=0;si=0
    while si<len(sigs):
        si_i=sigs[si]['idx']
        if not it:
            it=True;ep=d.iloc[si_i]['close_ask'];ct=ep+tp;cs=ep-sl
            ei=si_i;bh=0;si+=1;continue
        if si_i-ei>LONG_MAX_B:
            pnls.append(d.iloc[ei+LONG_MAX_B]['close_bid']-ep);it=False;continue
        ex_si=False
        for j in range(ei+bh+1,si_i+1):
            b=d.iloc[j];post=(b['ny_h']>NY_FC_H)or(b['ny_h']==NY_FC_H and b['ny_m']>=NY_FC_M)
            if post:
                pnls.append(b['close_bid']-ep);it=False
                if j==si_i:ex_si=True
                break
            if b['high']>=ct:
                pnls.append(tp);it=False
                if j==si_i:ex_si=True
                break
            if b['low']<=cs:
                pnls.append(-sl);it=False
                if j==si_i:ex_si=True
                break
        bh=si_i-ei
        if not it:
            if ex_si:si+=1;continue
            it=True;ep=d.iloc[si_i]['close_ask'];ct=ep+tp;cs=ep-sl
            ei=si_i;bh=0;si+=1;continue
        ne=d.iloc[si_i]['close_ask']
        ct=max(ct,ne+tp)
        cs=cs if cs<ne-sl else max(cs,ne-sl)
        ei=si_i;bh=0;si+=1
    if it:pnls.append(d.iloc[min(ei+LONG_MAX_B,len(d)-1)]['close_bid']-ep)
    return pnls

def stats(pnls):
    if not pnls:return{'t':0,'pnl':0,'wr':0,'pf':0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return{'t':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99}

print('='*72)
print('  OIL RETRACE TP/SL SWEEP + OVERLAP CHECK')
print(f'  Retrace: Dlow>{RET_DLOW} Rng>{RET_RNG} Chg<{RET_CHG} Wick<{RET_WICK}')
print('='*72)
d1m=load();d15=build_15m(d1m)

# Get both signal sets
wr90_sigs=find_wr90_signals(d15)
ret_sigs=find_ret_signals(d15)
print(f'\nWR90 Long signals: {len(wr90_sigs)}')
print(f'Oil Retrace signals: {len(ret_sigs)}')

# Overlap check: which retrace signals fire on same bar as WR90?
wr90_bars=set(s['idx'] for s in wr90_sigs)
ret_bars=set(s['idx'] for s in ret_sigs)
overlap_bars=wr90_bars & ret_bars
print(f'Overlapping bars (both signals fire): {len(overlap_bars)}')

# Split retrace signals into overlapping vs non-overlapping
ret_overlap=[s for s in ret_sigs if s['idx'] in overlap_bars]
ret_clean=[s for s in ret_sigs if s['idx'] not in overlap_bars]
print(f'Retrace signals with overlap: {len(ret_overlap)}')
print(f'Retrace signals without overlap: {len(ret_clean)}')

# Get standalone WR90 result
pnls_wr90=sim_long_advance(d15,wr90_sigs,LONG_TP,LONG_SL)
sw=stats(pnls_wr90)
print(f'\nWR90 Standalone: {sw["t"]}t PnL={sw["pnl"]:+.0f} WR={sw["wr"]:.1f}% PF={sw["pf"]:.2f}')

# Sweep TP/SL for retrace (overlap-excluded)
tp_range=[40,50,60,70,80,90,100]
sl_range=[10,15,20,25,30,40,50]
print(f'\nRetrace TP/SL Sweep (overlap-excluded, {len(ret_clean)} signals):')
print(f'  {"TP":>5s} {"SL":>5s} {"T":>5s} {"PnL":>9s} {"WR":>7s} {"PF":>7s} {"CombPnL":>10s}')
print(f'  {"-"*5} {"-"*5} {"-"*5} {"-"*9} {"-"*7} {"-"*7} {"-"*10}')
results=[]
for tp in tp_range:
    for sl in sl_range:
        # Standalone retrace
        pnls_ret=sim_retrace_advance(d15,ret_clean,tp,sl)
        sr=stats(pnls_ret)
        # Combined: WR90 + retrace (no overlap)
        combined_pnls=pnls_wr90+pnls_ret
        sc=stats(combined_pnls)
        results.append({'tp':tp,'sl':sl,'t':sr['t'],'pnl':sr['pnl'],'wr':sr['wr'],'pf':sr['pf'],'comb_pnl':sc['pnl']})

rdf=pd.DataFrame(results)
top=rdf.sort_values('pnl',ascending=False).head(15)
for _,r in top.iterrows():
    print(f'  {int(r["tp"]):>5d} {int(r["sl"]):>5d} {int(r["t"]):>5d} {int(r["pnl"]):>+9.0f} {r["wr"]:>6.1f}% {r["pf"]:>6.2f} {int(r["comb_pnl"]):>+10.0f}')

best=rdf.sort_values('comb_pnl',ascending=False).iloc[0]
print(f'\n  Best Combined: TP={int(best["tp"])} SL={int(best["sl"])} → '
      f'Retrace={int(best["pnl"]):+d}  Combined={int(best["comb_pnl"]):+d}pts  WR={best["wr"]:.1f}%  PF={best["pf"]:.2f}')

# Show overlap impact
print(f'\n  Overlap impact:')
pnls_ret_all=sim_retrace_advance(d15,ret_sigs,60,20)
sr_all=stats(pnls_ret_all)
pnls_ret_clean=sim_retrace_advance(d15,ret_clean,60,20)
sr_clean=stats(pnls_ret_clean)
print(f'  All retrace signals ({len(ret_sigs)}):  {sr_all["t"]}t PnL={sr_all["pnl"]:+.0f}')
print(f'  Clean (no overlap) ({len(ret_clean)}):    {sr_clean["t"]}t PnL={sr_clean["pnl"]:+.0f}')
print(f'  Combined with WR90 (overlap-free): {sw["t"]+sr_clean["t"]}t PnL={sw["pnl"]+sr_clean["pnl"]:+.0f}')
print(f'\nDONE.')
