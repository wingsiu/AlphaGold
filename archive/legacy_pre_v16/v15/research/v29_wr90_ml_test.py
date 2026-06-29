#!/usr/bin/env python3
"""WR90 ML Filter Test — Relaxed entry + XGBoost walk-forward
================================================================
Tests: WR<-70, CumVol≥8k, EpBars≥2 (relaxed from original WR<-80,CV≥15k,Ep≥3)
Goal: More signals → ML can filter out bad ones → better combined PnL?
"""
import sys; from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np; import pandas as pd; import xgboost as xgb
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_S,NY_E,NY_FC_H,NY_FC_M=3,12,14,28;LONG_MAX_B=60
LONG_TP,LONG_SL=60,20

def load():
    l=DataLoader();r=l.load_data('prices','2024-01-01','2026-06-30')
    r.index=pd.to_datetime(r['timestamp'],unit='ms');df=pd.DataFrame(index=r.index)
    for c,s in [('open','openPrice_ask'),('high','highPrice_ask'),('low','lowPrice_ask'),
                ('close_ask','closePrice_ask'),('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c]=r[s].astype(float)
    if df.index.tz is None:df.index=df.index.tz_localize('UTC')
    return df

def build(d1m):
    d=d1m.resample('15min',label='right',closed='right').agg(
        {'open':'first','high':'max','low':'min','close_ask':'last','close_bid':'last','volume':'sum'}).dropna()
    n=14;hh=d['high'].rolling(n).max();ll=d['low'].rolling(n).min()
    d['wr']=((hh-d['close_ask'])/(hh-ll+0.01))*-100
    ny=d.index.tz_convert('America/New_York')
    d['Dlow']=d['low'].groupby(ny.date).transform('min')
    d['range']=d['high']-d['low'];d['avg_r3']=d['range'].rolling(3,3).mean()
    d['wb']=np.minimum(d['open'],d['close_ask'])-d['low']
    d['bc']=d['close_ask']-d['open']
    d['cad']=d['close_ask']-d['Dlow']
    d['ny_h']=ny.hour;d['ny_m']=ny.minute
    d['ins']=(d['ny_h']>=NY_S)&(d['ny_h']<=NY_E)
    d['ret_1b']=d['close_ask'].pct_change(1);d['ret_3b']=d['close_ask'].pct_change(3)
    d['ret_5b']=d['close_ask'].pct_change(5)
    d['vol_r']=d['volume']/(d['volume'].rolling(20).mean()+0.01)
    d['h_dlow']=d['high']-d['Dlow'];d['l_dlow']=d['low']-d['Dlow']
    d['body']=abs(d['close_ask']-d['open'])
    d['up']=(d['close_ask']>d['open']).astype(int)
    d['up_p1']=d['up'].shift(1);d['up_p2']=d['up'].shift(2)
    d['body_p1']=d['body'].shift(1);d['range_p1']=d['range'].shift(1)
    return d

def wr_sigs(d,entry_wr,cv_min,ep_min):
    in_s=d['ins'];o=(d['wr']<entry_wr)&in_s
    s=[];ie=False;cv=0.0;bc=0
    for i in range(len(d)):
        if o.iloc[i]:
            if not ie:cv=0.0;bc=0
            ie=True;cv+=d['volume'].iloc[i];bc+=1
        elif ie:
            if i<len(d)-1 and in_s.iloc[i] and cv>=cv_min and bc>=ep_min:
                s.append({'idx':i})
            ie=False;cv=0.0;bc=0
    return s

def sim_full(d,sigs,tp,sl,stype='wr90'):
    p=[];tr=[];m=[];it=False;ct=cs=ep=ei=eb=0;si=0
    while si<len(sigs):
        si_i=sigs[si]['idx']
        if not it:
            it=True;eb=si_i;ep=d.iloc[si_i]['close_ask'];ct=ep+tp;cs=ep-sl;ei=si_i
            m.append(si);si+=1;continue
        ex=False;er='';ex_p=0.0;ex_i=ei
        for j in range(ei+1,si_i+1):
            b=d.iloc[j];post=(b['ny_h']>NY_FC_H)or(b['ny_h']==NY_FC_H and b['ny_m']>=NY_FC_M)
            if post:ex=True;er='ny_close';ex_p=b['close_bid'];ex_i=j;break
            if b['high']>=ct:ex=True;er='tp';ex_p=ep+tp;ex_i=j;break
            if b['low']<=cs:ex=True;er='sl';ex_p=ep-sl;ex_i=j;break
        if ex:
            pnl=ex_p-ep;p.append(pnl)
            tr.append({'entry':d.index[eb],'exit':d.index[ex_i],'pnl':pnl,'r':er,'type':stype,'side':1})
            it=False
            if ex_i==si_i:si+=1
            continue
        if si_i-ei>LONG_MAX_B:
            pnl=max(d.iloc[ei+LONG_MAX_B]['close_bid']-ep,-sl);p.append(pnl)
            tr.append({'entry':d.index[eb],'exit':d.index[ei+LONG_MAX_B],'pnl':pnl,'r':'timeout','type':stype,'side':1})
            it=False;continue
        ct=max(ct,d.iloc[si_i]['close_ask']+tp);cs=cs if cs<d.iloc[si_i]['close_ask']-sl else max(cs,d.iloc[si_i]['close_ask']-sl);si+=1
    if it:
        last=min(ei+LONG_MAX_B,len(d)-1);pnl=max(d.iloc[last]['close_bid']-ep,-sl);p.append(pnl)
        tr.append({'entry':d.index[eb],'exit':d.index[last],'pnl':pnl,'r':'timeout','type':stype,'side':1})
    return p,tr,m

WR_FEATS=['wr','volume','range','avg_r3','cad','ret_1b','ret_3b','vol_r','h_dlow','l_dlow','body','up','up_p1']

def train_ml_wr(d,sigs,tp,sl):
    p,tr,m=sim_full(d,sigs,tp,sl,'wr90')
    if len(p)<30:return None
    X=np.array([[float(d.iloc[sigs[si]['idx']][f]) for f in WR_FEATS] for si in m])
    y=np.array([1.0 if x>0 else 0.0 for x in p])
    tdates=pd.DatetimeIndex([d.index[sigs[si]['idx']] for si in m])
    months=sorted(set(pd.Period(dt,'M') for dt in tdates))
    tstart=pd.Period('2024-07',freq='M')
    pr=np.zeros(len(p))
    for tm in [mo for mo in months if mo>=tstart]:
        train_m=[mo for mo in months if mo<tm]
        tst=np.array([pd.Period(dt,'M')==tm for dt in tdates])
        trn=np.array([pd.Period(dt,'M') in train_m for dt in tdates])
        if trn.sum()<20 or tst.sum()<3:continue
        w=np.where(y[trn]==1)[0];l=np.where(y[trn]==0)[0];nm=min(len(w),len(l))
        if nm<5:continue
        rng=np.random.RandomState(42+tm.ordinal)
        bal=np.concatenate([rng.choice(w,nm,0),rng.choice(l,nm,0)])
        Xb,yb=X[trn][bal],y[trn][bal];spw=len(l)/max(1,len(w))
        model=xgb.XGBClassifier(n_estimators=80,max_depth=3,learning_rate=0.05,subsample=0.8,
                               scale_pos_weight=spw,random_state=42,verbosity=0)
        model.fit(Xb,yb);prib=model.predict_proba(X[tst])[:,1]
        for j,idx in enumerate(np.where(tst)[0]):pr[idx]=prib[j]
    return p,tr,pr

print('='*72)
print('  WR90 ML FILTER TEST — Relaxed Entry')
print('='*72)
d1m=load();d=build(d1m)
print(f'Data: {len(d):,} 15m bars\n')

# Original baseline
print(f'  {"Config":>20s} {"Sigs":>6s} {"T":>5s} {"PnL":>9s} {"WR":>7s} {"PF":>7s} {"Avg":>8s}')
sigs_o=wr_sigs(d,-80,15000,3)
p_o,tr_o,_=sim_full(d,sigs_o,LONG_TP,LONG_SL,'wr90')
n=len(p_o);t=sum(p_o);wr=sum(1 for x in p_o if x>0)/n*100
ps=sum(x for x in p_o if x>0);ns=abs(sum(x for x in p_o if x<0))
print(f'  {"Original WR<-80,CV≥15k,Ep≥3":>20s} {len(sigs_o):>6d} {n:>5d} {t:>+9.0f} {wr:>6.1f}% {ps/ns if ns>0 else 99:>6.2f} {t/n:>+8.2f}')

# Sweep relaxed configs
print(f'\n  Sweeping relaxed parameters with ML filter:')
print(f'  {"WR<":>5s} {"CV≥":>6s} {"Ep≥":>5s} {"T":>5s} {"PnL":>9s} {"WR":>7s}  ML-Thr {"T_ml":>5s} {"PnL_ml":>9s} {"WR_ml":>7s}')
best=None
for ew in [-70,-75]:
    for cv in [5000,8000,10000]:
        for ep in [2,3]:
            sigs=wr_sigs(d,ew,cv,ep)
            if len(sigs)<30:continue
            res=train_ml_wr(d,sigs,LONG_TP,LONG_SL)
            if res is None:continue
            pnls,tr,probas=res
            n=len(pnls);t=sum(pnls);wr=sum(1 for x in pnls if x>0)/n*100
            # find best ML threshold
            best_th,best_pnl=0.50,-999999
            for th in [0.50,0.52,0.55,0.58,0.60,0.65,0.70]:
                idx=[i for i in range(n) if probas[i]>=th]
                if len(idx)<5:continue
                tf=sum(pnls[i] for i in idx)
                if tf>best_pnl:best_pnl=tf;best_th=th
            idx_ml=[i for i in range(n) if probas[i]>=best_th]
            n_ml=len(idx_ml);t_ml=sum(pnls[i] for i in idx_ml);wr_ml=sum(1 for i in idx_ml if pnls[i]>0)/n_ml*100
            tag=' *' if best is None or t_ml>(best[3] if best else -1e9) else ''
            print(f'  {ew:>+5d} {cv:>6d} {ep:>5d} {n:>5d} {t:>+9.0f} {wr:>6.1f}%  ML≥{best_th:.2f} {n_ml:>5d} {t_ml:>+9.0f} {wr_ml:>6.1f}%{tag}')
            if best is None or t_ml>best[3]:best=(ew,cv,ep,t_ml,t_ml,n_ml,wr_ml,best_th,pnls,tr,probas)

# Detail for best
if best:
    ew,cv,ep,t_ml_raw,t_ml,n_ml,wr_ml,best_th,pnls,tr,probas=best
    print(f'\n  Best: WR<{ew}, CV≥{cv}, Ep≥{ep} → ML≥{best_th:.2f}: {n_ml}t, +{t_ml:.0f}pts, WR={wr_ml:.1f}%')
    print(f'\n  ML Threshold Detail:')
    print(f'  Unfiltered: {len(pnls)}t  PnL={sum(pnls):+.0f}  WR={sum(1 for x in pnls if x>0)/len(pnls)*100:.1f}%')
    print(f'  {"Thresh":>7s} {"T":>5s} {"PnL":>9s} {"WR":>7s} {"PF":>7s} {"Avg":>8s}')
    for th in [0.50,0.52,0.55,0.58,0.60,0.65,0.70]:
        idx=[i for i in range(len(pnls)) if probas[i]>=th]
        if len(idx)<5:continue
        fp=[pnls[i] for i in idx]
        nf=len(fp);tf=sum(fp);wf=sum(1 for x in fp if x>0)/nf*100
        ps2=sum(x for x in fp if x>0);ns2=abs(sum(x for x in fp if x<0))
        print(f'  WL≥{th:.2f}: {nf:>5d} {tf:>+9.0f} {wf:>6.1f}% {ps2/ns2 if ns2>0 else 99:>6.2f} {tf/nf:>+8.2f}')

print('\nDONE.')
