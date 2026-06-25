#!/usr/bin/env python3
"""Oil Retrace — TP/SL Sweep + XGBoost ML Filter + Full Stats
================================================================
FIXED: No cascade trades. One entry per signal. Timeout capped at -SL.

Set SKIP_SWEEP=1 env var to skip Parts 1-2 (TP/SL + entry param sweeps).
Part 3 (ML filter) and Part 4 (combined stats) always run.
"""
import sys; from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np; import pandas as pd; import xgboost as xgb; import os as _os
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

SKIP_SWEEP = _os.environ.get('SKIP_SWEEP', '0') == '1'

NY_S,NY_E,NY_FC_H,NY_FC_M=3,12,14,28;LONG_MAX_B=60
LONG_TP,LONG_SL=60,20
RET_CHG,RET_WICK=-10,16
SI_CHANGE_MAX,SI_VOL_MIN=-14.0,800
SI_TP,SI_SL,SI_MAX_B=120,80,90;SI_PROB=0.55

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

FEATS=['cad','avg_r3','bc','wb','range','ret_1b','ret_3b','ret_5b','vol_r','h_dlow','l_dlow','body','up','up_p1','up_p2','body_p1','range_p1']

# ---- simulation: original proven logic (from v29 that ran correctly) ----
def sim_no_cascade(d,sigs,tp,sl,stype='ret'):
    p=[];tr=[];it=False;ct=cs=ep=ei=eb=0;si=0
    while si<len(sigs):
        si_i=sigs[si]['idx']
        if not it:
            it=True;eb=si_i;ep=d.iloc[si_i]['close_ask'];ct=ep+tp;cs=ep-sl;ei=si_i;si+=1;continue
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
        ne=d.iloc[si_i]['close_ask'];ct=max(ct,ne+tp);cs=cs if cs<ne-sl else max(cs,ne-sl);si+=1
    if it:
        last=min(ei+LONG_MAX_B,len(d)-1)
        pnl=max(d.iloc[last]['close_bid']-ep,-sl);p.append(pnl)
        tr.append({'entry':d.index[eb],'exit':d.index[last],'pnl':pnl,'r':'timeout','type':stype,'side':1})
    return p,tr

def sim_full(d,sigs,tp,sl,stype='ret'):
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

def ret_sigs(d,dlow,rng):
    mask=(d['cad']>dlow)&(d['avg_r3']>rng)&(d['bc']<RET_CHG)&(d['wb']<RET_WICK)&d['ins']
    return [{'idx':i} for i in range(len(d)) if mask.iloc[i]]

def wr_sigs(d):
    in_s=d['ins'];o=(d['wr']<-80)&in_s
    s=[];ie=False;cv=0.0;bc=0
    for i in range(len(d)):
        if o.iloc[i]:
            if not ie:cv=0.0;bc=0
            ie=True;cv+=d['volume'].iloc[i];bc+=1
        elif ie:
            if i<len(d)-1 and in_s.iloc[i] and cv>=15000 and bc>=3:
                s.append({'idx':i})
            ie=False;cv=0.0;bc=0
    return s

# ---- MAIN ----
print('='*72)
print('  OIL RETRACE — SWEEP + ML FILTER + FULL STATS')
if SKIP_SWEEP:print('  (SKIP_SWEEP=1 — Parts 1-2 skipped)')
print('='*72)
d1m=load();d=build(d1m)
print(f'Data: {len(d):,} 15m bars\n')

# ---- PART 1: TP/SL SWEEP ----
if not SKIP_SWEEP:
    print('='*72)
    print('  PART 1: TP/SL Sweep for Retrace (Dlow>40, Rng>50)')
    print('='*72)
    sigs_r=ret_sigs(d,40,50)
    print(f'Signal bars: {len(sigs_r)}')
    tp_range=[30,40,50,60,70,80,100];sl_range=[15,20,25,30,40,50,60]
    print(f'  {"TP":>5s} {"SL":>5s} {"T":>5s} {"PnL":>9s} {"WR":>7s} {"PF":>7s} {"Avg":>8s} {"DD":>9s}')
    results=[]
    for tp in tp_range:
        for sl in sl_range:
            p,tr=sim_no_cascade(d,sigs_r,tp,sl,'ret')
            if not p:continue
            n=len(p);t=sum(p);wr=sum(1 for x in p if x>0)/n*100
            ps=sum(x for x in p if x>0);ns=abs(sum(x for x in p if x<0))
            pf=ps/ns if ns>0 else 99
            cs=pd.Series(p).cumsum();dd=float((cs-cs.cummax()).min())
            results.append({'tp':tp,'sl':sl,'t':n,'pnl':t,'wr':wr,'pf':pf,'avg':t/n,'dd':dd})
    rdf=pd.DataFrame(results).sort_values('pnl',ascending=False)
    for _,r in rdf.head(10).iterrows():
        print(f'  {int(r["tp"]):>5d} {int(r["sl"]):>5d} {int(r["t"]):>5d} {int(r["pnl"]):>+9.0f} {r["wr"]:>6.1f}% {r["pf"]:>6.2f} {r["avg"]:>+8.2f} {int(r["dd"]):>+9.0f}')

# ---- PART 2: SWEEP ENTRY PARAMS ----
if not SKIP_SWEEP:
    print(f'\n{"="*72}')
    print('  PART 2: Entry Param Sweep (TP=50/SL=50)')
    print('='*72)
    print(f'  {"Dlow":>5s} {"Rng3":>5s} {"Sigs":>6s} {"T":>5s} {"PnL":>9s} {"WR":>6s} {"PF":>6s}')
    es=[]
    for dl in [20,30,40,50,60,80,100]:
        for r3 in [30,40,50,60,80]:
            sigs=ret_sigs(d,dl,r3)
            if len(sigs)<20:continue
            p,tr=sim_no_cascade(d,sigs,50,50,'ret')
            n=len(p);t=sum(p);wr=sum(1 for x in p if x>0)/n*100
            ps=sum(x for x in p if x>0);ns=abs(sum(x for x in p if x<0))
            pf=ps/ns if ns>0 else 99
            es.append({'dl':dl,'r3':r3,'sigs':len(sigs),'t':n,'pnl':t,'wr':wr,'pf':pf})
    edf=pd.DataFrame(es).sort_values('pnl',ascending=False)
    for _,r in edf.head(10).iterrows():
        print(f'  {int(r["dl"]):>5d} {int(r["r3"]):>5d} {int(r["sigs"]):>6d} {int(r["t"]):>5d} {int(r["pnl"]):>+9.0f} {r["wr"]:>5.1f}% {r["pf"]:>5.2f}')
    best_row=edf.iloc[0] if len(edf)>0 else None
    if best_row is not None:BDL,BR3=int(best_row['dl']),int(best_row['r3'])
    else:BDL,BR3=40,50
    print(f'\nBest entry: Dlow>{BDL} Rng3>{BR3}')

# ---- PART 3: ML FILTER ----
BDL,BR3=20,30
print(f'\n{"="*72}')
print(f'  PART 3: XGBoost ML Filter (Dlow>{BDL} Rng3>{BR3})')
print('='*72)

sigs=ret_sigs(d,BDL,BR3)

def train_ml(d,sigs,tp,sl):
    p,tr,m=sim_full(d,sigs,tp,sl,'ret')
    if len(p)<30:return None
    n_m=len(m)
    X=np.array([[float(d.iloc[sigs[si]['idx']][f]) for f in FEATS] for si in m])
    y=np.array([1.0 if p[i]>0 else 0.0 for i in range(n_m)])
    p=p[:n_m];tr=tr[:n_m]
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
        model=xgb.XGBClassifier(n_estimators=120,max_depth=4,learning_rate=0.03,subsample=0.8,
                               scale_pos_weight=spw,random_state=42,verbosity=0)
        model.fit(Xb,yb);prib=model.predict_proba(X[tst])[:,1]
        for j,idx in enumerate(np.where(tst)[0]):pr[idx]=prib[j]
    return p,tr,pr

print(f'\n{"="*72}')
print(f'  PART 3a: TP/SL Sweep with ML Filter')
print('='*72)
tp_sl_combos=[(40,15),(30,15),(40,20),(30,20),(50,20),(40,25),(50,25),(60,25),(40,30),(50,30)]
print(f'  {"TP":>4s} {"SL":>4s} {"T":>5s} {"PnL":>9s} {"WR":>7s}  Best-ML-Thr {"T_ml":>5s} {"PnL_ml":>9s} {"WR_ml":>7s}')
best_overall={'tp':0,'sl':0,'th':0,'pnl':-999999,'pnls':[],'trades':[],'probas':[]}
for tp,sl in tp_sl_combos:
    res=train_ml(d,sigs,tp,sl)
    if res is None:continue
    pnls,tr,probas=res
    n=len(pnls);t=sum(pnls);wr=sum(1 for x in pnls if x>0)/n*100
    best_th,best_pnl_i=0.50,-999999
    for th in [0.50,0.52,0.55,0.58,0.60,0.65,0.70,0.75,0.80]:
        idx=[i for i in range(n) if probas[i]>=th]
        if len(idx)<5:continue
        tf=sum(pnls[i] for i in idx)
        if tf>best_pnl_i:best_pnl_i=tf;best_th=th
    idx_ml=[i for i in range(n) if probas[i]>=best_th]
    n_ml=len(idx_ml);t_ml=sum(pnls[i] for i in idx_ml);wr_ml=sum(1 for i in idx_ml if pnls[i]>0)/n_ml*100
    print(f'  {tp:>4d} {sl:>4d} {n:>5d} {t:>+9.0f} {wr:>6.1f}%  ML≥{best_th:.2f} {n_ml:>5d} {t_ml:>+9.0f} {wr_ml:>6.1f}%')
    if t_ml>best_overall['pnl']:
        best_overall={'tp':tp,'sl':sl,'th':best_th,'pnl':t_ml,'pnls':pnls,'trades':tr,'probas':probas}

# Override: use specific combo from sweep if FORCE_CONFIG set (format: TP,SL,THRESH)
_fc = _os.environ.get('FORCE_CONFIG', '')
if _fc:
    _parts = _fc.split(',')
    _ftp, _fsl, _fth = int(_parts[0]), int(_parts[1]), float(_parts[2])
    # find that combo in the sweep results
    for _tp, _sl in tp_sl_combos:
        if _tp == _ftp and _fsl == _fsl:
            _res = train_ml(d, sigs, _ftp, _fsl)
            if _res:
                pnls_all, tr_all, probas = _res
                best_overall = {'tp': _ftp, 'sl': _fsl, 'th': _fth, 'pnl': sum(pnls_all), 'pnls': pnls_all, 'trades': tr_all, 'probas': probas}
            break

BTP=best_overall['tp'];BSL=best_overall['sl'];BTH=best_overall['th']
pnls_all=best_overall['pnls'];tr_all=best_overall['trades'];probas=best_overall['probas']
print(f'\nBest ret config: TP={BTP} SL={BSL} ML≥{BTH:.2f}  →  PnL={best_overall["pnl"]:+.0f}')

print(f'\n  PART 3b: ML Threshold Detail (TP={BTP} SL={BSL})')
print('='*72)
n_all=len(pnls_all);t_all=sum(pnls_all);wr_all=sum(1 for x in pnls_all if x>0)/n_all*100
print(f'  Unfiltered: {n_all}t  PnL={t_all:+.0f}  WR={wr_all:.1f}%')
print(f'  {"Thresh":>7s} {"T":>5s} {"PnL":>9s} {"WR":>7s} {"PF":>7s} {"Avg":>8s}')
for th in [0.50,0.52,0.55,0.58,0.60,0.65,0.70,0.75,0.80]:
    idx=[i for i in range(n_all) if probas[i]>=th]
    if len(idx)<5:continue
    fp=[pnls_all[i] for i in idx]
    nf=len(fp);tf=sum(fp);wf=sum(1 for x in fp if x>0)/nf*100
    ps=sum(x for x in fp if x>0);ns=abs(sum(x for x in fp if x<0))
    pf2=ps/ns if ns>0 else 99
    print(f'  ML≥{th:.2f}: {nf:>5d} {tf:>+9.0f} {wf:>6.1f}% {pf2:>6.2f} {tf/nf:>+8.2f}')

best_ml_th=BTH

# ---- PART 4: COMBINED FULL STATS ----
print(f'\n{"="*72}')
print(f'  PART 4: Full V14-Style Stats (TP={BTP}/SL={BSL} ML≥{best_ml_th:.2f})')
print('='*72)

ret_idx=[i for i in range(len(pnls_all)) if probas[i]>=best_ml_th]
pnls_ret_ml=[pnls_all[i] for i in ret_idx]
trades_ret_ml=[tr_all[i] for i in ret_idx]

# ---- WR90 Long (ML-filtered, relaxed entry) ----
WR_ENTRY, WR_CV, WR_EP, WR_ML_TH = -75, 5000, 2, 0.65
print(f'\n  WR90 ML: WR<{WR_ENTRY}, CV≥{WR_CV}, Ep≥{WR_EP}, ML≥{WR_ML_TH}')
def wr_sigs_loose(d,entry_wr,cv_min,ep_min):
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

WR_FEATS=['wr','volume','range','avg_r3','cad','ret_1b','ret_3b','vol_r','h_dlow','l_dlow','body','up','up_p1']
def sim_full_wr(d,sigs,tp,sl):
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
            tr.append({'entry':d.index[eb],'exit':d.index[ex_i],'pnl':pnl,'r':er,'type':'wr90','side':1})
            it=False
            if ex_i==si_i:si+=1
            continue
        if si_i-ei>LONG_MAX_B:
            pnl=max(d.iloc[ei+LONG_MAX_B]['close_bid']-ep,-sl);p.append(pnl)
            tr.append({'entry':d.index[eb],'exit':d.index[ei+LONG_MAX_B],'pnl':pnl,'r':'timeout','type':'wr90','side':1})
            it=False;continue
        ct=max(ct,d.iloc[si_i]['close_ask']+tp);cs=cs if cs<d.iloc[si_i]['close_ask']-sl else max(cs,d.iloc[si_i]['close_ask']-sl);si+=1
    if it:
        last=min(ei+LONG_MAX_B,len(d)-1);pnl=max(d.iloc[last]['close_bid']-ep,-sl);p.append(pnl)
        tr.append({'entry':d.index[eb],'exit':d.index[last],'pnl':pnl,'r':'timeout','type':'wr90','side':1})
    return p,tr,m

def train_ml_wr(d,sigs,tp,sl):
    p,tr,m=sim_full_wr(d,sigs,tp,sl)
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

sigs_wl=wr_sigs_loose(d,WR_ENTRY,WR_CV,WR_EP)
res_w=train_ml_wr(d,sigs_wl,LONG_TP,LONG_SL)
if res_w:
    pnls_w_raw,tr_w_raw,probas_w=res_w
    wr_idx=[i for i in range(len(pnls_w_raw)) if probas_w[i]>=WR_ML_TH]
    pnls_w=[pnls_w_raw[i] for i in wr_idx];tr_w=[tr_w_raw[i] for i in wr_idx]
    print(f'    WR90 ML≥{WR_ML_TH:.2f}: {len(pnls_w)}t  PnL={sum(pnls_w):+.0f}  WR={sum(1 for x in pnls_w if x>0)/len(pnls_w)*100:.1f}%')
else:
    sigs_w=wr_sigs(d)
    pnls_w,tr_w=sim_no_cascade(d,sigs_w,LONG_TP,LONG_SL,'wr90')
    print(f'    WR90 fallback: {len(pnls_w)}t  PnL={sum(pnls_w):+.0f}')

# Short Impulse
d1m_si=d1m.copy()
d1m_si['chg']=d1m_si['close_ask']-d1m_si['open']
d1m_si['pc']=d1m_si['chg'].shift(1);d1m_si['pc2']=d1m_si['chg'].shift(2)
d1m_si['plw']=d1m_si['close_ask'].shift(1)-d1m_si['low'].shift(1)
d1m_si['pv']=d1m_si['volume'].shift(1)
d1m_si['pr']=d1m_si['high'].shift(1)-d1m_si['low'].shift(1)
d1m_si['ps']=d1m_si['close_ask'].shift(1)-d1m_si['close_bid'].shift(1)
tr_si=pd.concat([d1m_si['high']-d1m_si['low'],abs(d1m_si['high']-d1m_si['close_ask'].shift()),
              abs(d1m_si['low']-d1m_si['close_ask'].shift())],axis=1).max(axis=1)
d1m_si['atr']=tr_si.rolling(14).mean()
d1m_si['atr_r']=d1m_si['pr']/(d1m_si['atr']+0.01)
d1m_si['r1']=d1m_si['close_ask'].pct_change()
d1m_si['r3']=d1m_si['r1'].rolling(3,1).sum()
d1m_si['r5']=d1m_si['r1'].rolling(5,1).sum()
d1m_si['vma']=d1m_si['volume'].rolling(20,5).mean()
d1m_si['vr20']=d1m_si['pv']/(d1m_si['vma']+0.01)
d1m_si['nyhr']=d1m_si.index.tz_convert('America/New_York').hour.isin(list(range(3,13)))
dtmp=d1m_si.resample('15min',label='right',closed='right').agg({'open':'first','close_ask':'last'}).dropna()
dtmp['u']=np.where(dtmp['close_ask']>dtmp['open'],1,np.where(dtmp['close_ask']<dtmp['open'],-1,0))
dtmp['uc3']=dtmp['u'].rolling(3,1).sum()
dtmp['r15_3']=dtmp['close_ask'].pct_change().rolling(3,1).sum()
dtmp['r15_5']=dtmp['close_ask'].pct_change().rolling(5,1).sum()
f15=dtmp[['uc3','r15_3','r15_5']].reset_index()
m15=pd.merge_asof(d1m_si.reset_index().sort_values('timestamp'),f15.rename(columns={'timestamp':'t15'}),
                   left_on='timestamp',right_on='t15',direction='backward',tolerance=pd.Timedelta(minutes=15))
m15.index=m15['timestamp'];d1m_si['uc3_15']=m15['uc3']
d1m_si['r15_3']=m15['r15_3'];d1m_si['r15_5']=m15['r15_5']
dh=d1m_si['high'].resample('D').max().reindex(d1m_si.index,method='ffill')
d1m_si['ddh']=dh-d1m_si['close_ask']
si_mask=((d1m_si['pc']<SI_CHANGE_MAX)&(d1m_si['pc2']<10.0)&(d1m_si['pc2']>-14.0)&
          (d1m_si['plw']<35.0)&(d1m_si['pv']>SI_VOL_MIN)&d1m_si['nyhr']&
          (d1m_si['uc3_15']!=-3)&(d1m_si['ddh']<180.0))
si_sigs=sorted(d1m_si.index[si_mask].tolist())
def sim_si(ei,ep,df):
    stop=ep+SI_SL;target=ep-SI_TP;horizon=min(SI_MAX_B,len(df)-ei-1)
    nyz=df.index.tz_convert('America/New_York')
    for i in range(1,horizon+1):
        b=df.iloc[ei+i];bh=nyz[ei+i]
        if bh.hour>NY_FC_H or (bh.hour==NY_FC_H and bh.minute>=NY_FC_M):return df.iloc[ei+i]['close_ask'],i,'ny_close',df.index[ei+i]
        if b['high']>=stop:return stop,i,'sl',df.index[ei+i]
        if b['low']<=target:return target,i,'tp',df.index[ei+i]
    return df.iloc[ei+horizon]['close_ask'],horizon,'timeout',df.index[ei+horizon]
si_recs=[];in_si=False;si_exit_bar=-1
for sig in si_sigs:
    ei=d1m_si.index.get_loc(sig)
    if ei+SI_MAX_B>=len(d1m_si):continue
    if in_si and ei<=si_exit_bar:continue
    ep=d1m_si.iloc[ei]['close_bid'];ex,bars,reason,et=sim_si(ei,ep,d1m_si)
    si_recs.append({'entry_idx':sig,'pnl':ep-ex,'reason':reason,'exit_ts':et})
    in_si=True;si_exit_bar=ei+bars
si_features=['pc','pc2','plw','pv','pr','ps','atr','atr_r','r1','r3','r5','vr20','uc3_15','r15_3','r15_5','ddh']
dates_si=pd.DatetimeIndex([r['entry_idx'] for r in si_recs])
months_si=sorted(set(pd.Period(dd,'M') for dd in dates_si))
si_probas=np.zeros(len(si_recs))
for tm in [m for m in months_si if m>=pd.Period('2024-07',freq='M')]:
    train_m=[m for m in months_si if m<tm]
    tst=np.array([pd.Period(dd,'M')==tm for dd in dates_si]);trn=np.array([pd.Period(dd,'M') in train_m for dd in dates_si])
    X=np.array([[float(d1m_si.loc[r['entry_idx']].get(f,0)) for f in si_features] for r in si_recs])
    y=np.array([1.0 if r['pnl']>0 else 0.0 for r in si_recs])
    X_tr,y_tr=X[trn],y[trn];X_te=X[tst]
    if len(X_tr)<20 or len(X_te)<3:continue
    w=np.where(y_tr==1)[0];l=np.where(y_tr==0)[0];nm=min(len(w),len(l))
    if nm<5:continue
    rng=np.random.RandomState(42+tm.ordinal)
    bal=np.concatenate([rng.choice(w,nm,0),rng.choice(l,nm,0)])
    Xb,yb=X_tr[bal],y_tr[bal];spw=len(l)/max(1,len(w))
    model=xgb.XGBClassifier(n_estimators=80,max_depth=3,learning_rate=0.05,subsample=0.8,scale_pos_weight=spw,random_state=42,verbosity=0)
    model.fit(Xb,yb);probas_te=model.predict_proba(X_te)[:,1]
    for j,idx in enumerate(np.where(tst)[0]):si_probas[idx]=probas_te[j]

si_pnls=[r['pnl'] for i,r in enumerate(si_recs) if si_probas[i]>=SI_PROB]
trades_si=[{'entry':r['entry_idx'],'exit':r['exit_ts'],'pnl':r['pnl'],'r':r['reason'],'type':'short_impulse','side':-1}
           for i,r in enumerate(si_recs) if si_probas[i]>=SI_PROB]

all_trades=tr_w+trades_si+trades_ret_ml
tdf=pd.DataFrame(all_trades)
tdf['pnl']=tdf['pnl'].astype(float)
tdf['entry']=pd.to_datetime(tdf['entry'],utc=True)

n=len(tdf);wins=int((tdf['pnl']>0).sum());net=float(tdf['pnl'].sum())
wr_pct=wins/n*100;cs2=tdf['pnl'].cumsum();mdd=float((cs2-cs2.cummax()).min())
gw=float(tdf[tdf['pnl']>0]['pnl'].sum());gl=abs(float(tdf[tdf['pnl']<0]['pnl'].sum()))
pf=gw/gl if gl>0 else float('inf')
tdf['day']=tdf['entry'].dt.tz_convert('America/New_York').dt.floor('D')
dpnl=tdf.groupby('day')['pnl'].sum()
mday=float(dpnl.mean()) if len(dpnl) else 0
sday=float(dpnl.std(ddof=1)) if len(dpnl)>1 else 0
dd_s=dpnl[dpnl<0];dstd=float(dd_s.std(ddof=1)) if len(dd_s)>1 else 0
sharpe=(mday/sday)*np.sqrt(252) if sday>0 else 0
sortino=(mday/dstd)*np.sqrt(252) if dstd>0 else 0

print(f'  Trades       : {n}  (W:{wins}  L:{n-wins})')
print(f'  Win Rate     : {wr_pct:.1f}%')
print(f'  Net PnL      : {net:+.1f} pts')
print(f'  Avg/Trade    : {net/n:+.2f} pts')
print(f'  Max DD       : {mdd:+.1f} pts')
print(f'  Profit Factor: {pf:.2f}')
print(f'  Sharpe (ann) : {sharpe:.2f}')
print(f'  Sortino (ann): {sortino:.2f}')
print(f'  Recovery     : {net/abs(mdd):.2f}')

for pat,grp in tdf.groupby('type'):
    pw=(grp['pnl']>0).mean()*100;pn=len(grp);ps=grp['pnl'].sum()
    print(f'    {pat:20s}: {pn:4d}t  PnL={ps:+8.1f}  WR={pw:5.1f}%  avg={ps/pn:+7.2f}')
print(f'  Exit Breakdown:')
for r,grp in tdf.groupby('r'):
    rw=(grp['pnl']>0).mean()*100;rn=len(grp);rs=grp['pnl'].sum()
    print(f'    {str(r):18s}: {rn:4d}t  WR={rw:5.1f}%  avg={rs/rn:+7.2f}')

print(f'\n  Yearly:')
tdf['year']=tdf['entry'].dt.year
for y in sorted(tdf['year'].unique()):
    gy=tdf[tdf['year']==y];yn=len(gy);yt=gy['pnl'].sum();yw=(gy['pnl']>0).mean()*100
    print(f'    {y}: {yn:4d}t  PnL={yt:+8.1f}  WR={yw:5.1f}%')

print(f'\n  Monthly:')
monthly=tdf.copy()
monthly['month']=monthly['entry'].dt.tz_convert('Asia/Hong_Kong').dt.strftime('%Y-%m')
mg=monthly.groupby('month')['pnl'].agg(['sum','count'])
mg['wr']=monthly.groupby('month')['pnl'].apply(lambda x:(x>0).mean()*100)
print(f'  {"Month":>8s} {"T":>4s} {"PnL":>8s} {"WR":>5s} {"Cum":>9s}')
cm=0
for m in sorted(mg.index):
    r=mg.loc[m];cm+=r['sum']
    print(f'  {m:>8s} {int(r["count"]):>4d} {r["sum"]:>+8.0f} {r["wr"]:>4.0f}% {cm:>+9.0f}')

print(f'\n{"="*72}')
print(f'  LAST 20 TRADES (HKT)')
print(f'{"="*72}')
last20=tdf.tail(20).copy()
last20['entry_hkt']=pd.to_datetime(last20['entry'],utc=True).dt.tz_convert('Asia/Hong_Kong').dt.strftime('%m-%d %H:%M')
last20['exit_hkt']=pd.to_datetime(last20['exit'],utc=True).dt.tz_convert('Asia/Hong_Kong').dt.strftime('%m-%d %H:%M')
last20['dir']=last20['side'].map({1:'L',-1:'S'})
for _,r in last20.iterrows():
    print(f'  {r["dir"]:>2s} {r["entry_hkt"]:>11s} [{r["exit_hkt"]:>11s}] {r["pnl"]:>+8.1f} {str(r.get("r","?"))[:8]:>8s} {str(r.get("type","?"))[:14]}')
print(f'\n  Net last 20: {last20["pnl"].sum():+.1f} pts')
print(f'\nDONE.')
