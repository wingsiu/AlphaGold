#!/usr/bin/env python3
"""Oil Retrace Pattern — parameter sweep at actual data scale (60-70 range).
Thresholds: close-Dlow>60, avgRange3>35, cl-op<-10, wick<8, up-up-down pattern.
These are the CORRECT scales — oil data is ~$60-70, not ×100.
"""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd; import xgboost as xgb
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_S=3; NY_E=12; NY_FC_H=14; NY_FC_M=28
LONG_MAX_B=60

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
    ny=d.index.tz_convert('America/New_York');d['ny_h']=ny.hour;d['ny_m']=ny.minute
    d['in_sess']=(d['ny_h']>=NY_S)&(d['ny_h']<=NY_E)
    ny=d.index.tz_convert('America/New_York')
    d['Dlow']=d['low'].groupby(ny.date).transform('min')
    d['range']=d['high']-d['low']
    d['avg_range3']=d['range'].rolling(3,min_periods=3).mean()
    d['bar_up']=(d['close_ask']>d['open']).astype(int)
    d['bar_down']=(d['close_ask']<d['open']).astype(int)
    d['pat_up_up_down']=((d['bar_up'].shift(2)==1)&(d['bar_up'].shift(1)==1)&(d['bar_down']==1))
    d['wick_below']=np.minimum(d['open'],d['close_ask'])-d['low']
    d['bar_change']=d['close_ask']-d['open']
    d['close_above_dlow']=d['close_ask']-d['Dlow']
    return d

def find_signals(d, dlow_min, avg_range_min, bar_chg_max, wick_max, require_pattern=True):
    in_s=d['in_sess']
    o=(d['close_above_dlow']>dlow_min)& \
      (d['avg_range3']>avg_range_min)& \
      (d['bar_change']<bar_chg_max)& \
      (d['wick_below']<wick_max)& \
      in_s
    if require_pattern: o=o&d['pat_up_up_down']
    sigs=[]
    for i in range(len(d)):
        if o.iloc[i]:sigs.append({'idx':i})
    return sigs

def sim_long_advance(d15,sigs,tp_val,sl_val):
    pnls=[];in_trade=False;ct=0;cs=0;ep=0;ei=0;bh=0;sig_idx=0
    while sig_idx<len(sigs):
        si=sigs[sig_idx]['idx']
        if not in_trade:
            in_trade=True;ep=d15.iloc[si]['close_ask'];ct=ep+tp_val;cs=ep-sl_val
            ei=si;bh=0;sig_idx+=1;continue
        if si-ei>LONG_MAX_B:
            px=d15.iloc[ei+LONG_MAX_B]['close_bid'];pnls.append(px-ep);in_trade=False;continue
        exit_at_si=False
        for j in range(ei+bh+1,si+1):
            b=d15.iloc[j];post=(b['ny_h']>NY_FC_H)or(b['ny_h']==NY_FC_H and b['ny_m']>=NY_FC_M)
            if post:
                px=b['close_bid'];pnls.append(px-ep);in_trade=False
                if j==si:exit_at_si=True
                break
            if b['high']>=ct:
                pnls.append(tp_val);in_trade=False
                if j==si:exit_at_si=True
                break
            if b['low']<=cs:
                pnls.append(-sl_val);in_trade=False
                if j==si:exit_at_si=True
                break
        bh=si-ei
        if not in_trade:
            if exit_at_si:sig_idx+=1;continue
            in_trade=True;ep=d15.iloc[si]['close_ask'];ct=ep+tp_val;cs=ep-sl_val
            ei=si;bh=0;sig_idx+=1;continue
        ne=d15.iloc[si]['close_ask']
        ct=max(ct,ne+tp_val)
        cs=cs if cs<ne-sl_val else max(cs,ne-sl_val)
        ei=si;bh=0;sig_idx+=1
    if in_trade:
        last=min(ei+LONG_MAX_B,len(d15)-1);px=d15.iloc[last]['close_bid'];pnls.append(px-ep)
    return pnls

def stats(pnls):
    if not pnls:return dict(t=0,pnl=0,wr=0,pf=0)
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return dict(t=n,pnl=t,wr=wr,pf=ps/ns if ns>0 else 99)

print('='*72)
print('  OIL RETRACE — Parameter Sweep')
print('='*72)
d1m=load();d15=build_15m(d1m)

dlow_range=[40,50,60,70,80,100,120]
avg_range_range=[20,25,30,35,40,50,60]
bar_chg_range=[-5,-8,-10,-12,-15,-20,-25]
wick_range=[4,6,8,10,12,15,20]
tp_range=[30,40,50,60,70,80,90]
sl_range=[10,15,20,25,30,40,50]

results=[]
for dlow_min in dlow_range:
    for avg_range_min in avg_range_range:
        for bar_chg_max in bar_chg_range:
            for wick_max in wick_range:
                sigs=find_signals(d15,dlow_min,avg_range_min,bar_chg_max,wick_max,True)
                if len(sigs)==0:continue
                best_pnl=-999999;best_tp=0;best_sl=0;best_t=0;best_wr=0;best_pf=0
                for tp in tp_range:
                    for sl in sl_range:
                        pnls=sim_long_advance(d15,sigs,tp,sl)
                        s=stats(pnls)
                        if s['pnl']>best_pnl:
                            best_pnl=s['pnl'];best_tp=tp;best_sl=sl
                            best_t=s['t'];best_wr=s['wr'];best_pf=s['pf']
                results.append({'dlow':dlow_min,'range':avg_range_min,'chg':bar_chg_max,
                    'wick':wick_max,'sigs':len(sigs),'t':best_t,'pnl':best_pnl,
                    'wr':best_wr,'pf':best_pf,'tp':best_tp,'sl':best_sl})

rdf=pd.DataFrame(results).sort_values('pnl',ascending=False)
print(f'\nTop 15 combos (with best TP/SL per combo):')
print(f'  {"Dlow":>5s} {"Rng":>5s} {"Chg":>5s} {"Wick":>5s} {"Sigs":>5s} {"T":>5s} {"PnL":>9s} {"WR":>6s} {"PF":>6s} {"TP":>5s} {"SL":>5s}')
for _,r in rdf.head(15).iterrows():
    print(f'  {int(r["dlow"]):>5d} {int(r["range"]):>5d} {int(r["chg"]):>5d} {int(r["wick"]):>5d} '
          f'{int(r["sigs"]):>5d} {int(r["t"]):>5d} {int(r["pnl"]):>+9.0f} {r["wr"]:>5.1f}% {r["pf"]:>5.2f} '
          f'{int(r["tp"]):>5d} {int(r["sl"]):>5d}')

# Without pattern
print(f'\nTop 10 WITHOUT up-up-down:')
results_np=[]
for dlow_min in [40,60,80,100]:
    for avg_range_min in [20,30,40,50]:
        for bar_chg_max in [-5,-10,-15,-20]:
            for wick_max in [4,8,12,16]:
                sigs=find_signals(d15,dlow_min,avg_range_min,bar_chg_max,wick_max,False)
                if len(sigs)==0:continue
                best_pnl=-999999
                for tp in tp_range:
                    for sl in sl_range:
                        pnls=sim_long_advance(d15,sigs,tp,sl)
                        s=stats(pnls)
                        if s['pnl']>best_pnl:
                            best_pnl=s['pnl'];best_tp=tp;best_sl=sl
                            best_t=s['t'];best_wr=s['wr'];best_pf=s['pf']
                results_np.append({'dlow':dlow_min,'range':avg_range_min,'chg':bar_chg_max,
                    'wick':wick_max,'sigs':len(sigs),'t':best_t,'pnl':best_pnl,
                    'wr':best_wr,'pf':best_pf,'tp':best_tp,'sl':best_sl})
rdf2=pd.DataFrame(results_np).sort_values('pnl',ascending=False)
for _,r in rdf2.head(10).iterrows():
    print(f'  {int(r["dlow"]):>5d} {int(r["range"]):>5d} {int(r["chg"]):>5d} {int(r["wick"]):>5d} '
          f'{int(r["sigs"]):>5d} {int(r["t"]):>5d} {int(r["pnl"]):>+9.0f} {r["wr"]:>5.1f}% {r["pf"]:>5.2f} '
          f'{int(r["tp"]):>5d} {int(r["sl"]):>5d}')

best=rdf.iloc[0] if len(rdf)>0 else None
if best is not None and best['pnl']>0:
    print(f'\nBEST: Dlow>{int(best["dlow"])} Rng>{int(best["range"])} Chg<{int(best["chg"])} Wick<{int(best["wick"])} '
          f'TP={int(best["tp"])} SL={int(best["sl"])} PnL={int(best["pnl"]):+d} WR={best["wr"]:.1f}% PF={best["pf"]:.2f}')
    sigs=find_signals(d15,int(best["dlow"]),int(best["range"]),int(best["chg"]),int(best["wick"]),True)
    feats=['close_above_dlow','avg_range3','bar_change','wick_below','range']
    X=np.array([[d15.iloc[s['idx']][f] for f in feats] for s in sigs])
    tp_v,sl_v=int(best['tp']),int(best['sl'])
    pnls=sim_long_advance(d15,sigs,tp_v,sl_v)
    y=np.array([1.0 if p>0 else 0.0 for p in pnls])
    dates=pd.DatetimeIndex([d15.index[s['idx']] for s in sigs])
    months=sorted(set(d.to_period('M') for d in dates))
    test_start=pd.Period('2024-07',freq='M')
    probas=np.zeros(len(sigs))
    for tm in [m for m in months if m>=test_start]:
        train_m=[m for m in months if m<tm]
        tst=np.array([d.to_period('M')==tm for d in dates])
        trn=np.array([d.to_period('M') in train_m for d in dates])
        X_tr,y_tr=X[trn],y[trn];X_te=X[tst]
        if len(X_tr)<20 or len(X_te)<3:continue
        w=np.where(y_tr==1)[0];l=np.where(y_tr==0)[0];nm=min(len(w),len(l))
        if nm<5:continue
        rng=np.random.RandomState(42+tm.ordinal)
        bal=np.concatenate([rng.choice(w,nm,replace=False),rng.choice(l,nm,replace=False)])
        Xb,yb=X_tr[bal],y_tr[bal];spw=len(l)/max(1,len(w))
        model=xgb.XGBClassifier(n_estimators=80,max_depth=3,learning_rate=0.05,subsample=0.8,
                                 scale_pos_weight=spw,random_state=42,verbosity=0)
        model.fit(Xb,yb);probas_te=model.predict_proba(X_te)[:,1]
        for j,idx in enumerate(np.where(tst)[0]):probas[idx]=probas_te[j]
    print(f'\n  {"Threshold":>12s} {"T":>5s} {"PnL":>9s} {"WR":>7s} {"PF":>7s}')
    for thresh in [0.50,0.52,0.55,0.58,0.60,0.65,0.70]:
        fpnls=[pnls[i] for i in range(len(pnls)) if probas[i]>=thresh]
        s=stats(fpnls);mark=' *' if s['pnl']>best['pnl'] else ''
        print(f'  ML≥{thresh:.2f}:  {s["t"]:>5d} {s["pnl"]:>+9.0f} {s["wr"]:>6.1f}% {s["pf"]:>6.2f}{mark}')
elif best is not None:
    print(f'\nBest still negative ({int(best["pnl"]):+d}) — retrace not viable for oil.')
print(f'\nDONE.')
