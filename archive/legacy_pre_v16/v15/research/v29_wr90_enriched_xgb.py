#!/usr/bin/env python3
'''v29 WR90 CumVol Episode + Enriched Features + XGBoost.
Key improvement: more predictive features per episode, binary target = reaches -20?'''

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd; import xgboost as xgb
from sklearn.metrics import roc_auc_score
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

def load_oil(s='2024-01-01',e='2026-05-22'):
    loader=DataLoader();raw=loader.load_data(table_name='prices',start_date=s,end_date=e)
    raw.index=pd.to_datetime(raw['timestamp'],unit='ms')
    df=pd.DataFrame(index=raw.index)
    for c,src in [('open','openPrice_ask'),('high','highPrice_ask'),('low','lowPrice_ask'),
                   ('close_ask','closePrice_ask'),('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c]=raw[src].astype(float)
    if df.index.tz is None: df.index=df.index.tz_localize('UTC')
    return df

def build_15m(df1m):
    df15=df1m.resample('15min',label='right',closed='right').agg(
        {'open':'first','high':'max','low':'min','close_ask':'last','close_bid':'last','volume':'sum'}).dropna()
    n=14;hh=df15['high'].rolling(n).max();ll=df15['low'].rolling(n).min()
    df15['wr']=((hh-df15['close_ask'])/(hh-ll+0.01))*-100
    df15['hour']=df15.index.hour; df15['day']=df15.index.date
    df15['is_uk']=df15['hour'].isin([7,8,9,10,11,12,13,14,15,16])
    df15['is_us']=df15['hour'].isin([12,13,14,15,16,17,18,19,20])
    df15['vol_ma20']=df15['volume'].rolling(20,min_periods=5).mean()
    df15['vol_ratio']=df15['volume']/(df15['vol_ma20']+0.01)
    df15['range']=df15['high']-df15['low']
    df15['range_ma20']=df15['range'].rolling(20,min_periods=5).mean()
    df15['range_ratio']=df15['range']/(df15['range_ma20']+0.01)
    df15['ret_5']=df15['close_ask'].pct_change(5)
    df15['ret_10']=df15['close_ask'].pct_change(10)
    df15['ret_20']=df15['close_ask'].pct_change(20)
    df15['price']=df15['close_ask']
    # Day-level features
    day_high=df15.groupby('day')['high'].transform('max')
    day_low=df15.groupby('day')['low'].transform('min')
    day_open=df15.groupby('day')['open'].transform('first')
    df15['day_range']=day_high-day_low
    df15['day_fullness']=(df15['close_ask']-day_open)/(df15['day_range']+0.01)
    # Prev day
    df15['prev_day_range']=df15['day_range'].shift(96)  # ~96 15m bars/day
    df15['dow']=pd.to_datetime(df15.index).dayofweek
    # ATR
    tr=pd.concat([df15['high']-df15['low'],
                  abs(df15['high']-df15['close_ask'].shift()),
                  abs(df15['low']-df15['close_ask'].shift())],axis=1).max(axis=1)
    df15['ATR']=tr.rolling(14).mean()
    return df15

def find_enriched_episodes(df15, entry_th=-80, cvmin=10000, tp=80, sl=40, session='uk'):
    in_s=df15['is_uk'] if session=='uk' else (df15['is_us'] if session=='us' else df15['is_uk']|df15['is_us'])
    oversold=(df15['wr']<entry_th)&in_s
    episodes=[]; in_ep=False; ep_start=None; cv=0.0; bc=0; wr_min=-999
    for i in range(len(df15)):
        if oversold.iloc[i]:
            if not in_ep: ep_start=i; cv=0.0; bc=0; wr_min=-999
            in_ep=True; cv+=df15['volume'].iloc[i]; bc+=1
            wr_min=max(wr_min,df15['wr'].iloc[i])  # least negative = max value
        else:
            if in_ep:
                ebi=i
                if ebi<len(df15)-1 and in_s.iloc[ebi] and cv>=cvmin:
                    r=df15.iloc[ebi]
                    eps=df15.iloc[ep_start]
                    # Sim the trade to get actual outcome for this episode
                    ep_p=r['close_ask'];h60=min(60,len(df15)-ebi-1)
                    reached=False; exit_pnl=0
                    for j in range(1,h60+1):
                        bj=df15.iloc[ebi+j]
                        if bj['high']>=ep_p+tp: exit_pnl=tp; break
                        if bj['low']<=ep_p-sl: exit_pnl=-sl; break
                        if bj['wr']>=-20: reached=True
                        if reached and bj.name.hour==16: exit_pnl=bj['close_bid']-ep_p; break
                    else:
                        exit_pnl=df15.iloc[ebi+h60]['close_bid']-ep_p
                    # Binary target: 1 = profitable (tp or ride_end positive), 0 = loss
                    target=1.0 if exit_pnl>0 else 0.0
                    episodes.append({
                        'entry':df15.index[ebi],'entry_pos':ebi,'cum_vol':cv,'bars':bc,'wr_min':wr_min,
                        'wr_entry':r['wr'],'hour':r['hour'],'dow':r['dow'],
                        'vol_ratio':r['vol_ratio'],'range_ratio':r['range_ratio'],
                        'ret_5':r['ret_5'],'ret_10':r['ret_10'],'ret_20':r['ret_20'],
                        'ATR':r['ATR'],'price':r['price'],
                        'day_range':r['day_range'],'day_fullness':r['day_fullness'],
                        'prev_day_range':r['prev_day_range'],
                        'reached_20':reached, 'target':target, 'exit_pnl':exit_pnl,
                    })
                in_ep=False; cv=0.0; bc=0
    return episodes

def stats(pnls):
    if not pnls: return {'trades':0,'pnl':0.0,'wr':0.0,'pf':0.0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'trades':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99}

# XGBoost feature columns
FEAT=['cum_vol','bars','wr_min','wr_entry','hour','dow',
      'vol_ratio','range_ratio','ret_5','ret_10','ret_20',
      'ATR','price','day_range','day_fullness','prev_day_range']

def sim_one(ep,df15,tp,sl,max_bars=60,recovery=-20,weak=-50,wto=12,sess_end=16):
    ei=ep['entry'];ep_p=df15.iloc[ei]['close_ask'];h=min(max_bars,len(df15)-ei-1)
    reached=False;wc=0
    for i in range(1,h+1):
        b=df15.iloc[ei+i]
        if b['high']>=ep_p+tp: return ep_p+tp,i,'tp'
        if b['low']<=ep_p-sl: return ep_p-sl,i,'sl'
        if b['wr']>=recovery: reached=True
        if b['wr']<weak: wc+=1
        else: wc=0
        if reached and b.name.hour==sess_end: return b['close_bid'],i,'ride_end'
        if not reached and wc>=wto: return b['close_bid'],i,'weak'
    return df15.iloc[ei+h]['close_bid'],h,'timeout'

def main():
    print('='*72); print('v29 WR90 Enriched Features + XGBoost'); print('='*72)
    print(); print('[1] Loading...'); df1m=load_oil(); df15=build_15m(df1m)
    print(f'  {len(df15):,} 15m bars')

    # Sweep entry thresholds + CumVol filters + XGBoost
    print(f"\n  {'Entry':>8s} {'CV>':>10s} {'Eps':>7s} {'Win%':>7s} {'WF AUC':>8s} "
          f"{'BaseTrd':>8s} {'BasePnL':>10s} {'BasePF':>7s}")
    print(f"  {'-'*90}")
    
    for eth in [-80,-70]:
        for cmin in [0,5000,10000,15000,20000]:
            eps=find_enriched_episodes(df15,eth,cvmin=cmin)
            if len(eps)<50: continue
            n=len(eps)
            X=np.array([[e[f] for f in FEAT] for e in eps]).astype(float)
            y=np.array([e['target'] for e in eps])
            nv=~np.isnan(X).any(axis=1); X=X[nv];y=y[nv]
            if len(y)<30: continue
            winpct=y.sum()/len(y)*100
            
            dates=pd.to_datetime([eps[i]['entry'] for i in range(len(eps)) if not np.isnan(
                np.array([eps[i][f] for f in FEAT])).any()])
            months=sorted(set(dates.to_period('M')))
            tms=[m for m in months if m>=pd.Period('2024-09',freq='M')]
            probas=np.full(len(y),np.nan)
            for tm in tms:
                tr=dates.to_period('M')<tm; te=dates.to_period('M')==tm
                ti=np.where(tr)[0]; ei=np.where(te)[0]
                if len(ti)<20 or len(ei)<5: continue
                yt=y[ti];w=ti[yt==1];l=ti[yt==0]
                if len(l)>0 and len(w)>0:
                    mn=min(len(w),len(l));rng=np.random.RandomState(42+tm.ordinal)
                    w2=rng.choice(w,mn,replace=False) if len(w)>mn else w
                    l2=rng.choice(l,mn,replace=False) if len(l)>mn else l
                    ti=np.concatenate([w2,l2])
                if len(set(y[ti]))<2: continue
                sc=max(1.0,(len(yt)-yt.sum())/max(yt.sum(),1))
                m=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,
                                     subsample=0.8,colsample_bytree=0.8,scale_pos_weight=sc,
                                     random_state=42,eval_metric='logloss')
                m.fit(X[ti],y[ti]); probas[ei]=m.predict_proba(X[ei])[:,1]
            valid=~np.isnan(probas); nv2=valid.sum()
            auc_str='N/A'
            if nv2>10 and len(set(y[valid]))>1:
                try: auc_val=roc_auc_score(y[valid],probas[valid]); auc_str=f'{auc_val:.3f}'
                except: pass
            
            # Base stats (without ML filter)
            pnls=[e['exit_pnl'] for e in eps]; s=stats(pnls)
            print(f"  {eth:>+8d} {cmin:>10d} {n:>7d} {winpct:>6.1f}% {auc_str:>8s} "
                  f"{s['trades']:>8d} {s['pnl']:>+10.1f} {s['pf']:>6.2f}")

    # Best config: take ALL entry<-80 episodes, train XGBoost, filter with prob
    print(f"\n[3] Prob threshold sweep on ALL Entry<-80 episodes (no CumVol pre-filter)...")
    eps_final=find_enriched_episodes(df15,-70,tp=80,sl=40,cvmin=0)
    Xf=np.array([[e[f] for f in FEAT] for e in eps_final]).astype(float)
    yf=np.array([e['target'] for e in eps_final])
    nvf=~np.isnan(Xf).any(axis=1); Xf=Xf[nvf];yf=yf[nvf]
    dates_f=pd.to_datetime([eps_final[i]['entry'] for i in range(len(eps_final)) if not np.isnan(
        np.array([eps_final[i][f] for f in FEAT])).any()])
    months_f=sorted(set(dates_f.to_period('M')))
    tms_f=[m for m in months_f if m>=pd.Period('2024-09',freq='M')]
    probas_f=np.full(len(yf),np.nan)
    for tm in tms_f:
        tr_f=dates_f.to_period('M')<tm; te_f=dates_f.to_period('M')==tm
        ti_f=np.where(tr_f)[0]; ei_f=np.where(te_f)[0]
        if len(ti_f)<20 or len(ei_f)<5: continue
        yt_f=yf[ti_f];w_f=ti_f[yt_f==1];l_f=ti_f[yt_f==0]
        if len(l_f)>0 and len(w_f)>0:
            mn_f=min(len(w_f),len(l_f));rng_f=np.random.RandomState(42+tm.ordinal)
            w2_f=rng_f.choice(w_f,mn_f,replace=False) if len(w_f)>mn_f else w_f
            l2_f=rng_f.choice(l_f,mn_f,replace=False) if len(l_f)>mn_f else l_f
            ti_f=np.concatenate([w2_f,l2_f])
        if len(set(yf[ti_f]))<2: continue
        sc_f=max(1.0,(len(yt_f)-yt_f.sum())/max(yt_f.sum(),1))
        mf=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,
                              subsample=0.8,colsample_bytree=0.8,scale_pos_weight=sc_f,
                              random_state=42,eval_metric='logloss')
        mf.fit(Xf[ti_f],yf[ti_f]); probas_f[ei_f]=mf.predict_proba(Xf[ei_f])[:,1]
    valid_f=~np.isnan(probas_f)
    oot_idx=np.where(valid_f)[0]
    oot_pnls=[eps_final[i]['exit_pnl'] for i in oot_idx]
    base_s=stats(oot_pnls)
    print(f'  OOT Base: {base_s["trades"]}t, {base_s["pnl"]:+.0f}pts, WR={base_s["wr"]:.1f}%, PF={base_s["pf"]:.2f}')
    if len(set(yf[valid_f]))>1:
        auc_f=roc_auc_score(yf[valid_f],probas_f[valid_f])
        print(f'  OOT WF AUC: {auc_f:.3f}')
    print(f"\n  {'Thr':>5s} {'Pass':>8s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'dPnL':>8s}")
    print(f"  {'-'*65}")
    best_xgb=None
    for th in [0.50,0.52,0.55,0.58,0.60,0.62,0.65]:
        passed=probas_f[valid_f]>=th
        fp=[oot_pnls[i] for i in range(len(oot_pnls)) if passed[i]]; fs=stats(fp)
        pn=passed.sum(); delta=fs['pnl']-base_s['pnl']
        if fs['trades']>5:
            print(f"  {th:>4.2f} {pn:>4d}/{len(oot_pnls):<4d} {fs['trades']:>7d} {fs['pnl']:>+10.1f} "
                  f"{fs['wr']:>6.1f}% {fs['pf']:>5.2f} {delta:>+8.1f}")
        if best_xgb is None or (fs['trades']>20 and fs['pf']>best_xgb['pf']):
            best_xgb={'thresh':th,**fs,'delta':delta,'pass':pn}
    if best_xgb:
        print(f"\n  Best XGB: prob>{best_xgb['thresh']:.2f} -> {best_xgb['trades']}t, "
              f"{best_xgb['pnl']:+.0f}pts, WR={best_xgb['wr']:.1f}%, PF={best_xgb['pf']:.2f}, "
              f"dPnL={best_xgb['delta']:+.0f}")
    print('\nDONE.')

if __name__=='__main__': main()
