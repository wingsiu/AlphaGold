#!/usr/bin/env python3
'''v31 WR90 Zone Filter (-80 to -70) + TP/SL Sweep + XGBoost.
Only keep episodes where entry WR is between -80 and -70 (the sweet spot).
Then sweep TP/SL combos, train XGBoost per combo with matched target.
Show base vs XGB-filtered PnL/PF for each TP/SL.'''

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
    df15['vol_ma20']=df15['volume'].rolling(20,min_periods=5).mean()
    df15['vol_ratio']=df15['volume']/(df15['vol_ma20']+0.01)
    df15['range']=df15['high']-df15['low']
    df15['range_ma20']=df15['range'].rolling(20,min_periods=5).mean()
    df15['range_ratio']=df15['range']/(df15['range_ma20']+0.01)
    df15['ret_5']=df15['close_ask'].pct_change(5)
    df15['ret_10']=df15['close_ask'].pct_change(10)
    df15['ret_20']=df15['close_ask'].pct_change(20)
    df15['price']=df15['close_ask']
    day_high=df15.groupby('day')['high'].transform('max')
    day_low=df15.groupby('day')['low'].transform('min')
    day_open=df15.groupby('day')['open'].transform('first')
    df15['day_range']=day_high-day_low
    df15['day_fullness']=(df15['close_ask']-day_open)/(df15['day_range']+0.01)
    df15['prev_day_range']=df15['day_range'].shift(96)
    df15['dow']=pd.to_datetime(df15.index).dayofweek
    tr=pd.concat([df15['high']-df15['low'],abs(df15['high']-df15['close_ask'].shift()),
                  abs(df15['low']-df15['close_ask'].shift())],axis=1).max(axis=1)
    df15['ATR']=tr.rolling(14).mean()
    return df15

def find_episodes(df15, entry_th=-80, zone_low=-70, session='uk'):
    in_s=df15['is_uk'] if session=='uk' else True
    oversold=(df15['wr']<entry_th)&in_s
    eps=[]; in_ep=False; cv=0.0; bc=0; wr_min=-999
    for i in range(len(df15)):
        if oversold.iloc[i]:
            if not in_ep: cv=0.0; bc=0; wr_min=-999
            in_ep=True; cv+=df15['volume'].iloc[i]; bc+=1
            wr_min=max(wr_min,df15['wr'].iloc[i])
        else:
            if in_ep:
                ebi=i
                r=df15.iloc[ebi]
                if ebi<len(df15)-1 and in_s.iloc[ebi] and r['wr']>=entry_th and r['wr']<=zone_low:
                    eps.append({
                        'entry':df15.index[ebi],'entry_pos':ebi,'cum_vol':cv,'bars':bc,
                        'wr_min':wr_min,'wr_entry':r['wr'],'hour':r['hour'],'dow':r['dow'],
                        'vol_ratio':r['vol_ratio'],'range_ratio':r['range_ratio'],
                        'ret_5':r['ret_5'],'ret_10':r['ret_10'],'ret_20':r['ret_20'],
                        'ATR':r['ATR'],'price':r['price'],
                        'day_range':r['day_range'],'day_fullness':r['day_fullness'],
                        'prev_day_range':r['prev_day_range'],
                    })
                in_ep=False; cv=0.0; bc=0
    return eps

FEAT=['cum_vol','bars','wr_min','wr_entry','hour','dow',
      'vol_ratio','range_ratio','ret_5','ret_10','ret_20',
      'ATR','price','day_range','day_fullness','prev_day_range']

def sim_trade(ei,df15,tp,sl,max_bars=60,recovery=-20,sess_end=16):
    ep=df15.iloc[ei]['close_ask'];h=min(max_bars,len(df15)-ei-1)
    reached=False;wc=0
    for i in range(1,h+1):
        b=df15.iloc[ei+i]
        if b['high']>=ep+tp: return ep+tp,i,'tp'
        if b['low']<=ep-sl: return ep-sl,i,'sl'
        if b['wr']>=recovery: reached=True
        if b['wr']<-50: wc+=1
        else: wc=0
        if reached and b.name.hour==sess_end: return b['close_bid'],i,'ride_end'
        if not reached and wc>=12: return b['close_bid'],i,'weak'
    return df15.iloc[ei+h]['close_bid'],h,'timeout'

def stats(pnls):
    if not pnls: return {'trades':0,'pnl':0.0,'wr':0.0,'pf':0.0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'trades':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99}

def main():
    print('='*72); print('v31 WR90 Zone (-80 to -70) + TP/SL Sweep + XGBoost'); print('='*72)
    print(); print('[1] Loading...'); df1m=load_oil(); df15=build_15m(df1m)
    eps=find_episodes(df15,-80,-70,'uk')
    print(f'  {len(eps)} zone-filtered episodes (WR entry in [-80,-70])')

    # Sweep TP/SL combos
    tpsl=[(60,40),(70,40),(80,40),(80,50),(80,60),(100,50),(100,60)]
    print(f'\n[2] TP/SL sweep with XGBoost matched-target ({len(tpsl)} combos)...')
    print(f"  {'TP/SL':<10s} {'BaseTrd':>8s} {'BasePnL':>10s} {'BasePF':>7s} {'WF AUC':>8s} "
          f"{'XGBThr':>7s} {'XGBTrd':>7s} {'XGBPnL':>10s} {'XGBPF':>7s} {'dPnL':>10s}")
    print(f"  {'-'*105}")

    results=[]
    for tp,sl in tpsl:
        if sl>=tp: continue

        # Build X and y with THIS TP/SL as target
        X_list=[]; y_list=[]
        for e in eps:
            ei=e['entry_pos']; ep_p=df15.iloc[ei]['close_ask'];h=min(60,len(df15)-ei-1)
            reached=False; outcome='timeout'
            for j in range(1,h+1):
                b=df15.iloc[ei+j]
                if b['high']>=ep_p+tp: outcome='tp'; break
                if b['low']<=ep_p-sl: outcome='sl'; break
                if b['wr']>=-20: reached=True
                if reached and b.name.hour==16: outcome='ride_end'; break
            X_list.append([e[f] for f in FEAT])
            y_list.append(1.0 if outcome=='ride_end' else 0.0)  # predict ride_end

        X=np.array(X_list).astype(float);y=np.array(y_list)
        nv=~np.isnan(X).any(axis=1); X=X[nv];y=y[nv]
        if len(y)<50: continue

        # WF XGBoost
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

        # Base stats
        pnls_all=[y_list[i]*tp+(1-y_list[i])*(-sl) for i in range(len(y_list))]
        # Recalculate proper PnL from simulation
        base_trades=[]; xgb_feats_list=[]
        for e in eps:
            ei=e['entry_pos']; ep_p=df15.iloc[ei]['close_ask']
            ex,bars,reason=sim_trade(ei,df15,tp,sl)
            base_trades.append({'pnl':ex-ep_p,'entry_idx':e['entry']})
            xgb_feats_list.append([e[f] for f in FEAT])
        base_s=stats([t['pnl'] for t in base_trades])

        # AUC
        auc_str='N/A'
        if nv2>10 and len(set(y[valid]))>1:
            try: auc_str=f'{roc_auc_score(y[valid],probas[valid]):.3f}'
            except: pass

        # Filter with XGBoost prob
        best_xgb=None
        for th in [0.50,0.52,0.55,0.58,0.60,0.65]:
            passed=probas[valid]>=th
            vi=np.where(valid)[0]; fi=vi[passed]
            filt_trades=[base_trades[i] for i in fi]
            fs=stats([t['pnl'] for t in filt_trades])
            delta=fs['pnl']-base_s['pnl']
            if fs['trades']>10:
                if best_xgb is None or (fs['pf']>best_xgb['pf'] and fs['trades']>20):
                    best_xgb={'thresh':th,**fs,'delta':delta,'np':passed.sum()}

        if best_xgb:
            results.append({'tp':tp,'sl':sl,**base_s,'auc':auc_str,**best_xgb})
            print(f"  {tp:>4d}/{sl:<4d} {base_s['trades']:>8d} {base_s['pnl']:>+10.1f} "
                  f"{base_s['pf']:>6.2f} {auc_str:>8s} "
                  f"{best_xgb['thresh']:>6.2f} {best_xgb['trades']:>7d} "
                  f"{best_xgb['pnl']:>+10.1f} {best_xgb['pf']:>6.2f} {best_xgb['delta']:>+10.1f}")
        else:
            print(f"  {tp:>4d}/{sl:<4d} {base_s['trades']:>8d} {base_s['pnl']:>+10.1f} "
                  f"{base_s['pf']:>6.2f} {auc_str:>8s} {'--':>7s}")

    # Summary
    print(f'\n[3] Summary (sorted by XGB PnL)')
    print(f"  {'TP/SL':<10s} {'BasePnL':>10s} {'BasePF':>7s} {'AUC':>7s} {'XGBThr':>7s} "
          f"{'XGBTrd':>7s} {'XGBPnL':>10s} {'XGBPF':>7s} {'dPnL':>10s}")
    print(f"  {'-'*95}")
    for r in sorted(results, key=lambda r: r['pnl'], reverse=True):
        base_pnl=r.get('pnl',0); base_pf=r.get('pf',0)
        print(f"  {r['tp']:>4d}/{r['sl']:<4d} {base_pnl:>+10.1f} {base_pf:>6.2f} "
              f"{r.get('auc','N/A'):>7s} {r['thresh']:>6.2f} {r['trades']:>7d} "
              f"{r['pnl']:>+10.1f} {r['pf']:>6.2f} {r['delta']:>+10.1f}")
    print('\nDONE.')

if __name__=='__main__': main()
