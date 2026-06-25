#!/usr/bin/env python3
'''v26 WR90 + Volume Filter + XGBoost — same approach as v24, applied to WR90 signals.'''
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

def load_oil_data(s='2024-01-01', e='2026-05-22'):
    loader=DataLoader();raw=loader.load_data(table_name='prices',start_date=s,end_date=e)
    raw.index=pd.to_datetime(raw['timestamp'],unit='ms')
    df=pd.DataFrame(index=raw.index)
    for c,src in [('open','openPrice_ask'),('high','highPrice_ask'),('low','lowPrice_ask'),
                   ('close_ask','closePrice_ask'),('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c]=raw[src].astype(float)
    if df.index.tz is None: df.index=df.index.tz_localize('UTC')
    return df

def build_15m(df_1m):
    df15=df_1m.resample('15min',label='right',closed='right').agg(
        {'open':'first','high':'max','low':'min','close_ask':'last','close_bid':'last','volume':'sum'}).dropna()
    n=14;hh=df15['high'].rolling(n).max();ll=df15['low'].rolling(n).min()
    df15['wr']=((hh-df15['close_ask'])/(hh-ll+0.01))*-100
    df15['wr_prev']=df15['wr'].shift(1)
    df15['hour']=df15.index.hour
    df15['is_uk']=df15['hour'].isin([7,8,9,10,11,12,13,14,15,16])
    df15['is_us']=df15['hour'].isin([12,13,14,15,16,17,18,19,20])
    # Volume features
    df15['vol_ma_20']=df15['volume'].rolling(20,min_periods=5).mean()
    df15['vol_ratio']=df15['volume']/(df15['vol_ma_20']+0.01)
    # Range features
    df15['range']=df15['high']-df15['low']
    df15['range_ma_20']=df15['range'].rolling(20,min_periods=5).mean()
    df15['range_ratio']=df15['range']/(df15['range_ma_20']+0.01)
    # Retrace depth
    df15['ret_3']=df15['close_ask'].pct_change(3)
    df15['ret_5']=df15['close_ask'].pct_change(5)
    df15['ret_10']=df15['close_ask'].pct_change(10)
    return df15

def sim_long(ei,ep,df15,tp,sl,max_bars):
    horizon=min(max_bars,len(df15)-ei-1)
    for i in range(1,horizon+1):
        b=df15.iloc[ei+i]
        if b['low']<=ep-sl: return ep-sl,i,'sl'
        if b['high']>=ep+tp: return ep+tp,i,'tp'
    return df15.iloc[ei+horizon]['close_bid'],horizon,'timeout'

def find_entries(df15,entry_cross=-80,session='uk',first_of_day=True,
                 vol_min=None):  # vol_min = min vol_ratio for volume filter
    in_s=df15['is_uk'] if session=='uk' else (df15['is_us'] if session=='us' else df15['is_uk']|df15['is_us'])
    cross_up=(df15['wr_prev']<=entry_cross)&(df15['wr']>entry_cross)&in_s
    if vol_min is not None:
        cross_up&=df15['vol_ratio']>=vol_min
    if not first_of_day: return cross_up
    df15['day']=df15.index.date
    df15['cross_day_rank']=cross_up.groupby(df15['day']).cumsum()*cross_up
    return cross_up&(df15['cross_day_rank']==1)

def evaluate_with_features(df15,entry,tp,sl,max_bars,session,vol_min):
    mask=find_entries(df15,entry,session,vol_min=vol_min)
    trades=[]
    for idx in df15.index[mask]:
        ei=df15.index.get_loc(idx);ep=df15.iloc[ei]['close_ask']
        ex,bars,reason=sim_long(ei,ep,df15,tp,sl,max_bars)
        row=df15.iloc[ei]
        trades.append({'entry_idx':idx,'pnl':ex-ep,'reason':reason,'bars':bars,
                       'wr':row['wr'],'wr_prev':row['wr_prev'],
                       'vol_ratio':row['vol_ratio'],'range_ratio':row['range_ratio'],
                       'ret_3':row['ret_3'],'ret_5':row['ret_5'],'ret_10':row['ret_10'],
                       'hour':row['hour'],'is_uk':row['is_uk'],'is_us':row['is_us']})
    return trades

def stats(pnls):
    if not pnls: return {'trades':0,'pnl':0.0,'wr':0.0,'pf':0.0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'trades':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99}

# XGBoost features (from trade context)
XGB_COLS=['wr','wr_prev','vol_ratio','range_ratio','ret_3','ret_5','ret_10','hour','is_uk','is_us']

def train_wf_xgb(trades):
    """Walk-forward XGBoost on WR90 signals. Returns probs and labels on OOT data."""
    n=len(trades); y=np.array([1.0 if t['pnl']>0 else 0.0 for t in trades])
    dates=pd.to_datetime([t['entry_idx'] for t in trades])
    X_arr=np.array([[t.get(c,np.nan) for c in XGB_COLS] for t in trades]).astype(float)
    valid_rows=~np.isnan(X_arr).any(axis=1)
    if valid_rows.sum()<20: return None,None,None
    # Filter valid
    X_arr=X_arr[valid_rows]; y=y[valid_rows]; dates=dates[valid_rows]
    n=len(y)
    months=sorted(set(dates.to_period('M')))
    test_months=[m for m in months if m>=pd.Period('2024-07',freq='M')]
    probas=np.full(n,np.nan)
    for tm in test_months:
        train_mask=dates.to_period('M')<tm; test_mask=dates.to_period('M')==tm
        tr_idx=np.where(train_mask)[0]; te_idx=np.where(test_mask)[0]
        if len(tr_idx)<20 or len(te_idx)<3: continue
        y_tr=y[tr_idx];wins=tr_idx[y_tr==1];losses=tr_idx[y_tr==0]
        if len(losses)>len(wins) and len(wins)>0:
            losses=np.random.RandomState(42+tm.ordinal).choice(losses,len(wins),replace=False)
            tr_idx=np.concatenate([wins,losses])
        sc=max(1.0,(len(y_tr)-y_tr.sum())/max(y_tr.sum(),1))
        m=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,
                             subsample=0.8,colsample_bytree=0.8,scale_pos_weight=sc,
                             random_state=42,eval_metric='logloss')
        m.fit(X_arr[tr_idx],y[tr_idx]); probas[te_idx]=m.predict_proba(X_arr[te_idx])[:,1]
    test_mask=~np.isnan(probas)
    return probas[test_mask],y[test_mask],test_mask

def main():
    print('='*72); print('v26 WR90 + Volume Filter + XGBoost'); print('='*72)
    print(); print('[1] Loading...'); df1m=load_oil_data(); df15=build_15m(df1m)
    print(f'  {len(df15):,} 15m bars')

    # Volume threshold sweep
    print(); print('[2] Volume filter sweep (WR>-80 UK, TP=80/SL=40)...')
    print(f"  {'VolMin':>8s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s}")
    print(f"  {'-'*45}")
    for vmin in [1.0,1.2,1.5,1.8,2.0]:
        trades=evaluate_with_features(df15,-80,80,40,80,'uk',vmin)
        pnls=[t['pnl'] for t in trades]; s=stats(pnls)
        print(f"  {vmin:>8.1f} {s['trades']:>7d} {s['pnl']:>+10.1f} {s['wr']:>6.1f}% {s['pf']:>5.2f}")

    # XGBoost on best volume filter
    print(); print('[3] XGBoost WF filter (vol>1.0 baseline)...')
    trades=evaluate_with_features(df15,-80,80,40,80,'uk',1.0)
    pnls=[t['pnl'] for t in trades]
    base=stats(pnls)
    print(f'  Base: {base["trades"]}t, {base["pnl"]:+.0f}pts, {base["wr"]:.1f}% WR, PF={base["pf"]:.2f}')

    probas,labels,test_mask=train_wf_xgb(trades)
    if probas is None:
        print('  Not enough WF predictions.')
        return
    n_oot=len(probas)
    valid_indices=np.where(test_mask)[0]
    oot_pnls=[pnls[i] for i in valid_indices]

    try: auc=roc_auc_score((labels>0).astype(int),probas); print(f'  WF AUC: {auc:.3f}')
    except: auc=None

    print(f"\n  {'Thr':>5s} {'Pass':>7s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'dPnL':>8s}")
    print(f"  {'-'*60}")
    oot_base=stats(oot_pnls)
    best=None
    for thresh in [0.50,0.52,0.55,0.58,0.60,0.65]:
        passed=probas>=thresh
        fp=[oot_pnls[i] for i in range(len(oot_pnls)) if passed[i]]; fs=stats(fp)
        delta=fs['pnl']-oot_base['pnl']; np_=passed.sum()
        print(f"  {thresh:>4.2f} {np_:>5d}/{n_oot:<5d} {fs['trades']:>7d} {fs['pnl']:>+10.1f} "
              f"{fs['wr']:>6.1f}% {fs['pf']:>5.2f} {delta:>+8.1f}")
        if best is None or (fs['trades']>20 and fs['pf']>best['pf']):
            best={'thresh':thresh,**fs,'delta':delta}

    # Sweep: vol min vs XGB best
    print(); print('[4] Combined sweep: vol filter + XGB prob...')
    print(f"  {'VolMin':>8s} {'XGBThr':>7s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s}")
    print(f"  {'-'*55}")
    for vmin in [1.0,1.2,1.5]:
        trades2=evaluate_with_features(df15,-80,80,40,80,'uk',vmin)
        pnls2=[t['pnl'] for t in trades2]; b2=stats(pnls2)
        probs,_,tm=train_wf_xgb(trades2)
        if probs is None: continue
        n2=len(probs); vi=np.where(tm)[0]; op=[pnls2[i] for i in vi]
        for th in [0.55,0.60]:
            passed=probs>=th; fp2=[op[i] for i in range(len(op)) if passed[i]]; fs2=stats(fp2)
            if fs2['trades']<5: continue
            print(f"  {vmin:>8.1f} {th:>7.2f} {fs2['trades']:>7d} {fs2['pnl']:>+10.1f} "
                  f"{fs2['wr']:>6.1f}% {fs2['pf']:>5.2f}")

    print(); print('DONE.')

if __name__=='__main__': main()
