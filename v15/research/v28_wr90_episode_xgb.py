#!/usr/bin/env python3
'''v28 WR90 CumVol Episode + XGBoost + Param Sweep
===================================================
Entry: WR90 oversold episode ends with cumvol > threshold
Exit: TP/SL + ride-to-session if WR reaches -20
Adds XGBoost win/loss filter on episode features.
'''

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd; import xgboost as xgb
from sklearn.metrics import roc_auc_score
from data.data_loader import DataLoader
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
    df15['hour']=df15.index.hour
    df15['is_uk']=df15['hour'].isin([7,8,9,10,11,12,13,14,15,16])
    df15['is_us']=df15['hour'].isin([12,13,14,15,16,17,18,19,20])
    df15['vol_ma20']=df15['volume'].rolling(20,min_periods=5).mean()
    df15['vol_ratio']=df15['volume']/(df15['vol_ma20']+0.01)
    df15['range']=df15['high']-df15['low']
    df15['range_ma20']=df15['range'].rolling(20,min_periods=5).mean()
    df15['range_ratio']=df15['range']/(df15['range_ma20']+0.01)
    df15['ret_5']=df15['close_ask'].pct_change(5)
    df15['ret_10']=df15['close_ask'].pct_change(10)
    return df15

def find_episodes(df15, entry_thresh=-80, session='uk'):
    in_s=df15['is_uk'] if session=='uk' else (df15['is_us'] if session=='us' else df15['is_uk']|df15['is_us'])
    oversold=(df15['wr']<entry_thresh)&in_s
    episodes=[]; in_ep=False; ep_start=None; cv=0.0; bc=0
    for i in range(len(df15)):
        if oversold.iloc[i]:
            if not in_ep: ep_start=i; cv=0.0; bc=0
            in_ep=True; cv+=df15['volume'].iloc[i]; bc+=1
        else:
            if in_ep:
                ebi=i
                if ebi<len(df15)-1 and in_s.iloc[ebi]:
                    episodes.append({'start':ep_start,'entry':ebi,'cum_vol':cv,'bars':bc})
                in_ep=False; cv=0.0; bc=0
    return episodes

def sim_trade(ei, df15, tp, sl, max_bars=60, recovery=-20, weak=-50, weak_timeout=12, session_end=16):
    ep=df15.iloc[ei]['close_ask'];h=min(max_bars,len(df15)-ei-1)
    reached=-99;wc=0
    for i in range(1,h+1):
        b=df15.iloc[ei+i]
        if b['high']>=ep+tp: return ep+tp,i,'tp'
        if b['low']<=ep-sl: return ep-sl,i,'sl'
        if b['wr']>=recovery: reached=recovery
        if b['wr']<weak: wc+=1
        else: wc=0
        if reached==recovery and b.name.hour==session_end: return b['close_bid'],i,'ride_end'
        if reached!=recovery and wc>=weak_timeout: return b['close_bid'],i,'weak'
    return df15.iloc[ei+h]['close_bid'],h,'timeout'

def stats(pnls):
    if not pnls: return {'trades':0,'pnl':0.0,'wr':0.0,'pf':0.0,'avg':0.0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'trades':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99,'avg':t/n}

# XGBoost features at entry
XGB_COLS=['cum_vol','bars','vol_ratio','range_ratio','ret_5','ret_10','wr','hour']
def feats_from_episode(df15,ep):
    ei=ep['entry'];r=df15.iloc[ei]
    return [ep['cum_vol'],ep['bars'],r['vol_ratio'],r['range_ratio'],
            r['ret_5'],r['ret_10'],r['wr'],r['hour']]

def wf_xgb(trades):
    n=len(trades);y=np.array([1.0 if t['pnl']>0 else 0.0 for t in trades])
    X=np.array([t['xgb_feats'] for t in trades]).astype(float)
    nv=~np.isnan(X).any(axis=1)
    if nv.sum()<20: return None,None,None
    X=X[nv];y=y[nv];dates=pd.to_datetime([t['entry_idx'] for t in trades])[nv]
    n2=len(y);months=sorted(set(dates.to_period('M')))
    tms=[m for m in months if m>=pd.Period('2024-07',freq='M')]
    probas=np.full(n2,np.nan)
    for tm in tms:
        tr=dates.to_period('M')<tm; te=dates.to_period('M')==tm
        ti=np.where(tr)[0]; ei=np.where(te)[0]
        if len(ti)<20 or len(ei)<3: continue
        yt=y[ti];w=ti[yt==1];l=ti[yt==0]
        if len(l)>len(w) and len(w)>0: l=np.random.RandomState(42+tm.ordinal).choice(l,len(w),replace=False); ti=np.concatenate([w,l])
        sc=max(1.0,(len(yt)-yt.sum())/max(yt.sum(),1))
        m=xgb.XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.05,subsample=0.8,
                             colsample_bytree=0.8,scale_pos_weight=sc,random_state=42,eval_metric='logloss')
        m.fit(X[ti],y[ti]); probas[ei]=m.predict_proba(X[ei])[:,1]
    tm=~np.isnan(probas); return probas[tm],y[tm],tm

def main():
    print('='*72); print('v28 WR90 CumVol Episode + XGBoost + Full Sweep'); print('='*72)
    print(); print('[1] Loading...'); df1m=load_oil_data(); df15=build_15m(df1m)
    print(f'  {len(df15):,} 15m bars')

    # Sweep: entry threshold + cumvol + TP/SL + weakness_timeout
    print(f'\n[2] Entry threshold + CumVol + TP/SL sweep...')
    print(f"  {'Entry<':>8s} {'CumVol>':>10s} {'TP/SL':>10s} {'WTO':>6s} "
          f"{'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'Ride%':>7s}")
    print(f"  {'-'*85}")

    best_sweep=None
    best_overall=None
    for entry_th in [-80,-75,-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20]:
        episodes=find_episodes(df15,entry_th,'uk')
        if len(episodes)<20: continue
        for cmin in [5000,7500,10000,15000]:
            for tp,sl in [(60,40),(80,40),(80,50),(100,50),(100,60)]:
                if sl>=tp: continue
                for wto in [8,12,16]:
                    trades=[]
                    for ep in episodes:
                        if ep['cum_vol']<cmin: continue
                        ex,bars,reason=sim_trade(ep['entry'],df15,tp,sl,60,-20,-50,wto,16)
                        pnl=ex-df15.iloc[ep['entry']]['close_ask']
                        trades.append({'pnl':pnl,'reason':reason,'bars':bars,
                                       'entry_idx':df15.index[ep['entry']],
                                       'xgb_feats':feats_from_episode(df15,ep)})
                    if len(trades)<20: continue
                    pnls=[t['pnl'] for t in trades];s=stats(pnls)
                    rides=sum(1 for t in trades if t['reason']=='ride_end')
                    key={'entry':entry_th,'cmin':cmin,'tp':tp,'sl':sl,'wto':wto}
                    if best_sweep is None or s['pf']>best_sweep['pf']:
                        best_sweep={**key,**s,'trades_list':trades}
                    if best_overall is None or s['pf']>best_overall['pf']:
                        best_overall={**key,**s,'trades_list':trades}
                    if s['pf']>=1.30:
                        print(f"  {entry_th:>+8d} {cmin:>10d} {tp:>4d}/{sl:<4d} {wto:>6d} "
                              f"{s['trades']:>7d} {s['pnl']:>+10.1f} "
                              f"{s['wr']:>6.1f}% {s['pf']:>5.2f} "
                              f"{rides/(s['trades']+0.01)*100:>6.1f}%")

    if best_sweep is None: print('  No results.'); return
    bs=best_sweep
    print(f"\n  Best sweep: CumVol>{bs['cmin']} TP={bs['tp']}/SL={bs['sl']} WTO={bs['wto']} → "
          f"{bs['trades']}t, {bs['pnl']:+.0f}pts, WR={bs['wr']:.1f}%, PF={bs['pf']:.2f}")

    # XGBoost on best sweep config
    print(f'\n[3] XGBoost WF on best config...')
    trades=bs['trades_list']; pnls=[t['pnl'] for t in trades]; base=stats(pnls)
    probas,labels,tm=wf_xgb(trades)
    if probas is None: print('  Not enough WF data.'); return
    n_oot=len(probas); vi=np.where(tm)[0]; oot_p=[pnls[i] for i in vi]
    try: auc=roc_auc_score((labels>0).astype(int),probas); print(f'  WF AUC: {auc:.3f}')
    except: auc=None
    oot_b=stats(oot_p); print(f'  Base OOT: {oot_b["trades"]}t, {oot_b["pnl"]:+.0f}pts, '
                              f'WR={oot_b["wr"]:.1f}%, PF={oot_b["pf"]:.2f}')
    print(f"\n  {'Thr':>5s} {'Pass':>7s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'d':>8s}")
    print(f"  {'-'*65}")
    best_xgb=None
    for th in [0.50,0.52,0.55,0.58,0.60,0.62,0.65]:
        passed=probas>=th; fp=[oot_p[i] for i in range(len(oot_p)) if passed[i]]; fs=stats(fp)
        np_=passed.sum(); delta=fs['pnl']-oot_b['pnl']
        if fs['trades']<5: continue
        print(f"  {th:>4.2f} {np_:>5d}/{n_oot:<5d} {fs['trades']:>7d} {fs['pnl']:>+10.1f} "
              f"{fs['wr']:>6.1f}% {fs['pf']:>5.2f} {delta:>+8.1f}")
        if best_xgb is None or (fs['trades']>20 and fs['pf']>best_xgb['pf']):
            best_xgb={'thresh':th,**fs,'delta':delta}

    # Final summary
    print(f'\n{"="*72}')
    print(f"Base (no filter): {base['trades']}t, {base['pnl']:+.0f}pts, "
          f"WR={base['wr']:.1f}%, PF={base['pf']:.2f}, Avg={base['avg']:+.1f}/trade")
    if best_xgb:
        print(f"XGB filtered: {best_xgb['trades']}t, {best_xgb['pnl']:+.0f}pts, "
              f"WR={best_xgb['wr']:.1f}%, PF={best_xgb['pf']:.2f}, "
              f"Δ={best_xgb['delta']:+.0f}pts (prob≥{best_xgb['thresh']:.2f})")
    print('DONE.')

if __name__=='__main__': main()
