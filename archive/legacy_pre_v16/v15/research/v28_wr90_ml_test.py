#!/usr/bin/env python3
"""WR90 Long with XGBoost ML filter — quick test."""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd; import xgboost as xgb
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_S=3; NY_E=12; NY_FC_H=14; NY_FC_M=28
LONG_MAX_B=60; LONG_EP_MIN=3; LONG_ENTRY=-80; LONG_CV=15000
LONG_RECOVERY=-20; LONG_WEAK=-50; LONG_WT=12
LONG_TP=60; LONG_SL=20

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
    d['wr_m5']=d['wr'].shift(5); d['wr_m1']=d['wr'].shift(1)
    d['ret_1b']=d['close_ask'].pct_change()
    d['ret_3b']=d['ret_1b'].rolling(3,min_periods=1).sum()
    d['ret_5b']=d['ret_1b'].rolling(5,min_periods=1).sum()
    d['vol_chg']=d['volume']/(d['volume'].rolling(20).mean()+0.01)
    d['range_pct']=(d['high']-d['low'])/(d['close_ask']+0.01)
    ny=d.index.tz_convert('America/New_York');d['ny_h']=ny.hour;d['ny_m']=ny.minute
    d['ny_hour']=d['ny_h'].astype(float)
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

def sim_long_advance(d15,sigs):
    pnls=[];records=[]
    in_trade=False;ct=0;cs=0;ep=0;ei=0;bh=0;reached=False;wc=0;sig_idx=0
    while sig_idx<len(sigs):
        si=sigs[sig_idx]['idx']
        if not in_trade:
            in_trade=True;ep=d15.iloc[si]['close_ask'];ct=ep+LONG_TP;cs=ep-LONG_SL
            ei=si;bh=0;reached=False;wc=0;entry_ts=d15.index[si]
            records.append({'entry_idx':entry_ts,'pnl':0})
            records[-1]['_sig_data']={'cv':sigs[sig_idx]['cv'],'bc':sigs[sig_idx]['bc']}
            sig_idx+=1;continue
        end_bar=min(si,ei+LONG_MAX_B);exit_at_si=False
        for j in range(ei+bh+1,end_bar+1):
            b=d15.iloc[j];post=(b['ny_h']>NY_FC_H)or(b['ny_h']==NY_FC_H and b['ny_m']>=NY_FC_M)
            if post:
                px=b['close_bid'];pnls.append(px-ep);records[-1]['pnl']=px-ep;in_trade=False
                if j==si:exit_at_si=True
                break
            if b['high']>=ct:
                pnls.append(LONG_TP);records[-1]['pnl']=LONG_TP;in_trade=False
                if j==si:exit_at_si=True
                break
            if b['low']<=cs:
                pnls.append(-LONG_SL);records[-1]['pnl']=-LONG_SL;in_trade=False
                if j==si:exit_at_si=True
                break
            if b['wr']>=LONG_RECOVERY:reached=True
            if b['wr']<LONG_WEAK:wc+=1
            else:wc=0
            if reached and post:
                px=b['close_bid'];pnls.append(px-ep);records[-1]['pnl']=px-ep;in_trade=False
                if j==si:exit_at_si=True
                break
            if not reached and wc>=LONG_WT:
                px=b['close_bid'];pnls.append(px-ep);records[-1]['pnl']=px-ep;in_trade=False
                if j==si:exit_at_si=True
                break
        bh=si-ei
        if not in_trade:
            if exit_at_si:sig_idx+=1;continue
            in_trade=True;ep=d15.iloc[si]['close_ask'];ct=ep+LONG_TP;cs=ep-LONG_SL
            ei=si;bh=0;reached=False;wc=0;entry_ts=d15.index[si]
            records.append({'entry_idx':entry_ts,'pnl':0});sig_idx+=1;continue
        ne=d15.iloc[si]['close_ask'];ct=ne+LONG_TP;cs=max(cs,ne-LONG_SL)
        ei=si;bh=0;reached=False;wc=0;sig_idx+=1
    if in_trade:
        last=min(ei+LONG_MAX_B,len(d15)-1);px=d15.iloc[last]['close_bid'];pnls.append(px-ep);records[-1]['pnl']=px-ep
    return pnls, records

def stats(pnls):
    if not pnls: return {'t':0,'pnl':0,'wr':0,'pf':0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'t':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99}

FEATURES=['wr','wr_m5','wr_m1','ret_1b','ret_3b','ret_5b','vol_chg','range_pct','ny_hour']

def train_xgb(d15,records,test_start='2024-07'):
    X_list=[];y_list=[0.0 if r['pnl']<=0 else 1.0 for r in records]
    for r in records:
        try:row=d15.loc[r['entry_idx']];X_list.append([float(row.get(c,np.nan)) for c in FEATURES])
        except:X_list.append([np.nan]*len(FEATURES))
    X=np.array(X_list);valid=~np.isnan(X).any(axis=1)
    if valid.sum()<20:return np.ones(len(records))*0.5
    X=X[valid];y=np.array(y_list)[valid];recs=[records[i] for i in range(len(records)) if valid[i]]
    dates=pd.DatetimeIndex([r['entry_idx'] for r in recs])
    months=sorted(set(d.to_period('M') for d in dates))
    test_months=[m for m in months if m>=pd.Period(test_start,freq='M')]
    probas=np.zeros(len(records))
    for tm in test_months:
        train_m=[m for m in months if m<tm]
        test_mask=np.array([d.to_period('M')==tm for d in dates])
        train_mask=np.array([d.to_period('M') in train_m for d in dates])
        X_tr,y_tr=X[train_mask],y[train_mask];X_te=X[test_mask]
        if len(X_tr)<20 or len(X_te)<3:continue
        win_idx=np.where(y_tr==1)[0];lose_idx=np.where(y_tr==0)[0];n_min=min(len(win_idx),len(lose_idx))
        if n_min<5:continue
        rng=np.random.RandomState(42+tm.ordinal)
        bal=np.concatenate([rng.choice(win_idx,n_min,replace=False),rng.choice(lose_idx,n_min,replace=False)])
        Xb,yb=X_tr[bal],y_tr[bal];spw=len(lose_idx)/max(1,len(win_idx))
        model=xgb.XGBClassifier(n_estimators=80,max_depth=3,learning_rate=0.05,subsample=0.8,
                                 scale_pos_weight=spw,random_state=42,verbosity=0)
        model.fit(Xb,yb);probas_te=model.predict_proba(X_te)[:,1]
        for j,idx in enumerate(np.where(test_mask)[0]):
            r_idx=recs[idx].get('orig_idx',idx)
            if r_idx<len(probas):probas[r_idx]=probas_te[j]
    return probas

print('='*72)
print('  WR90 Long XGBoost ML Filter Test')
print('='*72)
d1m=load();d15=build_15m(d1m)
sigs=find_signals(d15)
print(f'Signals: {len(sigs)}')

pnls,records=sim_long_advance(d15,sigs)
sa=stats(pnls)
print(f'\nUnfiltered: {sa["t"]}t PnL={sa["pnl"]:+.0f} WR={sa["wr"]:.1f}% PF={sa["pf"]:.2f}')

# Train XGBoost
probas=train_xgb(d15,records)
for prob_thresh in [0.55,0.60,0.65,0.70]:
    fpnls=[r['pnl'] for i,r in enumerate(records) if i<len(probas) and probas[i]>=prob_thresh]
    s=stats(fpnls)
    print(f'  Filter≥{prob_thresh:.2f}: {s["t"]}t PnL={s["pnl"]:+.0f} WR={s["wr"]:.1f}% PF={s["pf"]:.2f}')
    if s['t']<30:print('    (too few trades)')
print(f'\nDONE.')
