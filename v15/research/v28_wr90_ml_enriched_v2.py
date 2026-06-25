#!/usr/bin/env python3
"""WR90 Long with XGBoost ML filter — per-1m-bar training.
Each of the 15 1-minute bars within the signal's 15m window is a sample.
All 15 bars share the same trade outcome label. At inference, 
aggregate the 15 individual predictions into one probability.
"""
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

def sim_long_advance(d15,sigs):
    pnls=[];records=[]
    in_trade=False;ct=0;cs=0;ep=0;ei=0;bh=0;reached=False;wc=0;sig_idx=0
    while sig_idx<len(sigs):
        si=sigs[sig_idx]['idx']
        if not in_trade:
            in_trade=True;ep=d15.iloc[si]['close_ask'];ct=ep+LONG_TP;cs=ep-LONG_SL
            ei=si;bh=0;reached=False;wc=0;entry_ts=d15.index[si]
            records.append({'entry_idx':entry_ts,'pnl':0,'_cv':sigs[sig_idx]['cv'],'_bc':sigs[sig_idx]['bc']})
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
            records.append({'entry_idx':entry_ts,'pnl':0,'_cv':sigs[sig_idx]['cv'],
                            '_bc':sigs[sig_idx]['bc']});sig_idx+=1;continue
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


def extract_per_bar_features(df_1m, bar_ts):
    """Extract features for a single 1-minute bar at bar_ts.
    Returns features describing this bar + its context within the 15m window.
    """
    features = []
    try:
        idx_loc = df_1m.index.get_loc(bar_ts)
        # Get surrounding bars for context
        start = max(0, idx_loc - 10)
        end = min(len(df_1m)-1, idx_loc + 3)
        context = df_1m.iloc[start:end+1]
        current = df_1m.iloc[idx_loc]
    except:
        return [np.nan]*25
    
    # Bar-level features
    features.append(float(current['close_ask'] - current['open']))  # change
    features.append(float(current['high'] - current['low']))  # range
    features.append(float(current['volume']))
    features.append(float(current['close_ask'] - current['low']))  # lower wick
    features.append(float(current['high'] - current['close_ask']))  # upper wick
    
    # Position within 15m bar (0-14)
    bar_minute = bar_ts.minute
    pos_in_15m = bar_minute % 15
    features.append(float(pos_in_15m if bar_minute>0 else 15))
    
    # Price relative to 15m bar's open
    bar_15m_end = bar_ts.ceil('15min')
    bar_15m_start = bar_15m_end - pd.Timedelta(minutes=15)
    try:
        bar_15m_open = df_1m.loc[bar_15m_start+pd.Timedelta(minutes=1):bar_15m_end].iloc[0]['open'] \
            if len(df_1m.loc[bar_15m_start+pd.Timedelta(minutes=1):bar_15m_end])>0 else current['close_ask']
    except:
        bar_15m_open = current['close_ask']
    features.append((current['close_ask']-bar_15m_open)/(abs(bar_15m_open)+0.01)*100)
    
    # Volume relative to recent average
    vol_ma = df_1m.iloc[max(0,idx_loc-20):idx_loc+1]['volume'].mean() if idx_loc>=0 else current['volume']
    features.append(current['volume']/(vol_ma+0.01))
    
    # Short-term momentum (last 3 bars)
    if idx_loc>=3:
        recent = df_1m.iloc[idx_loc-3:idx_loc+1]
        features.append((recent['close_ask'].iloc[-1]-recent['close_ask'].iloc[0])/(recent['close_ask'].iloc[0]+0.01)*100)
        features.append(recent['volume'].sum())
        up_bars = (recent['close_ask']>recent['open']).sum()
        features.append(up_bars/4.0)
    else:
        features.extend([0,0,0])
    
    # Spread
    features.append(float(current['close_ask']-current['close_bid']))
    
    # Day-high distance
    try:
        day_start = bar_ts.normalize().tz_convert('America/New_York').tz_convert('UTC') + pd.Timedelta(hours=NY_S)
        day_bars = df_1m.loc[day_start:bar_ts]
        day_high = day_bars['high'].max() if len(day_bars)>0 else current['close_ask']
        features.append(day_high - current['close_ask'])
    except:
        features.append(0)
    
    # Sector hour
    try:
        ny_h = bar_ts.tz_convert('America/New_York').hour
        features.append(float(ny_h))
    except:
        features.append(0)
    
    # Cumulative change from start of 15m window
    try:
        win_bars = df_1m.loc[bar_15m_start+pd.Timedelta(minutes=1):bar_ts]
        cum_chg = (win_bars['close_ask'].iloc[-1]-win_bars['close_ask'].iloc[0]) \
            if len(win_bars)>=2 else 0
        features.append(cum_chg/(abs(win_bars['close_ask'].iloc[0])+0.01)*100 if len(win_bars)>=2 else 0)
    except:
        features.append(0)
    
    return [float(x) if x==x and abs(x)<1e10 else 0.0 for x in features]

print('='*72)
print('  WR90 Long — Per-1m-Bar Training (4K+ samples)')
print('='*72)
d1m=load()
d15=build_15m(d1m)
sigs=find_signals(d15)
print(f'Signals: {len(sigs)}')

pnls,records=sim_long_advance(d15,sigs)
sa=stats(pnls)
print(f'\nUnfiltered: {sa["t"]}t PnL={sa["pnl"]:+.0f} WR={sa["wr"]:.1f}% PF={sa["pf"]:.2f}')

# Build per-bar training set
print(f'\nBuilding per-bar samples from {len(records)} trades...')
X_all=[];y_all=[];trade_ids=[]  # track which trade each bar belongs to
for tid,r in enumerate(records):
    bar_end=r['entry_idx'];bar_start=bar_end-pd.Timedelta(minutes=15)
    bar_slice=d1m.loc[bar_start+pd.Timedelta(minutes=1):bar_end]
    label=0.0 if r['pnl']<=0 else 1.0
    for ts in bar_slice.index:
        feats=extract_per_bar_features(d1m,ts)
        if not any(np.isnan(feats)):
            X_all.append(feats);y_all.append(label);trade_ids.append(tid)

X=np.array(X_all);y=np.array(y_all);trade_ids=np.array(trade_ids)
valid=~np.isnan(X).any(axis=1)
print(f'Valid bar samples: {valid.sum()} (from {len(set(trade_ids))} trades)')
X=X[valid];y=y[valid];trade_ids=trade_ids[valid]

# Walk-forward training (per month, using trade-level grouping)
dates=pd.DatetimeIndex([records[t]['entry_idx'] for t in range(len(records))])
months=sorted(set(d.to_period('M') for d in dates))
test_start=pd.Period('2024-07',freq='M')

probas=np.zeros(len(records))
for tm in [m for m in months if m>=test_start]:
    train_m=[m for m in months if m<tm]
    # Test trades = trades in this month
    test_trade_indices=[i for i,d in enumerate(dates) if d.to_period('M')==tm]
    train_trade_indices=[i for i,d in enumerate(dates) if d.to_period('M') in train_m]
    
    # Get bar-level samples for train
    train_mask=np.isin(trade_ids,train_trade_indices)
    test_mask=np.isin(trade_ids,test_trade_indices)
    
    X_tr,y_tr=X[train_mask],y[train_mask];X_te=X[test_mask]
    if len(X_tr)<50 or len(X_te)<30:continue
    
    # Balance training
    win_idx=np.where(y_tr==1)[0];lose_idx=np.where(y_tr==0)[0]
    n_min=min(len(win_idx),len(lose_idx))
    if n_min<10:continue
    rng=np.random.RandomState(42+tm.ordinal)
    bal=np.concatenate([rng.choice(win_idx,n_min,replace=False),rng.choice(lose_idx,n_min,replace=False)])
    Xb,yb=X_tr[bal],y_tr[bal];spw=len(lose_idx)/max(1,len(win_idx))
    
    model=xgb.XGBClassifier(n_estimators=150,max_depth=4,learning_rate=0.03,subsample=0.8,
                             scale_pos_weight=spw,random_state=42,verbosity=0)
    model.fit(Xb,yb)
    
    # Per-bar predictions on test
    bar_probas=model.predict_proba(X_te)[:,1]
    test_tids=trade_ids[test_mask]
    
    # Aggregate: mean probability per trade
    for tid in np.unique(test_tids):
        tmask=test_tids==tid
        tprob=bar_probas[tmask].mean()
        if tid<len(probas):probas[tid]=tprob

# Filtered results
print(f'\n  {"Threshold":>12s} {"T":>5s} {"PnL":>9s} {"WR":>7s} {"PF":>7s} {"Avg":>7s}')
print(f'  {"-"*12} {"-"*5} {"-"*9} {"-"*7} {"-"*7} {"-"*7}')
for thresh in [0.50,0.52,0.55,0.58,0.60,0.62,0.65,0.70,0.75,0.80]:
    fpnls=[r['pnl'] for i,r in enumerate(records) if i<len(probas) and probas[i]>=thresh]
    s=stats(fpnls);avg=s['pnl']/s['t'] if s['t']>0 else 0
    mark=' *' if s['pnl']>sa['pnl'] else ''
    print(f'  {thresh:>12.2f} {s["t"]:>5d} {s["pnl"]:>+9.0f} {s["wr"]:>6.1f}% {s["pf"]:>6.2f} {avg:>+7.1f}{mark}')
print(f'\nTotal bar samples: {len(X)} (from {len(set(trade_ids))} trades)')
print(f'DONE.')
