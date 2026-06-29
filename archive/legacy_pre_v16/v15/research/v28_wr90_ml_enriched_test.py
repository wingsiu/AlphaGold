#!/usr/bin/env python3
"""WR90 Long with XGBoost ML filter — enriched 1-minute bar features.
For each 15m signal bar, compute 15× 1-minute intra-bar features
from the 15m bar that just closed (the signal bar), plus context.
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

def extract_1m_features(df_1m, signal_ts):
    """For a 15min signal bar (e.g. 10:30 closed bar representing 10:15-10:29),
    extract features from the 15 individual 1-minute bars within that window.
    
    The 15m bar at time T uses 1m bars from T-15min to T-1min (inclusive).
    Signal at 10:30 → 1m bars at 10:16 through 10:30 (actually 10:15-10:29 data).
    """
    bar_end = signal_ts
    bar_start = bar_end - pd.Timedelta(minutes=15)
    slice_m = df_1m.loc[bar_start:bar_end]
    if len(slice_m) == 0:
        return [np.nan] * 95  # fallback
    
    # Use the LAST 15 bars (some may be missing from the full 15)
    bars = slice_m.tail(15)
    features = []
    
    # Per-bar features (15 bars × 4 stats = 60 features)
    changes = bars['close_ask'].values - bars['open'].values
    ranges = bars['high'].values - bars['low'].values
    volumes = bars['volume'].values
    wicks_lower = bars['close_ask'].values - bars['low'].values
    wicks_upper = bars['high'].values - bars['close_ask'].values
    
    # Pad to exactly 15 bars
    def pad15(arr):
        padded = np.zeros(15); n = min(15, len(arr)); padded[-n:] = arr[-n:] if len(arr)>0 else [0]
        return padded
    
    features.extend(pad15(changes))
    features.extend(pad15(ranges))
    features.extend(pad15(volumes))
    features.extend(pad15(wicks_lower))
    
    # Aggregate stats (15 features)
    features.append(np.sum(changes))           # cumulative change
    features.append(np.max(np.abs(changes)))   # max bar move
    features.append(np.sum(ranges))            # total range
    features.append(np.max(changes) if len(changes)>0 else 0)  # max up bar
    features.append(np.min(changes) if len(changes)>0 else 0)  # max down bar
    features.append(np.sum(volumes))           # total volume
    features.append(np.max(volumes) if len(volumes)>0 else 0)  # max bar vol
    features.append(np.sum(changes>0)/max(1,len(changes)))  # up bar ratio
    features.append(np.std(changes) if len(changes)>1 else 0)  # std of changes
    features.append(np.corrcoef(np.arange(len(changes)), changes)[0,1] if len(changes)>1 else 0)  # trend slope
    features.append(bars['close_ask'].iloc[-1] - bars['open'].iloc[0] if len(bars)>0 else 0)  # open→close
    features.append((bars['high'].max() - bars['low'].min()) if len(bars)>0 else 0)  # full range
    features.append(bars['close_ask'].iloc[-1] - bars['close_ask'].iloc[0] if len(bars)>1 else 0)  # start→end price
    features.append((bars['close_ask'].values[-1]-bars['open'].values[0])/(bars['open'].values[0]+0.01)*100 if len(bars)>0 else 0)  # bar pct change
    features.append(np.sum(volumes[-5:])/max(1,np.sum(volumes[:5] if len(volumes)>5 else 1))) # vol front/back
    
    # Context features (5): ATR, day high distance, spread
    try:
        tr = pd.concat([bars['high']-bars['low'],
                        abs(bars['high']-bars['close_ask'].shift()),
                        abs(bars['low']-bars['close_ask'].shift())], axis=1).max(axis=1)
        features.append(tr.rolling(14).mean().iloc[-1] if len(tr)>=14 else tr.mean())  # ATR on 1m
    except: features.append(0)
    features.append(bars['close_bid'].iloc[-1] - bars['close_ask'].iloc[-1] if len(bars)>0 else 0)  # spread
    
    return [float(x) if not np.isnan(x) and not np.isinf(x) else 0.0 for x in features]

print('='*72)
print('  WR90 Long — Enriched 1m-bar Features + XGBoost Filter')
print('='*72)
d1m=load()
# Pre-compute 1m derived columns
d1m['change_1m'] = d1m['close_ask'] - d1m['open']
d1m['range_1m'] = d1m['high'] - d1m['low']
d1m['ret_1m'] = d1m['close_ask'].pct_change()

d15=build_15m(d1m)
sigs=find_signals(d15)
print(f'Signals: {len(sigs)}')

pnls,records=sim_long_advance(d15,sigs)
sa=stats(pnls)
print(f'\nUnfiltered: {sa["t"]}t PnL={sa["pnl"]:+.0f} WR={sa["wr"]:.1f}% PF={sa["pf"]:.2f}')

# Build feature matrix from 1m bars
print(f'Extracting 1m features for {len(records)} trades...')
X_list=[];y_list=[]
for i,r in enumerate(records):
    feats=extract_1m_features(d1m, r['entry_idx'])
    # Add 15m bar level features
    try:
        b15=d15.loc[r['entry_idx']]
        feats.append(float(b15['wr']))
        feats.append(float(b15['volume']))
        feats.append((float(b15['high'])-float(b15['low']))/(float(b15['close_ask'])+0.01))  # range%
        feats.append(float(b15['close_ask'])-float(b15['open']))  # 15m change
    except:k=4
    X_list.append(feats)
    y_list.append(0.0 if r['pnl']<=0 else 1.0)
    if (i+1)%100==0:print(f'  {i+1}/{len(records)}')

X=np.array(X_list);valid=~np.isnan(X).any(axis=1)
print(f'Valid samples: {valid.sum()}/{len(valid)}')
if valid.sum()<20:
    print('Too few valid samples');exit()

X=X[valid];y=np.array(y_list)[valid];recs=[records[i] for i in range(len(records)) if valid[i]]
dates=pd.DatetimeIndex([r['entry_idx'] for r in recs])
months=sorted(set(d.to_period('M') for d in dates))
test_start=pd.Period('2024-07',freq='M')
test_months=[m for m in months if m>=test_start]

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
    if len(Xb)<10:continue
    model=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,subsample=0.8,
                             scale_pos_weight=spw,random_state=42,verbosity=0)
    model.fit(Xb,yb);probas_te=model.predict_proba(X_te)[:,1]
    for j,idx in enumerate(np.where(test_mask)[0]):
        r_idx=recs[idx].get('orig_idx',idx)
        if r_idx<len(probas):probas[r_idx]=probas_te[j]
    if len(test_months)>8 and tm==test_months[len(test_months)//2]:
        print(f'  ... mid-way ({tm}) ...')

print(f'\n  {"Threshold":>12s} {"T":>5s} {"PnL":>9s} {"WR":>7s} {"PF":>7s} {"Avg":>7s}')
print(f'  {"-"*12} {"-"*5} {"-"*9} {"-"*7} {"-"*7} {"-"*7}')
for thresh in [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85]:
    fpnls=[r['pnl'] for i,r in enumerate(records) if i<len(probas) and probas[i]>=thresh]
    s=stats(fpnls);avg=s['pnl']/s['t'] if s['t']>0 else 0
    mark=' *' if s['pnl']>sa['pnl'] else ''
    print(f'  {thresh:>12.2f} {s["t"]:>5d} {s["pnl"]:>+9.0f} {s["wr"]:>6.1f}% {s["pf"]:>6.2f} {avg:>+7.1f}{mark}')
print(f'\nTotal features: {X.shape[1]}')
print(f'DONE.')
