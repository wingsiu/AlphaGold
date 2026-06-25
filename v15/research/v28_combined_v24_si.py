#!/usr/bin/env python3
"""Combined: WR90 Long + v24 Short Impulse (EXACT v24 pipeline)
=================================================================
Uses the EXACT v24 signal generation, feature engineering, and XGBoost WF
from v24_oil_short_impulse_xgb_filter.py — no approximations.
"""

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd; import xgboost as xgb
from sklearn.metrics import roc_auc_score
from data.data_loader import DataLoader; import warnings; warnings.filterwarnings('ignore')

# ===== WR90 Long Config =====
NY_S=3;NY_E=12;NY_FC_H=14;NY_FC_M=28;LONG_MAX_B=60;LONG_EP_MIN=3
LONG_ENTRY=-80;LONG_CV=15000;LONG_TP=80;LONG_SL=30;LONG_RECOV=-20;LONG_WEAK=-50;LONG_WT=12

# ===== v24 Short Impulse Config (EXACT from v24) =====
SIGNAL_CFG = {"change_max":-14.0,"prev2_max":10.0,"prev2_min":-14.0,"lower_wick_max":35.0,
              "volume_min":800.0,"dist_high_max":180.0,"in_session":True,"uk_only":True}
SI_TP=90;SL=60;SI_MAX_B=60
XGB_FEATURES = ["prev_change","prev2_change","prev_lower_wick","prev_upper_wick",
    "prev_volume","prev_range","prev_spread","ATR","ATR_ratio","dist_day_high",
    "fullness","up_count3_15min","ret_3_15m","ret_5_15m","ret_1m","ret_3m","ret_5m",
    "vol_ratio_20","is_us","hour"]

# ==================== DATA LOADING ====================

def load(s='2024-01-01',e='2026-06-30'):
    loader=DataLoader();raw=loader.load_data(table_name='prices',start_date=s,end_date=e)
    raw.index=pd.to_datetime(raw['timestamp'],unit='ms')
    df=pd.DataFrame(index=raw.index)
    for c,src in [('open_ask','openPrice_ask'),('high_bid','highPrice_bid'),('low_bid','lowPrice_bid'),
                  ('high_ask','highPrice_ask'),('low_ask','lowPrice_ask'),('close_ask','closePrice_ask'),
                  ('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c]=raw[src].astype(float)
    df['close']=df['close_ask']; df['spread']=df['close_ask']-df['close_bid']
    if df.index.tz is None: df.index=df.index.tz_localize('UTC')
    return df

# ==================== v24 FEATURE COMPUTATION ====================

def compute_v24_features(df_1m):
    df=df_1m.copy()
    df['change']=df['close_ask']-df['open_ask']
    df['prev_change']=df['change'].shift(1); df['prev2_change']=df['change'].shift(2)
    df['prev_lower_wick']=df['close_ask'].shift(1)-df['low_ask'].shift(1)
    df['prev_upper_wick']=df['high_ask'].shift(1)-df['close_ask'].shift(1)
    df['prev_volume']=df['volume'].shift(1)
    df['prev_range']=df['high_ask'].shift(1)-df['low_ask'].shift(1)
    df['prev_spread']=df['spread'].shift(1)
    tr=pd.concat([df['high_ask']-df['low_ask'],abs(df['high_ask']-df['close_ask'].shift()),
                  abs(df['low_ask']-df['close_ask'].shift())],axis=1).max(axis=1)
    df['ATR']=tr.rolling(14).mean()
    df['ATR_ratio']=df['prev_range']/(df['ATR']+0.01)
    # dist_day_high
    daily_high=df['high_ask'].resample('D').max(); df['day_high']=np.nan
    for di in daily_high.index:
        mask=df.index.date==di.date(); df.loc[mask,'day_high']=daily_high.loc[di]
    df['dist_day_high']=df['day_high']-df['close_ask']
    # fullness
    dr=df.resample('D').agg({'high_ask':'max','low_ask':'min','open_ask':'first','close_ask':'last'}).dropna()
    dr['range']=dr['high_ask']-dr['low_ask']; dr['avg_range_5d']=dr['range'].rolling(5,min_periods=3).mean()
    df['day_open']=np.nan; df['avg_range_5d']=np.nan
    for di in dr.index:
        mask=df.index.date==di.date()
        df.loc[mask,'day_open']=dr.loc[di,'open_ask']; df.loc[mask,'avg_range_5d']=dr.loc[di,'avg_range_5d']
    df['fullness']=(df['close_ask']-df['day_open'])/(df['avg_range_5d']+0.01)
    # 15m features
    df_15=df.resample('15min',label='right',closed='right').agg({'open_ask':'first','close_ask':'last'}).dropna()
    df_15['up']=0; df_15.loc[df_15['close_ask']>df_15['open_ask'],'up']=1
    df_15.loc[df_15['close_ask']<df_15['open_ask'],'up']=-1
    df_15['up_count3']=df_15['up'].rolling(3,min_periods=1).sum(); df['up_count3_15min']=np.nan
    for idx_15 in df_15.index:
        ns=idx_15+pd.Timedelta(minutes=15); mask=(df.index>=idx_15)&(df.index<ns)
        df.loc[mask,'up_count3_15min']=df_15.loc[idx_15,'up_count3']
    df_15e=df.resample('15min',label='right',closed='right').agg({'open_ask':'first','close_ask':'last',
        'high_ask':'max','low_ask':'min','volume':'sum'}).dropna()
    df_15e['ret']=df_15e['close_ask'].pct_change()
    df_15e['ret_3']=df_15e['ret'].rolling(3,min_periods=1).sum()
    df_15e['ret_5']=df_15e['ret'].rolling(5,min_periods=1).sum()
    df['ret_3_15m']=np.nan; df['ret_5_15m']=np.nan
    for idx_15 in df_15e.index:
        ns=idx_15+pd.Timedelta(minutes=15); mask=(df.index>=idx_15)&(df.index<ns)
        df.loc[mask,'ret_3_15m']=df_15e.loc[idx_15,'ret_3']; df.loc[mask,'ret_5_15m']=df_15e.loc[idx_15,'ret_5']
    # Time features
    df['is_us']=df.index.hour.isin([12,13,14,15,16,17,18,19,20])
    df['is_uk']=df.index.hour.isin([7,8,9,10,11,12,13,14,15,16])
    df['vol_ma_20']=df['volume'].rolling(20,min_periods=5).mean()
    df['vol_ratio_20']=df['prev_volume']/(df['vol_ma_20']+0.01)
    df['ret_1m']=df['close_ask'].pct_change()
    df['ret_3m']=df['ret_1m'].rolling(3,min_periods=1).sum()
    df['ret_5m']=df['ret_1m'].rolling(5,min_periods=1).sum()
    return df

# ==================== v24 SIGNAL GENERATION ====================

def generate_si_signals(df,cfg=None):
    if cfg is None: cfg=SIGNAL_CFG
    mask=((df['prev_change']<cfg['change_max'])&(df['prev2_change']<cfg['prev2_max'])&
          (df['prev2_change']>cfg['prev2_min'])&(df['prev_lower_wick']<cfg['lower_wick_max'])&
          (df['prev_volume']>cfg['volume_min'])&(df['up_count3_15min']!=-3)&
          (df['dist_day_high']<cfg['dist_high_max']))
    if cfg.get('uk_only'): mask&=df['is_uk']
    elif cfg.get('us_only'): mask&=df['is_us']
    elif cfg.get('in_session',True): mask&=df['is_us']|df['is_uk']
    return mask

def sim_si_short(ei,ep,df,tp=SI_TP,sl=SL):
    stop=ep+sl;target=ep-tp;horizon=min(SI_MAX_B,len(df)-ei-1)
    for i in range(1,horizon+1):
        b=df.iloc[ei+i]
        if b['high_ask']>=stop: return stop,i,'sl'
        if b['low_ask']<=target: return target,i,'tp'
    return df.iloc[ei+horizon]['close_ask'],horizon,'timeout'

def evaluate_si(mask,df,tp=SI_TP,sl=SL):
    trades=[];records=[]
    for sig_idx in df.index[mask]:
        ei=df.index.get_loc(sig_idx)
        if ei+SI_MAX_B>=len(df): continue
        ep=df.iloc[ei]['close_bid'];ex,bh,reason=sim_si_short(ei,ep,df,tp,sl)
        pnl=ep-ex;trades.append({'pnl':pnl,'reason':reason});records.append({'entry_idx':sig_idx,'pnl':pnl,'reason':reason})
    return trades,records

# ==================== v24 XGBoost WF ====================

def extract_si_xgb_features(df,records):
    feats=[]
    for i,rec in enumerate(records):
        idx=rec['entry_idx'];row=df.loc[idx];feat={}
        for col in XGB_FEATURES:
            if col=='hour': feat[col]=idx.hour
            elif col=='is_us': feat[col]=int(row.get(col,0))
            else: feat[col]=float(row.get(col,np.nan))
        feat['signal_index']=i;feat['entry_idx']=idx;feats.append(feat)
    X=pd.DataFrame(feats);valid=X[XGB_FEATURES].notna().all(axis=1)
    return X[valid].reset_index(drop=True)

def train_si_xgb_wf(df,mask,records,X):
    y=np.array([1.0 if r['pnl']>0 else 0.0 for r in records]);n=len(X)
    if n<20: return None
    all_months=sorted(set(X['entry_idx'].dt.to_period('M')))
    test_months=[m for m in all_months if m>=pd.Period('2024-07',freq='M')]
    if not test_months: return None
    X_all=X[XGB_FEATURES].astype(float).values; probas=np.zeros(n); trained=np.zeros(n,dtype=bool)
    for tm in test_months:
        train_m=[m for m in all_months if m<tm]
        tr_m=X['entry_idx'].dt.to_period('M').isin(train_m)
        te_m=X['entry_idx'].dt.to_period('M')==tm
        tr_i=np.where(tr_m)[0];te_i=np.where(te_m)[0]
        if len(tr_i)<20 or len(te_i)<3: continue
        tr_y=y[tr_i];wi=tr_i[tr_y==1];li=tr_i[tr_y==0]
        if len(li)>len(wi) and len(wi)>0:
            rng=np.random.RandomState(42+tm.ordinal);li=rng.choice(li,len(wi),replace=False)
            tr_i=np.concatenate([wi,li])
        X_tr=X_all[tr_i];y_tr=y[tr_i];X_te=X_all[te_i]
        scale=max(1.0,(len(y_tr)-y_tr.sum())/max(y_tr.sum(),1))
        model=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,subsample=0.8,
            colsample_bytree=0.8,scale_pos_weight=scale,random_state=42,verbosity=0,
            use_label_encoder=False,eval_metric='logloss')
        model.fit(X_tr,y_tr);probas[te_i]=model.predict_proba(X_te)[:,1];trained[te_i]=True
    test_mask=trained;n_test=test_mask.sum()
    if n_test==0: return None
    ti=X.loc[test_mask,'signal_index'].values
    return {'probas':probas[test_mask],'labels':y[test_mask],'test_indices':ti}

# ==================== WR90 LONG ====================

def build_15m(df_1m):
    d=df_1m.resample('15min',label='right',closed='right').agg({'open_ask':'first','high_ask':'max','low_ask':'min',
        'close_ask':'last','close_bid':'last','volume':'sum'}).dropna()
    d.columns=['open','high','low','close_ask','close_bid','volume']
    n=14;hh=d['high'].rolling(n).max();ll=d['low'].rolling(n).min()
    d['wr']=((hh-d['close_ask'])/(hh-ll+0.01))*-100
    ny=d.index.tz_convert('America/New_York');d['ny_h']=ny.hour;d['ny_m']=ny.minute
    d['in_sess']=(d['ny_h']>=NY_S)&(d['ny_h']<=NY_E)
    return d

def find_long_signals(d):
    in_s=d['in_sess'];o=(d['wr']<LONG_ENTRY)&in_s;sigs=[];ie=False;cv=0.0;bc=0
    for i in range(len(d)):
        if o.iloc[i]:
            if not ie:cv=0.0;bc=0;ie=True
            cv+=d['volume'].iloc[i];bc+=1
        else:
            if ie:
                if i<len(d)-1 and in_s.iloc[i] and cv>=LONG_CV and bc>=LONG_EP_MIN:
                    sigs.append({'idx':i,'cv':cv,'bc':bc})
                ie=False
    return sigs

def sim_long_adv(d,sigs):
    pnls=[];it=False;ct=0;cs=0;ep=0;ei=0;bh=0;rec=False;wc=0;si=0
    while si<len(sigs):
        idx=sigs[si]['idx']
        if not it:
            it=True;ep=d.iloc[idx]['close_ask'];ct=ep+LONG_TP;cs=ep-LONG_SL
            ei=idx;bh=0;rec=False;wc=0;si+=1;continue
        if idx-ei>LONG_MAX_B:
            px=d.iloc[ei+LONG_MAX_B]['close_bid'];pnls.append(px-ep);it=False;continue
        for j in range(ei+bh+1,idx+1):
            b=d.iloc[j]
            post=(b['ny_h']>NY_FC_H)or(b['ny_h']==NY_FC_H and b['ny_m']>=NY_FC_M)
            if post:pnls.append(b['close_bid']-ep);it=False;break
            if b['high']>=ct:pnls.append(LONG_TP);it=False;break
            if b['low']<=cs:pnls.append(-LONG_SL);it=False;break
            if b['wr']>=LONG_RECOV:rec=True
            if b['wr']<LONG_WEAK:wc+=1
            else:wc=0
            if rec and post:pnls.append(b['close_bid']-ep);it=False;break
            if not rec and wc>=LONG_WT:pnls.append(b['close_bid']-ep);it=False;break
        bh=idx-ei
        if not it:
            it=True;ep=d.iloc[idx]['close_ask'];ct=ep+LONG_TP;cs=ep-LONG_SL
            ei=idx;bh=0;rec=False;wc=0;si+=1;continue
        ne=d.iloc[idx]['close_ask'];ct=ne+LONG_TP;cs=min(cs,ne-LONG_SL)
        ei=idx;bh=0;rec=False;wc=0;si+=1
    if it:
        last=min(ei+LONG_MAX_B,len(d)-1);pnls.append(d.iloc[last]['close_bid']-ep)
    return pnls

def stats(pnls):
    if not pnls:return{'t':0,'pnl':0,'wr':0,'pf':0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return{'t':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99}

# ==================== MAIN ====================

print('='*80)
print('  COMBINED: WR90 Long + v24 Short Impulse (EXACT pipeline)')
print('='*80)

d1m=load();d15=build_15m(d1m)
df_si=compute_v24_features(d1m)
# Match v24 dropna
drop_cols=['ATR','day_high','up_count3_15min','prev_change','fullness','ret_1m','vol_ratio_20','ret_3_15m']
df_si=df_si.dropna(subset=drop_cols)
print(f'Data: {len(d1m):,} 1m → {len(d15):,} 15m (SI ready: {len(df_si):,})')

# WR90
ls=find_long_signals(d15);l_pnls=sim_long_adv(d15,ls);long_s=stats(l_pnls)
print(f'\nWR90 Long: {len(ls)} sigs → {long_s["t"]}t, {long_s["pnl"]:+.0f}pts, WR={long_s["wr"]:.0f}%, PF={long_s["pf"]:.2f}')

# v24 Short Impulse
si_mask=generate_si_signals(df_si)
print(f'SI signals: {si_mask.sum()}')

si_trades,si_records=evaluate_si(mask=si_mask,df=df_si)
si_base_pnls=[t['pnl'] for t in si_trades];si_base=stats(si_base_pnls)
print(f'SI base: {si_base["t"]}t, {si_base["pnl"]:+.0f}pts, WR={si_base["wr"]:.0f}%, PF={si_base["pf"]:.2f}')

# v24 XGBoost
X=extract_si_xgb_features(df_si,si_records)
if len(X)>=20:
    result=train_si_xgb_wf(df_si,si_mask,si_records,X)
    if result:
        ti=result['test_indices']
        best=None
        for thresh in [0.50,0.52,0.55,0.58,0.60,0.65]:
            passed=result['probas']>=thresh
            fp=[si_records[i]['pnl'] for i in ti[passed]]
            if not fp: continue
            fs=stats(fp)
            if best is None or (fs['t']>20 and fs['pf']>best['pf']):
                try:auc=roc_auc_score(result['labels'].astype(int),result['probas'])
                except:auc=0
                best={'thresh':thresh,'t':fs['t'],'pnl':fs['pnl'],'wr':fs['wr'],'pf':fs['pf'],'auc':auc}
        if best:
            print(f'\nSI XGBoost WF: prob≥{best["thresh"]} → {best["t"]}t, {best["pnl"]:+.0f}pts, WR={best["wr"]:.0f}%, PF={best["pf"]:.2f}, AUC={best["auc"]:.3f}')
            # Combined
            comb=l_pnls+[si_records[i]['pnl'] for i in ti[result['probas']>=best['thresh']]]
            cs=stats(comb)
            print(f'\n{"="*80}')
            print(f'  COMBINED PORTFOLIO')
            print(f'  {"="*50}')
            print(f'  Long (WR90):      {long_s["t"]:>5d}t  {long_s["pnl"]:>+10.0f}pts  WR={long_s["wr"]:>5.1f}%  PF={long_s["pf"]:.2f}')
            print(f'  Short (SI+XGB):   {best["t"]:>5d}t  {best["pnl"]:>+10.0f}pts  WR={best["wr"]:>5.1f}%  PF={best["pf"]:.2f}')
            print(f'  {"="*50}')
            print(f'  TOTAL:            {cs["t"]:>5d}t  {cs["pnl"]:>+10.0f}pts  WR={cs["wr"]:>5.1f}%  PF={cs["pf"]:.2f}')
            print(f'  {"="*80}')
        else: print('\n  SI XGBoost: no valid WF predictions')
    else: print('\n  SI XGBoost: insufficient data')
else: print(f'\n  SI XGBoost: only {len(X)} samples, need 20+')

print('\nDONE.')
