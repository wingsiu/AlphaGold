#!/usr/bin/env python3
"""v28 WR90 Best Config + XGBoost Walk-Forward Filter (RELAXED TRAINING)
===================================================================
Train on relaxed signals (CumVol≥5k, EpBars≥1, WR<-70) — broad dataset.
Filter on strict signals (CumVol≥15k, EpBars≥3, WR<-80) — 284 best.
Best mechanical: TP=80/SL=30, advance target, NY 03-12, force-close 14:28.
"""

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import xgboost as xgb
import warnings; warnings.filterwarnings('ignore')

NY_SESSION_START = 3; NY_SESSION_END = 12
NY_FORCE_CLOSE_H = 14; NY_FORCE_CLOSE_M = 28
MAX_BARS = 60; RECOVERY = -20; WEAK = -50; WEAKNESS_TIMEOUT = 12
TP = 80; SL = 30

def load(s='2024-01-01', e='2026-06-30'):
    loader=DataLoader();raw=loader.load_data(table_name='prices',start_date=s,end_date=e)
    raw.index=pd.to_datetime(raw['timestamp'],unit='ms')
    df=pd.DataFrame(index=raw.index)
    for c,src in [('open','openPrice_ask'),('high','highPrice_ask'),('low','lowPrice_ask'),
                   ('close_ask','closePrice_ask'),('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c]=raw[src].astype(float)
    if df.index.tz is None: df.index=df.index.tz_localize('UTC')
    return df

def build(df):
    d=df.resample('15min',label='right',closed='right').agg({'open':'first','high':'max','low':'min','close_ask':'last','close_bid':'last','volume':'sum'}).dropna()
    n=14;hh=d['high'].rolling(n).max();ll=d['low'].rolling(n).min()
    d['wr']=((hh-d['close_ask'])/(hh-ll+0.01))*-100
    ny=d.index.tz_convert('America/New_York'); d['ny_h']=ny.hour; d['ny_m']=ny.minute
    d['in_sess']=(d['ny_h']>=NY_SESSION_START)&(d['ny_h']<=NY_SESSION_END)
    d['atr14']=(d['high']-d['low']).rolling(14).mean()
    d['ret_1']=d['close_ask'].pct_change()
    d['ret_5']=d['close_ask'].pct_change(5)
    d['ret_20']=d['close_ask'].pct_change(20)
    d['vol_ma']=d['volume'].rolling(10).mean()
    d['vol_ratio']=d['volume']/d['vol_ma'].replace(0,1)
    d['range']=d['high']-d['low']
    d['range_ma']=d['range'].rolling(10).mean()
    d['range_ratio']=d['range']/d['range_ma'].replace(0,1)
    d['hour']=d.index.hour; d['dow']=d.index.dayofweek
    return d

def find_signals(d, entry_wr_thresh=-80, min_cv=15000, min_bars=3):
    in_s=d['in_sess']; oversold=(d['wr']<entry_wr_thresh)&in_s
    sigs=[]; in_ep=False; cv=0.0; bc=0
    for i in range(len(d)):
        if oversold.iloc[i]:
            if not in_ep: ep_start=i; cv=0.0; bc=0
            in_ep=True; cv+=d['volume'].iloc[i]; bc+=1
        else:
            if in_ep:
                ebi=i
                if ebi<len(d)-1 and in_s.iloc[ebi] and cv>=min_cv and bc>=min_bars:
                    sigs.append({'idx':ebi,'cv':cv,'bc':bc,'start':ep_start})
                in_ep=False; cv=0.0; bc=0
    return sigs

def extract_features(d, sig):
    i=sig['idx']; row=d.iloc[i]; ep_bars=sig['bc']
    feats={
        'wr':row['wr'], 'cum_vol':np.log1p(sig['cv']), 'ep_bars':ep_bars,
        'atr14':row['atr14'], 'volume':np.log1p(row['volume']),
        'vol_ratio':row['vol_ratio'] if not np.isnan(row['vol_ratio']) else 1.0,
        'range_ratio':row['range_ratio'] if not np.isnan(row['range_ratio']) else 1.0,
        'ret_1':row['ret_1']*100 if not np.isnan(row['ret_1']) else 0,
        'ret_5':row['ret_5']*100 if not np.isnan(row['ret_5']) else 0,
        'ret_20':row['ret_20']*100 if not np.isnan(row['ret_20']) else 0,
        'hour':row['hour'], 'dow':row['dow'], 'ny_hour':row['ny_h'],
    }
    if ep_bars>0:
        ep_wr_vals=d['wr'].iloc[sig['start']:i+1]
        feats['mean_wr']=ep_wr_vals.mean(); feats['min_wr']=ep_wr_vals.min()
        feats['wr_range']=ep_wr_vals.max()-ep_wr_vals.min()
    else:
        feats['mean_wr']=feats['min_wr']=row['wr']; feats['wr_range']=0
    feats['bar_drop']=d.iloc[sig['start']]['close_ask']-d.iloc[i]['close_ask']
    return feats

def sim_trade_single(ei, d, tp=TP, sl=SL):
    ep_p=d.iloc[ei]['close_ask']; h=min(MAX_BARS,len(d)-ei-1)
    reached=-99; wc=0
    for i in range(1,h+1):
        b=d.iloc[ei+i]
        post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
        if post: return b['close_bid']-ep_p
        if b['high']>=ep_p+tp: return tp
        if b['low']<=ep_p-sl: return -sl
        if b['wr']>=RECOVERY: reached=RECOVERY
        if b['wr']<WEAK: wc+=1
        else: wc=0
        if reached==RECOVERY and post: return b['close_bid']-ep_p
        if reached!=RECOVERY and wc>=WEAKNESS_TIMEOUT: return b['close_bid']-ep_p
    return d.iloc[ei+h]['close_bid']-ep_p

def sim_with_advance(sigs, d, tp=TP, sl=SL):
    pnls=[]
    in_trade=False; current_tp=0; current_sl=0; ep_p=0; entry_idx=0; bars_held=0
    reached=-99; wc=0; sig_idx=0
    while sig_idx<len(sigs):
        si=sigs[sig_idx]['idx']
        if not in_trade:
            in_trade=True; ep_p=d.iloc[si]['close_ask']
            current_tp=ep_p+tp; current_sl=ep_p-sl
            entry_idx=si; bars_held=0; reached=-99; wc=0
            sig_idx+=1; continue
        if si-entry_idx>MAX_BARS:
            pnls.append(d.iloc[entry_idx+MAX_BARS]['close_bid']-ep_p); in_trade=False; continue
        for j in range(entry_idx+bars_held+1, si+1):
            b=d.iloc[j]
            post=(b['ny_h']>NY_FORCE_CLOSE_H)or(b['ny_h']==NY_FORCE_CLOSE_H and b['ny_m']>=NY_FORCE_CLOSE_M)
            if post: pnls.append(b['close_bid']-ep_p); in_trade=False; break
            if b['high']>=current_tp: pnls.append(current_tp-ep_p); in_trade=False; break
            if b['low']<=current_sl: pnls.append(current_sl-ep_p); in_trade=False; break
            if b['wr']>=RECOVERY: reached=RECOVERY
            if b['wr']<WEAK: wc+=1
            else: wc=0
            if reached==RECOVERY and post: pnls.append(b['close_bid']-ep_p); in_trade=False; break
            if reached!=RECOVERY and wc>=WEAKNESS_TIMEOUT: pnls.append(b['close_bid']-ep_p); in_trade=False; break
        bars_held=si-entry_idx
        if not in_trade:
            in_trade=True; ep_p=d.iloc[si]['close_ask']; current_tp=ep_p+tp; current_sl=ep_p-sl
            entry_idx=si; bars_held=0; reached=-99; wc=0; sig_idx+=1; continue
        new_entry=d.iloc[si]['close_ask']; current_tp=new_entry+tp; current_sl=min(current_sl,new_entry-sl)
        entry_idx=si; bars_held=0; reached=-99; wc=0; sig_idx+=1
    if in_trade:
        last=min(entry_idx+MAX_BARS,len(d)-1); pnls.append(d.iloc[last]['close_bid']-ep_p)
    return pnls

def stats(pnls):
    if not pnls: return {'t':0,'pnl':0,'wr':0,'pf':0,'avg':0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'t':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99,'avg':t/n}

# ===== MAIN =====
print('='*72)
print('  V28 ML Filter — RELAXED TRAINING')
print(f'  Base: TP={TP}/SL={SL}, advance target, NY 03-12, force-close 14:28')
print('='*72)

d1=load(); d15=build(d1)

# --- RELAXED training set ---
train_sigs=find_signals(d15, entry_wr_thresh=-70, min_cv=5000, min_bars=1)
print(f'\nRelaxed training signals (WR<-70, CumVol≥5k, EpBars≥1): {len(train_sigs)}')

# --- STRICT filter set (the 284 best) ---
strict_sigs=find_signals(d15, entry_wr_thresh=-80, min_cv=15000, min_bars=3)
print(f'Strict filter signals (WR<-80, CumVol≥15k, EpBars≥3): {len(strict_sigs)}')

# Build strict index lookup for WF filtering
strict_indices={s['idx'] for s in strict_sigs}

# Label training set with single-trade PnL
print('Labeling training set...')
X_train_list=[]; y_train_list=[]; train_dates=[]
for s in train_sigs:
    f=extract_features(d15,s); X_train_list.append(f)
    pnl=sim_trade_single(s['idx'],d15)
    y_train_list.append(1 if pnl>0 else 0)
    train_dates.append(d15.index[s['idx']])

X_train=pd.DataFrame(X_train_list).fillna(0)
y_train=np.array(y_train_list)
train_date_strs=[d.strftime('%Y-%m') for d in train_dates]

# Label strict set
print('Labeling strict set...')
X_strict_list=[]; y_strict_list=[]; strict_pnls=[]; strict_dates=[]
for s in strict_sigs:
    f=extract_features(d15,s); X_strict_list.append(f)
    pnl=sim_trade_single(s['idx'],d15)
    strict_pnls.append(pnl)
    y_strict_list.append(1 if pnl>0 else 0)
    strict_dates.append(d15.index[s['idx']])

X_strict=pd.DataFrame(X_strict_list).fillna(0)
strict_date_strs=[d.strftime('%Y-%m') for d in strict_dates]

# Walk-forward: train on relaxed, filter on strict
months=sorted(set(strict_date_strs))
test_months=[m for m in months if m>='2024-07']
if not test_months: test_months=months[-3:]

print(f'\nWF test months: {test_months[0]} → {test_months[-1]} ({len(test_months)} folds)')
print(f'\n{"="*85}')
print(f'  {"Prob":>5s} {"Trades":>7s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s} {"AUC":>6s} {"Keep%":>7s}')
print(f'  {"-"*56}')

all_results={}
best_by_pnl=None

for pth in [0.50,0.52,0.55,0.58,0.60,0.65]:
    filtered_pnls=[]; all_probs=[]; all_labels=[]; in_sample=int(0)
    for tm in test_months:
        train_idx=[i for i,m in enumerate(train_date_strs) if m<tm]
        test_idx=[i for i,m in enumerate(strict_date_strs) if m==tm]
        if len(train_idx)<30 or len(test_idx)<3: continue
        X_tr=X_train.iloc[train_idx]; y_tr=y_train[train_idx]
        win_idx=[i for i,l in enumerate(y_tr) if l==1]
        lose_idx=[i for i,l in enumerate(y_tr) if l==0]
        n_min=min(len(win_idx),len(lose_idx))
        if n_min<10: continue
        np.random.seed(42)
        bal_idx=np.concatenate([np.random.choice(win_idx,n_min,replace=False),
                                 np.random.choice(lose_idx,n_min,replace=False)])
        Xb=X_tr.iloc[bal_idx]; yb=y_tr[bal_idx]
        spw=len(lose_idx)/max(1,len(win_idx))
        try:
            mdl=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,subsample=0.8,
                                   scale_pos_weight=spw,random_state=42,verbosity=0,
                                   use_label_encoder=False,eval_metric='logloss')
            mdl.fit(Xb,yb)
            X_test=X_strict.iloc[test_idx]; probs=mdl.predict_proba(X_test)[:,1]
            for j,ti in enumerate(test_idx):
                all_probs.append(probs[j]); all_labels.append(y_strict_list[ti])
                if probs[j]>=pth: filtered_pnls.append(strict_pnls[ti])
        except: pass
    if not filtered_pnls: continue
    s=stats(filtered_pnls)
    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc=roc_auc_score(all_labels,all_probs) if len(set(all_labels))>1 else 0.5
    except: auc=0.5
    keep_pct=len(filtered_pnls)/max(1,sum(1 for ti in range(len(strict_sigs)) if strict_date_strs[ti]>=test_months[0]))*100
    all_results[pth]={'s':s,'auc':auc,'keep':keep_pct}
    print(f'  {pth:>4.2f} {s["t"]:>7d} {s["pnl"]:>+10.0f} {s["wr"]:>6.1f}% {s["pf"]:>5.2f} {s["avg"]:>+8.1f} {auc:>5.3f} {keep_pct:>6.0f}%')
    if best_by_pnl is None or s['pnl']>best_by_pnl['s']['pnl']:
        best_by_pnl={'pth':pth,'s':s,'auc':auc,'keep':keep_pct}

# Baseline
pnls_adv=sim_with_advance(strict_sigs,d15)
sb=stats(pnls_adv)
print(f'\n{"="*72}')
print(f'  BASELINE (advance, no ML): {sb["t"]}t, {sb["pnl"]:+.0f}pts, WR={sb["wr"]:.1f}%, PF={sb["pf"]:.2f}')

if best_by_pnl:
    b=best_by_pnl
    print(f'  BEST ML (prob≥{b["pth"]}): {b["s"]["t"]}t, {b["s"]["pnl"]:+.0f}pts, WR={b["s"]["wr"]:.1f}%, PF={b["s"]["pf"]:.2f}, AUC={b["auc"]:.3f}')
    print(f'  Keeps {b["keep"]:.0f}% of strict signals')
    print(f'  Delta: {b["s"]["pnl"]-sb["pnl"]:+.0f} pts, WR Δ: {b["s"]["wr"]-sb["wr"]:+.1f}%')
else:
    print('  ML produced no trades — cannot improve on mechanical edge.')

print('\nDONE.')
