#!/usr/bin/env python3
"""Short Impulse advance-target sweep: test different TP/SL combos with XGBoost filter."""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import xgboost as xgb
import warnings; warnings.filterwarnings('ignore')

SI_CHANGE_MAX=-14.0; SI_VOL_MIN=800; SI_MAX_B=90; SI_FC_H=14; SI_FC_M=28
SI_PROB=0.55

def load(s='2024-01-01', e='2026-06-30'):
    loader=DataLoader(); raw=loader.load_data('prices',s,e)
    raw.index=pd.to_datetime(raw['timestamp'],unit='ms')
    df=pd.DataFrame(index=raw.index)
    for c,src in [('open','openPrice_ask'),('high','highPrice_ask'),('low','lowPrice_ask'),
                   ('close_ask','closePrice_ask'),('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c]=raw[src].astype(float)
    if df.index.tz is None: df.index=df.index.tz_localize('UTC')
    return df

def compute_si_features(df):
    df['change']=df['close_ask']-df['open']
    df['prev_change']=df['change'].shift(1)
    df['prev2_change']=df['change'].shift(2)
    df['prev_lower_wick']=df['close_ask'].shift(1)-df['low'].shift(1)
    df['prev_upper_wick']=df['high'].shift(1)-df['close_ask'].shift(1)
    df['prev_volume']=df['volume'].shift(1)
    df['prev_range']=df['high'].shift(1)-df['low'].shift(1)
    df['prev_spread']=df['close_ask'].shift(1)-df['close_bid'].shift(1)
    tr=pd.concat([df['high']-df['low'],abs(df['high']-df['close_ask'].shift()),
                  abs(df['low']-df['close_ask'].shift())],axis=1).max(axis=1)
    df['ATR']=tr.rolling(14).mean()
    df['ATR_ratio']=df['prev_range']/(df['ATR']+0.01)
    ny_idx=df.index.tz_convert('America/New_York'); df['ny_hour']=ny_idx.hour.isin(list(range(3,13)))
    df['vol_ma_20']=df['volume'].rolling(20,min_periods=5).mean()
    df['vol_ratio_20']=df['prev_volume']/(df['vol_ma_20']+0.01)
    df['ret_1m']=df['close_ask'].pct_change()
    df['ret_3m']=df['ret_1m'].rolling(3,min_periods=1).sum()
    df['ret_5m']=df['ret_1m'].rolling(5,min_periods=1).sum()
    df_15=df.resample('15min',label='right',closed='right').agg({'open':'first','close_ask':'last'}).dropna()
    df_15['up']=np.where(df_15['close_ask']>df_15['open'],1,np.where(df_15['close_ask']<df_15['open'],-1,0))
    df_15['up_count3']=df_15['up'].rolling(3,min_periods=1).sum()
    f15=df_15[['up_count3']].reset_index(); df_idx=df.reset_index()
    m15=pd.merge_asof(df_idx.sort_values('timestamp'),f15.rename(columns={'timestamp':'t15'}),
                       left_on='timestamp',right_on='t15',direction='backward',tolerance=pd.Timedelta(minutes=15))
    m15.index=m15['timestamp']; df['up_count3_15min']=m15['up_count3']
    df_15e=df.resample('15min',label='right',closed='right').agg({'close_ask':'last'}).dropna()
    df_15e['ret']=df_15e['close_ask'].pct_change()
    df_15e['ret_3_15m']=df_15e['ret'].rolling(3,min_periods=1).sum()
    df_15e['ret_5_15m']=df_15e['ret'].rolling(5,min_periods=1).sum()
    f15e=df_15e[['ret_3_15m','ret_5_15m']].reset_index()
    m15e=pd.merge_asof(df_idx.sort_values('timestamp'),f15e.rename(columns={'timestamp':'t15'}),
                        left_on='timestamp',right_on='t15',direction='backward',tolerance=pd.Timedelta(minutes=15))
    m15e.index=m15e['timestamp']
    df['ret_3_15m']=m15e['ret_3_15m']; df['ret_5_15m']=m15e['ret_5_15m']
    daily_high=df['high'].resample('D').max().rename('day_high').reset_index()
    dh_m=pd.merge_asof(df_idx.sort_values('timestamp'),daily_high.rename(columns={'timestamp':'day_ts'}),
                        left_on='timestamp',right_on='day_ts',direction='backward')
    dh_m.index=dh_m['timestamp']; df['dist_day_high']=dh_m['day_high']-df['close_ask']
    return df

def find_si_signals(df):
    mask=((df['prev_change']<SI_CHANGE_MAX)&(df['prev2_change']<10.0)&
          (df['prev2_change']>-14.0)&(df['prev_lower_wick']<35.0)&
          (df['prev_volume']>SI_VOL_MIN)&df['ny_hour']&
          (df['up_count3_15min']!=-3)&(df['dist_day_high']<180.0))
    return mask

def sim_si(ei, ep, sl, tp, df):
    """Simulate short from ei with given sl/tp. Returns (exit_price, bars, reason)."""
    stop=ep+sl; target=ep-tp
    horizon=min(SI_MAX_B, len(df)-ei-1)
    nyz=df.index.tz_convert('America/New_York')
    for i in range(1, horizon+1):
        b=df.iloc[ei+i]; bh=nyz[ei+i]
        if bh.hour>SI_FC_H or (bh.hour==SI_FC_H and bh.minute>=SI_FC_M):
            return df.iloc[ei+i]['close_ask'], i, 'ny_close'
        if b['high']>=stop: return stop, i, 'sl'
        if b['low']<=target: return target, i, 'tp'
    last=min(ei+SI_MAX_B, len(df)-1)
    return df.iloc[last]['close_ask'], last-ei, 'timeout'

def run_advance_simulation(df, si_sig_order, sl, tp, with_advance):
    """Run short impulse simulation. Returns list of (entry_idx, pnl, reason)."""
    results=[]
    if not with_advance:
        # No-advance: skip overlaps
        in_trade=False; last_exit=-1
        for sig_idx in si_sig_order:
            ei=df.index.get_loc(sig_idx)
            if ei+SI_MAX_B>=len(df): continue
            if in_trade and ei<=last_exit: continue
            ep=df.iloc[ei]['close_bid']
            ex,bars,reason=sim_si(ei, ep, sl, tp, df)
            pnl=ep-ex
            results.append({'entry_idx':sig_idx,'pnl':pnl,'reason':reason,'sl':sl,'tp':tp})
            in_trade=(reason=='timeout')
            last_exit=ei+bars
    else:
        # With advance: advance TP/SL when better, keep trade open
        in_trade=False
        current_ep=0.0; current_sl=sl; current_tp=tp; current_ei=0
        for sig_idx in si_sig_order:
            ei=df.index.get_loc(sig_idx)
            if ei+SI_MAX_B>=len(df): continue
            if in_trade:
                ex_chk=sim_si(current_ei, current_ep, current_sl, current_tp, df)
                if current_ei+ex_chk[1]<=ei:
                    results.append({'entry_idx':df.index[current_ei],'pnl':current_ep-ex_chk[0],
                                    'reason':ex_chk[2],'sl':current_sl,'tp':current_tp})
                    in_trade=False
                else:
                    new_ep=df.iloc[ei]['close_bid']
                    new_target=new_ep-tp; new_stop=new_ep+sl
                    current_target=current_ep-current_tp
                    current_stop_val=current_ep+current_sl
                    advanced=False
                    if new_target<current_target:
                        current_tp=current_ep-new_target; advanced=True
                    if new_stop<current_stop_val:
                        current_sl=new_stop-current_ep; advanced=True
                    if advanced:
                        ex_chk2=sim_si(current_ei, current_ep, current_sl, current_tp, df)
                        if current_ei+ex_chk2[1]<=ei:
                            results.append({'entry_idx':df.index[current_ei],'pnl':current_ep-ex_chk2[0],
                                            'reason':ex_chk2[2],'sl':current_sl,'tp':current_tp})
                            in_trade=False
                continue
            current_ep=df.iloc[ei]['close_bid']
            current_sl=sl; current_tp=tp; current_ei=ei
            ex,bars,reason=sim_si(ei, current_ep, sl, tp, df)
            results.append({'entry_idx':sig_idx,'pnl':current_ep-ex,'reason':reason,'sl':sl,'tp':tp})
            in_trade=(reason=='timeout')
        if in_trade:
            ex_chk=sim_si(current_ei, current_ep, current_sl, current_tp, df)
            results.append({'entry_idx':df.index[current_ei],'pnl':current_ep-ex_chk[0],
                            'reason':ex_chk[2],'sl':current_sl,'tp':current_tp})
    return results

SI_XGB_FEATURES=['prev_change','prev2_change','prev_lower_wick','prev_upper_wick',
    'prev_volume','prev_range','prev_spread','ATR','ATR_ratio',
    'ret_1m','ret_3m','ret_5m','vol_ratio_20',
    'up_count3_15min','ret_3_15m','ret_5_15m','dist_day_high']

def train_xgb(df_feat, records, test_start='2024-07'):
    X_list=[]
    for r in records:
        try:
            row=df_feat.loc[r['entry_idx']]
            feat=[float(row.get(c,np.nan)) for c in SI_XGB_FEATURES]
            X_list.append(feat)
        except:
            X_list.append([np.nan]*len(SI_XGB_FEATURES))
    X=np.array(X_list)
    valid=~np.isnan(X).any(axis=1)
    if valid.sum()<20: return np.ones(len(records))*0.5
    X=X[valid]; y=np.array([1.0 if r['pnl']>0 else 0.0 for r in records])[valid]
    recs=[records[i] for i in range(len(records)) if valid[i]]
    dates=pd.DatetimeIndex([r['entry_idx'] for r in recs])
    months=sorted(set(d.to_period('M') for d in dates))
    test_months=[m for m in months if m>=pd.Period(test_start,freq='M')]
    probas=np.zeros(len(records))
    for tm in test_months:
        train_m=[m for m in months if m<tm]
        test_mask=np.array([d.to_period('M')==tm for d in dates])
        train_mask=np.array([d.to_period('M') in train_m for d in dates])
        X_tr,y_tr=X[train_mask],y[train_mask]; X_te=X[test_mask]
        if len(X_tr)<20 or len(X_te)<3: continue
        win_idx=np.where(y_tr==1)[0]; lose_idx=np.where(y_tr==0)[0]
        n_min=min(len(win_idx),len(lose_idx))
        if n_min<5: continue
        rng=np.random.RandomState(42+tm.ordinal)
        bal=np.concatenate([rng.choice(win_idx,n_min,replace=False),
                             rng.choice(lose_idx,n_min,replace=False)])
        Xb,yb=X_tr[bal],y_tr[bal]
        spw=len(lose_idx)/max(1,len(win_idx))
        model=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,
                                 subsample=0.8,scale_pos_weight=spw,random_state=42,
                                 verbosity=0,use_label_encoder=False,eval_metric='logloss')
        model.fit(Xb,yb)
        probas_te=model.predict_proba(X_te)[:,1]
        for j,idx in enumerate(np.where(test_mask)[0]):
            r_idx=recs[idx].get('orig_idx',idx)
            if r_idx<len(probas): probas[r_idx]=probas_te[j]
    return probas

def stats(pnls):
    if not pnls: return (0,0,0,0)
    n=len(pnls); t=sum(pnls); wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0); ns=abs(sum(p for p in pnls if p<0))
    return (n,int(t),round(wr,1),round(ps/ns if ns>0 else 99,2))

print('='*72)
print('  SHORT IMPULSE ADVANCE TARGET SWEEP')
print('='*72)
print('Loading data...')
d1m=load()
d1m_si=compute_si_features(d1m)
si_mask=find_si_signals(d1m_si)
si_indices=d1m_si.index[si_mask].tolist()
print(f'Raw signals: {len(si_indices)}')
si_sig_order=sorted(si_indices)

# Sweep grid
tp_range=[60,70,80,90,100,110,120]
sl_range=[40,50,60,70,80,90]
results_table=[]

for tp in tp_range:
    for sl in sl_range:
        # Run advance simulation
        recs=run_advance_simulation(d1m_si, si_sig_order, sl, tp, with_advance=True)
        # Train XGBoost
        probas=train_xgb(d1m_si, recs)
        # Filter
        fpnls=[]
        for i,r in enumerate(recs):
            if i<len(probas) and probas[i]>=SI_PROB:
                fpnls.append(r['pnl'])
        # Also compute no-advance for comparison
        recs_no=run_advance_simulation(d1m_si, si_sig_order, sl, tp, with_advance=False)
        probas_no=train_xgb(d1m_si, recs_no)
        fpnls_no=[]
        for i,r in enumerate(recs_no):
            if i<len(probas_no) and probas_no[i]>=SI_PROB:
                fpnls_no.append(r['pnl'])
        
        sa=stats(fpnls); sb=stats(fpnls_no)
        results_table.append({
            'tp':tp,'sl':sl,
            'adv_t':sa[0],'adv_pnl':sa[1],'adv_wr':sa[2],'adv_pf':sa[3],
            'no_t':sb[0],'no_pnl':sb[1],'no_wr':sb[2],'no_pf':sb[3],
        })

# Print results
rdf=pd.DataFrame(results_table)
print(f'\nAdvance Target Sweep ({len(results_table)} combos, XGB filtered):')
print(f'\n{"TP":>5s} {"SL":>5s} {"ADV T":>6s} {"ADV PnL":>9s} {"ADV WR":>7s} {"ADV PF":>7s} {"|":>3s} {"NO T":>6s} {"NO PnL":>9s} {"NO WR":>7s} {"NO PF":>7s}')
print(f'{"-"*5} {"-"*5} {"-"*6} {"-"*9} {"-"*7} {"-"*7} {"-"*3} {"-"*6} {"-"*9} {"-"*7} {"-"*7}')

# Best by PnL for advance
best_adv=rdf.sort_values('adv_pnl',ascending=False).head(10)
for _,r in best_adv.iterrows():
    print(f'{int(r["tp"]):>5d} {int(r["sl"]):>5d} {int(r["adv_t"]):>6d} {int(r["adv_pnl"]):>+9d} {r["adv_wr"]:>6.1f}% {r["adv_pf"]:>6.2f}  {"|":>3s} {int(r["no_t"]):>6d} {int(r["no_pnl"]):>+9d} {r["no_wr"]:>6.1f}% {r["no_pf"]:>6.2f}')

print(f'\nBest Advance PnL: TP={int(best_adv.iloc[0]["tp"])} SL={int(best_adv.iloc[0]["sl"])} → {int(best_adv.iloc[0]["adv_pnl"]):+d}pts')
print(f'Best No-Adv PnL:   {rdf.sort_values("no_pnl",ascending=False).iloc[0]["no_pnl"]:+d}pts')

print(f'\nDONE.')
