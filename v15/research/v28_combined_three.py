#!/usr/bin/env python3
"""Oil Combined Backtest — WR90 Long + Short Impulse + Oil Retrace
===================================================================
Run: python3 v15/research/v28_combined_three.py

FIXED: No cascading trades — one entry per signal, advance target on
subsequent signals but no re-entry until exit.

Config:
  WR90 Long      : WR<-80, CV>=15K, EpB>=3, TP=60/SL=20, NY 3-12, advance target
  Short Impulse  : prev_change<-14, vol>800, NY 3-12, TP=120/SL=80, XGB>=0.55
  Oil Retrace    : close-Dlow>40, avgRange3>50, cl-op<-10, wick<16, TP=50/SL=50, no pattern
"""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd; import xgboost as xgb
from data.data_loader import DataLoader
from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl
import warnings; warnings.filterwarnings('ignore')

# ======================== CONFIG ========================
NY_S,NY_E,NY_FC_H,NY_FC_M = 3,12,14,28
LONG_MAX_B=60; LONG_EP_MIN=3; LONG_ENTRY=-80; LONG_CV=15000
LONG_RECOVERY,LONG_WEAK,LONG_WT = -20,-50,12
LONG_TP,LONG_SL = 60,20

SI_CHANGE_MAX,SI_VOL_MIN = -14.0,800
SI_TP,SI_SL,SI_MAX_B = 120,80,90
SI_PROB = 0.55

RET_DLOW,RET_RNG,RET_CHG,RET_WICK = 40,50,-10,16
RET_TP,RET_SL = 50,50

# ======================== SIMULATION ========================
print('='*72)
print('  OIL COMBINED BACKTEST — Three Legs (No Cascade)')
print(f'  WR90: WR<{LONG_ENTRY}  TP={LONG_TP}/SL={LONG_SL}  |  SI: TP={SI_TP}/SL={SI_SL}  |  Retrace: TP={RET_TP}/SL={RET_SL}')
print('='*72)

def load():
    loader=DataLoader();raw=loader.load_data('prices','2024-01-01','2026-06-30')
    raw.index=pd.to_datetime(raw['timestamp'],unit='ms')
    df=pd.DataFrame(index=raw.index)
    for c,src in [('open','openPrice_ask'),('high','highPrice_ask'),('low','lowPrice_ask'),
                   ('close_ask','closePrice_ask'),('close_bid','closePrice_bid'),('volume','lastTradedVolume')]:
        df[c]=raw[src].astype(float)
    if df.index.tz is None:df.index=df.index.tz_localize('UTC')
    return df

def build_15m(df_1m):
    d=df_1m.resample('15min',label='right',closed='right').agg(
        {'open':'first','high':'max','low':'min','close_ask':'last','close_bid':'last','volume':'sum'}).dropna()
    n=14;hh=d['high'].rolling(n).max();ll=d['low'].rolling(n).min()
    d['wr']=((hh-d['close_ask'])/(hh-ll+0.01))*-100
    ny=d.index.tz_convert('America/New_York')
    d['Dlow']=d['low'].groupby(ny.date).transform('min')
    d['range']=d['high']-d['low'];d['avg_range3']=d['range'].rolling(3,min_periods=3).mean()
    d['wick_below']=np.minimum(d['open'],d['close_ask'])-d['low']
    d['bar_change']=d['close_ask']-d['open']
    d['close_above_dlow']=d['close_ask']-d['Dlow']
    d['ny_h']=ny.hour;d['ny_m']=ny.minute
    d['in_sess']=(d['ny_h']>=NY_S)&(d['ny_h']<=NY_E)
    return d

def sim_no_cascade(d_d15,sigs,tp,sl,signal_type='wr90_long'):
    """One trade per signal. If already in trade, advance target (skip re-entry)."""
    pnls=[];trades=[]
    in_trade=False;ct=cs=ep=ei=entry_bar=0;rec=False;wc=0;sig_idx=0
    while sig_idx<len(sigs):
        si=sigs[sig_idx]['idx']
        if not in_trade:
            in_trade=True;entry_bar=si;ep=d_d15.iloc[si]['close_ask'];ct=ep+tp;cs=ep-sl
            ei=si;rec=False;wc=0;sig_idx+=1;continue
        
        # Already in trade - check exits from ei+1 to si
        ex=False;ex_reason='';ex_price=0.0;ex_idx=ei
        for j in range(ei+1,si+1):
            b=d_d15.iloc[j];post=(b['ny_h']>NY_FC_H)or(b['ny_h']==NY_FC_H and b['ny_m']>=NY_FC_M)
            if post:ex=True;ex_reason='ny_close';ex_price=b['close_bid'];ex_idx=j;break
            if b['high']>=ct:ex=True;ex_reason='tp';ex_price=ep+tp;ex_idx=j;break
            if b['low']<=cs:ex=True;ex_reason='sl';ex_price=ep-sl;ex_idx=j;break
            if signal_type=='wr90_long':
                if b['wr']>=LONG_RECOVERY:rec=True
                if b['wr']<LONG_WEAK:wc+=1
                else:wc=0
                if rec and post:ex=True;ex_reason='ride_end';ex_price=b['close_bid'];ex_idx=j;break
                if not rec and wc>=LONG_WT:ex=True;ex_reason='weak';ex_price=b['close_bid'];ex_idx=j;break
        
        if ex:
            pnl=ex_price-ep;pnls.append(pnl)
            trades.append({'entry_time':d_d15.index[entry_bar],'exit_time':d_d15.index[ex_idx],
                           'pnl':pnl,'exit_reason':ex_reason,'pattern':signal_type,'side':1})
            in_trade=False
            if ex_idx==si:sig_idx+=1
            continue
        
        # Timeout
        if si-ei>LONG_MAX_B:
            raw_pnl=d_d15.iloc[ei+LONG_MAX_B]['close_bid']-ep
            pnl=max(raw_pnl,-sl)
            pnls.append(pnl)
            trades.append({'entry_time':d_d15.index[entry_bar],'exit_time':d_d15.index[ei+LONG_MAX_B],
                           'pnl':pnl,'exit_reason':'timeout','pattern':signal_type,'side':1})
            in_trade=False;continue
        
        # Still in trade - advance target, skip re-entry
        ne=d_d15.iloc[si]['close_ask']
        ct=max(ct,ne+tp)
        cs=cs if cs<ne-sl else max(cs,ne-sl)
        sig_idx+=1
    
    if in_trade:
        last=min(ei+LONG_MAX_B,len(d_d15)-1)
        raw_pnl=d_d15.iloc[last]['close_bid']-ep
        pnl=max(raw_pnl,-sl)
        pnls.append(pnl)
        trades.append({'entry_time':d_d15.index[entry_bar],'exit_time':d_d15.index[last],
                       'pnl':pnl,'exit_reason':'timeout','pattern':signal_type,'side':1})
    return pnls,trades

d1m=load();d15=build_15m(d1m)
print(f'Data: {len(d1m):,} 1m bars, {len(d15):,} 15m bars')

# ---- WR90 Long ----
in_s=d15['in_sess'];o=(d15['wr']<LONG_ENTRY)&in_s
sigs_wr=[];ie=False;cv=0.0;bc=0
for i in range(len(d15)):
    if o.iloc[i]:
        if not ie:cv=0.0;bc=0
        ie=True;cv+=d15['volume'].iloc[i];bc+=1
    elif ie:
        ebi=i
        if ebi<len(d15)-1 and in_s.iloc[ebi] and cv>=LONG_CV and bc>=LONG_EP_MIN:
            sigs_wr.append({'idx':ebi})
        ie=False;cv=0.0;bc=0

pnls_wr,trades_wr=sim_no_cascade(d15,sigs_wr,LONG_TP,LONG_SL,'wr90_long')
wr_wr = sum(1 for p in pnls_wr if p>0)/max(len(pnls_wr),1)*100
print(f'WR90 Long: {len(sigs_wr)} sigs → {len(pnls_wr)}t  PnL={sum(pnls_wr):+.0f}  WR={wr_wr:.1f}%')

# ---- Oil Retrace ----
mask_ret=((d15['close_above_dlow']>RET_DLOW)&(d15['avg_range3']>RET_RNG)&
           (d15['bar_change']<RET_CHG)&(d15['wick_below']<RET_WICK)&d15['in_sess'])
sigs_ret=[{'idx':i} for i in range(len(d15)) if mask_ret.iloc[i]]
wr_bars=set(s['idx'] for s in sigs_wr)
sigs_ret_clean=[s for s in sigs_ret if s['idx'] not in wr_bars]

pnls_ret,trades_ret=sim_no_cascade(d15,sigs_ret_clean,RET_TP,RET_SL,'oil_retrace')
wr_ret = sum(1 for p in pnls_ret if p>0)/max(len(pnls_ret),1)*100
print(f'Oil Retrace: {len(sigs_ret)} raw ({len(sigs_ret_clean)} clean) → {len(pnls_ret)}t  PnL={sum(pnls_ret):+.0f}  WR={wr_ret:.1f}%')

# ---- Short Impulse ----
d1m_si=d1m.copy()
d1m_si['change']=d1m_si['close_ask']-d1m_si['open']
d1m_si['prev_change']=d1m_si['change'].shift(1)
d1m_si['prev2_change']=d1m_si['change'].shift(2)
d1m_si['prev_lower_wick']=d1m_si['close_ask'].shift(1)-d1m_si['low'].shift(1)
d1m_si['prev_volume']=d1m_si['volume'].shift(1)
d1m_si['prev_range']=d1m_si['high'].shift(1)-d1m_si['low'].shift(1)
d1m_si['prev_spread']=d1m_si['close_ask'].shift(1)-d1m_si['close_bid'].shift(1)
tr=pd.concat([d1m_si['high']-d1m_si['low'],abs(d1m_si['high']-d1m_si['close_ask'].shift()),
              abs(d1m_si['low']-d1m_si['close_ask'].shift())],axis=1).max(axis=1)
d1m_si['ATR']=tr.rolling(14).mean()
d1m_si['ATR_ratio']=d1m_si['prev_range']/(d1m_si['ATR']+0.01)
d1m_si['ret_1m']=d1m_si['close_ask'].pct_change()
d1m_si['ret_3m']=d1m_si['ret_1m'].rolling(3,min_periods=1).sum()
d1m_si['ret_5m']=d1m_si['ret_1m'].rolling(5,min_periods=1).sum()
d1m_si['vol_ma_20']=d1m_si['volume'].rolling(20,min_periods=5).mean()
d1m_si['vol_ratio_20']=d1m_si['prev_volume']/(d1m_si['vol_ma_20']+0.01)
d1m_si['ny_hour']=d1m_si.index.tz_convert('America/New_York').hour.isin(list(range(3,13)))
d15_tmp=d1m_si.resample('15min',label='right',closed='right').agg({'open':'first','close_ask':'last'}).dropna()
d15_tmp['up']=np.where(d15_tmp['close_ask']>d15_tmp['open'],1,np.where(d15_tmp['close_ask']<d15_tmp['open'],-1,0))
d15_tmp['up_count3']=d15_tmp['up'].rolling(3,min_periods=1).sum()
d15_tmp['ret']=d15_tmp['close_ask'].pct_change()
d15_tmp['ret_3_15m']=d15_tmp['ret'].rolling(3,min_periods=1).sum()
d15_tmp['ret_5_15m']=d15_tmp['ret'].rolling(5,min_periods=1).sum()
f15=d15_tmp[['up_count3','ret_3_15m','ret_5_15m']].reset_index()
m15=pd.merge_asof(d1m_si.reset_index().sort_values('timestamp'),f15.rename(columns={'timestamp':'t15'}),
                   left_on='timestamp',right_on='t15',direction='backward',tolerance=pd.Timedelta(minutes=15))
m15.index=m15['timestamp'];d1m_si['up_count3_15min']=m15['up_count3']
d1m_si['ret_3_15m']=m15['ret_3_15m'];d1m_si['ret_5_15m']=m15['ret_5_15m']
dh=d1m_si['high'].resample('D').max().reindex(d1m_si.index,method='ffill')
d1m_si['dist_day_high']=dh-d1m_si['close_ask']
si_mask=((d1m_si['prev_change']<SI_CHANGE_MAX)&(d1m_si['prev2_change']<10.0)&(d1m_si['prev2_change']>-14.0)&
          (d1m_si['prev_lower_wick']<35.0)&(d1m_si['prev_volume']>SI_VOL_MIN)&d1m_si['ny_hour']&
          (d1m_si['up_count3_15min']!=-3)&(d1m_si['dist_day_high']<180.0))
si_sigs=sorted(d1m_si.index[si_mask].tolist())

def sim_si(ei,ep,df):
    stop=ep+SI_SL;target=ep-SI_TP;horizon=min(SI_MAX_B,len(df)-ei-1)
    nyz=df.index.tz_convert('America/New_York')
    for i in range(1,horizon+1):
        b=df.iloc[ei+i];bh=nyz[ei+i]
        if bh.hour>NY_FC_H or (bh.hour==NY_FC_H and bh.minute>=NY_FC_M):
            return df.iloc[ei+i]['close_ask'],i,'ny_close',df.index[ei+i]
        if b['high']>=stop:return stop,i,'sl',df.index[ei+i]
        if b['low']<=target:return target,i,'tp',df.index[ei+i]
    px=df.iloc[ei+horizon]['close_ask']
    if ep-px<-SI_SL:
        return (ep+SI_SL,horizon,'timeout',df.index[ei+horizon])
    return px,horizon,'timeout',df.index[ei+horizon]

si_features=['prev_change','prev2_change','prev_lower_wick','prev_volume','prev_range',
             'prev_spread','ATR','ATR_ratio','ret_1m','ret_3m','ret_5m','vol_ratio_20',
             'up_count3_15min','ret_3_15m','ret_5_15m','dist_day_high']

si_recs=[];in_si=False;si_exit_bar=-1
for sig in si_sigs:
    ei=d1m_si.index.get_loc(sig)
    if ei+SI_MAX_B>=len(d1m_si):continue
    if in_si and ei<=si_exit_bar:continue
    ep=d1m_si.iloc[ei]['close_bid'];ex,bars,reason,et=sim_si(ei,ep,d1m_si)
    si_recs.append({'entry_idx':sig,'pnl':ep-ex,'reason':reason,'exit_ts':et,'bars':bars})
    in_si=True;si_exit_bar=ei+bars

dates_si=pd.DatetimeIndex([r['entry_idx'] for r in si_recs])
months=sorted(set(d.to_period('M') for d in dates_si))
test_start=pd.Period('2024-07',freq='M')
si_probas=np.zeros(len(si_recs))
for tm in [m for m in months if m>=test_start]:
    train_m=[m for m in months if m<tm]
    tst=np.array([d.to_period('M')==tm for d in dates_si]);trn=np.array([d.to_period('M') in train_m for d in dates_si])
    X=np.array([[float(d1m_si.loc[r['entry_idx']].get(f,0)) for f in si_features] for r in si_recs])
    y=np.array([1.0 if r['pnl']>0 else 0.0 for r in si_recs])
    X_tr,y_tr=X[trn],y[trn];X_te=X[tst]
    if len(X_tr)<20 or len(X_te)<3:continue
    w=np.where(y_tr==1)[0];l=np.where(y_tr==0)[0];nm=min(len(w),len(l))
    if nm<5:continue
    rng=np.random.RandomState(42+tm.ordinal)
    bal=np.concatenate([rng.choice(w,nm,replace=False),rng.choice(l,nm,replace=False)])
    Xb,yb=X_tr[bal],y_tr[bal];spw=len(l)/max(1,len(w))
    model=xgb.XGBClassifier(n_estimators=80,max_depth=3,learning_rate=0.05,subsample=0.8,
                             scale_pos_weight=spw,random_state=42,verbosity=0)
    model.fit(Xb,yb);probas_te=model.predict_proba(X_te)[:,1]
    for j,idx in enumerate(np.where(tst)[0]):si_probas[idx]=probas_te[j]

si_pnls=[r['pnl'] for i,r in enumerate(si_recs) if si_probas[i]>=SI_PROB]
trades_si=[{'entry_time':r['entry_idx'],'exit_time':r['exit_ts'],
            'pnl':r['pnl'],'exit_reason':r['reason'],'pattern':'short_impulse','side':-1}
           for i,r in enumerate(si_recs) if si_probas[i]>=SI_PROB]
for t in trades_si:t['duration_min']=(pd.Timestamp(t['exit_time'])-pd.Timestamp(t['entry_time'])).total_seconds()/60
wr_si = sum(1 for p in si_pnls if p>0)/max(len(si_pnls),1)*100
print(f'Short Impulse: {len(si_sigs)} raw → {len(si_recs)} recs → {len(si_pnls)}t  PnL={sum(si_pnls):+.0f}  WR={wr_si:.1f}%')

# ======================== COMBINED & STATS ========================
all_trades=trades_wr+trades_si+trades_ret
tdf=pd.DataFrame(all_trades)
tdf['pnl']=tdf['pnl'].astype(float)
tdf['entry_time']=pd.to_datetime(tdf['entry_time'],utc=True)
csv_path='runtime/oil_combined_backtest_trades.csv'
tdf.to_csv(csv_path,index=False)

n=len(tdf);wins=int((tdf['pnl']>0).sum());net=float(tdf['pnl'].sum())
wr_pct=wins/n*100;pnl_s=pd.Series(tdf['pnl']);cum=pnl_s.cumsum();max_dd=float((cum-cum.cummax()).min())
gross_win=float(tdf[tdf['pnl']>0]['pnl'].sum())
gross_loss=abs(float(tdf[tdf['pnl']<0]['pnl'].sum()))
pf=(gross_win/gross_loss) if gross_loss>0 else float('inf')
tdf['trade_day']=tdf['entry_time'].dt.tz_convert('America/New_York').dt.floor('D')
daily_pnl=tdf.groupby('trade_day')['pnl'].sum()
mean_day=float(daily_pnl.mean()) if len(daily_pnl) else 0.0
std_day=float(daily_pnl.std(ddof=1)) if len(daily_pnl)>1 else 0.0
downside=daily_pnl[daily_pnl<0];down_std=float(downside.std(ddof=1)) if len(downside)>1 else 0.0
sharpe=(mean_day/std_day)*np.sqrt(252.0) if std_day>0 else 0.0
sortino=(mean_day/down_std)*np.sqrt(252.0) if down_std>0 else 0.0
tpd = len(tdf) / max(len(trade_days := tdf['trade_day'].unique()), 1)

print(f'\n{"="*72}')
print(f'  FULL STATS — Oil Combined (Fixed: No Cascade)')
print(f'{"="*72}')
print(f'  Trades       : {n}  (W:{wins}  L:{n-wins})')
print(f'  Win Rate     : {wr_pct:.1f}%')
print(f'  Net PnL      : {net:+.1f} pts')
print(f'  Avg/Trade    : {net/n:+.2f} pts')
print(f'  Max DD       : {max_dd:+.1f} pts')
print(f'  Profit Factor: {pf:.2f}')

for pat,grp in tdf.groupby('pattern'):
    pw=(grp['pnl']>0).mean()*100;pn=len(grp);ps=grp['pnl'].sum()
    print(f'    {pat:20s}: {pn:4d}t  PnL={ps:+8.1f}  WR={pw:5.1f}%  avg={ps/pn:+7.2f}')

print(f'\n  Exit Breakdown:')
for reason,grp in tdf.groupby('exit_reason'):
    rw=(grp['pnl']>0).mean()*100;rn=len(grp);rs=grp['pnl'].sum()
    print(f'    {str(reason):18s}: {rn:4d}t  WR={rw:5.1f}%  avg={rs/rn:+7.2f}')

print(f'\n  Risk-Adjusted:')
expectancy=net/n;recovery=net/abs(max_dd) if max_dd<0 else float('inf')
print(f'    Expectancy/Trade   : {expectancy:+.2f} pts')
print(f'    Recovery Factor    : {recovery:.3f}')
print(f'    Sharpe  (ann): {sharpe:.2f}')
print(f'    Sortino (ann): {sortino:.2f}')

print(f'\n  Yearly:')
tdf['year']=tdf['entry_time'].dt.year
for y in sorted(tdf['year'].unique()):
    gy=tdf[tdf['year']==y];yn=len(gy);yt=gy['pnl'].sum();yw=(gy['pnl']>0).mean()*100
    yl=gy[gy['side']==1];ys=gy[gy['side']==-1]
    print(f'    {y}: {yn:4d}t  PnL={yt:+8.1f}  WR={yw:5.1f}%  Long:{len(yl):3d}t/{yl["pnl"].sum():+.0f}  Short:{len(ys):3d}t/{ys["pnl"].sum():+.0f}')

print(f'\n  Monthly:')
monthly=tdf.copy()
monthly['month']=monthly['entry_time'].dt.tz_convert('Asia/Hong_Kong').dt.strftime('%Y-%m')
mg=monthly.groupby('month')['pnl'].agg(['sum','count'])
mg['wr']=monthly.groupby('month')['pnl'].apply(lambda x:(x>0).mean()*100)
mg=mg.fillna(0)
print(f'  {"Month":>8s} {"T":>4s} {"PnL":>8s} {"WR":>5s} {"Cum":>9s}')
cum_m=0.0
for m in sorted(mg.index):
    r=mg.loc[m];cum_m+=r['sum']
    print(f'  {m:>8s} {int(r["count"]):>4d} {r["sum"]:>+8.0f} {r["wr"]:>4.0f}% {cum_m:>+9.0f}')

print(f'\n{"="*72}')
print(f'  LAST 20 TRADES (HKT)')
print(f'{"="*72}')
last20=tdf.tail(20).copy()
last20['exit_time']=pd.to_datetime(last20['exit_time'],utc=True)
last20['entry_hkt']=last20['entry_time'].dt.tz_convert('Asia/Hong_Kong').dt.strftime('%m-%d %H:%M')
last20['exit_hkt']=last20['exit_time'].dt.tz_convert('Asia/Hong_Kong').dt.strftime('%m-%d %H:%M')
last20['dir']=last20['side'].map({1:'L',-1:'S'})
for _,r in last20.iterrows():
    print(f'  {r["dir"]:>2s} {r["entry_hkt"]:>11s} [{r["exit_hkt"]:>11s}] {r["pnl"]:>+8.1f} {str(r.get("exit_reason","?"))[:8]:>8s} {str(r.get("pattern","?"))[:14]}')
print(f'\n  Net last 20: {last20["pnl"].sum():+.1f} pts')
print(f'\n  CSV: {csv_path}')
print(f'DONE.')
