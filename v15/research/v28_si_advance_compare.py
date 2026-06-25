#!/usr/bin/env python3
"""Short Impulse standalone: No-Advance vs With-Advance comparison."""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

SI_CHANGE_MAX=-14.0; SI_VOL_MIN=800; SI_TP=90; SI_SL=60; SI_MAX_B=90; SI_FC_H=14; SI_FC_M=28

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
    ny_idx=df.index.tz_convert('America/New_York'); df['ny_hour']=ny_idx.hour.isin(list(range(3,13)))
    df['ret_1m']=df['close_ask'].pct_change()
    df['ret_3m']=df['ret_1m'].rolling(3,min_periods=1).sum()
    df_15=df.resample('15min',label='right',closed='right').agg({'open':'first','close_ask':'last'}).dropna()
    df_15['up']=np.where(df_15['close_ask']>df_15['open'],1,np.where(df_15['close_ask']<df_15['open'],-1,0))
    df_15['up_count3']=df_15['up'].rolling(3,min_periods=1).sum()
    f15=df_15[['up_count3']].reset_index(); df_idx=df.reset_index()
    m15=pd.merge_asof(df_idx.sort_values('timestamp'),f15.rename(columns={'timestamp':'t15'}),
                       left_on='timestamp',right_on='t15',direction='backward',tolerance=pd.Timedelta(minutes=15))
    m15.index=m15['timestamp']; df['up_count3_15min']=m15['up_count3']
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

def stats(pnls):
    if not pnls: return {'t':0,'pnl':0,'wr':0,'pf':0}
    n=len(pnls); t=sum(pnls); wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0); ns=abs(sum(p for p in pnls if p<0))
    return {'t':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99}

print('='*72)
print('  SHORT IMPULSE — No-Advance vs With-Advance')
print(f'  change<{SI_CHANGE_MAX}  vol>{SI_VOL_MIN}  TP={SI_TP}/SL={SI_SL}  max_bars={SI_MAX_B}')
print('='*72)

d1m=load()
d1m_si=compute_si_features(d1m)
si_mask=find_si_signals(d1m_si)
si_indices=d1m_si.index[si_mask].tolist()
print(f'\nRaw signals: {len(si_indices)}  |  Period: {d1m.index[0]} → {d1m.index[-1]}')
si_sig_order=sorted(si_indices)

# ========================
# A) NO ADVANCE: skip overlapping signals
# ========================
a_pnls=[]; a_results=[]
in_trade=False; last_exit_bar=-1
for sig_idx in si_sig_order:
    ei=d1m_si.index.get_loc(sig_idx)
    if ei+SI_MAX_B>=len(d1m_si): continue
    if in_trade and ei<=last_exit_bar: continue
    ep=d1m_si.iloc[ei]['close_bid']
    ex,bars,reason=sim_si(ei, ep, SI_SL, SI_TP, d1m_si)
    pnl=ep-ex
    a_pnls.append(pnl)
    a_results.append({'entry_time':sig_idx,'pnl':pnl,'reason':reason,'bars':bars,'advanced':0})
    in_trade=(reason=='timeout')
    last_exit_bar=ei+bars
sa=stats(a_pnls)

# ========================
# B) WITH ADVANCE: same-dir signal = advance TP/SL, keep trade open
# ========================
# When a new signal fires while short trade is active:
#   - Advance TP lower if new_entry_price - SI_TP < current_target (better target)
#   - Tighten SL lower if new_entry_price + SI_SL < current_stop (tighter stop)
#   - Do NOT close the trade, do NOT change the entry bar
# Only record PnL when the trade actually exits (TP/SL/ny_close/timeout)
b_pnls=[]; b_results=[]
in_trade=False
current_ep=0.0; current_sl=SI_SL; current_tp=SI_TP; current_ei=0  # active trade state
last_check_bar=0  # last bar we checked up to

for sig_idx in si_sig_order:
    ei=d1m_si.index.get_loc(sig_idx)
    if ei+SI_MAX_B>=len(d1m_si): continue
    
    if in_trade:
        # Check if current trade exited between last_check_bar and this signal
        ex_chk=sim_si(current_ei, current_ep, current_sl, current_tp, d1m_si)
        ex_bar=current_ei+ex_chk[1]
        if ex_bar<=ei:
            # Trade exited before this signal
            b_pnls.append(current_ep-ex_chk[0])
            b_results.append({'entry_time':d1m_si.index[current_ei],'pnl':current_ep-ex_chk[0],
                              'reason':ex_chk[2],'bars':ex_chk[1],'advanced':1 if current_sl!=SI_SL or current_tp!=SI_TP else 0})
            in_trade=False
        else:
            # Trade still alive — possibly advance TP/SL
            new_ep=d1m_si.iloc[ei]['close_bid']
            new_target=new_ep-SI_TP  # lower = better
            new_stop=new_ep+SI_SL   # lower = tighter (for short, tighter stop is lower)
            current_target=current_ep-current_tp
            current_stop_val=current_ep+current_sl
            advanced=False
            if new_target<current_target:  # lower TP target = more profit
                current_tp=current_ep-new_target  # recalc tp distance
                advanced=True
            if new_stop<current_stop_val:  # lower stop = tighter
                current_sl=new_stop-current_ep  # recalc sl distance
                advanced=True
            if advanced:
                # Re-sim from current entry with new TP/SL
                ex_chk2=sim_si(current_ei, current_ep, current_sl, current_tp, d1m_si)
                ex_bar2=current_ei+ex_chk2[1]
                if ex_bar2<=ei:
                    # Already exited with new targets before this signal
                    b_pnls.append(current_ep-ex_chk2[0])
                    b_results.append({'entry_time':d1m_si.index[current_ei],'pnl':current_ep-ex_chk2[0],
                                      'reason':ex_chk2[2],'bars':ex_chk2[1],'advanced':2})
                    in_trade=False
            last_check_bar=ei
        continue
    
    # No active trade — enter new
    current_ep=d1m_si.iloc[ei]['close_bid']
    current_sl=SI_SL; current_tp=SI_TP; current_ei=ei; last_check_bar=ei
    ex,bars,reason=sim_si(ei, current_ep, SI_SL, SI_TP, d1m_si)
    pnl=current_ep-ex
    b_pnls.append(pnl)
    b_results.append({'entry_time':sig_idx,'pnl':pnl,'reason':reason,'bars':bars,'advanced':0})
    in_trade=(reason=='timeout')

# Close any final open trade
if in_trade:
    ex_chk=sim_si(current_ei, current_ep, current_sl, current_tp, d1m_si)
    b_pnls.append(current_ep-ex_chk[0])
    b_results.append({'entry_time':d1m_si.index[current_ei],'pnl':current_ep-ex_chk[0],
                      'reason':ex_chk[2],'bars':ex_chk[1],'advanced':1 if current_sl!=SI_SL or current_tp!=SI_TP else 0})
sb=stats(b_pnls)

print(f'\n{"="*65}')
print(f'  {"Metric":>20s} {"No-Advance":>14s} {"Advance":>14s} {"Delta":>12s}')
print(f'  {"-"*20} {"-"*14} {"-"*14} {"-"*12}')
print(f'  {"Trades":>20s} {sa["t"]:>14d} {sb["t"]:>14d} {sb["t"]-sa["t"]:>+12d}')
print(f'  {"PnL (pts)":>20s} {sa["pnl"]:>+14.0f} {sb["pnl"]:>+14.0f} {sb["pnl"]-sa["pnl"]:>+12.0f}')
print(f'  {"Win Rate":>20s} {sa["wr"]:>13.1f}% {sb["wr"]:>13.1f}% {sb["wr"]-sa["wr"]:>+11.1f}%')
print(f'  {"Profit Factor":>20s} {sa["pf"]:>13.2f}  {sb["pf"]:>13.2f}  {sb["pf"]-sa["pf"]:>+11.2f}')
print(f'  {"Avg PnL":>20s} {sa["pnl"]/max(sa["t"],1):>+14.2f} {sb["pnl"]/max(sb["t"],1):>+14.2f}')

# Exit reason breakdown
print(f'\n  Exit Breakdown:')
for label,pnls,res in [('No-Adv',a_pnls,a_results),('Advance',b_pnls,b_results)]:
    reasons={}
    for r in res:
        rr=r['reason']
        if rr not in reasons: reasons[rr]=[]
        reasons[rr].append(r['pnl'])
    print(f'    {label}:')
    for rr,pnls2 in sorted(reasons.items()):
        n2=len(pnls2); t2=sum(pnls2); wr2=sum(1 for p in pnls2 if p>0)/max(n2,1)*100
        print(f'      {rr:18s}: {n2:4d}t  PnL={t2:+8.0f}  WR={wr2:5.1f}%')

# Monthly
print(f'\n  Monthly (HKT):')
bdf=pd.DataFrame(b_results)
bdf['time']=pd.to_datetime(bdf['entry_time'],utc=True)
bdf['month']=bdf['time'].dt.tz_convert('Asia/Hong_Kong').dt.strftime('%Y-%m')
monthly=bdf.groupby('month')['pnl'].agg(['sum','count','mean'])
print(f'    {"Month":>8s} {"T":>4s} {"PnL":>8s} {"Avg":>7s} {"WR":>5s}')
for m in sorted(monthly.index)[-12:]:
    s=monthly.loc[m]; wins=sum(1 for _,r in bdf[bdf['month']==m].iterrows() if r['pnl']>0)
    wr=wins/int(s['count'])*100
    print(f'    {m} {int(s["count"]):>4d} {s["sum"]:>+8.0f} {s["mean"]:>+7.1f} {wr:>4.0f}%')

print(f'\nDONE.')
