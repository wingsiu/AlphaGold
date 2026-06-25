#!/usr/bin/env python3
"""v28 WR90 — Winner vs Loser Analysis
======================================
Identifies mechanical filters that separate +80 TP hits from -40 SL losses.
Goal: find improvements without ML.
"""

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

ENTRY_THRESH = -80; CUMVOL_MIN = 15000; TP = 80; SL = 40
MAX_BARS = 60; RECOVERY = -20; WEAK = -50; WEAKNESS_TIMEOUT = 12

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
    df15['vol_ma20']=df15['volume'].rolling(20,min_periods=5).mean()
    df15['vol_ratio']=df15['volume']/(df15['vol_ma20']+0.01)
    df15['range']=df15['high']-df15['low']
    df15['range_ma20']=df15['range'].rolling(20,min_periods=5).mean()
    df15['range_ratio']=df15['range']/(df15['range_ma20']+0.01)
    df15['ret_5']=df15['close_ask'].pct_change(5)
    df15['ret_10']=df15['close_ask'].pct_change(10)
    # Added features for analysis
    df15['ma50']=df15['close_ask'].rolling(50).mean()
    df15['atr14']=(df15['high']-df15['low']).rolling(14).mean()
    df15['dayofweek']=df15.index.dayofweek  # 0=Mon, 4=Fri
    return df15

def find_episodes(df15):
    in_s=df15['is_uk']; oversold=(df15['wr']<ENTRY_THRESH)&in_s
    episodes=[]; in_ep=False; cv=0.0; bc=0
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

def sim_trade(ei, df15):
    ep=df15.iloc[ei]['close_ask'];h=min(MAX_BARS,len(df15)-ei-1)
    reached=-99;wc=0
    for i in range(1,h+1):
        b=df15.iloc[ei+i]
        if b['high']>=ep+TP: return ep+TP,i,'tp'
        if b['low']<=ep-SL: return ep-SL,i,'sl'
        if b['wr']>=RECOVERY: reached=RECOVERY
        if b['wr']<WEAK: wc+=1
        else: wc=0
        if reached==RECOVERY and b.name.hour==16: return b['close_bid'],i,'ride_end'
        if reached!=RECOVERY and wc>=WEAKNESS_TIMEOUT: return b['close_bid'],i,'weak'
    return df15.iloc[ei+h]['close_bid'],h,'timeout'

def main():
    print('='*72)
    print('  v28 WR90 — Winner vs Loser Feature Analysis')
    print('='*72)

    df1m=load_oil_data(); df15=build_15m(df1m)
    episodes=find_episodes(df15)
    episodes=[ep for ep in episodes if ep['cum_vol']>=CUMVOL_MIN]
    print(f'Episodes: {len(episodes)}')

    trades=[]
    for ep in episodes:
        ex,bars,reason=sim_trade(ep['entry'], df15)
        pnl=ex-df15.iloc[ep['entry']]['close_ask']
        row=df15.iloc[ep['entry']]
        trades.append({
            'pnl':pnl,'reason':reason,'bars_to_exit':bars,
            'entry_wr':row['wr'],'cum_vol':ep['cum_vol'],'ep_bars':ep['bars'],
            'hour':row['hour'],'dow':row['dayofweek'],
            'vol_ratio':row['vol_ratio'],'range_ratio':row['range_ratio'],
            'ret_5':row['ret_5'],'ret_10':row['ret_10'],
            'close':row['close_ask'],'ma50':row['ma50'],
            'atr14':row['atr14'],
        })

    tdf=pd.DataFrame(trades)
    tdf['win']=tdf['pnl']>0
    tdf['is_tp']=tdf['reason']=='tp'
    tdf['is_sl']=tdf['reason']=='sl'

    # Filter to just TP vs SL for clean signal/noise
    tp_sl=tdf[tdf['reason'].isin(['tp','sl'])].copy()
    print(f'\nTP+SL trades: {len(tp_sl)} (TP={tp_sl["is_tp"].sum()}, SL={tp_sl["is_sl"].sum()})')
    print(f'TP rate: {tp_sl["is_tp"].mean()*100:.1f}%')

    # ─── Feature analysis: TP vs SL ───────────────────────────────────────
    features=['entry_wr','cum_vol','ep_bars','hour','dow','vol_ratio','range_ratio','ret_5','ret_10','atr14']
    print(f'\n{"="*72}')
    print('  FEATURE MEANS: TP HITS vs SL HITS')
    print(f'{"="*72}')
    print(f'{"Feature":>15s} {"TP_mean":>10s} {"SL_mean":>10s} {"Diff":>10s} {"Ratio":>8s}')
    print(f'{"-"*58}')
    for f in features:
        tp_mean=tp_sl[tp_sl['is_tp']][f].mean()
        sl_mean=tp_sl[tp_sl['is_sl']][f].mean()
        diff=tp_mean-sl_mean
        ratio=tp_mean/sl_mean if sl_mean!=0 else 99
        print(f'{f:>15s} {tp_mean:>10.2f} {sl_mean:>10.2f} {diff:>+10.2f} {ratio:>8.2f}')

    # ─── Best hour analysis ───────────────────────────────────────────────
    print(f'\n{"="*72}')
    print('  TP RATE BY HOUR (UK session)')
    print(f'{"="*72}')
    for h in sorted(tp_sl['hour'].unique()):
        sub=tp_sl[tp_sl['hour']==h]
        if len(sub)<3: continue
        print(f'  {h:2d}h UTC ({h+8:2d}h HKT): {len(sub):3d} trades  TP={sub["is_tp"].mean()*100:.0f}%  PnL={sub["pnl"].sum():+.0f}')

    # ─── Best WR depth ────────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print('  TP RATE BY ENTRY WR DEPTH')
    print(f'{"="*72}')
    for lo,hi,label in [(-120,-90,'-120 to -90'),(-90,-85,'-90 to -85'),(-85,-80,'-85 to -80')]:
        sub=tp_sl[(tp_sl['entry_wr']>=lo)&(tp_sl['entry_wr']<hi)]
        if len(sub)<3: continue
        print(f'  WR {label:>12s}: {len(sub):3d} trades  TP={sub["is_tp"].mean()*100:.0f}%  PnL={sub["pnl"].sum():+.0f}')

    # ─── CumVol bins ──────────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print('  TP RATE BY CUMVOL BIN')
    print(f'{"="*72}')
    for lo,hi in [(15000,20000),(20000,30000),(30000,50000),(50000,200000)]:
        sub=tp_sl[(tp_sl['cum_vol']>=lo)&(tp_sl['cum_vol']<hi)]
        if len(sub)<3: continue
        print(f'  CumVol {lo//1000:>4d}k-{hi//1000:<4d}k: {len(sub):3d} trades  TP={sub["is_tp"].mean()*100:.0f}%  PnL={sub["pnl"].sum():+.0f}')

    # ─── Day of week ──────────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print('  TP RATE BY DAY OF WEEK')
    print(f'{"="*72}')
    days=['Mon','Tue','Wed','Thu','Fri']
    for d in range(5):
        sub=tp_sl[tp_sl['dow']==d]
        if len(sub)<3: continue
        print(f'  {days[d]:>4s}: {len(sub):3d} trades  TP={sub["is_tp"].mean()*100:.0f}%  PnL={sub["pnl"].sum():+.0f}')

    # ─── Trend alignment ──────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print('  TP RATE BY TREND (price vs MA50)')
    print(f'{"="*72}')
    tp_sl['above_ma']=tp_sl['close']>tp_sl['ma50']
    for label,sub in [('Above MA50 (uptrend)',tp_sl[tp_sl['above_ma']]),
                       ('Below MA50 (downtrend)',tp_sl[~tp_sl['above_ma']])]:
        if len(sub)<3: continue
        print(f'  {label:>22s}: {len(sub):3d} trades  TP={sub["is_tp"].mean()*100:.0f}%  PnL={sub["pnl"].sum():+.0f}')

    # ─── Vol regime (ATR) ─────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print('  TP RATE BY ATR TERTILE')
    print(f'{"="*72}')
    tp_sl['atr_tertile']=pd.qcut(tp_sl['atr14'],3,labels=['Low','Med','High'])
    for label,sub in tp_sl.groupby('atr_tertile'):
        if len(sub)<3: continue
        print(f'  ATR {label:>4s}: {len(sub):3d} trades  TP={sub["is_tp"].mean()*100:.0f}%  PnL={sub["pnl"].sum():+.0f}')

    # ─── Ep bars (exhaustion) ─────────────────────────────────────────────
    print(f'\n{"="*72}')
    print('  TP RATE BY EPISODE BARS')
    print(f'{"="*72}')
    for lo,hi,label in [(1,2,'1-2 bars'),(2,4,'2-4 bars'),(4,7,'4-7 bars'),(7,20,'7+ bars')]:
        sub=tp_sl[(tp_sl['ep_bars']>=lo)&(tp_sl['ep_bars']<hi)]
        if len(sub)<3: continue
        print(f'  EpBars {label:>10s}: {len(sub):3d} trades  TP={sub["is_tp"].mean()*100:.0f}%  PnL={sub["pnl"].sum():+.0f}')

    # ─── Best combo: deeper WR + high CumVol ──────────────────────────────
    print(f'\n{"="*72}')
    print('  COMBO FILTERS (All Trades, not just TP/SL)')
    print(f'{"="*72}')
    all_t=tdf.copy()
    def eval_filter(mask,label):
        sub=all_t[mask]
        if len(sub)<5: return
        wr=(sub['pnl']>0).mean()*100
        pf_num=sub[sub['pnl']>0]['pnl'].sum()
        pf_den=abs(sub[sub['pnl']<0]['pnl'].sum())
        pf=pf_num/pf_den if pf_den>0 else 99
        print(f'  {label:>30s}: {len(sub):3d}t  PnL={sub["pnl"].sum():+.0f}  WR={wr:.0f}%  PF={pf:.2f}')

    # Baseline
    eval_filter(slice(None),'BASELINE (all)')

    # Deeper WR
    for wr_th in [-85,-90,-95]:
        eval_filter(all_t['entry_wr']<wr_th, f'WR < {wr_th}')
    # Higher CumVol
    for cv in [20000,25000,30000,40000]:
        eval_filter(all_t['cum_vol']>cv, f'CumVol > {cv//1000}k')

    # WR depth + CumVol combos
    for wr_th in [-85,-90]:
        for cv in [20000,25000,30000]:
            m=(all_t['entry_wr']<wr_th)&(all_t['cum_vol']>cv)
            eval_filter(m, f'WR<{wr_th} & CV>{cv//1000}k')

    # Best hours
    for hr_range,label in [([9,10,11,12,13,14,15],'09-15h'),([10,11,12,13,14,15],'10-15h'),
                            ([11,12,13,14,15],'11-15h'),([12,13,14,15],'12-15h')]:
        eval_filter(all_t['hour'].isin(hr_range), f'Hours {label}')

    # ATR filter
    atr_med=all_t['atr14'].median()
    eval_filter(all_t['atr14']>atr_med, f'ATR > median ({atr_med:.0f})')
    eval_filter(all_t['atr14']<atr_med, f'ATR < median ({atr_med:.0f})')

    # Consecutive episode bars >= 3
    eval_filter(all_t['ep_bars']>=3, 'EpBars >= 3')
    eval_filter(all_t['ep_bars']>=4, 'EpBars >= 4')

    # Volume ratio
    eval_filter(all_t['vol_ratio']>1.0, 'VolRatio > 1.0')
    eval_filter(all_t['vol_ratio']>1.5, 'VolRatio > 1.5')

    # Range ratio
    eval_filter(all_t['range_ratio']>1.0, 'RangeRatio > 1.0')

    print('\nDONE.')

if __name__=='__main__':
    main()
