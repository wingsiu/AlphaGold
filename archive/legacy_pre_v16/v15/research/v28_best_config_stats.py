#!/usr/bin/env python3
"""v28 WR90 CumVol Episode — Best Config Full Stats
=====================================================
Config: Entry WR<-80, CumVol>15k, TP=80/SL=40, UK 7-16 UTC
Ride-to-session-end if WR reaches -20, weakness timeout=12
No XGBoost (ML degrades this pattern).

Produces v14-style full stat dump.
"""

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
ENTRY_THRESH = -80
CUMVOL_MIN = 15000
TP = 80
SL = 40
MAX_BARS = 60
RECOVERY = -20       # WR level to trigger ride-to-session-end
WEAK = -50           # WR level considered "weak"
WEAKNESS_TIMEOUT = 12
SESSION = 'uk'       # UK 7-16 UTC
SESSION_END = 16     # hour

# ─── Load ─────────────────────────────────────────────────────────────────────
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

# ─── Episode Detection ────────────────────────────────────────────────────────
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

# ─── Trade Simulation ─────────────────────────────────────────────────────────
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

# ─── Stats Helpers ────────────────────────────────────────────────────────────
def trade_stats(trades):
    if not trades: return {'trades':0,'pnl':0.0,'wr':0.0,'pf':0.0,'avg':0.0}
    pnls=[t['pnl'] for t in trades]
    n=len(pnls);t=sum(pnls)
    wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'trades':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99,'avg':t/n,
            'max_win':max(pnls),'max_loss':min(pnls)}

def monthly_breakdown(trades):
    months={}
    for t in trades:
        m=t['entry_time'].strftime('%Y-%m')
        if m not in months: months[m]=[]
        months[m].append(t['pnl'])
    rows=[]
    for m in sorted(months.keys()):
        pnls=months[m];n=len(pnls);s=sum(pnls);wr=sum(1 for p in pnls if p>0)/max(n,1)*100
        ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
        pf=ps/ns if ns>0 else 99
        rows.append({'Month':m,'Trades':n,'PnL':f'{s:+.0f}','WR':f'{wr:.0f}%','PF':f'{pf:.2f}','Avg':f'{s/n:+.1f}'})
    return rows

def equity_stats(pnls):
    cum=np.cumsum(pnls)
    peak=np.maximum.accumulate(cum)
    dd=cum-peak
    max_dd=dd.min()
    max_dd_pct=max_dd/peak[np.argmin(dd)]*100 if max_dd<0 else 0
    # longest dd duration in trades
    in_dd=False;dd_start=0;longest=0;current=0
    for i in range(len(cum)):
        if cum[i]<peak[i]:
            if not in_dd: dd_start=i;in_dd=True
            current=i-dd_start+1
        else:
            if in_dd: longest=max(longest,current);in_dd=False
    if in_dd: longest=max(longest,current)
    return {'final_equity':cum[-1],'peak':peak[-1],'max_dd':max_dd,'max_dd_pct':max_dd_pct,'max_dd_trades':longest}

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('='*72)
    print('  v28 WR90 CumVol Episode — Best Config Full Stats')
    print(f'  Entry WR<{ENTRY_THRESH}  CumVol>{CUMVOL_MIN:,}  TP={TP}/SL={SL}')
    print(f'  Ride-to-session-end @ WR≥{RECOVERY}  Weak timeout={WEAKNESS_TIMEOUT}')
    print('='*72)

    print('\n[1] Loading data...')
    df1m=load_oil_data()
    df15=build_15m(df1m)
    print(f'  1m bars: {len(df1m):,}  →  15m bars: {len(df15):,}')
    print(f'  Date range: {df15.index[0]} → {df15.index[-1]}')

    print(f'\n[2] Finding WR90 episodes (WR<{ENTRY_THRESH}, UK 7-16 UTC)...')
    episodes=find_episodes(df15, ENTRY_THRESH, SESSION)
    print(f'  Raw episodes: {len(episodes)}')
    episodes=[ep for ep in episodes if ep['cum_vol']>=CUMVOL_MIN]
    print(f'  After CumVol>{CUMVOL_MIN:,}: {len(episodes)}')

    print(f'\n[3] Simulating trades (TP={TP}/SL={SL}, max {MAX_BARS} bars)...')
    trades=[]
    for ep in episodes:
        ex,bars,reason=sim_trade(ep['entry'],df15,TP,SL,MAX_BARS,RECOVERY,WEAK,WEAKNESS_TIMEOUT,SESSION_END)
        pnl=ex-df15.iloc[ep['entry']]['close_ask']
        trades.append({
            'entry_time':df15.index[ep['entry']],
            'entry_price':df15.iloc[ep['entry']]['close_ask'],
            'exit_price':ex,'pnl':pnl,'bars':bars,'reason':reason,
            'cum_vol':ep['cum_vol'],'ep_bars':ep['bars'],
            'entry_wr':df15.iloc[ep['entry']]['wr'],
        })

    # ─── Full Trade Table ─────────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print(f'  TRADE LOG ({len(trades)} trades)')
    print(f'{"="*72}')
    print(f'{"#":>4s} {"Entry Time (UTC)":>20s} {"Entry":>8s} {"Exit":>8s} {"PnL":>8s} {"Bars":>5s} {"Reason":>10s} {"EntryWR":>8s} {"CumVol":>10s}')
    print(f'{"-"*88}')
    for i,t in enumerate(trades):
        print(f'{i+1:>4d} {str(t["entry_time"]):>20s} {t["entry_price"]:>8.1f} {t["exit_price"]:>8.1f} {t["pnl"]:>+8.1f} {t["bars"]:>5d} {t["reason"]:>10s} {t["entry_wr"]:>8.1f} {t["cum_vol"]:>10.0f}')

    # ─── Summary Stats ────────────────────────────────────────────────────────
    s=trade_stats(trades)
    pnls=[t['pnl'] for t in trades]
    eq=equity_stats(pnls)

    print(f'\n{"="*72}')
    print(f'  SUMMARY STATS (like v14)')
    print(f'{"="*72}')
    print(f'  Trades       : {s["trades"]}')
    print(f'  Net PnL      : {s["pnl"]:+.1f} pts')
    print(f'  Win Rate     : {s["wr"]:.1f}%')
    print(f'  Profit Factor: {s["pf"]:.2f}')
    print(f'  Avg Trade    : {s["avg"]:+.2f} pts')
    print(f'  Max Win      : {s["max_win"]:+.1f}')
    print(f'  Max Loss     : {s["max_loss"]:+.1f}')
    print(f'  Final Equity : {eq["final_equity"]:+.1f}')
    print(f'  Peak Equity  : {eq["peak"]:+.1f}')
    print(f'  Max DD       : {eq["max_dd"]:+.1f} pts ({eq["max_dd_pct"]:.1f}%)')
    print(f'  Longest DD   : {eq["max_dd_trades"]} trades')

    # ─── Exit Reason Distribution ─────────────────────────────────────────────
    reasons={}
    for t in trades:
        r=t['reason'];reasons[r]=reasons.get(r,0)+1
    print(f'\n  Exit Reasons:')
    for r,c in sorted(reasons.items(),key=lambda x:-x[1]):
        rpnls=[t['pnl'] for t in trades if t['reason']==r]
        rs=trade_stats([{'pnl':p} for p in rpnls])
        print(f'    {r:>10s}: {c:>4d} trades  PnL={sum(rpnls):+.0f}  WR={rs["wr"]:.0f}%  Avg={rs["avg"]:+.1f}')

    # ─── Monthly Breakdown ────────────────────────────────────────────────────
    monthly=monthly_breakdown(trades)
    print(f'\n{"="*72}')
    print(f'  MONTHLY BREAKDOWN')
    print(f'{"="*72}')
    print(f'{"Month":>8s} {"Trades":>7s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s}')
    print(f'{"-"*52}')
    cum_pnl=0.0
    for m in monthly:
        cum_pnl+=float(m['PnL'])
        print(f'{m["Month"]:>8s} {m["Trades"]:>7d} {m["PnL"]:>10s} {m["WR"]:>7s} {m["PF"]:>6s} {m["Avg"]:>8s}')

    # ─── Yearly Summary ───────────────────────────────────────────────────────
    yearly={}
    for t in trades:
        y=t['entry_time'].year
        if y not in yearly: yearly[y]=[]
        yearly[y].append(t['pnl'])
    print(f'\n  YEARLY:')
    for y in sorted(yearly.keys()):
        p=yearly[y];n=len(p);s=sum(p);wr=sum(1 for x in p if x>0)/max(n,1)*100
        print(f'    {y}: {n} trades  PnL={s:+.0f}  WR={wr:.0f}%')

    # ─── PnL Distribution ─────────────────────────────────────────────────────
    pnls_sorted=sorted(pnls)
    print(f'\n  PnL DISTRIBUTION:')
    print(f'    Min: {pnls_sorted[0]:+.1f}  P05: {np.percentile(pnls,5):+.1f}  P25: {np.percentile(pnls,25):+.1f}')
    print(f'    Med: {np.percentile(pnls,50):+.1f}  P75: {np.percentile(pnls,75):+.1f}  P95: {np.percentile(pnls,95):+.1f}')
    print(f'    Max: {pnls_sorted[-1]:+.1f}')

    # ─── Sharpe-ish ───────────────────────────────────────────────────────────
    if len(pnls)>1:
        mean_ret=np.mean(pnls);std_ret=np.std(pnls)
        sharpe=mean_ret/std_ret*np.sqrt(len(pnls)/len(pnls)) if std_ret>0 else 0
        print(f'\n  Mean±Std: {mean_ret:+.2f} ± {std_ret:.2f}  (SR-like: {mean_ret/std_ret:.3f} if zero risk-free)')

    # ─── Config Summary ───────────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print(f'  CONFIG')
    print(f'{"="*72}')
    print(f'  Entry      : WR < {ENTRY_THRESH}')
    print(f'  CumVol     : > {CUMVOL_MIN:,}')
    print(f'  TP / SL    : {TP} / {SL}')
    print(f'  Max Bars   : {MAX_BARS}')
    print(f'  Ride if WR : ≥ {RECOVERY} at session end ({SESSION_END}h UTC)')
    print(f'  Weak exit  : WR ≤ {WEAK} for {WEAKNESS_TIMEOUT} bars')
    print(f'  Session    : {SESSION.upper()} (7-16 UTC)')
    print(f'  No XGBoost : ML degrades this pattern (v26/v29/v31 evidence)')

    print(f'\n{"="*72}')
    print('  DONE.')
    print(f'{"="*72}')

if __name__=='__main__':
    main()
