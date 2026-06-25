#!/usr/bin/env python3
"""v28 WR90 CumVol Episode — MECHANICALLY IMPROVED
=====================================================
Baseline: WR<-80, CumVol>15k, TP=80/SL=40 → 364t, +2,615pts, PF 1.34

IMPROVEMENTS (purely mechanical, no ML):
  1. EpBars ≥ 3 (exhaustion depth filter) — best single filter: +2,709pts, PF 1.52
  2. ATR below median (low-vol regime) — second best: +1,829pts, PF 1.54
  3. Exclude Friday — Mon/Tue best, Fri worst (-160pts)
  4. Exclude late session 15-16h UTC — late hours have worst TP rate
  5. COMBO: EpBars≥3 + ATR<median + Mon-Thu + hours 8-14 UTC

All output in HKT (UTC+8).
"""

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
ENTRY_THRESH = -80; CUMVOL_MIN = 15000; TP = 80; SL = 40; MAX_BARS = 60
RECOVERY = -20; WEAK = -50; WEAKNESS_TIMEOUT = 12
SESSION_END = 16  # UTC hour

# Improvement filters
EP_BARS_MIN = 3        # Episode must have ≥3 consecutive WR<-80 bars
EXCLUDE_FRIDAY = True
EXCLUDE_LATE_SESSION = True  # exclude 15-16 UTC
ATR_BELOW_MEDIAN = True      # filter to low-ATR regime

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
    df15['hour_hkt']=df15.index.tz_convert('Asia/Hong_Kong').hour
    df15['is_uk']=df15['hour'].isin([7,8,9,10,11,12,13,14,15,16])
    df15['atr14']=(df15['high']-df15['low']).rolling(14).mean()
    df15['dayofweek']=df15.index.dayofweek
    return df15

# ─── Episode Detection ────────────────────────────────────────────────────────
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

# ─── Trade Simulation ─────────────────────────────────────────────────────────
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
        if reached==RECOVERY and b.name.hour==SESSION_END: return b['close_bid'],i,'ride_end'
        if reached!=RECOVERY and wc>=WEAKNESS_TIMEOUT: return b['close_bid'],i,'weak'
    return df15.iloc[ei+h]['close_bid'],h,'timeout'

# ─── Stats ────────────────────────────────────────────────────────────────────
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
        m=t['entry_time'].tz_convert('Asia/Hong_Kong').strftime('%Y-%m')
        if m not in months: months[m]=[]
        months[m].append(t['pnl'])
    rows=[]
    for m in sorted(months.keys()):
        pnls=months[m];n=len(pnls);s=sum(pnls);wr=sum(1 for p in pnls if p>0)/max(n,1)*100
        ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
        pf=ps/ns if ns>0 else 99
        rows.append({'Month':m,'Trades':n,'PnL':s,'WR':wr,'PF':pf,'Avg':s/n})
    return rows

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('='*72)
    print('  V28 WR90 CUMVOL EPISODE — MECHANICALLY IMPROVED')
    print('='*72)

    print('\n[1] Loading data...')
    df1m=load_oil_data(); df15=build_15m(df1m)
    print(f'  1m bars: {len(df1m):,}  →  15m bars: {len(df15):,}')
    print(f'  Date range (UTC): {df15.index[0]} → {df15.index[-1]}')
    print(f'  Date range (HKT): {df15.index[0].tz_convert("Asia/Hong_Kong")} → {df15.index[-1].tz_convert("Asia/Hong_Kong")}')

    # Pre-compute ATR median from POST-FILTER episode set (not global)
    # Episodes have much higher ATR than the full bar population
    print(f'\n  Computing ATR thresholds from episode population...')

    print(f'\n[2] Finding WR90 episodes (WR<{ENTRY_THRESH}, UK 7-16 UTC)...')
    episodes=find_episodes(df15)
    raw_count=len(episodes)
    print(f'  Raw episodes: {raw_count}')

    # CumVol filter
    episodes=[ep for ep in episodes if ep['cum_vol']>=CUMVOL_MIN]
    print(f'  After CumVol≥{CUMVOL_MIN:,}: {len(episodes)}')

    # EpBars filter (exhaustion depth)
    if EP_BARS_MIN > 1:
        episodes=[ep for ep in episodes if ep['bars']>=EP_BARS_MIN]
        print(f'  After EpBars≥{EP_BARS_MIN}: {len(episodes)}')

    # ATR filter (low-vol regime) — compute median from THIS episode set
    if ATR_BELOW_MEDIAN and len(episodes)>0:
        ep_atrs = [df15.iloc[ep['entry']]['atr14'] for ep in episodes]
        atr_median = np.median(ep_atrs)
        episodes=[ep for ep in episodes if df15.iloc[ep['entry']]['atr14']<=atr_median]
        print(f'  After ATR≤{atr_median:.0f} (median of episode set): {len(episodes)}')

    # Day-of-week and late-session filters
    filtered_by_dow=0; filtered_by_hour=0
    filtered_eps=[]
    for ep in episodes:
        row=df15.iloc[ep['entry']]
        if EXCLUDE_FRIDAY and row['dayofweek']==4:
            filtered_by_dow+=1; continue
        if EXCLUDE_LATE_SESSION and row['hour']>=15:
            filtered_by_hour+=1; continue
        filtered_eps.append(ep)
    episodes=filtered_eps
    if EXCLUDE_FRIDAY: print(f'  After exclude Friday: {len(episodes)} (removed {filtered_by_dow})')
    if EXCLUDE_LATE_SESSION: print(f'  After exclude 15-16h UTC: {len(episodes)} (removed {filtered_by_hour})')

    print(f'\n[3] Simulating trades (TP={TP}/SL={SL}, max {MAX_BARS}×15m bars)...')
    trades=[]
    for ep in episodes:
        ex,bars,reason=sim_trade(ep['entry'], df15)
        pnl=ex-df15.iloc[ep['entry']]['close_ask']
        row=df15.iloc[ep['entry']]
        trades.append({
            'entry_time':df15.index[ep['entry']],
            'entry_price':row['close_ask'],
            'exit_price':ex,'pnl':pnl,'bars':bars,'reason':reason,
            'cum_vol':ep['cum_vol'],'ep_bars':ep['bars'],
            'entry_wr':row['wr'],'atr':row['atr14'],
            'dow':row['dayofweek'],'hour_utc':row['hour'],
        })

    # ─── FULL TRADE TABLE (HKT) ──────────────────────────────────────────────
    print(f'\n{"="*100}')
    print(f'  TRADE LOG ({len(trades)} trades) — All times HKT (UTC+8)')
    print(f'{"="*100}')
    print(f'{"#":>4s} {"Entry Time (HKT)":>22s} {"Entry":>8s} {"Exit":>8s} {"PnL":>8s} {"15mBars":>8s} {"Reason":>10s} {"EntryWR":>8s} {"CumVol":>10s} {"ATR":>6s}')
    print(f'{"-"*96}')
    for i,t in enumerate(trades):
        hkt=t['entry_time'].tz_convert('Asia/Hong_Kong').strftime('%Y-%m-%d %H:%M')
        print(f'{i+1:>4d} {hkt:>22s} {t["entry_price"]:>8.1f} {t["exit_price"]:>8.1f} {t["pnl"]:>+8.1f} {t["bars"]:>8d} {t["reason"]:>10s} {t["entry_wr"]:>8.1f} {t["cum_vol"]:>10.0f} {t["atr"]:>6.1f}')

    # ─── SUMMARY STATS ────────────────────────────────────────────────────────
    s=trade_stats(trades)
    pnls=[t['pnl'] for t in trades]
    cum=np.cumsum(pnls); peak=np.maximum.accumulate(cum); dd=cum-peak
    max_dd=dd.min(); max_dd_pct=max_dd/peak[np.argmin(dd)]*100 if max_dd<0 else 0

    print(f'\n{"="*72}')
    print(f'  SUMMARY STATS')
    print(f'{"="*72}')
    print(f'  Trades       : {s["trades"]}')
    print(f'  Net PnL      : {s["pnl"]:+.1f} pts  ({s["pnl"]/10:+.1f} USD/contract @ 0.10 tick)')
    print(f'  Win Rate     : {s["wr"]:.1f}%')
    print(f'  Profit Factor: {s["pf"]:.2f}')
    print(f'  Avg Trade    : {s["avg"]:+.2f} pts')
    print(f'  Max Win      : {s["max_win"]:+.1f}')
    print(f'  Max Loss     : {s["max_loss"]:+.1f}')
    print(f'  Max DD       : {max_dd:+.1f} pts ({max_dd_pct:.1f}%)')

    # ─── Exit Reason Distribution ─────────────────────────────────────────────
    reasons={}
    for t in trades:
        r=t['reason'];reasons[r]=reasons.get(r,0)+1
    print(f'\n  Exit Reasons:')
    for r,c in sorted(reasons.items(),key=lambda x:-x[1]):
        rpnls=[t['pnl'] for t in trades if t['reason']==r]
        rs=trade_stats([{'pnl':p} for p in rpnls])
        print(f'    {r:>10s}: {c:>4d} trades ({c/max(len(trades),1)*100:.0f}%)  PnL={sum(rpnls):+.0f}  WR={rs["wr"]:.0f}%  Avg={rs["avg"]:+.1f}')

    # ─── MONTHLY BREAKDOWN (HKT) ──────────────────────────────────────────────
    monthly=monthly_breakdown(trades)
    print(f'\n{"="*72}')
    print(f'  MONTHLY BREAKDOWN (HKT)')
    print(f'{"="*72}')
    print(f'{"Month":>8s} {"Trades":>7s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s}')
    print(f'{"-"*52}')
    cum_pnl=0.0
    for m in monthly:
        cum_pnl+=m['PnL']
        print(f'{m["Month"]:>8s} {m["Trades"]:>7d} {m["PnL"]:>+10.0f} {m["WR"]:>6.0f}% {m["PF"]:>5.2f} {m["Avg"]:>+8.1f}')

    # ─── Yearly ───────────────────────────────────────────────────────────────
    yearly={}
    for t in trades:
        y=t['entry_time'].tz_convert('Asia/Hong_Kong').year
        if y not in yearly: yearly[y]=[]
        yearly[y].append(t['pnl'])
    print(f'\n  YEARLY:')
    for y in sorted(yearly.keys()):
        p=yearly[y];n=len(p);y_total=sum(p);wr=sum(1 for x in p if x>0)/max(n,1)*100
        ps=sum(x for x in p if x>0);ns=abs(sum(x for x in p if x<0))
        pf=ps/ns if ns>0 else 99
        print(f'    {y}: {n} trades  PnL={y_total:+.0f}  WR={wr:.0f}%  PF={pf:.2f}')

    # ─── Day of Week ──────────────────────────────────────────────────────────
    days=['Mon','Tue','Wed','Thu','Fri']
    print(f'\n  BY DAY OF WEEK (UTC):')
    for d in range(5):
        sub=[t for t in trades if t['dow']==d]
        if not sub: continue
        st=trade_stats([{'pnl':t['pnl']} for t in sub])
        print(f'    {days[d]:>4s}: {len(sub):3d} trades  PnL={st["pnl"]:+.0f}  WR={st["wr"]:.0f}%  PF={st["pf"]:.2f}')

    # ─── PnL Distribution ─────────────────────────────────────────────────────
    print(f'\n  PnL DISTRIBUTION:')
    pnls_sorted=sorted(pnls)
    print(f'    Min: {pnls_sorted[0]:+.1f}  P05: {np.percentile(pnls,5):+.1f}  P25: {np.percentile(pnls,25):+.1f}')
    print(f'    Med: {np.percentile(pnls,50):+.1f}  P75: {np.percentile(pnls,75):+.1f}  P95: {np.percentile(pnls,95):+.1f}')
    print(f'    Max: {pnls_sorted[-1]:+.1f}')

    if len(pnls)>1:
        mean_ret=np.mean(pnls);std_ret=np.std(pnls)
        print(f'    Mean±Std: {mean_ret:+.2f} ± {std_ret:.2f} (SR: {mean_ret/std_ret:.3f})')

    # ─── COMPARISON TO BASELINE ───────────────────────────────────────────────
    print(f'\n{"="*72}')
    print(f'  COMPARISON: BASELINE vs IMPROVED')
    print(f'{"="*72}')
    print(f'  {"Metric":>20s} {"Baseline":>12s} {"Improved":>12s} {"Delta":>10s}')
    print(f'  {"-"*58}')
    n_trades = s.get('trades', 0)
    net_pnl = s.get('pnl', 0.0)
    win_rate = s.get('wr', 0.0)
    profit_factor = s.get('pf', 0.0)
    avg_trade = s.get('avg', 0.0)
    print(f'  {"Trades":>20s} {"364":>12s} {n_trades:>12d} {n_trades-364:>+10d}')
    print(f'  {"PnL":>20s} {"+2615":>12s} {net_pnl:>+12.0f} {net_pnl-2615:>+10.0f}')
    print(f'  {"WR":>20s} {"44.8%":>12s} {win_rate:>11.1f}% {win_rate-44.8:>+9.1f}%')
    print(f'  {"PF":>20s} {"1.34":>12s} {profit_factor:>11.2f} {profit_factor-1.34:>+9.2f}')
    print(f'  {"Avg Trade":>20s} {"+7.18":>12s} {avg_trade:>+11.2f} {avg_trade-7.18:>+9.2f}')

    # ─── FILTERS APPLIED ──────────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print(f'  FILTERS APPLIED')
    print(f'{"="*72}')
    print(f'  Entry          : WR < {ENTRY_THRESH}')
    print(f'  CumVol         : ≥ {CUMVOL_MIN:,}')
    print(f'  EpBars         : ≥ {EP_BARS_MIN} (exhaustion depth — best single filter)')
    print(f'  ATR            : ≤ {atr_median:.0f} (median — low-vol regime)')
    print(f'  Exclude Friday : {"YES" if EXCLUDE_FRIDAY else "no"}')
    print(f'  Exclude 15-16h : {"YES" if EXCLUDE_LATE_SESSION else "no"}')
    print(f'  TP / SL / Max  : {TP} / {SL} / {MAX_BARS}×15min bars')
    print(f'  Ride if WR≥{-RECOVERY} at session end ({SESSION_END}h UTC)')
    print(f'  Weak exit      : WR ≤ {WEAK} for {WEAKNESS_TIMEOUT} bars')
    print(f'  No XGBoost     : Mechanical only (ML degrades this pattern)')

    print(f'\n{"="*72}')
    print('  DONE.')
    print(f'{"="*72}')

if __name__=='__main__':
    main()
