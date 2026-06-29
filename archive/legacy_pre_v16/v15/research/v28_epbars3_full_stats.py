#!/usr/bin/env python3
"""v28 WR90 + EpBars≥3 — BEST CONFIG FULL STATS (HKT output)
Session: NY 03:00–12:00 local (America/New_York tz, handles DST automatically).
"""

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_SESSION_START = 3; NY_SESSION_END = 12  # NY local hours (03-12, entry window)
NY_FORCE_CLOSE_HOUR = 14; NY_FORCE_CLOSE_MIN = 28  # Hard close at NY 14:28
ENTRY_THRESH = -80; CUMVOL_MIN = 15000; TP = 80; SL = 40; MAX_BARS = 60
RECOVERY = -20; WEAK = -50; WEAKNESS_TIMEOUT = 12; EP_BARS_MIN = 3

def load_oil_data(s='2024-01-01', e='2026-06-30'):
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
    # Convert to NY local hour for session filtering (handles DST correctly)
    ny_time = df15.index.tz_convert('America/New_York')
    df15['ny_hour'] = ny_time.hour
    df15['ny_minute'] = ny_time.minute
    df15['is_session'] = (df15['ny_hour'] >= NY_SESSION_START) & (df15['ny_hour'] <= NY_SESSION_END)
    df15['hour'] = df15.index.hour  # keep UTC hour for display
    df15['atr14'] = (df15['high'] - df15['low']).rolling(14).mean()
    df15['dayofweek'] = df15.index.dayofweek
    return df15

def find_episodes(df15):
    in_s = df15['is_session']; oversold = (df15['wr'] < ENTRY_THRESH) & in_s
    episodes = []; in_ep = False; cv = 0.0; bc = 0
    for i in range(len(df15)):
        if oversold.iloc[i]:
            if not in_ep: ep_start = i; cv = 0.0; bc = 0
            in_ep = True; cv += df15['volume'].iloc[i]; bc += 1
        else:
            if in_ep:
                ebi = i
                if ebi < len(df15) - 1 and in_s.iloc[ebi]:
                    episodes.append({'start': ep_start, 'entry': ebi, 'cum_vol': cv, 'bars': bc})
                in_ep = False; cv = 0.0; bc = 0
    return episodes

def sim_trade(ei, df15):
    ep = df15.iloc[ei]['close_ask']; h = min(MAX_BARS, len(df15) - ei - 1)
    reached = -99; wc = 0
    for i in range(1, h + 1):
        b = df15.iloc[ei + i]
        # Hard force-close at NY 14:28
        post_close = (b['ny_hour'] > NY_FORCE_CLOSE_HOUR or 
                      (b['ny_hour'] == NY_FORCE_CLOSE_HOUR and b['ny_minute'] >= NY_FORCE_CLOSE_MIN))
        if post_close:
            return b['close_bid'], i, 'ny_close'
        if b['high'] >= ep + TP: return ep + TP, i, 'tp'
        if b['low'] <= ep - SL: return ep - SL, i, 'sl'
        if b['wr'] >= RECOVERY: reached = RECOVERY
        if b['wr'] < WEAK: wc += 1
        else: wc = 0
        # Ride-to-close if WR recovered and we pass force-close time
        if reached == RECOVERY and post_close:
            return b['close_bid'], i, 'ride_end'
        if reached != RECOVERY and wc >= WEAKNESS_TIMEOUT: return b['close_bid'], i, 'weak'
    return df15.iloc[ei + h]['close_bid'], h, 'timeout'

def trade_stats(trades):
    if not trades: return {'trades':0,'pnl':0.0,'wr':0.0,'pf':0.0,'avg':0.0}
    pnls=[t['pnl'] for t in trades]
    n=len(pnls);t=sum(pnls)
    wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'trades':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99,'avg':t/n,
            'max_win':max(pnls),'max_loss':min(pnls)}

def main():
    print('='*72)
    print('  V28 WR90 + EpBars>=3 — BEST CONFIG FULL STATS')
    print(f'  Session: NY {NY_SESSION_START:02d}:00-{NY_SESSION_END:02d}:00 (America/New_York, DST-aware)')
    print(f'  Entry WR<{ENTRY_THRESH}  CumVol>={CUMVOL_MIN:,}  EpBars>={EP_BARS_MIN}')
    print(f'  TP={TP}/SL={SL}  Max {MAX_BARS} x 15m bars')
    print('='*72)

    df1m=load_oil_data(); df15=build_15m(df1m)
    print(f'\n[1] Data: {len(df1m):,} 1m bars -> {len(df15):,} 15m bars')
    print(f'    UTC: {df15.index[0]} -> {df15.index[-1]}')
    print(f'    HKT: {df15.index[0].tz_convert("Asia/Hong_Kong")} -> {df15.index[-1].tz_convert("Asia/Hong_Kong")}')
    # Show DST transition capture
    ny = df15.index.tz_convert('America/New_York')
    print(f'    NY tz starts: {ny[0]} (offset={ny[0].utcoffset()})')
    print(f'    NY tz ends:   {ny[-1]} (offset={ny[-1].utcoffset()})')

    episodes=find_episodes(df15)
    raw=len(episodes)
    episodes=[ep for ep in episodes if ep['cum_vol']>=CUMVOL_MIN]
    print(f'\n[2] Episodes: raw={raw}  CumVol>={CUMVOL_MIN:,}={len(episodes)}')
    before_epbars=len(episodes)
    episodes=[ep for ep in episodes if ep['bars']>=EP_BARS_MIN]
    print(f'    EpBars>={EP_BARS_MIN}: {len(episodes)} (removed {before_epbars-len(episodes)} shallow)')

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
            'ny_hour':row['ny_hour'],
        })

    # FULL TRADE TABLE (HKT)
    print(f'\n{"="*105}')
    print(f'  TRADE LOG ({len(trades)} trades) — Times in HKT (UTC+8)')
    print(f'{"="*105}')
    print(f'{"#":>4s} {"Entry Time (HKT)":>22s} {"Entry":>8s} {"Exit":>8s} {"PnL":>8s} {"15mB":>5s} {"Reason":>10s} {"WR":>7s} {"NYh":>4s} {"CumVol":>10s} {"ATR":>6s}')
    print(f'{"-"*101}')
    for i,t in enumerate(trades):
        hkt=t['entry_time'].tz_convert('Asia/Hong_Kong').strftime('%Y-%m-%d %H:%M')
        print(f'{i+1:>4d} {hkt:>22s} {t["entry_price"]:>8.1f} {t["exit_price"]:>8.1f} {t["pnl"]:>+8.1f} {t["bars"]:>5d} {t["reason"]:>10s} {t["entry_wr"]:>+7.1f} {t["ny_hour"]:>4d} {t["cum_vol"]:>10.0f} {t["atr"]:>6.1f}')

    # SUMMARY
    s=trade_stats(trades)
    pnls=[t['pnl'] for t in trades]
    cum_pnl=np.cumsum(pnls); peak=np.maximum.accumulate(cum_pnl); dd=cum_pnl-peak
    max_dd=dd.min(); max_dd_pct=max_dd/peak[np.argmin(dd)]*100 if max_dd<0 else 0

    in_dd=False; longest_dd=0; current_dd=0
    for i in range(len(cum_pnl)):
        if cum_pnl[i]<peak[i]:
            if not in_dd: in_dd=True
            current_dd+=1
        else:
            if in_dd: longest_dd=max(longest_dd,current_dd); current_dd=0; in_dd=False
    if in_dd: longest_dd=max(longest_dd,current_dd)

    print(f'\n{"="*72}')
    print(f'  SUMMARY STATS')
    print(f'{"="*72}')
    print(f'  Trades       : {s["trades"]}')
    print(f'  Net PnL      : {s["pnl"]:+.1f} pts ({s["pnl"]/10:+.1f} USD/contract)')
    print(f'  Win Rate     : {s["wr"]:.1f}%')
    print(f'  Profit Factor: {s["pf"]:.2f}')
    print(f'  Avg Trade    : {s["avg"]:+.2f} pts')
    print(f'  Max Win      : {s["max_win"]:+.1f}')
    print(f'  Max Loss     : {s["max_loss"]:+.1f}')
    print(f'  Peak Equity  : {peak[-1]:+.1f}')
    print(f'  Final Equity : {cum_pnl[-1]:+.1f}')
    print(f'  Max DD       : {max_dd:+.1f} pts ({max_dd_pct:.1f}%)')
    print(f'  Longest DD   : {longest_dd} trades')

    # EXIT REASONS
    reasons={}
    for t in trades:
        r=t['reason'];reasons[r]=reasons.get(r,0)+1
    print(f'\n  Exit Reasons:')
    for r,c in sorted(reasons.items(),key=lambda x:-x[1]):
        rpnls=[t['pnl'] for t in trades if t['reason']==r]
        rs=trade_stats([{'pnl':p} for p in rpnls])
        print(f'    {r:>10s}: {c:>4d} ({c/len(trades)*100:.0f}%)  PnL={sum(rpnls):+.0f}  WR={rs["wr"]:.0f}%  Avg={rs["avg"]:+.1f}')

    # MONTHLY BREAKDOWN (HKT)
    months={}
    for t in trades:
        m=t['entry_time'].tz_convert('Asia/Hong_Kong').strftime('%Y-%m')
        if m not in months: months[m]=[]
        months[m].append(t['pnl'])
    print(f'\n{"="*72}')
    print(f'  MONTHLY BREAKDOWN (HKT)')
    print(f'{"="*72}')
    print(f'{"Month":>8s} {"Trades":>7s} {"PnL":>10s} {"WR":>7s} {"PF":>6s} {"Avg":>8s} {"Cum":>10s}')
    print(f'{"-"*58}')
    running=0.0
    for m in sorted(months.keys()):
        p=months[m];n=len(p);s2=sum(p);wr=sum(1 for x in p if x>0)/n*100
        ps=sum(x for x in p if x>0);ns=abs(sum(x for x in p if x<0))
        pf=ps/ns if ns>0 else 99
        running+=s2
        print(f'{m:>8s} {n:>7d} {s2:>+10.0f} {wr:>6.0f}% {pf:>5.2f} {s2/n:>+8.1f} {running:>+10.0f}')

    # YEARLY
    yearly={}
    for t in trades:
        y=t['entry_time'].tz_convert('Asia/Hong_Kong').year
        if y not in yearly: yearly[y]=[]
        yearly[y].append(t['pnl'])
    print(f'\n  YEARLY:')
    for y in sorted(yearly.keys()):
        p=yearly[y];n=len(p);yt=sum(p);wr=sum(1 for x in p if x>0)/max(n,1)*100
        ps=sum(x for x in p if x>0);ns=abs(sum(x for x in p if x<0))
        pf=ps/ns if ns>0 else 99
        print(f'    {y}: {n} trades  PnL={yt:+.0f}  WR={wr:.0f}%  PF={pf:.2f}')

    # DAY OF WEEK
    days=['Mon','Tue','Wed','Thu','Fri']
    print(f'\n  BY DAY OF WEEK (UTC):')
    for d in range(5):
        sub=[t for t in trades if t['dow']==d]
        if not sub: continue
        st=trade_stats([{'pnl':t['pnl']} for t in sub])
        print(f'    {days[d]:>4s}: {len(sub):3d}  PnL={st["pnl"]:+.0f}  WR={st["wr"]:.0f}%  PF={st["pf"]:.2f}')

    # BY NY HOUR
    print(f'\n  BY NY HOUR:')
    for h in sorted(set(t['ny_hour'] for t in trades)):
        sub=[t for t in trades if t['ny_hour']==h]
        st=trade_stats([{'pnl':t['pnl']} for t in sub])
        print(f'    NY {h:2d}:00: {len(sub):3d}  PnL={st["pnl"]:+.0f}  WR={st["wr"]:.0f}%  PF={st["pf"]:.2f}')

    # PnL DISTRIBUTION
    print(f'\n  PnL DISTRIBUTION:')
    for pct,label in [(0,'Min'),(5,'P05'),(10,'P10'),(25,'P25'),(50,'Med'),(75,'P75'),(90,'P90'),(95,'P95'),(100,'Max')]:
        print(f'    {label:>4s}: {np.percentile(pnls,pct):+.1f}')

    if len(pnls)>1:
        mean_ret=np.mean(pnls);std_ret=np.std(pnls)
        neg=[p for p in pnls if p<0]
        downside_std=np.std(neg) if len(neg)>1 else std_ret
        sharpe=mean_ret/std_ret if std_ret>0 else 0
        sortino=mean_ret/downside_std if downside_std>0 else 0
        print(f'    Sharpe (mu/sigma): {sharpe:.3f}')
        print(f'    Sortino (mu/sigma_down): {sortino:.3f}')
        print(f'    Mean +/- Std: {mean_ret:+.2f} +/- {std_ret:.2f}')

    # CONSECUTIVE
    wins=[1 if p>0 else 0 for p in pnls]
    max_ws=0; max_ls=0; cw=0; cl=0
    for w in wins:
        if w==1: cw+=1; cl=0; max_ws=max(max_ws,cw)
        else: cl+=1; cw=0; max_ls=max(max_ls,cl)
    print(f'    Max Win Streak: {max_ws} trades')
    print(f'    Max Lose Streak: {max_ls} trades')

    print(f'\n{"="*72}')
    print('  DONE.')
    print(f'{"="*72}')

if __name__=='__main__':
    main()
