#!/usr/bin/env python3
'''v27 WR90 Episode — CumVol filter + Recovery Ride.

Logic:
  1. WR90 enters oversold zone (< entry_threshold), start accumulating volume
  2. WR90 exits oversold (crosses above entry_threshold) — this is ONE episode
  3. Only enter IF cumulative volume during oversold > cumvol_min
  4. Entry: on the first bar where WR > entry_threshold after episode
  5. Exit A: WR90 reaches -20 → hold until UK session end (16 UTC)
  6. Exit B: WR90 stays < -50 for timeout bars → weak recovery, exit
  7. Exit C: TP/SL fallback
'''

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

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
    return df15

def find_episodes(df15, entry_thresh=-80, session='uk'):
    """Find WR90 oversold episodes: contiguous bars where WR < entry_thresh.
    Returns list of (start_bar_idx, entry_bar_idx, cum_vol, episode_bars)."""
    in_s = df15['is_uk'] if session=='uk' else (df15['is_us'] if session=='us' else df15['is_uk']|df15['is_us'])
    oversold = (df15['wr'] < entry_thresh) & in_s
    episodes = []
    in_episode = False; ep_start = None; cum_vol = 0.0; bar_count = 0
    for i in range(len(df15)):
        if oversold.iloc[i]:
            if not in_episode: ep_start = i; cum_vol = 0.0; bar_count = 0
            in_episode = True; cum_vol += df15['volume'].iloc[i]; bar_count += 1
        else:
            if in_episode:
                entry_bar_idx = i  # first bar after oversold ends = entry
                if entry_bar_idx < len(df15)-1 and in_s.iloc[entry_bar_idx]:
                    episodes.append({'start_ep': ep_start, 'entry_bar': entry_bar_idx,
                                    'cum_vol': cum_vol, 'bars': bar_count})
                in_episode = False; cum_vol = 0.0; bar_count = 0
    return episodes

def sim_episode_trade(ei, df15, tp, sl, max_bars, recovery_thresh=-20,
                       weakness_thresh=-50, weakness_timeout=12, session_end_hour=16):
    """Simulate one long trade with episode-aware exit logic."""
    ep = df15.iloc[ei]['close_ask']
    horizon = min(max_bars, len(df15)-ei-1)
    reached_recovery = False; weak_count = 0

    for i in range(1, horizon+1):
        b = df15.iloc[ei+i]; bar_hour = b.name.hour
        # TP/SL
        if b['high'] >= ep+tp: return ep+tp, i, 'tp'
        if b['low'] <= ep-sl: return ep-sl, i, 'sl'
        # Track WR recovery
        if b['wr'] >= recovery_thresh:
            reached_recovery = True
        # Weakness timeout
        if b['wr'] < weakness_thresh:
            weak_count += 1
        else:
            weak_count = 0
        # If recovered, hold until session end
        if reached_recovery and bar_hour == session_end_hour:
            return b['close_bid'], i, 'ride_end'
        # If never recovered and weak for too long, exit
        if not reached_recovery and weak_count >= weakness_timeout:
            return b['close_bid'], i, 'weak_timeout'
    return df15.iloc[ei+horizon]['close_bid'], horizon, 'timeout'

def stats(pnls):
    if not pnls: return {'trades':0,'pnl':0.0,'wr':0.0,'pf':0.0,'avg':0.0}
    n=len(pnls);t=sum(pnls);wr=sum(1 for p in pnls if p>0)/n*100
    ps=sum(p for p in pnls if p>0);ns=abs(sum(p for p in pnls if p<0))
    return {'trades':n,'pnl':t,'wr':wr,'pf':ps/ns if ns>0 else 99,'avg':t/n}

def main():
    print('='*72); print('v27 WR90 CumVol Episode + Recovery Ride'); print('='*72)
    print(); print('[1] Loading...'); df1m=load_oil_data(); df15=build_15m(df1m)
    print(f'  {len(df15):,} 15m bars')

    # 1. Find all episodes
    episodes = find_episodes(df15, -80, 'uk')
    print(f'\n[2] Found {len(episodes)} WR<-80 episodes (UK session)')
    # CumVol distribution
    cvs = [e['cum_vol'] for e in episodes]
    bars_list = [e['bars'] for e in episodes]
    print(f'  CumVol: p10={np.percentile(cvs,10):.0f} p25={np.percentile(cvs,25):.0f} '
          f'p50={np.percentile(cvs,50):.0f} p75={np.percentile(cvs,75):.0f} p90={np.percentile(cvs,90):.0f}')
    print(f'  Bars in episode: p25={np.percentile(bars_list,25):.0f} p50={np.percentile(bars_list,50):.0f} '
          f'p75={np.percentile(bars_list,75):.0f} p90={np.percentile(bars_list,90):.0f} max={max(bars_list)}')

    # 2. CumVol filter sweep
    print(f'\n[3] CumVol filter sweep (TP=80/SL=40)...')
    print(f"  {'CumVol>':>10s} {'Episodes':>10s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} "
          f"{'Avg':>8s} {'Ride%':>7s}")
    print(f"  {'-'*80}")

    for cmin in [0, 1000, 3000, 5000, 7500, 10000, 15000]:
        trades = []
        n_ride = 0
        for ep in episodes:
            if ep['cum_vol'] < cmin: continue
            ei = ep['entry_bar']
            ex, bars, reason = sim_episode_trade(ei, df15, 80, 40, 60)
            pnl = ex - df15.iloc[ei]['close_ask']
            trades.append({'pnl': pnl, 'reason': reason, 'bars': bars, 'cum_vol': ep['cum_vol']})
            if reason == 'ride_end': n_ride += 1
        pnls = [t['pnl'] for t in trades]; s = stats(pnls)
        n_ep_used = sum(1 for e in episodes if e['cum_vol']>=cmin)
        print(f"  {cmin:>10d} {n_ep_used:>10d} {s['trades']:>7d} {s['pnl']:>+10.1f} "
              f"{s['wr']:>6.1f}% {s['pf']:>5.2f} {s['avg']:>+8.2f} {n_ride/(s['trades']+0.01)*100:>6.1f}%")

    # 3. Sweep entry threshold
    print(f'\n[4] Entry threshold sweep (CumVol>0, TP=80/SL=40)...')
    print(f"  {'Entry<':>8s} {'Episodes':>10s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'Avg':>8s}")
    print(f"  {'-'*65}")
    for thresh in [-95, -90, -85, -80, -75]:
        eps = find_episodes(df15, thresh, 'uk')
        trades=[]
        for ep in eps:
            ex, bars, reason = sim_episode_trade(ep['entry_bar'], df15, 80, 40, 60)
            trades.append({'pnl': ex - df15.iloc[ep['entry_bar']]['close_ask'], 'reason': reason})
        pnls=[t['pnl'] for t in trades]; s=stats(pnls)
        print(f"  {thresh:>+8d} {len(eps):>10d} {s['trades']:>7d} {s['pnl']:>+10.1f} "
              f"{s['wr']:>6.1f}% {s['pf']:>5.2f} {s['avg']:>+8.2f}")

    # 4. Ride logic: best config detail
    print(f'\n[5] Best config detail (WR<-80, CumVol>5000, TP=80/SL=40)...')
    trades_detail=[]
    for ep in episodes:
        if ep['cum_vol'] < 5000: continue
        ex, bars, reason = sim_episode_trade(ep['entry_bar'], df15, 80, 40, 60)
        pnl = ex - df15.iloc[ep['entry_bar']]['close_ask']
        trades_detail.append({'pnl':pnl,'reason':reason,'bars':bars,'cum_vol':ep['cum_vol']})
    pnls=[t['pnl'] for t in trades_detail]; s=stats(pnls)
    reasons={r:sum(1 for t in trades_detail if t['reason']==r) for r in set(t['reason'] for t in trades_detail)}
    print(f"  {s['trades']}t, {s['pnl']:+.0f}pts, {s['wr']:.1f}% WR, PF={s['pf']:.2f}, Avg={s['avg']:+.1f}")
    print(f"  Exit reasons: {reasons}")
    # Ride ends detail
    ride_trades=[t for t in trades_detail if t['reason']=='ride_end']
    if ride_trades: print(f"  Ride ends: {len(ride_trades)} trades, "
                          f"{sum(t['pnl'] for t in ride_trades):+.0f}pts, "
                          f"avg bars={np.mean([t['bars'] for t in ride_trades]):.0f}")
    weak_trades=[t for t in trades_detail if t['reason']=='weak_timeout']
    if weak_trades: print(f"  Weak timeouts: {len(weak_trades)} trades, "
                          f"{sum(t['pnl'] for t in weak_trades):+.0f}pts")

    # 5. Recovery-ride-only: what if we only trade episodes that reach -20?
    print(f'\n[6] Recovery-ride only: enter on WR>-80, hold to session end IF WR reaches -20')
    print(f"  {'CumVol>':>10s} {'Trades':>7s} {'Rides':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s}")
    print(f"  {'-'*55}")
    for cmin in [0, 3000, 5000, 7500, 10000]:
        all_trades=[]; rides=0
        for ep in episodes:
            if ep['cum_vol'] < cmin: continue
            ei=ep['entry_bar'];ep_p=df15.iloc[ei]['close_ask'];h=min(60,len(df15)-ei-1)
            reached_20=False; exit_p=ep_p; exit_bars=h
            for i in range(1,h+1):
                b=df15.iloc[ei+i]
                if b['low']<=ep_p-40: exit_p=ep_p-40; exit_bars=i; break
                if b['wr']>=-20: reached_20=True
                if reached_20 and b.name.hour==16: exit_p=b['close_bid']; exit_bars=i; break
            all_trades.append({'pnl':exit_p-ep_p,'ride':reached_20,'bars':exit_bars})
            if reached_20: rides+=1
        pnls=[t['pnl'] for t in all_trades]; s=stats(pnls)
        print(f"  {cmin:>10d} {s['trades']:>7d} {rides:>7d} {s['pnl']:>+10.1f} "
              f"{s['wr']:>6.1f}% {s['pf']:>5.2f}")

    print(); print('DONE.')

if __name__=='__main__': main()
