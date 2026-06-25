#!/usr/bin/env python3
"""Oil Combined Backtest — WR90 Long + Short Impulse + Oil Retrace
===================================================================
Full V14-style stats: Sharpe, Sortino, MaxDD, yearly/monthly breakdown.
Saves feature snapshots to SQLite for live bot comparison.
Saves CSV: runtime/oil_combined_backtest_trades.csv

Config (from v29 research best combos):
  WR90 Long      : WR<-75, CV>=5K, Ep>=2, TP=60/SL=20, XGBoost>=0.65
  Short Impulse  : prev_change<-14, vol>800, TP=120/SL=80, XGB>=0.55
  Oil Retrace    : close-Dlow>20, avgRange3>30, cl-op<-10, wick<16,
                   TP=30/SL=15, XGBoost>=0.60

Run: python3 v15/backtest/backtest_oil.py [start_date] [end_date]
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from data.data_loader import DataLoader
from v15.research.v29_oil_journal import OilSignalJournal
import warnings
warnings.filterwarnings('ignore')

# ======================== CONFIG ========================
NY_S, NY_E, NY_FC_H, NY_FC_M = 3, 12, 14, 28
LONG_MAX_B = 60
LONG_ENTRY = -75
LONG_CV = 5000
LONG_EP_MIN = 2
LONG_TP, LONG_SL = 60, 20
LONG_WR_ML_TH = 0.65

SI_CHANGE_MAX, SI_VOL_MIN = -14.0, 800
SI_TP, SI_SL, SI_MAX_B = 120, 80, 90
SI_PROB = 0.55

RET_DLOW, RET_RNG = 20, 30
RET_CHG, RET_WICK = -10, 16
RET_TP, RET_SL = 30, 15
RET_ML_TH = 0.60

# ======================== DATA LOADING ========================
def load(start='2024-01-01', end='2026-06-30'):
    loader = DataLoader()
    raw = loader.load_data('prices', start, end)
    raw.index = pd.to_datetime(raw['timestamp'], unit='ms')
    df = pd.DataFrame(index=raw.index)
    for c, src in [('open', 'openPrice_ask'), ('high', 'highPrice_ask'),
                   ('low', 'lowPrice_ask'), ('close_ask', 'closePrice_ask'),
                   ('close_bid', 'closePrice_bid'), ('volume', 'lastTradedVolume')]:
        df[c] = raw[src].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def build_15m(df1m):
    d = df1m.resample('15min', label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min',
         'close_ask': 'last', 'close_bid': 'last', 'volume': 'sum'}).dropna()
    n = 14
    hh = d['high'].rolling(n).max()
    ll = d['low'].rolling(n).min()
    d['wr'] = ((hh - d['close_ask']) / (hh - ll + 0.01)) * -100
    ny = d.index.tz_convert('America/New_York')
    d['Dlow'] = d['low'].groupby(ny.date).transform('min')
    d['range'] = d['high'] - d['low']
    d['avg_r3'] = d['range'].rolling(3, 3).mean()
    d['wb'] = np.minimum(d['open'], d['close_ask']) - d['low']
    d['bc'] = d['close_ask'] - d['open']
    d['cad'] = d['close_ask'] - d['Dlow']
    d['ny_h'] = ny.hour
    d['ny_m'] = ny.minute
    d['ins'] = (d['ny_h'] >= NY_S) & (d['ny_h'] <= NY_E)
    d['ret_1b'] = d['close_ask'].pct_change(1)
    d['ret_3b'] = d['close_ask'].pct_change(3)
    d['ret_5b'] = d['close_ask'].pct_change(5)
    d['vol_r'] = d['volume'] / (d['volume'].rolling(20).mean() + 0.01)
    d['h_dlow'] = d['high'] - d['Dlow']
    d['l_dlow'] = d['low'] - d['Dlow']
    d['body'] = abs(d['close_ask'] - d['open'])
    d['up'] = (d['close_ask'] > d['open']).astype(int)
    d['up_p1'] = d['up'].shift(1)
    d['up_p2'] = d['up'].shift(2)
    d['body_p1'] = d['body'].shift(1)
    d['range_p1'] = d['range'].shift(1)
    return d


# ======================== SIMULATION (SL-capped timeout) ========================
def sim_no_cascade(d, sigs, tp, sl, stype='wr90'):
    """One trade per signal. FIX: ALL timeout exits cap loss at SL."""
    pnls = []; trades = []; it = False; ct = cs = ep = ei = eb = 0; si = 0
    while si < len(sigs):
        si_i = sigs[si]['idx']
        if not it:
            it = True; eb = si_i
            ep = d.iloc[si_i]['close_ask']; ct = ep + tp; cs = ep - sl
            ei = si_i; si += 1; continue
        ex = False; er = ''; ex_p = 0.0; ex_i = ei
        for j in range(ei + 1, si_i + 1):
            b = d.iloc[j]
            post = (b['ny_h'] > NY_FC_H) or (b['ny_h'] == NY_FC_H and b['ny_m'] >= NY_FC_M)
            if post: ex = True; er = 'ny_close'; ex_p = b['close_bid']; ex_i = j; break
            if b['high'] >= ct: ex = True; er = 'tp'; ex_p = ep + tp; ex_i = j; break
            if b['low'] <= cs: ex = True; er = 'sl'; ex_p = ep - sl; ex_i = j; break
        if ex:
            pnl = ex_p - ep; pnls.append(pnl)
            trades.append({'entry': d.index[eb], 'exit': d.index[ex_i],
                           'pnl': pnl, 'reason': er, 'type': stype, 'side': 1,
                           'entry_price': ep, 'exit_price': ex_p})
            it = False
            if ex_i == si_i: si += 1
            continue
        if si_i - ei > LONG_MAX_B:
            raw = d.iloc[ei + LONG_MAX_B]['close_bid'] - ep
            pnl = max(raw, -sl)
            pnls.append(pnl)
            trades.append({'entry': d.index[eb], 'exit': d.index[ei + LONG_MAX_B],
                           'pnl': pnl, 'reason': 'timeout', 'type': stype, 'side': 1,
                           'entry_price': ep, 'exit_price': d.iloc[ei + LONG_MAX_B]['close_bid']})
            it = False; continue
        ne = d.iloc[si_i]['close_ask']
        ct = max(ct, ne + tp)
        cs = cs if cs < ne - sl else max(cs, ne - sl)
        si += 1
    if it:
        last = min(ei + LONG_MAX_B, len(d) - 1)
        raw = d.iloc[last]['close_bid'] - ep
        pnl = max(raw, -sl)
        pnls.append(pnl)
        trades.append({'entry': d.index[eb], 'exit': d.index[last],
                       'pnl': pnl, 'reason': 'timeout', 'type': stype, 'side': 1,
                       'entry_price': ep, 'exit_price': d.iloc[last]['close_bid']})
    return pnls, trades


def sim_full(d, sigs, tp, sl, stype='ret'):
    """Full sim tracking entry indices for ML training. Timeout SL-capped."""
    p = []; tr = []; m = []; it = False; ct = cs = ep = ei = eb = 0; si = 0
    while si < len(sigs):
        si_i = sigs[si]['idx']
        if not it:
            it = True; eb = si_i; ep = d.iloc[si_i]['close_ask']; ct = ep + tp; cs = ep - sl
            ei = si_i; m.append(si); si += 1; continue
        ex = False; er = ''; ex_p = 0.0; ex_i = ei
        for j in range(ei + 1, si_i + 1):
            b = d.iloc[j]
            post = (b['ny_h'] > NY_FC_H) or (b['ny_h'] == NY_FC_H and b['ny_m'] >= NY_FC_M)
            if post: ex = True; er = 'ny_close'; ex_p = b['close_bid']; ex_i = j; break
            if b['high'] >= ct: ex = True; er = 'tp'; ex_p = ep + tp; ex_i = j; break
            if b['low'] <= cs: ex = True; er = 'sl'; ex_p = ep - sl; ex_i = j; break
        if ex:
            pnl = ex_p - ep; p.append(pnl)
            tr.append({'entry': d.index[eb], 'exit': d.index[ex_i],
                       'pnl': pnl, 'reason': er, 'type': stype, 'side': 1,
                       'entry_price': ep, 'exit_price': ex_p})
            it = False
            if ex_i == si_i: si += 1
            continue
        if si_i - ei > LONG_MAX_B:
            raw = d.iloc[ei + LONG_MAX_B]['close_bid'] - ep
            pnl = max(raw, -sl); p.append(pnl)
            tr.append({'entry': d.index[eb], 'exit': d.index[ei + LONG_MAX_B],
                       'pnl': pnl, 'reason': 'timeout', 'type': stype, 'side': 1,
                       'entry_price': ep, 'exit_price': d.iloc[ei + LONG_MAX_B]['close_bid']})
            it = False; continue
        ct = max(ct, d.iloc[si_i]['close_ask'] + tp)
        cs = cs if cs < d.iloc[si_i]['close_ask'] - sl else max(cs, d.iloc[si_i]['close_ask'] - sl)
        si += 1
    if it:
        last = min(ei + LONG_MAX_B, len(d) - 1)
        raw = d.iloc[last]['close_bid'] - ep
        pnl = max(raw, -sl); p.append(pnl)
        tr.append({'entry': d.index[eb], 'exit': d.index[last],
                   'pnl': pnl, 'reason': 'timeout', 'type': stype, 'side': 1,
                   'entry_price': ep, 'exit_price': d.iloc[last]['close_bid']})
    return p, tr, m


# ======================== FEATURE LISTS ========================
RET_FEATS = ['cad', 'avg_r3', 'bc', 'wb', 'range', 'ret_1b', 'ret_3b', 'ret_5b',
             'vol_r', 'h_dlow', 'l_dlow', 'body', 'up', 'up_p1', 'up_p2', 'body_p1', 'range_p1']

WR_FEATS = ['wr', 'volume', 'range', 'avg_r3', 'cad', 'ret_1b', 'ret_3b',
            'vol_r', 'h_dlow', 'l_dlow', 'body', 'up', 'up_p1']

SI_FEATS = ['prev_change', 'prev2_change', 'prev_lower_wick', 'prev_volume',
            'prev_range', 'prev_spread', 'ATR', 'ATR_ratio',
            'ret_1m', 'ret_3m', 'ret_5m', 'vol_ratio_20',
            'up_count3_15min', 'ret_3_15m', 'ret_5_15m', 'dist_day_high']


# ======================== ML TRAIN ========================
def train_ml(d, sigs, tp, sl, feats, stype='ret', ml_th=0.60):
    """Walk-forward XGBoost training. Returns (pnls, trades, probas)."""
    p, tr, m = sim_full(d, sigs, tp, sl, stype)
    if len(p) < 30:
        return None
    n_m = len(m)
    X = np.array([[float(d.iloc[sigs[si]['idx']][f]) for f in feats] for si in m])
    y = np.array([1.0 if p[i] > 0 else 0.0 for i in range(n_m)])
    p = p[:n_m]
    tr = tr[:n_m]
    tdates = pd.DatetimeIndex([d.index[sigs[si]['idx']] for si in m])
    months = sorted(set(pd.Period(dt, 'M') for dt in tdates))
    tstart = pd.Period('2024-07', freq='M')
    pr = np.zeros(len(p))
    for tm in [mo for mo in months if mo >= tstart]:
        train_m = [mo for mo in months if mo < tm]
        tst = np.array([pd.Period(dt, 'M') == tm for dt in tdates])
        trn = np.array([pd.Period(dt, 'M') in train_m for dt in tdates])
        if trn.sum() < 20 or tst.sum() < 3:
            continue
        w = np.where(y[trn] == 1)[0]
        l = np.where(y[trn] == 0)[0]
        nm = min(len(w), len(l))
        if nm < 5:
            continue
        rng = np.random.RandomState(42 + tm.ordinal)
        bal = np.concatenate([rng.choice(w, nm, 0), rng.choice(l, nm, 0)])
        Xb, yb = X[trn][bal], y[trn][bal]
        spw = len(l) / max(1, len(w))
        model = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, scale_pos_weight=spw,
                                   random_state=42, verbosity=0)
        model.fit(Xb, yb)
        prib = model.predict_proba(X[tst])[:, 1]
        for j, idx in enumerate(np.where(tst)[0]):
            pr[idx] = prib[j]
    return p, tr, pr


# ======================== SHORT IMPULSE ========================
def compute_si_features(d1m):
    df = d1m.copy()
    df['change'] = df['close_ask'] - df['open']
    df['prev_change'] = df['change'].shift(1)
    df['prev2_change'] = df['change'].shift(2)
    df['prev_lower_wick'] = df['close_ask'].shift(1) - df['low'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_range'] = df['high'].shift(1) - df['low'].shift(1)
    df['prev_spread'] = df['close_ask'].shift(1) - df['close_bid'].shift(1)
    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close_ask'].shift()),
                    abs(df['low'] - df['close_ask'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['ATR_ratio'] = df['prev_range'] / (df['ATR'] + 0.01)
    df['ret_1m'] = df['close_ask'].pct_change()
    df['ret_3m'] = df['ret_1m'].rolling(3, 1).sum()
    df['ret_5m'] = df['ret_1m'].rolling(5, 1).sum()
    df['vol_ma_20'] = df['volume'].rolling(20, 5).mean()
    df['vol_ratio_20'] = df['prev_volume'] / (df['vol_ma_20'] + 0.01)
    df['ny_hour'] = df.index.tz_convert('America/New_York').hour.isin(list(range(3, 13)))
    # 15-min context
    d15 = df.resample('15min', label='right', closed='right').agg(
        {'open': 'first', 'close_ask': 'last'}).dropna()
    d15['up'] = np.where(d15['close_ask'] > d15['open'], 1,
                         np.where(d15['close_ask'] < d15['open'], -1, 0))
    d15['up_count3'] = d15['up'].rolling(3, 1).sum()
    d15['ret'] = d15['close_ask'].pct_change()
    d15['ret_3_15m'] = d15['ret'].rolling(3, 1).sum()
    d15['ret_5_15m'] = d15['ret'].rolling(5, 1).sum()
    f15 = d15[['up_count3', 'ret_3_15m', 'ret_5_15m']].reset_index()
    m15 = pd.merge_asof(df.reset_index().sort_values('timestamp'),
                         f15.rename(columns={'timestamp': 't15'}),
                         left_on='timestamp', right_on='t15',
                         direction='backward', tolerance=pd.Timedelta(minutes=15))
    m15.index = m15['timestamp']
    df['up_count3_15min'] = m15['up_count3']
    df['ret_3_15m'] = m15['ret_3_15m']
    df['ret_5_15m'] = m15['ret_5_15m']
    daily_high = df['high'].resample('D').max().reindex(df.index, method='ffill')
    df['dist_day_high'] = daily_high - df['close_ask']
    return df


def sim_si_fixed(ei, ep, df):
    """Short impulse sim. FIX: timeout capped at SI_SL."""
    stop = ep + SI_SL
    target = ep - SI_TP
    hz = min(SI_MAX_B, len(df) - ei - 1)
    nyz = df.index.tz_convert('America/New_York')
    for i in range(1, hz + 1):
        b = df.iloc[ei + i]
        bh = nyz[ei + i]
        if bh.hour > NY_FC_H or (bh.hour == NY_FC_H and bh.minute >= NY_FC_M):
            return b['close_ask'], i, 'ny_close'
        if b['high'] >= stop:
            return stop, i, 'sl'
        if b['low'] <= target:
            return target, i, 'tp'
    px = df.iloc[ei + hz]['close_ask']
    pnl = ep - px
    return (ep + SI_SL, hz, 'timeout') if pnl < -SI_SL else (px, hz, 'timeout')


# ======================== STATS ========================
def full_stats(trades, label='Combined'):
    """Print full v14-style stats."""
    tdf = pd.DataFrame(trades)
    tdf['pnl'] = tdf['pnl'].astype(float)
    tdf['entry'] = pd.to_datetime(tdf['entry'], utc=True)

    n = len(tdf)
    wins = int((tdf['pnl'] > 0).sum())
    net = float(tdf['pnl'].sum())
    wr_pct = wins / n * 100 if n else 0
    cs2 = tdf['pnl'].cumsum()
    mdd = float((cs2 - cs2.cummax()).min())
    gw = float(tdf[tdf['pnl'] > 0]['pnl'].sum())
    gl = abs(float(tdf[tdf['pnl'] < 0]['pnl'].sum()))
    pf = gw / gl if gl > 0 else float('inf')

    tdf['day'] = tdf['entry'].dt.tz_convert('America/New_York').dt.floor('D')
    dp = tdf.groupby('day')['pnl'].sum()
    md = float(dp.mean()) if len(dp) else 0
    sd = float(dp.std(ddof=1)) if len(dp) > 1 else 0
    ds_ = dp[dp < 0]
    dw = float(ds_.std(ddof=1)) if len(ds_) > 1 else 0
    shp = (md / sd) * 252 ** 0.5 if sd > 0 else 0
    sor = (md / dw) * 252 ** 0.5 if dw > 0 else 0

    print(f'\n{"="*72}')
    print(f'  {label}')
    print(f'{"="*72}')
    print(f'  Trades       : {n}  (W:{wins}  L:{n - wins})')
    print(f'  Win Rate     : {wr_pct:.1f}%')
    print(f'  Net PnL      : {net:+.1f} pts')
    print(f'  Avg/Trade    : {net / n:+.2f} pts' if n else '  Avg/Trade: N/A')
    print(f'  Max DD       : {mdd:+.1f} pts')
    print(f'  Profit Factor: {pf:.2f}')
    print(f'  Sharpe       : {shp:.2f}')
    print(f'  Sortino      : {sor:.2f}')

    print(f'\n  By Type:')
    if 'type' in tdf.columns:
        for pat, grp in tdf.groupby('type'):
            pw = (grp['pnl'] > 0).mean() * 100
            pn = len(grp)
            ps = grp['pnl'].sum()
            print(f'    {pat:20s}: {pn:4d}t  PnL={ps:+8.1f}  WR={pw:5.1f}%  '
                  f'avg={ps / pn:+7.2f}')

    print(f'\n  Exit Reason:')
    if 'reason' in tdf.columns:
        for r, grp in tdf.groupby('reason'):
            rw = (grp['pnl'] > 0).mean() * 100
            rn = len(grp)
            rs = grp['pnl'].sum()
            print(f'    {str(r):18s}: {rn:4d}t  WR={rw:5.1f}%  avg={rs / rn:+7.2f}')

    print(f'\n  Yearly:')
    tdf['year'] = tdf['entry'].dt.year
    for y in sorted(tdf['year'].unique()):
        gy = tdf[tdf['year'] == y]
        yn = len(gy)
        yt = gy['pnl'].sum()
        yw = (gy['pnl'] > 0).mean() * 100
        yl = gy[gy['side'] == 1] if 'side' in gy.columns else gy
        ys = gy[gy['side'] == -1] if 'side' in gy.columns else type('', (), {})()
        long_info = f'Long:{len(yl):3d}t/{yl["pnl"].sum():+.0f}' if len(yl) else ''
        short_info = f'Short:{len(ys):3d}t/{ys["pnl"].sum():+.0f}' if len(ys) else ''
        print(f'    {y}: {yn:4d}t  PnL={yt:+8.1f}  WR={yw:5.1f}%  '
              f'{long_info}  {short_info}')

    print(f'\n  Monthly:')
    monthly = tdf.copy()
    monthly['month'] = monthly['entry'].dt.tz_convert('Asia/Hong_Kong').dt.strftime('%Y-%m')
    mg = monthly.groupby('month')['pnl'].agg(['sum', 'count'])
    mg['wr'] = monthly.groupby('month')['pnl'].apply(lambda x: (x > 0).mean() * 100)
    mg = mg.fillna(0)
    print(f'  {"Month":>8s} {"T":>4s} {"PnL":>8s} {"WR":>5s} {"Cum":>9s}')
    cm = 0.0
    for m in sorted(mg.index):
        r = mg.loc[m]
        cm += r['sum']
        print(f'  {m:>8s} {int(r["count"]):>4d} {r["sum"]:>+8.0f} {r["wr"]:>4.0f}% {cm:>+9.0f}')

    print(f'\n  Past 20 (by entry time):')
    tdf_sorted = tdf.sort_values('entry')
    l20 = tdf_sorted.tail(20).copy()
    l20['ehkt'] = l20['entry'].dt.tz_convert('Asia/Hong_Kong').dt.strftime('%m-%d %H:%M')
    if 'exit' in l20.columns:
        l20['xhkt'] = pd.to_datetime(l20['exit'], utc=True).dt.tz_convert('Asia/Hong_Kong').dt.strftime('%m-%d %H:%M')
    else:
        l20['xhkt'] = '?'
    l20['d'] = l20['side'].map({1: 'L', -1: 'S'})
    for _, r in l20.iterrows():
        print(f'  {r["d"]:>2s} {r["ehkt"]:>11s} [{r["xhkt"]:>11s}] '
              f'{r["pnl"]:>+8.1f} {str(r.get("reason", "?"))[:8]:>8s} '
              f'{str(r.get("type", "?"))[:14]}')
    if len(l20):
        print(f'\n  Net last 20: {l20["pnl"].sum():+.1f} pts')

    return {'n': n, 'net': net, 'wr': wr_pct, 'pf': pf, 'sharpe': shp, 'sortino': sor, 'mdd': mdd}


# ======================== MAIN ========================
def main():
    args = sys.argv[1:]
    bt_start = args[0] if args else '2024-01-01'
    bt_end = args[1] if len(args) > 1 else '2026-06-30'

    print('=' * 72)
    print('  OIL COMBINED BACKTEST v29 — ML-Filtered Three Legs')
    print(f'  Period: {bt_start} → {bt_end}')
    print(f'  WR90:   WR<{LONG_ENTRY} CV>={LONG_CV} Ep>={LONG_EP_MIN} '
          f'TP={LONG_TP}/SL={LONG_SL}  ML≥{LONG_WR_ML_TH}')
    print(f'  Retrace: Dlow>{RET_DLOW} Rng>{RET_RNG} Chg<{RET_CHG} '
          f'Wick<{RET_WICK}  TP={RET_TP}/SL={RET_SL}  ML≥{RET_ML_TH}')
    print(f'  SI:     prev_chg<{SI_CHANGE_MAX} Vol>{SI_VOL_MIN} '
          f'TP={SI_TP}/SL={SI_SL}  ML≥{SI_PROB}')
    print('=' * 72)

    journal = OilSignalJournal()

    # ---- LOAD ----
    d1m = load(bt_start, bt_end)
    d15 = build_15m(d1m)
    print(f'\nData: {len(d1m):,} 1m bars → {len(d15):,} 15m bars')

    all_trades = []

    # ======================== WR90 LONG ========================
    print('\n[1] WR90 Long (relaxed + ML)...')
    in_s = d15['ins']
    o = (d15['wr'] < LONG_ENTRY) & in_s
    sigs_w = []
    ie = False
    cv = 0.0
    bc = 0
    for i in range(len(d15)):
        if o.iloc[i]:
            if not ie:
                cv = 0.0
                bc = 0
            ie = True
            cv += d15['volume'].iloc[i]
            bc += 1
        elif ie:
            ebi = i
            if ebi < len(d15) - 1 and in_s.iloc[ebi] and cv >= LONG_CV and bc >= LONG_EP_MIN:
                sigs_w.append({'idx': ebi})
            ie = False
            cv = 0.0
            bc = 0

    res_w = train_ml(d15, sigs_w, LONG_TP, LONG_SL, WR_FEATS, 'wr90', LONG_WR_ML_TH)
    if res_w:
        pnls_w, tr_w, probas_w = res_w
        wr_idx = [i for i in range(len(pnls_w)) if probas_w[i] >= LONG_WR_ML_TH]
        w_pnls = [pnls_w[i] for i in wr_idx]
        w_tr = [tr_w[i] for i in wr_idx]
        print(f'  Unfiltered: {len(pnls_w)}t  PnL={sum(pnls_w):+.0f}  '
              f'WR={sum(1 for x in pnls_w if x > 0) / len(pnls_w) * 100:.1f}%')
        w_pnl_count = len(w_pnls)
        w_pnl_wr = sum(1 for x in w_pnls if x > 0) / w_pnl_count * 100 if w_pnl_count else 0
        print(f'  ML≥{LONG_WR_ML_TH:.2f}: {w_pnl_count}t  PnL={sum(w_pnls):+.0f}  '
              f'WR={w_pnl_wr:.1f}%')
        all_trades.extend(w_tr)

        # Save features
        wr_bars = [d15.index[sigs_w[si]['idx']] for si in wr_idx]
        for w_bar, w_rec in zip(wr_bars, w_tr):
            try:
                row = d15.loc[w_bar]
                feats = {f: float(row.get(f, 0)) for f in WR_FEATS}
                feats['entry_price'] = w_rec.get('entry')
                feats['pnl'] = w_rec.get('pnl')
                journal.record_bar_feature(
                    str(w_bar),
                    json.dumps(feats, default=str),
                )
            except Exception:
                pass

    # ======================== OIL RETRACE ========================
    print('\n[2] Oil Retrace (ML-filtered)...')
    mask = ((d15['cad'] > RET_DLOW) & (d15['avg_r3'] > RET_RNG) &
            (d15['bc'] < RET_CHG) & (d15['wb'] < RET_WICK) & d15['ins'])
    sigs_r = [{'idx': i} for i in range(len(d15)) if mask.iloc[i]]

    res_r = train_ml(d15, sigs_r, RET_TP, RET_SL, RET_FEATS, 'ret', RET_ML_TH)
    if res_r:
        pnls_r, tr_r, probas_r = res_r
        r_idx = [i for i in range(len(pnls_r)) if probas_r[i] >= RET_ML_TH]
        r_pnls = [pnls_r[i] for i in r_idx]
        r_tr = [tr_r[i] for i in r_idx]
        print(f'  Unfiltered: {len(pnls_r)}t  PnL={sum(pnls_r):+.0f}  '
              f'WR={sum(1 for x in pnls_r if x > 0) / len(pnls_r) * 100:.1f}%')
        r_pnl_count = len(r_pnls)
        r_pnl_wr = sum(1 for x in r_pnls if x > 0) / r_pnl_count * 100 if r_pnl_count else 0
        print(f'  ML≥{RET_ML_TH:.2f}: {r_pnl_count}t  PnL={sum(r_pnls):+.0f}  '
              f'WR={r_pnl_wr:.1f}%')
        all_trades.extend(r_tr)

    # ======================== SHORT IMPULSE ========================
    print('\n[3] Short Impulse...')
    d1m_s = compute_si_features(d1m)
    si_mask = ((d1m_s['prev_change'] < SI_CHANGE_MAX) &
               (d1m_s['prev2_change'] < 10.0) &
               (d1m_s['prev2_change'] > -14.0) &
               (d1m_s['prev_lower_wick'] < 35.0) &
               (d1m_s['prev_volume'] > SI_VOL_MIN) &
               d1m_s['ny_hour'] &
               (d1m_s['up_count3_15min'] != -3) &
               (d1m_s['dist_day_high'] < 180.0))
    si_sigs = sorted(d1m_s.index[si_mask].tolist())

    si_recs = []
    in_si = False
    si_ex = -1
    for sig in si_sigs:
        ei = d1m_s.index.get_loc(sig)
        if ei + SI_MAX_B >= len(d1m_s):
            continue
        if in_si and ei <= si_ex:
            continue
        ep = d1m_s.iloc[ei]['close_bid']
        ex_price, bars, reason = sim_si_fixed(ei, ep, d1m_s)
        si_recs.append({
            'entry_idx': sig,
            'pnl': ep - ex_price,
            'reason': reason,
            'exit_ts': d1m_s.index[ei + bars],
        })
        in_si = True
        si_ex = ei + bars

    # XGBoost filter
    ds = pd.DatetimeIndex([r['entry_idx'] for r in si_recs])
    ms = sorted(set(d.to_period('M') for d in ds))
    tsp = pd.Period('2024-07', freq='M')
    sp = np.zeros(len(si_recs))
    for tm in [m for m in ms if m >= tsp]:
        tr_m = [m for m in ms if m < tm]
        tst = np.array([d.to_period('M') == tm for d in ds])
        trn = np.array([d.to_period('M') in tr_m for d in ds])
        X = np.array([[float(d1m_s.loc[r['entry_idx']].get(f, 0)) for f in SI_FEATS]
                      for r in si_recs])
        y = np.array([1.0 if r['pnl'] > 0 else 0.0 for r in si_recs])
        X_tr, y_tr = X[trn], y[trn]
        X_te = X[tst]
        if len(X_tr) < 20 or len(X_te) < 3:
            continue
        w = np.where(y_tr == 1)[0]
        l = np.where(y_tr == 0)[0]
        nm = min(len(w), len(l))
        if nm < 5:
            continue
        rng = np.random.RandomState(42 + tm.ordinal)
        bal = np.concatenate([rng.choice(w, nm, 0), rng.choice(l, nm, 0)])
        Xb, yb = X_tr[bal], y_tr[bal]
        spw_val = len(l) / max(1, len(w))
        mx = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05,
                                subsample=0.8, scale_pos_weight=spw_val,
                                random_state=42, verbosity=0)
        mx.fit(Xb, yb)
        prib = mx.predict_proba(X_te)[:, 1]
        for j, idx in enumerate(np.where(tst)[0]):
            sp[idx] = prib[j]

    si_p = [r['pnl'] for i, r in enumerate(si_recs) if sp[i] >= SI_PROB]
    si_tr = []
    for i, r in enumerate(si_recs):
        if sp[i] >= SI_PROB:
            sig = r['entry_idx']
            ei = d1m_s.index.get_loc(sig)
            ep_val = d1m_s.iloc[ei]['close_bid']
            ex_ts = r['exit_ts']
            ex_i = d1m_s.index.get_loc(ex_ts)
            ex_val = d1m_s.iloc[ex_i]['close_ask']
            si_tr.append({'entry': sig, 'exit': ex_ts,
                          'pnl': r['pnl'], 'reason': r['reason'],
                          'type': 'short_impulse', 'side': -1,
                          'entry_price': ep_val, 'exit_price': ex_val})
    print(f'  Raw signals: {len(si_sigs)} → records: {len(si_recs)} → '
          f'ML≥{SI_PROB:.2f}: {len(si_p)}t  PnL={sum(si_p):+.0f}  '
          f'WR={sum(1 for x in si_p if x > 0) / len(si_p) * 100:.1f}%') if si_p else print('SI (all): 0 trades')
    all_trades.extend(si_tr)

    # ======================== COMBINED ========================
    if not all_trades:
        print("\nNo trades in period.")
        return

    tdf = pd.DataFrame(all_trades)
    tdf['pnl'] = tdf['pnl'].astype(float)
    tdf['entry'] = pd.to_datetime(tdf['entry'], utc=True)
    csv_path = PROJECT_ROOT / 'runtime' / 'oil_combined_backtest_trades.csv'
    tdf.to_csv(csv_path, index=False)

    stats = full_stats(all_trades, 'OIL COMBINED v29 — Full Backtest')
    print(f'\n  CSV: {csv_path}')
    print(f'DONE.')


if __name__ == '__main__':
    main()
