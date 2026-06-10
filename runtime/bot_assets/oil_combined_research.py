#!/usr/bin/env python3
"""Combined Backtest: WR90 Long + Short Impulse
=================================================
WR90 Long (15m): WR<-80, CumVol≥15k, EpBars≥3, NY 03-12, TP=80/SL=30, advance target
Short Impulse (1m): prev_change<-14, vol>800, UK 7-16, TP=90/SL=60, XGBoost prob≥0.55

Both are proven independently. This tests combined portfolio performance.
"""

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd
from data.data_loader import DataLoader
import xgboost as xgb; import warnings; warnings.filterwarnings('ignore')

# ===== WR90 Long Config =====
NY_S = 3; NY_E = 12; NY_FC_H = 14; NY_FC_M = 28
LONG_MAX_B = 60; LONG_EP_MIN = 3
LONG_ENTRY = -80; LONG_CV = 15000
LONG_RECOVERY = -20; LONG_WEAK = -50
LONG_WT = 12
LONG_TP = 80; LONG_SL = 30

# ===== Short Impulse Config =====
SI_CHANGE_MAX = -14.0; SI_VOL_MIN = 800
SI_UK_HOURS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SI_TP = 90; SI_SL = 60; SI_MAX_B = 60
SI_PROB = 0.55  # XGBoost probability threshold

def load(s='2024-01-01', e='2026-06-30'):
    loader = DataLoader(); raw = loader.load_data(table_name='prices', start_date=s, end_date=e)
    raw.index = pd.to_datetime(raw['timestamp'], unit='ms')
    df = pd.DataFrame(index=raw.index)
    for c, src in [('open', 'openPrice_ask'), ('high', 'highPrice_ask'), ('low', 'lowPrice_ask'),
                   ('close_ask', 'closePrice_ask'), ('close_bid', 'closePrice_bid'), ('volume', 'lastTradedVolume')]:
        df[c] = raw[src].astype(float)
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    return df

# ===== WR90 Long Functions (15m) =====

def build_15m(df_1m):
    d = df_1m.resample('15min', label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close_ask': 'last', 'close_bid': 'last', 'volume': 'sum'}).dropna()
    n = 14; hh = d['high'].rolling(n).max(); ll = d['low'].rolling(n).min()
    d['wr'] = ((hh - d['close_ask']) / (hh - ll + 0.01)) * -100
    ny = d.index.tz_convert('America/New_York'); d['ny_h'] = ny.hour; d['ny_m'] = ny.minute
    d['in_sess'] = (d['ny_h'] >= NY_S) & (d['ny_h'] <= NY_E)
    return d

def find_long_signals(d):
    in_s = d['in_sess']; o = (d['wr'] < LONG_ENTRY) & in_s
    sigs = []; ie = False; cv = 0.0; bc = 0
    for i in range(len(d)):
        if o.iloc[i]:
            if not ie: ep_s = i; cv = 0.0; bc = 0
            ie = True; cv += d['volume'].iloc[i]; bc += 1
        else:
            if ie:
                ebi = i
                if ebi < len(d) - 1 and in_s.iloc[ebi] and cv >= LONG_CV and bc >= LONG_EP_MIN:
                    sigs.append({'idx': ebi, 'cv': cv, 'bc': bc})
                ie = False; cv = 0.0; bc = 0
    return sigs

def sim_long_with_advance(d15, sigs):
    pnls = []; results = []
    in_trade = False; ct = 0; cs = 0; ep = 0; ei = 0; bh = 0
    reached = False; wc = 0; sig_idx = 0
    while sig_idx < len(sigs):
        si = sigs[sig_idx]['idx']
        if not in_trade:
            in_trade = True; ep = d15.iloc[si]['close_ask']; ct = ep + LONG_TP; cs = ep - LONG_SL
            ei = si; bh = 0; reached = False; wc = 0
            results.append({'entry_time': d15.index[si], 'entry_price': ep, 'direction': 'LONG', 'signal_type': 'WR90'})
            sig_idx += 1; continue
        if si - ei > LONG_MAX_B:
            px = d15.iloc[ei + LONG_MAX_B]['close_bid']; pnls.append(px - ep)
            results[-1]['exit_time'] = d15.index[ei + LONG_MAX_B]; results[-1]['exit_price'] = px
            results[-1]['pnl'] = px - ep; results[-1]['reason'] = 'timeout'
            in_trade = False; continue
        exit_at_si = False
        for j in range(ei + bh + 1, si + 1):
            b = d15.iloc[j]
            post = (b['ny_h'] > NY_FC_H) or (b['ny_h'] == NY_FC_H and b['ny_m'] >= NY_FC_M)
            if post:
                px = b['close_bid']; pnls.append(px - ep)
                in_trade = False
                results[-1]['exit_time'] = d15.index[j]; results[-1]['exit_price'] = px
                results[-1]['pnl'] = px - ep; results[-1]['reason'] = 'ny_close'
                if j == si: exit_at_si = True
                break
            if b['high'] >= ct:
                pnls.append(LONG_TP); in_trade = False
                results[-1]['exit_time'] = d15.index[j]; results[-1]['exit_price'] = b['close_bid']
                results[-1]['pnl'] = LONG_TP; results[-1]['reason'] = 'tp'
                if j == si: exit_at_si = True
                break
            if b['low'] <= cs:
                pnls.append(-LONG_SL); in_trade = False
                results[-1]['exit_time'] = d15.index[j]; results[-1]['exit_price'] = b['close_bid']
                results[-1]['pnl'] = -LONG_SL; results[-1]['reason'] = 'sl'
                if j == si: exit_at_si = True
                break
            if b['wr'] >= LONG_RECOVERY: reached = True
            if b['wr'] < LONG_WEAK: wc += 1
            else: wc = 0
            if reached and post:
                px = b['close_bid']; pnls.append(px - ep); in_trade = False
                results[-1]['exit_time'] = d15.index[j]; results[-1]['exit_price'] = px
                results[-1]['pnl'] = px - ep; results[-1]['reason'] = 'ride_end'
                if j == si: exit_at_si = True
                break
            if not reached and wc >= LONG_WT:
                px = b['close_bid']; pnls.append(px - ep); in_trade = False
                results[-1]['exit_time'] = d15.index[j]; results[-1]['exit_price'] = px
                results[-1]['pnl'] = px - ep; results[-1]['reason'] = 'weak'
                if j == si: exit_at_si = True
                break
        bh = si - ei
        if not in_trade:
            if exit_at_si:
                # Exit and signal triggered at the exact same bar — skip this double-trade signal.
                sig_idx += 1; continue
            # Exit happened on an earlier bar (j < si). Enter new trade at the next signal (si).
            in_trade = True; ep = d15.iloc[si]['close_ask']; ct = ep + LONG_TP; cs = ep - LONG_SL
            ei = si; bh = 0; reached = False; wc = 0
            results.append({'entry_time': d15.index[si], 'entry_price': ep, 'direction': 'LONG', 'signal_type': 'WR90'})
            sig_idx += 1; continue
        ne = d15.iloc[si]['close_ask']; ct = ne + LONG_TP; cs = min(cs, ne - LONG_SL)
        ei = si; bh = 0; reached = False; wc = 0; sig_idx += 1
    if in_trade:
        last = min(ei + LONG_MAX_B, len(d15) - 1)
        px = d15.iloc[last]['close_bid']; pnls.append(px - ep)
        results[-1]['exit_time'] = d15.index[last]; results[-1]['exit_price'] = px
        results[-1]['pnl'] = px - ep; results[-1]['reason'] = 'timeout'
    return pnls, results

# ===== Short Impulse Functions (1m) =====

def compute_si_features(df):
    df['change'] = df['close_ask'] - df['open']
    df['prev_change'] = df['change'].shift(1)
    df['prev2_change'] = df['change'].shift(2)
    df['prev_lower_wick'] = df['close_ask'].shift(1) - df['low'].shift(1)
    df['prev_upper_wick'] = df['high'].shift(1) - df['close_ask'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_range'] = df['high'].shift(1) - df['low'].shift(1)
    df['prev_spread'] = df['close_ask'].shift(1) - df['close_bid'].shift(1)
    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close_ask'].shift()),
                    abs(df['low'] - df['close_ask'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['ATR_ratio'] = df['prev_range'] / (df['ATR'] + 0.01)
    df['uk_hour'] = df.index.hour.isin(SI_UK_HOURS)
    df['vol_ma_20'] = df['volume'].rolling(20, min_periods=5).mean()
    df['vol_ratio_20'] = df['prev_volume'] / (df['vol_ma_20'] + 0.01)
    df['ret_1m'] = df['close_ask'].pct_change()
    df['ret_3m'] = df['ret_1m'].rolling(3, min_periods=1).sum()
    df['ret_5m'] = df['ret_1m'].rolling(5, min_periods=1).sum()
    # 15m context features (v24)
    df_15 = df.resample('15min', label='right', closed='right').agg(
        {'open': 'first', 'close_ask': 'last'}).dropna()
    df_15['up'] = np.where(df_15['close_ask'] > df_15['open'], 1,
                           np.where(df_15['close_ask'] < df_15['open'], -1, 0))
    df_15['up_count3'] = df_15['up'].rolling(3, min_periods=1).sum()
    f15 = df_15[['up_count3']].reset_index()
    df_idx = df.reset_index()
    m15 = pd.merge_asof(df_idx.sort_values('timestamp'),
                         f15.rename(columns={'timestamp': 't15'}),
                         left_on='timestamp', right_on='t15', direction='backward',
                         tolerance=pd.Timedelta(minutes=15))
    m15.index = m15['timestamp']; df['up_count3_15min'] = m15['up_count3']
    df_15e = df.resample('15min', label='right', closed='right').agg(
        {'close_ask': 'last'}).dropna()
    df_15e['ret'] = df_15e['close_ask'].pct_change()
    df_15e['ret_3_15m'] = df_15e['ret'].rolling(3, min_periods=1).sum()
    df_15e['ret_5_15m'] = df_15e['ret'].rolling(5, min_periods=1).sum()
    f15e = df_15e[['ret_3_15m', 'ret_5_15m']].reset_index()
    m15e = pd.merge_asof(df_idx.sort_values('timestamp'),
                          f15e.rename(columns={'timestamp': 't15'}),
                          left_on='timestamp', right_on='t15', direction='backward',
                          tolerance=pd.Timedelta(minutes=15))
    m15e.index = m15e['timestamp']
    df['ret_3_15m'] = m15e['ret_3_15m']; df['ret_5_15m'] = m15e['ret_5_15m']
    daily_high = df['high'].resample('D').max().rename('day_high').reset_index()
    dh_m = pd.merge_asof(df_idx.sort_values('timestamp'),
                          daily_high.rename(columns={'timestamp': 'day_ts'}),
                          left_on='timestamp', right_on='day_ts', direction='backward')
    dh_m.index = dh_m['timestamp']
    df['dist_day_high'] = dh_m['day_high'] - df['close_ask']
    df['hour'] = df.index.hour.astype(float)
    return df

def find_si_signals(df):
    mask = ((df['prev_change'] < SI_CHANGE_MAX) & (df['prev2_change'] < 10.0) &
            (df['prev2_change'] > -14.0) & (df['prev_lower_wick'] < 35.0) &
            (df['prev_volume'] > SI_VOL_MIN) & df['uk_hour'] &
            (df['up_count3_15min'] != -3) & (df['dist_day_high'] < 180.0))
    return mask

def sim_si_short(ei, ep, df, tp=SI_TP, sl=SI_SL):
    stop = ep + sl; target = ep - tp
    horizon = min(SI_MAX_B, len(df) - ei - 1)
    for i in range(1, horizon + 1):
        b = df.iloc[ei + i]
        if b['high'] >= stop: return stop, i, 'sl'
        if b['low'] <= target: return target, i, 'tp'
    return df.iloc[ei + horizon]['close_ask'], horizon, 'timeout'

# ===== Simple XGBoost for SI filtering =====
SI_XGB_FEATURES = [
    'prev_change', 'prev2_change', 'prev_lower_wick', 'prev_upper_wick',
    'prev_volume', 'prev_range', 'prev_spread', 'ATR', 'ATR_ratio',
    'ret_1m', 'ret_3m', 'ret_5m', 'vol_ratio_20',
    'up_count3_15min', 'ret_3_15m', 'ret_5_15m', 'dist_day_high'
]

def train_si_xgb(df, records, test_start='2024-07'):
    """Walk-forward XGBoost for short impulse filtering."""
    X_list = []
    for r in records:
        row = df.loc[r['entry_idx']]
        feat = [float(row.get(c, np.nan)) for c in SI_XGB_FEATURES]
        X_list.append(feat)
    X = np.array(X_list)
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    y = np.array([1.0 if r['pnl'] > 0 else 0.0 for r in records])[valid]
    recs = [records[i] for i in range(len(records)) if valid[i]]
    n = len(X)
    if n < 20: return np.ones(len(records)) * 0.5  # neutral prob

    dates = pd.DatetimeIndex([r['entry_idx'] for r in recs])
    months = sorted(set(d.to_period('M') for d in dates))
    test_months = [m for m in months if m >= pd.Period(test_start, freq='M')]
    probas = np.zeros(len(records))

    for tm in test_months:
        train_m = [m for m in months if m < tm]
        test_mask = np.array([d.to_period('M') == tm for d in dates])
        train_mask = np.array([d.to_period('M') in train_m for d in dates])
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te = X[test_mask]
        if len(X_tr) < 20 or len(X_te) < 3: continue
        win_idx = np.where(y_tr == 1)[0]; lose_idx = np.where(y_tr == 0)[0]
        n_min = min(len(win_idx), len(lose_idx))
        if n_min < 5: continue
        rng = np.random.RandomState(42 + tm.ordinal)
        bal = np.concatenate([rng.choice(win_idx, n_min, replace=False),
                              rng.choice(lose_idx, n_min, replace=False)])
        Xb, yb = X_tr[bal], y_tr[bal]
        spw = len(lose_idx) / max(1, len(win_idx))
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, scale_pos_weight=spw, random_state=42,
                                   verbosity=0, use_label_encoder=False, eval_metric='logloss')
        model.fit(Xb, yb)
        probas_te = model.predict_proba(X_te)[:, 1]
        for j, idx in enumerate(np.where(test_mask)[0]):
            rec_idx = recs[idx].get('orig_idx', idx)
            if rec_idx < len(probas):
                probas[rec_idx] = probas_te[j]
    return probas

# ===== Stats =====
def stats(pnls):
    if not pnls: return {'t': 0, 'pnl': 0, 'wr': 0, 'pf': 0}
    n = len(pnls); t = sum(pnls); wr = sum(1 for p in pnls if p > 0) / n * 100
    ps = sum(p for p in pnls if p > 0); ns = abs(sum(p for p in pnls if p < 0))
    return {'t': n, 'pnl': t, 'wr': wr, 'pf': ps / ns if ns > 0 else 99}

# ===== MAIN =====
print('=' * 80)
print('  COMBINED BACKTEST: WR90 Long + Short Impulse')
print(f'  WR90: WR<{LONG_ENTRY}, CV≥{LONG_CV}, EpB≥{LONG_EP_MIN}, TP={LONG_TP}/SL={LONG_SL}, advance')
print(f'  SI:   change<{SI_CHANGE_MAX}, vol>{SI_VOL_MIN}, UK 7-16, TP={SI_TP}/SL={SI_SL}, XGB≥{SI_PROB}')
print('=' * 80)

d1m = load(); d15 = build_15m(d1m)
d1m_si = compute_si_features(d1m)

print(f'\nData: {len(d1m):,} 1m bars, {len(d15):,} 15m bars')

# ---- WR90 Long ----
long_sigs = find_long_signals(d15)
print(f'WR90 Long signals: {len(long_sigs)}')
pnls_long, long_results = sim_long_with_advance(d15, long_sigs)
sl = stats(pnls_long)
print(f'  Long: {sl["t"]}t, {sl["pnl"]:+.0f}pts, WR={sl["wr"]:.0f}%, PF={sl["pf"]:.2f}')

# ---- Short Impulse (raw) ----
si_mask = find_si_signals(d1m_si)
si_indices = d1m_si.index[si_mask].tolist()
print(f'\nShort Impulse raw signals: {len(si_indices)}')

si_records = []
for sig_idx in si_indices:
    ei = d1m_si.index.get_loc(sig_idx)
    if ei + SI_MAX_B >= len(d1m_si): continue
    ep = d1m_si.iloc[ei]['close_bid']  # short entry at bid
    ex, bars, reason = sim_si_short(ei, ep, d1m_si)
    pnl = ep - ex  # PnL = entry - exit (short)
    si_records.append({'entry_idx': sig_idx, 'entry_price': ep,
                        'exit_price': ex, 'pnl': pnl, 'reason': reason,
                        'orig_idx': len(si_records), 'bars': bars})

# ---- Short Impulse XGBoost WF ----
print('Training SI XGBoost WF filter...')
si_probas = train_si_xgb(d1m_si, si_records)

# Apply XGBoost filter
si_filtered_pnls = []
si_filtered_records = []
for i, r in enumerate(si_records):
    if si_probas[i] >= SI_PROB:
        si_filtered_pnls.append(r['pnl'])
        si_filtered_records.append(r)

ss = stats(si_filtered_pnls)
print(f'  SI XGB≥{SI_PROB}: {ss["t"]}t, {ss["pnl"]:+.0f}pts, WR={ss["wr"]:.0f}%, PF={ss["pf"]:.2f}')

# ---- Combined ----
all_pnls = pnls_long + si_filtered_pnls
sc = stats(all_pnls)
print(f'\n{"=" * 80}')
print(f'  COMBINED PORTFOLIO')
print(f'  {"=" * 50}')
print(f'  Long (WR90):       {sl["t"]:>5d}t  {sl["pnl"]:>+10.0f}pts  WR={sl["wr"]:>5.1f}%  PF={sl["pf"]:.2f}')
print(f'  Short (Impulse):   {ss["t"]:>5d}t  {ss["pnl"]:>+10.0f}pts  WR={ss["wr"]:>5.1f}%  PF={ss["pf"]:.2f}')
print(f'  {"=" * 50}')
print(f'  TOTAL:             {sc["t"]:>5d}t  {sc["pnl"]:>+10.0f}pts  WR={sc["wr"]:>5.1f}%  PF={sc["pf"]:.2f}')
print(f'  {"=" * 80}')

# ---- Monthly breakdown ----
all_trades = []
for r in long_results:
    all_trades.append({'time': r['entry_time'], 'pnl': r['pnl'], 'type': 'WR90_LONG'})
for i, pnl in enumerate(si_filtered_pnls):
    r = si_filtered_records[i]
    all_trades.append({'time': r['entry_idx'], 'pnl': pnl, 'type': 'SI_SHORT'})

all_trades.sort(key=lambda x: x['time'])
print(f'\n  Monthly Combined (HKT):')
months = {}
for t in all_trades:
    m = t['time'].tz_convert('Asia/Hong_Kong').strftime('%Y-%m')
    if m not in months: months[m] = {'pnls': [], 'long': [], 'short': []}
    months[m]['pnls'].append(t['pnl'])
    if t['type'] == 'WR90_LONG': months[m]['long'].append(t['pnl'])
    else: months[m]['short'].append(t['pnl'])

print(f'  {"Month":>8s} {"T":>4s} {"Long":>8s} {"Short":>8s} {"Comb":>8s} {"WR":>6s} {"Cum":>9s}')
cum = 0.0
for m in sorted(months.keys())[-18:]:
    d = months[m]; n = len(d['pnls']); s2 = sum(d['pnls'])
    ls = sum(d['long']) if d['long'] else 0
    ss2 = sum(d['short']) if d['short'] else 0
    wr = sum(1 for p in d['pnls'] if p > 0) / max(n, 1) * 100
    cum += s2
    print(f'  {m:>8s} {n:>4d} {ls:>+8.0f} {ss2:>+8.0f} {s2:>+8.0f} {wr:>5.0f}% {cum:>+9.0f}')

# Yearly
print(f'\n  Yearly:')
yearly = {}
for t in all_trades:
    y = t['time'].tz_convert('Asia/Hong_Kong').year
    if y not in yearly: yearly[y] = []
    yearly[y].append(t['pnl'])
for y in sorted(yearly.keys()):
    p = yearly[y]; n = len(p); yt = sum(p); wr = sum(1 for x in p if x > 0) / max(n, 1) * 100
    print(f'    {y}: {n}t  PnL={yt:+.0f}  WR={wr:.0f}%')

# ---- Full v14-style stats ----
from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl

# Build trade CSV
trade_rows = []
for r in long_results:
    dur = (r['exit_time'] - r['entry_time']).total_seconds() / 60 if 'exit_time' in r else 0
    trade_rows.append({
        'entry_time': str(r['entry_time']),
        'pnl': r['pnl'], 'side': 1, 'pattern': 'wr90_long',
        'source': 'oil', 'exit_reason': r.get('reason', 'unknown'),
        'duration_min': dur
    })
for r in si_filtered_records:
    r_time = r['entry_idx']
    dur = r.get('bars', 0)
    # Compute exit_time from entry + bars
    et = pd.to_datetime(r_time) + pd.Timedelta(minutes=int(dur)) if dur else pd.to_datetime(r_time)
    trade_rows.append({
        'entry_time': str(r_time), 'exit_time': str(et),
        'pnl': r['pnl'], 'side': -1, 'pattern': 'short_impulse',
        'source': 'oil', 'exit_reason': r.get('reason', 'unknown'),
        'duration_min': dur
    })
tdf = pd.DataFrame(trade_rows)
csv_path = 'runtime/oil_combined_backtest_trades.csv'
tdf.to_csv(csv_path, index=False)

# Full stats
import numpy as np
all_stats = rebuild_directional_pnl(csv_path).get('all', {})

# Core
n = len(tdf); wins = int((tdf['pnl'] > 0).sum()); net = float(tdf['pnl'].sum())
wr = wins / n * 100; cum = tdf['pnl'].cumsum(); max_dd = float((cum - cum.cummax()).min())
gross_win = float(all_stats.get('gross_profit', 0) or 0)
gross_loss = abs(float(all_stats.get('gross_loss', 0) or 0))
pf = (gross_win / gross_loss) if gross_loss > 0 else float('inf')

print(f'\n{"=" * 72}')
print(f'  FULL STATS — Oil Combined Backtest')
print(f'  {tdf.iloc[0]["entry_time"][:10]} → {tdf.iloc[-1]["entry_time"][:10]}')
print(f'{"=" * 72}')
print(f'  Trades       : {n}  (W:{wins}  L:{n-wins})')
print(f'  Win Rate     : {wr:.1f}%')
print(f'  Net PnL      : {net:+.1f}pts')
print(f'  Avg/Trade    : {net/n:+.2f}pts')
print(f'  Max DD       : {max_dd:+.1f}pts')
print(f'  Profit Factor: {pf:.2f}')

# By pattern
print(f'\n  By Pattern:')
for pat, grp in tdf.groupby('pattern'):
    pw = (grp['pnl'] > 0).mean() * 100; pn = len(grp); ps = grp['pnl'].sum()
    pavg = ps / pn
    print(f'    {pat:20s}: {pn:4d}t  PnL={ps:+8.1f}  WR={pw:5.1f}%  avg={pavg:+7.2f}')

# By side
print(f'\n  By Side:')
for sd, grp in tdf.groupby('side'):
    sw = (grp['pnl'] > 0).mean() * 100; sn = len(grp); ss = grp['pnl'].sum()
    label = 'LONG' if sd == 1 else 'SHORT'
    print(f'    {label:6s}: {sn:4d}t  PnL={ss:+8.1f}  WR={sw:5.1f}%  avg={ss/sn:+7.2f}')

# By exit reason
print(f'\n  Exit Breakdown:')
for reason, grp in tdf.groupby('exit_reason'):
    rw = (grp['pnl'] > 0).mean() * 100; rn = len(grp); rs = grp['pnl'].sum()
    print(f'    {str(reason):18s}: {rn:4d}t  WR={rw:5.1f}%  avg={rs/rn:+7.2f}')

# Risk-adjusted
avg_win = float(all_stats.get('avg_win', 0) or 0)
avg_loss = float(all_stats.get('avg_loss', 0) or 0)
expectancy = (wr / 100.0) * avg_win + (1.0 - wr / 100.0) * avg_loss
tpd = float(all_stats.get('avg_trades_per_day', 0) or 0)
recovery = (net / abs(max_dd)) if max_dd < 0 else float('inf')

tdf_copy = tdf.copy(); tdf_copy['entry_time'] = pd.to_datetime(tdf_copy['entry_time'], utc=True)
tdf_copy['trade_day'] = tdf_copy['entry_time'].dt.tz_convert('America/New_York').dt.floor('D')
daily_pnl = tdf_copy.groupby('trade_day')['pnl'].sum().astype(float)
mean_day = float(daily_pnl.mean()) if len(daily_pnl) else 0.0
std_day = float(daily_pnl.std(ddof=1)) if len(daily_pnl) > 1 else 0.0
downside = daily_pnl[daily_pnl < 0]
down_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
sharpe = (mean_day / std_day) * np.sqrt(252.0) if std_day > 0 else 0.0
sortino = (mean_day / down_std) * np.sqrt(252.0) if down_std > 0 else 0.0

st = all_stats.get('streaks', {}) if isinstance(all_stats, dict) else {}
print(f'\n  Risk-Adjusted:')
print(f'    Expectancy/Trade   : {expectancy:+.2f}pts')
print(f'    Expectancy/Day     : {expectancy * tpd:+.2f}pts')
print(f'    Recovery Factor    : {recovery:.3f}')
print(f'    Sharpe  (annualized): {sharpe:.2f}')
print(f'    Sortino (annualized): {sortino:.2f}')
print(f'    Max Win Streak     : {int(st.get("max_win_streak", 0))}')
print(f'    Max Loss Streak    : {int(st.get("max_loss_streak", 0))}')
print(f'    Avg Duration       : {float(all_stats.get("avg_duration_min", 0) or 0):.1f} min')

# Yearly table
print(f'\n  Yearly:')
tdf_copy['year'] = tdf_copy['entry_time'].dt.year
for y in sorted(tdf_copy['year'].unique()):
    gy = tdf_copy[tdf_copy['year'] == y]
    yn = len(gy); yt = gy['pnl'].sum(); yw = (gy['pnl'] > 0).mean() * 100
    ylong = gy[gy['side'] == 1]; yshort = gy[gy['side'] == -1]
    print(f'    {y}: {yn:4d}t  PnL={yt:+8.1f}  WR={yw:5.1f}%  '
          f'Long:{len(ylong):3d}t/{ylong["pnl"].sum():+.0f}  '
          f'Short:{len(yshort):3d}t/{yshort["pnl"].sum():+.0f}')

# Monthly table (last 18 months)
print(f'\n  Monthly (last 18):')
tdf_copy['month'] = tdf_copy['entry_time'].dt.strftime('%Y-%m')
monthly = tdf_copy.groupby('month')['pnl'].agg(['sum', 'count'])
monthly['wr'] = tdf_copy.groupby('month')['pnl'].apply(lambda x: (x > 0).mean() * 100)
ms = sorted(monthly.index)[-18:]
print(f'    {"Month":>8s}  {"T":>3s}  {"PnL":>10s}  {"WR":>6s}  {"Cum":>10s}')
cum_m = 0.0
for m in ms:
    s = monthly.loc[m]; cum_m += s['sum']
    print(f'    {m:>8s}  {int(s["count"]):>3d}  {s["sum"]:+10.1f}  {s["wr"]:>5.0f}%  {cum_m:+10.1f}')

print(f'\n  Trade CSV: {csv_path}')
print('\nDONE.')
