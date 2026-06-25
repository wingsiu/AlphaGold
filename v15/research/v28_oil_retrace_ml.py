#!/usr/bin/env python3
"""Oil Retrace (no pattern) with XGBoost ML filter."""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np; import pandas as pd; import xgboost as xgb
from data.data_loader import DataLoader
import warnings; warnings.filterwarnings('ignore')

NY_S, NY_E, NY_FC_H, NY_FC_M = 3, 12, 14, 28
LONG_MAX_B = 60
# Best entry params from sweep: Dlow>40, Rng>50, Chg<-10, Wick<16
DLOW, RNG, CHG, WICK = 40, 50, -10, 16
TP, SL = 60, 20

def load():
    loader = DataLoader()
    raw = loader.load_data('prices', '2024-01-01', '2026-06-30')
    raw.index = pd.to_datetime(raw['timestamp'], unit='ms')
    df = pd.DataFrame(index=raw.index)
    for c, src in [('open', 'openPrice_ask'), ('high', 'highPrice_ask'), ('low', 'lowPrice_ask'),
                   ('close_ask', 'closePrice_ask'), ('close_bid', 'closePrice_bid'), ('volume', 'lastTradedVolume')]:
        df[c] = raw[src].astype(float)
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    return df

def build_15m(df_1m):
    d = df_1m.resample('15min', label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close_ask': 'last', 'close_bid': 'last', 'volume': 'sum'}).dropna()
    ny = d.index.tz_convert('America/New_York')
    d['ny_h'] = ny.hour; d['ny_m'] = ny.minute
    d['in_sess'] = (d['ny_h'] >= NY_S) & (d['ny_h'] <= NY_E)
    d['Dlow'] = d['low'].groupby(ny.date).transform('min')
    d['range'] = d['high'] - d['low']
    d['avg_range3'] = d['range'].rolling(3, min_periods=3).mean()
    d['wick_below'] = np.minimum(d['open'], d['close_ask']) - d['low']
    d['bar_change'] = d['close_ask'] - d['open']
    d['close_above_dlow'] = d['close_ask'] - d['Dlow']
    d['ret_1b'] = d['close_ask'].pct_change()
    d['ret_3b'] = d['ret_1b'].rolling(3, min_periods=1).sum()
    d['ret_5b'] = d['ret_1b'].rolling(5, min_periods=1).sum()
    d['vol_ratio'] = d['volume'] / (d['volume'].rolling(20).mean() + 0.01)
    d['bar_up'] = (d['close_ask'] > d['open']).astype(int)
    d['bar_down'] = (d['close_ask'] < d['open']).astype(int)
    d['bar_dir'] = d['bar_up'] - d['bar_down']  # +1 up, -1 down, 0 flat
    d['bar_dir_prev'] = d['bar_dir'].shift(1)
    d['bar_dir_prev2'] = d['bar_dir'].shift(2)
    return d

def find_signals(d):
    in_s = d['in_sess']
    o = ((d['close_above_dlow'] > DLOW) & (d['avg_range3'] > RNG) &
         (d['bar_change'] < CHG) & (d['wick_below'] < WICK) & in_s)
    return [{'idx': i} for i in range(len(d)) if o.iloc[i]]

def sim(d, sigs):
    pnls = []; records = []
    si = 0
    while si < len(sigs):
        si_i = sigs[si]['idx']
        # New trade
        ep = d.iloc[si_i]['close_ask']; ct = ep + TP; cs = ep - SL
        ei = si_i; bh = 0
        records.append({'entry_idx': d.index[si_i], 'pnl': 0})
        si += 1
        # Run until exit
        while True:
            if si >= len(sigs):
                si_i_next = min(ei + LONG_MAX_B, len(d) - 1)
            else:
                si_i_next = sigs[si]['idx']
            end_bar = min(si_i_next, ei + LONG_MAX_B)
            ex_si = False
            for j in range(ei + bh + 1, end_bar + 1):
                b = d.iloc[j]; post = (b['ny_h'] > NY_FC_H) or (b['ny_h'] == NY_FC_H and b['ny_m'] >= NY_FC_M)
                if post:
                    pnls.append(b['close_bid'] - ep); records[-1]['pnl'] = b['close_bid'] - ep
                    if si < len(sigs) and j == sigs[si]['idx']: ex_si = True
                    break
                if b['high'] >= ct:
                    pnls.append(TP); records[-1]['pnl'] = TP
                    if si < len(sigs) and j == sigs[si]['idx']: ex_si = True
                    break
                if b['low'] <= cs:
                    pnls.append(-SL); records[-1]['pnl'] = -SL
                    if si < len(sigs) and j == sigs[si]['idx']: ex_si = True
                    break
            else:
                # No exit — advance or timeout
                bh = end_bar - ei
                if si >= len(sigs):
                    # Final
                    px = d.iloc[end_bar]['close_bid']; pnls.append(px - ep); records[-1]['pnl'] = px - ep
                    break
                if end_bar == ei + LONG_MAX_B:
                    px = d.iloc[ei + LONG_MAX_B]['close_bid']; pnls.append(px - ep); records[-1]['pnl'] = px - ep
                    break
                if ex_si:
                    si += 1
                    break
                # Advance targets
                ne = d.iloc[sigs[si]['idx']]['close_ask']
                ct = max(ct, ne + TP)
                cs = cs if cs < ne - SL else max(cs, ne - SL)
                ei = sigs[si]['idx']; bh = 0; si += 1
                continue
            # Exit happened
            if not ex_si and si < len(sigs):
                # Exit happened before this signal — start new trade
                break
            si += 1
            break
    return pnls, records

def stats(pnls):
    if not pnls: return {'t': 0, 'pnl': 0, 'wr': 0, 'pf': 0}
    n = len(pnls); t = sum(pnls); wr = sum(1 for p in pnls if p > 0) / n * 100
    ps = sum(p for p in pnls if p > 0); ns = abs(sum(p for p in pnls if p < 0))
    return {'t': n, 'pnl': t, 'wr': wr, 'pf': ps / ns if ns > 0 else 99}

FEATURES = ['close_above_dlow', 'avg_range3', 'bar_change', 'wick_below', 'range',
            'ret_1b', 'ret_3b', 'ret_5b', 'vol_ratio', 'bar_dir', 'bar_dir_prev', 'bar_dir_prev2']

print('=' * 60)
print(f'  OIL RETRACE ML Filter — Dlow>{DLOW} Rng>{RNG} Chg<{CHG} Wick<{WICK}')
print('=' * 60)
d1m = load(); d15 = build_15m(d1m)
sigs = find_signals(d15)
print(f'\nSignals: {len(sigs)}')

pnls, records = sim(d15, sigs)
sa = stats(pnls)
print(f'\nUnfiltered: {sa["t"]}t PnL={sa["pnl"]:+.0f} WR={sa["wr"]:.1f}% PF={sa["pf"]:.2f}')

# Build feature matrix: one row per TRADE
X_list = []
for r in records:
    ts = r['entry_idx']
    if ts in d15.index:
        row = d15.loc[ts]
        X_list.append([float(row.get(f, 0)) for f in FEATURES])
    else:
        X_list.append([0.0] * len(FEATURES))
X = np.array(X_list)
y = np.array([1.0 if r['pnl'] > 0 else 0.0 for r in records])

dates = pd.DatetimeIndex([r['entry_idx'] for r in records])
months = sorted(set(d.to_period('M') for d in dates))
test_start = pd.Period('2024-07', freq='M')
test_months = [m for m in months if m >= test_start]

probas = np.zeros(len(sigs))
for tm in test_months:
    train_m = [m for m in months if m < tm]
    tst = np.array([d.to_period('M') == tm for d in dates])
    trn = np.array([d.to_period('M') in train_m for d in dates])
    X_tr, y_tr = X[trn], y[trn]; X_te = X[tst]
    if len(X_tr) < 20 or len(X_te) < 3: continue
    w = np.where(y_tr == 1)[0]; l = np.where(y_tr == 0)[0]; nm = min(len(w), len(l))
    if nm < 5: continue
    rng = np.random.RandomState(42 + tm.ordinal)
    bal = np.concatenate([rng.choice(w, nm, replace=False), rng.choice(l, nm, replace=False)])
    Xb, yb = X_tr[bal], y_tr[bal]; spw = len(l) / max(1, len(w))
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
                               scale_pos_weight=spw, random_state=42, verbosity=0)
    model.fit(Xb, yb); probas_te = model.predict_proba(X_te)[:, 1]
    for j, idx in enumerate(np.where(tst)[0]): probas[idx] = probas_te[j]

print(f'\n  {"Threshold":>12s} {"T":>5s} {"PnL":>9s} {"WR":>7s} {"PF":>7s} {"Avg":>7s}')
print(f'  {"-" * 12} {"-" * 5} {"-" * 9} {"-" * 7} {"-" * 7} {"-" * 7}')
for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75]:
    fpnls = [pnls[i] for i in range(len(pnls)) if probas[i] >= thresh]
    s = stats(fpnls); avg = s['pnl'] / s['t'] if s['t'] > 0 else 0
    mark = ' *' if s['pnl'] > sa['pnl'] else ''
    print(f'  ML≥{thresh:.2f}:  {s["t"]:>5d} {s["pnl"]:>+9.0f} {s["wr"]:>6.1f}% {s["pf"]:>6.2f} {avg:>+7.1f}{mark}')

print(f'\nDONE.')
