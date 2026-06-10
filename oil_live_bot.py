#!/usr/bin/env python3
"""Oil Live Trading Bot — WR90 Long + Short Impulse with XGBoost.
=================================================================
Runs on-demand (cron/call): loads recent data, scans for signals,
prints trade instructions if signals detected with probability filter.

Usage: python3 oil_live_bot.py
"""

import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np; import pandas as pd; import xgboost as xgb
from data.data_loader import DataLoader
from datetime import datetime, timezone, timedelta
import json; import os; import warnings
warnings.filterwarnings('ignore')

# ======================== CONFIG ========================

# WR90 Long
NY_S, NY_E = 3, 12
NY_FC_H, NY_FC_M = 14, 28
LONG_MAX_B, LONG_EP_MIN = 60, 3
LONG_ENTRY, LONG_CV = -80, 15000
LONG_RECOVERY, LONG_WEAK, LONG_WT = -20, -50, 12
LONG_TP, LONG_SL = 80, 30

# Short Impulse
SI_CHANGE_MAX, SI_VOL_MIN = -14.0, 800
SI_UK_HOURS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SI_TP, SI_SL, SI_MAX_B = 90, 60, 60
SI_PROB = 0.55

# Data window: last N days for feature computation
LOOKBACK_DAYS = 90

# State file to track open positions
STATE_FILE = 'runtime/oil_bot_state.json'

# ======================== DATA LOADING ========================

def load_data():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    loader = DataLoader()
    raw = loader.load_data(table_name='prices',
                           start_date=start.strftime('%Y-%m-%d'),
                           end_date=end.strftime('%Y-%m-%d'))
    raw.index = pd.to_datetime(raw['timestamp'], unit='ms')
    df = pd.DataFrame(index=raw.index)
    cols = [('open', 'openPrice_ask'), ('high', 'highPrice_ask'),
            ('low', 'lowPrice_ask'), ('close_ask', 'closePrice_ask'),
            ('close_bid', 'closePrice_bid'), ('volume', 'lastTradedVolume')]
    for c, src in cols: df[c] = raw[src].astype(float)
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    return df

# ======================== 15m BUILD ========================

def build_15m(df1m):
    d = df1m.resample('15min', label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min',
         'close_ask': 'last', 'close_bid': 'last', 'volume': 'sum'}).dropna()
    n = 14; hh = d['high'].rolling(n).max(); ll = d['low'].rolling(n).min()
    d['wr'] = ((hh - d['close_ask']) / (hh - ll + 0.01)) * -100
    ny = d.index.tz_convert('America/New_York')
    d['ny_h'], d['ny_m'] = ny.hour, ny.minute
    d['in_sess'] = (d['ny_h'] >= NY_S) & (d['ny_h'] <= NY_E)
    return d

# ======================== WR90 SIGNAL DETECTION ========================

def detect_wr90_signal(d15):
    """Check the most recent completed 15m bar for WR90 long signal."""
    if len(d15) < 15: return None
    # Check if a WR episode just ended
    in_s = d15['in_sess']; o = (d15['wr'] < LONG_ENTRY) & in_s
    # Find last episode
    cv, bc = 0.0, 0; in_ep = False
    for i in range(len(d15)):
        if o.iloc[i]:
            if not in_ep: cv, bc = 0.0, 0
            in_ep = True; cv += d15['volume'].iloc[i]; bc += 1
        else:
            if in_ep:
                ebi = i
                if ebi == len(d15) - 1 and in_s.iloc[ebi] and cv >= LONG_CV and bc >= LONG_EP_MIN:
                    bar = d15.iloc[ebi]
                    return {
                        'type': 'WR90_LONG', 'entry_price': float(bar['close_ask']),
                        'bar_time': str(d15.index[ebi]),
                        'cum_vol': float(cv), 'bars': bc,
                        'wr': float(bar['wr']), 'ny_hour': int(bar['ny_h']),
                        'tp': LONG_TP, 'sl': LONG_SL
                    }
                in_ep = False; cv, bc = 0.0, 0
    return None

# ======================== SHORT IMPULSE + XGBOOST ========================

SI_XGB_FEATURES = [
    'prev_change', 'prev2_change', 'prev_lower_wick', 'prev_upper_wick',
    'prev_volume', 'prev_range', 'prev_spread', 'ATR', 'ATR_ratio',
    'ret_1m', 'ret_3m', 'ret_5m', 'vol_ratio_20',
    'up_count3_15min', 'ret_3_15m', 'ret_5_15m', 'dist_day_high'
]

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
    # 15m context
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

def check_si_signal(df):
    """Check if latest bar is a short impulse signal."""
    if len(df) < 30: return None
    recent = df.iloc[-1]
    required = SI_XGB_FEATURES + ['uk_hour']
    if any(pd.isna(recent.get(c)) for c in required): return None
    if not (recent['prev_change'] < SI_CHANGE_MAX and recent['prev2_change'] < 10.0
            and recent['prev2_change'] > -14.0 and recent['prev_lower_wick'] < 35.0
            and recent['prev_volume'] > SI_VOL_MIN and recent['uk_hour']
            and float(recent.get('up_count3_15min', 0)) != -3
            and float(recent.get('dist_day_high', 999)) < 180.0):
        return None
    feat = [float(recent.get(c, np.nan)) for c in SI_XGB_FEATURES]
    if any(np.isnan(feat)): return None
    return {
        'type': 'SHORT_IMPULSE',
        'entry_price': float(recent['close_bid']),
        'bar_time': str(df.index[-1]),
        'features': feat,
        'tp': SI_TP, 'sl': SI_SL
    }

def train_si_xgb_live(df, lookback_days=60):
    """Train XGBoost on recent historical data for prob filter."""
    from datetime import timedelta
    cutoff = df.index[-1] - timedelta(days=lookback_days)
    hist = df[df.index < cutoff]
    future = df[df.index >= cutoff]
    if len(hist) < 1000 or len(future) < 10: return None

    # Generate signals and labels from historical data
    mask = ((hist['prev_change'] < SI_CHANGE_MAX) & (hist['prev2_change'] < 10.0) &
            (hist['prev2_change'] > -14.0) & (hist['prev_lower_wick'] < 35.0) &
            (hist['prev_volume'] > SI_VOL_MIN) & hist['uk_hour'] &
            (hist['up_count3_15min'] != -3) & (hist['dist_day_high'] < 180.0))

    X_list, y_list = [], []
    for idx in hist.index[mask]:
        ei = hist.index.get_loc(idx)
        if ei + SI_MAX_B >= len(hist): continue
        ep = hist.iloc[ei]['close_bid']
        ex, bars, reason = None, 0, 'timeout'
        h = min(SI_MAX_B, len(hist) - ei - 1)
        for i in range(1, h + 1):
            b = hist.iloc[ei + i]
            if b['high'] >= ep + SI_SL: ex = ep + SI_SL; reason = 'sl'; break
            if b['low'] <= ep - SI_TP: ex = ep - SI_TP; reason = 'tp'; break
        else:
            ex = hist.iloc[ei + h]['close_ask']
        pnl = ep - ex
        row = hist.iloc[ei]
        feat = [float(row.get(c, np.nan)) for c in SI_XGB_FEATURES]
        if any(np.isnan(feat)): continue
        X_list.append(feat); y_list.append(1.0 if pnl > 0 else 0.0)

    if len(X_list) < 20: return None
    X, y = np.array(X_list), np.array(y_list)
    wi = np.where(y == 1)[0]; li = np.where(y == 0)[0]
    n_min = min(len(wi), len(li))
    if n_min < 5: return None
    rng = np.random.RandomState(42)
    bal = np.concatenate([rng.choice(wi, n_min, replace=False),
                           rng.choice(li, n_min, replace=False)])
    spw = len(li) / max(1, len(wi))
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                               subsample=0.8, scale_pos_weight=spw, random_state=42,
                               verbosity=0, use_label_encoder=False, eval_metric='logloss')
    model.fit(X[bal], y[bal])
    return model

# ======================== STATE MANAGEMENT ========================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'positions': []}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=2, default=str)

# ======================== MAIN ========================

def main():
    now = datetime.now(timezone.utc)
    print(f"\n{'='*60}")
    print(f"  OIL LIVE BOT — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    state = load_state()
    open_pos = state.get('positions', [])
    if open_pos:
        print(f"\n  Open positions: {len(open_pos)}")
        for p in open_pos:
            print(f"    {p['type']} @ {p['entry_price']} — TP={p['tp']} SL={p['sl']}")

    # Load data
    print("\n[1] Loading data...")
    df = load_data()
    print(f"    {len(df):,} 1m bars ({df.index[0]} -> {df.index[-1]})")

    # 15m build for WR90
    d15 = build_15m(df)

    # Detect WR90 signal
    wr90 = detect_wr90_signal(d15)
    if wr90:
        print(f"\n[WR90] SIGNAL: WR90 Long @ {wr90['entry_price']:.1f}")
        print(f"       Bar: {wr90['bar_time']}, WR={wr90['wr']:.0f}")
        print(f"       TP={wr90['tp']}, SL={wr90['sl']}")
        print(f"       CumVol={wr90['cum_vol']:.0f}, EpBars={wr90['bars']}")
        print(f"       >>> LONG {wr90['entry_price']:.1f}, TP={wr90['entry_price']+wr90['tp']:.1f}, "
              f"SL={wr90['entry_price']-wr90['sl']:.1f}")
    else:
        print("\n[WR90] No signal")

    # Short Impulse check
    d1m_si = compute_si_features(df)
    si = check_si_signal(d1m_si)
    if si:
        # Train XGBoost and get probability
        model = train_si_xgb_live(d1m_si)
        prob = 0.5  # default neutral
        if model is not None:
            X_sig = np.array([si['features']])
            prob = float(model.predict_proba(X_sig)[0, 1])
        si['prob'] = prob
        print(f"\n[SI] SIGNAL: Short Impulse @ {si['entry_price']:.1f}")
        print(f"       Bar: {si['bar_time']}, XGB Prob: {prob:.3f}")
        if prob >= SI_PROB:
            print(f"       TP={si['tp']}, SL={si['sl']}")
            print(f"       >>> SHORT {si['entry_price']:.1f}, TP={si['entry_price']-si['tp']:.1f}, "
                  f"SL={si['entry_price']+si['sl']:.1f}")
            print(f"       Prob≥{SI_PROB} — CONFIRMED")
        else:
            print(f"       Prob<{SI_PROB} — REJECTED")
    else:
        print("\n[SI] No signal")

    # Summary
    signals_found = sum(1 for s in [wr90, si] if s is not None and
                       (s['type'] != 'SHORT_IMPULSE' or s.get('prob', 0) >= SI_PROB))
    print(f"\n  Total confirmed signals: {signals_found}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
