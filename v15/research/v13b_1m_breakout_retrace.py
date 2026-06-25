#!/usr/bin/env python3
"""
v13b 1-Min Breakout Retrace FVG — all on 1m bars
==================================================
Same 3-step logic as v13 but entirely on 1-min bars:
  1. BREAKOUT: 1m bar closes beyond prior N-bar range
  2. RETRACE to FVG: within next M bars, price reverses toward unfilled FVG
  3. CONFIRM: 1m reverse engulf candle at FVG boundary = entry

Also tests simpler 2-step variant (skip engulf, just retrace to FVG).
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from itertools import product
from data.data_loader import DataLoader


def load_data(start_date="2025-01-01", end_date="2026-06-09"):
    loader = DataLoader()
    raw = loader.load_data(table_name="gold_prices", start_date=start_date, end_date=end_date)
    raw.index = pd.to_datetime(raw['timestamp'], unit='ms')
    df = pd.DataFrame(index=raw.index)
    df['open_ask'] = raw['openPrice_ask'].astype(float)
    df['high_bid'] = raw['highPrice_bid'].astype(float)
    df['low_bid'] = raw['lowPrice_bid'].astype(float)
    df['high_ask'] = raw['highPrice_ask'].astype(float)
    df['low_ask'] = raw['lowPrice_ask'].astype(float)
    df['close_ask'] = raw['closePrice_ask'].astype(float)
    df['close_bid'] = raw['closePrice_bid'].astype(float)
    df['close'] = df['close_ask']; df['open'] = df['open_ask']
    df['volume'] = raw['lastTradedVolume'].astype(float)
    df['spread'] = df['close_ask'] - df['close_bid']
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def compute_1m_indicators(df):
    df['atr14'] = (df['high_ask'] - df['low_ask']).rolling(14).mean()
    df['ema50_slope'] = df['close'].ewm(50).mean().diff(10)
    # FVG on 1m: gap 2 bars apart
    df['fvg_bull'] = df['low_ask'] > df['high_ask'].shift(2)
    df['fvg_bear'] = df['high_ask'] < df['low_ask'].shift(2)
    df['fvg_bull_top'] = df['low_ask']
    df['fvg_bull_bot'] = df['high_ask'].shift(2)
    df['fvg_bear_top'] = df['low_ask'].shift(2)
    df['fvg_bear_bot'] = df['high_ask']
    # Low-vol regime
    low_vol = df['atr14'] < 3.5
    weak_trend = df['ema50_slope'].abs() < 0.15
    df['in_regime'] = (low_vol & weak_trend).fillna(False)
    return df


def gen_1m_signals(df, range_lookback=15, fvg_max_bars=10, require_engulf=True):
    """Generate signals on 1-min bars.

    range_lookback: bars for prior range (15 1m bars ≈ 15 min)
    fvg_max_bars: bars after breakout to look for retrace+FVG
    """
    sig = pd.DataFrame(index=df.index)
    sig['long'] = False
    sig['short'] = False

    range_high = df['high_ask'].shift(1).rolling(range_lookback).max()
    range_low = df['low_ask'].shift(1).rolling(range_lookback).min()

    for i in range(range_lookback + 3, len(df) - fvg_max_bars):
        if not df['in_regime'].iloc[i]:
            continue

        bar = df.iloc[i]
        broke_up = bar['close'] > range_high.iloc[i] and bar['close'] > bar['open']
        broke_down = bar['close'] < range_low.iloc[i] and bar['close'] < bar['open']
        if not broke_up and not broke_down:
            continue

        # Search next bars for retrace into FVG
        for j in range(i+1, min(i+1+fvg_max_bars, len(df))):
            rbar = df.iloc[j]

            if broke_up:
                # Bull trap: find bear FVG with retrace back down
                if (df['fvg_bear'].iloc[j] and
                    rbar['close'] < bar['close'] and
                    rbar['close'] < range_high.iloc[i]):
                    fvg_low = df['fvg_bear_bot'].iloc[j]
                    fvg_high = df['fvg_bear_top'].iloc[j]

                    if require_engulf:
                        # Find engulf within FVG zone (next 2 bars)
                        for k in range(j, min(j+3, len(df)-1)):
                            prev = df.iloc[k]; curr = df.iloc[k+1]
                            price = curr['close_bid']
                            if price < fvg_low or price > fvg_high:
                                continue
                            if (prev['close_bid'] > prev['open_ask'] and
                                curr['close_bid'] < curr['open_ask'] and
                                curr['close_bid'] <= prev['open_ask']):
                                sig.loc[df.index[k+1], 'short'] = True
                                break
                    else:
                        # Entry on next bar after retrace bar
                        if j+1 < len(df):
                            sig.loc[df.index[j+1], 'short'] = True
                    break

            else:  # broke_down, bear trap → long
                if (df['fvg_bull'].iloc[j] and
                    rbar['close'] > bar['close'] and
                    rbar['close'] > range_low.iloc[i]):
                    fvg_low = df['fvg_bull_bot'].iloc[j]
                    fvg_high = df['fvg_bull_top'].iloc[j]

                    if require_engulf:
                        for k in range(j, min(j+3, len(df)-1)):
                            prev = df.iloc[k]; curr = df.iloc[k+1]
                            price = curr['close_ask']
                            if price < fvg_low or price > fvg_high:
                                continue
                            if (prev['close_ask'] < prev['open_ask'] and
                                curr['close_ask'] > curr['open_ask'] and
                                curr['close_ask'] >= prev['open_ask']):
                                sig.loc[df.index[k+1], 'long'] = True
                                break
                    else:
                        if j+1 < len(df):
                            sig.loc[df.index[j+1], 'long'] = True
                    break

    return sig


# Vectorized exit
MAX_BARS = 60

def _build_fwd(df, max_bars=60):
    n = len(df); N = max_bars
    def _s(col):
        out = np.full((n, N), np.nan, dtype=np.float64)
        vals = df[col].values
        for k in range(N): shift = k+1; out[:n-shift, k] = vals[shift:]
        return out
    return {'fh_bid': _s('high_bid'), 'fl_bid': _s('low_bid'),
            'fh_ask': _s('high_ask'), 'fl_ask': _s('low_ask'),
            'fc_bid': _s('close_bid'), 'fc_ask': _s('close_ask')}

def v_long(ei, ep, fwd, tp, sl):
    n = len(ei); pnls = np.full(n, np.nan)
    if n == 0: return pnls
    st, tg = ep - sl, ep + tp
    fl = fwd['fl_bid'][ei]; fh = fwd['fh_bid'][ei]; fc = fwd['fc_bid'][ei]
    sh = fl <= st[:, None]; th = fh >= tg[:, None]; N = fl.shape[1]
    sb = np.argmax(sh, axis=1); tb = np.argmax(th, axis=1)
    sb[~sh.any(axis=1)] = N; tb[~th.any(axis=1)] = N
    for i in range(n):
        if sb[i] == N and tb[i] == N: lc = fc[i, -1]; pnls[i] = (lc-ep[i]) if not np.isnan(lc) else 0.0
        elif sb[i] <= tb[i]: pnls[i] = st[i] - ep[i]
        else: pnls[i] = tg[i] - ep[i]
    return pnls

def v_short(ei, ep, fwd, tp, sl):
    n = len(ei); pnls = np.full(n, np.nan)
    if n == 0: return pnls
    st, tg = ep + sl, ep - tp
    fh = fwd['fh_ask'][ei]; fl = fwd['fl_ask'][ei]; fc = fwd['fc_ask'][ei]
    sh = fh >= st[:, None]; th = fl <= tg[:, None]; N = fl.shape[1]
    sb = np.argmax(sh, axis=1); tb = np.argmax(th, axis=1)
    sb[~sh.any(axis=1)] = N; tb[~th.any(axis=1)] = N
    for i in range(n):
        if sb[i] == N and tb[i] == N: lc = fc[i, -1]; pnls[i] = (ep[i]-lc) if not np.isnan(lc) else 0.0
        elif sb[i] <= tb[i]: pnls[i] = ep[i] - st[i]
        else: pnls[i] = ep[i] - tg[i]
    return pnls


def main():
    print("=" * 72)
    print("v13b 1-Min Breakout Retrace FVG")
    print("=" * 72)

    print("\n[1/3] Loading...")
    df = load_data()
    df = compute_1m_indicators(df)
    df = df.dropna(subset=['atr14', 'ema50_slope'])
    print(f"  {len(df):,} bars, {df['in_regime'].sum():,} in-regime")

    print("\n[2/3] Signal sweep...")
    configs = [
        {'lookback': 10, 'fvg_bars': 5,  'engulf': True,  'label': 'LB=10 FVG=5 +engulf'},
        {'lookback': 10, 'fvg_bars': 5,  'engulf': False, 'label': 'LB=10 FVG=5 no_engulf'},
        {'lookback': 10, 'fvg_bars': 10, 'engulf': True,  'label': 'LB=10 FVG=10 +engulf'},
        {'lookback': 10, 'fvg_bars': 10, 'engulf': False, 'label': 'LB=10 FVG=10 no_engulf'},
        {'lookback': 15, 'fvg_bars': 5,  'engulf': True,  'label': 'LB=15 FVG=5 +engulf'},
        {'lookback': 15, 'fvg_bars': 5,  'engulf': False, 'label': 'LB=15 FVG=5 no_engulf'},
        {'lookback': 15, 'fvg_bars': 10, 'engulf': True,  'label': 'LB=15 FVG=10 +engulf'},
        {'lookback': 15, 'fvg_bars': 10, 'engulf': False, 'label': 'LB=15 FVG=10 no_engulf'},
        {'lookback': 20, 'fvg_bars': 10, 'engulf': False, 'label': 'LB=20 FVG=10 no_engulf'},
    ]

    fwd = _build_fwd(df, MAX_BARS)
    tp_grid = [5, 8, 10, 12, 15, 20]
    sl_grid = [5, 8, 10, 12, 15, 20]

    best_overall = {'pnl': -999, 'label': '', 'tp': 0, 'sl': 0, 'trades': 0}

    for cfg in configs:
        sig = gen_1m_signals(df, cfg['lookback'], cfg['fvg_bars'], cfg['engulf'])
        nl = sig['long'].sum(); ns = sig['short'].sum()
        total_sig = nl + ns
        if total_sig == 0:
            print(f"  {cfg['label']:<30s}: ZERO signals")
            continue

        # Quick TP/SL sweep
        best_pnl = -999; best_tp = 0; best_sl = 0; best_trades = 0
        for tp, sl in product(tp_grid, sl_grid):
            pnls = []
            if nl:
                li = np.array([df.index.get_loc(i) for i in sig.index[sig['long']]], dtype=np.int64)
                lp = df['close_ask'].values[li]
                lr = v_long(li, lp, fwd, tp, sl); lr = lr[~np.isnan(lr)]; pnls.extend(lr.tolist())
            if ns:
                si = np.array([df.index.get_loc(i) for i in sig.index[sig['short']]], dtype=np.int64)
                sp = df['close_bid'].values[si]
                sr = v_short(si, sp, fwd, tp, sl); sr = sr[~np.isnan(sr)]; pnls.extend(sr.tolist())
            if len(pnls) < 3: continue
            a = np.array(pnls); total = a.sum()
            if total > best_pnl:
                best_pnl = total; best_tp = tp; best_sl = sl; best_trades = len(a)

        if best_trades > 0:
            wr = (np.array(pnls) > 0).mean() * 100 if 'pnls' in dir() else 0
            mj_mask = (sig.index >= '2026-05-01') & (sig.index <= '2026-06-09')
            mj_sig = sig.loc[mj_mask]
            print(f"  {cfg['label']:<30s}: {total_sig:>4d}sig {nl:>3d}L/{ns:>3d}S "
                  f"best TP={best_tp}/{best_sl} → {best_trades}t, {best_pnl:+.1f}pt, "
                  f"MJ:{mj_sig['long'].sum()+mj_sig['short'].sum()}sig")
            if best_pnl > best_overall['pnl']:
                best_overall = {'pnl': best_pnl, 'label': cfg['label'],
                               'tp': best_tp, 'sl': best_sl, 'trades': best_trades,
                               'sig': sig, 'nl': nl, 'ns': ns}

    if best_overall['trades'] == 0:
        print("\nZERO across all configs. Exiting.")
        return

    print(f"\n[3/3] Best: {best_overall['label']}")
    sig = best_overall['sig']
    tp = best_overall['tp']; sl = best_overall['sl']

    # Full TP/SL table for best config
    print(f"\n  Full TP/SL for best config (TP={tp}, SL={sl}):")
    nl = best_overall['nl']; ns = best_overall['ns']
    for tp2, sl2 in product(tp_grid, sl_grid):
        pnls = []
        if nl:
            li = np.array([df.index.get_loc(i) for i in sig.index[sig['long']]], dtype=np.int64)
            lp = df['close_ask'].values[li]
            lr = v_long(li, lp, fwd, tp2, sl2); lr = lr[~np.isnan(lr)]; pnls.extend(lr.tolist())
        if ns:
            si = np.array([df.index.get_loc(i) for i in sig.index[sig['short']]], dtype=np.int64)
            sp = df['close_bid'].values[si]
            sr = v_short(si, sp, fwd, tp2, sl2); sr = sr[~np.isnan(sr)]; pnls.extend(sr.tolist())
        if len(pnls) < 3: continue
        a = np.array(pnls); n = len(a); total = a.sum(); wr = (a>0).mean()*100
        pos = a[a>0].sum(); neg = abs(a[a<0].sum()); pf = pos/neg if neg>0 else 99
        print(f"    TP={tp2:>4.0f} SL={sl2:>4.0f}  {n:>4d}t  {total:>+8.1f}pt  {wr:>5.1f}% WR  PF={pf:.2f}  avg={total/n:>+.2f}")

    # May-June
    mj_mask = (df.index >= '2026-05-01') & (df.index <= '2026-06-09')
    mj_sig = sig.loc[mj_mask]
    mj_nl = mj_sig['long'].sum(); mj_ns = mj_sig['short'].sum()
    if mj_nl + mj_ns > 0:
        pnls = []
        if mj_nl:
            li = np.array([df.index.get_loc(i) for i in mj_sig.index[mj_sig['long']]], dtype=np.int64)
            lp = df['close_ask'].values[li]; lr = v_long(li, lp, fwd, tp, sl); pnls.extend(lr[~np.isnan(lr)].tolist())
        if mj_ns:
            si = np.array([df.index.get_loc(i) for i in mj_sig.index[mj_sig['short']]], dtype=np.int64)
            sp = df['close_bid'].values[si]; sr = v_short(si, sp, fwd, tp, sl); pnls.extend(sr[~np.isnan(sr)].tolist())
        mj_a = np.array(pnls) if pnls else np.array([])
        print(f"\n  May-June 2026: {len(mj_a)} trades, {mj_a.sum():+.1f}pt, {(mj_a>0).mean()*100:.1f}% WR" if len(mj_a) else "\n  May-June 2026: signals but no valid exits")
    else:
        print(f"\n  May-June 2026: ZERO signals")

    print(f"\nDONE.")


if __name__ == '__main__':
    main()
