#!/usr/bin/env python3
"""
v13 Breakout Retrace FVG Pattern — Production Research Script
===============================================================
Three-step entry logic:
  1. BREAKOUT: 15m bar closes beyond prior N-bar range (false breakout)
  2. RETRACE: Price reverses back INTO the range, toward an unfilled FVG
  3. CONFIRM: 1-min reverse engulfing candle at FVG boundary triggers entry

This captures "breakout trap" / "liquidity sweep" patterns:
  - Bull trap: breaks above range high → reverses → bear engulf on 1m at FVG → SHORT
  - Bear trap: breaks below range low → reverses → bull engulf on 1m at FVG → LONG

TP/SL sweep on fine grid. Walk-forward monthly for stability.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
from data.data_loader import DataLoader

# =============================================================================
# Data
# =============================================================================

def load_askbid_data(start_date="2025-01-01", end_date="2026-06-09"):
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
    df['close'] = df['close_ask']
    df['volume'] = raw['lastTradedVolume'].astype(float)
    df['spread'] = df['close_ask'] - df['close_bid']
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df

# =============================================================================
# 15-min resample + indicators
# =============================================================================

def resample_15m(df_1m):
    df = df_1m.resample('15min', label='left', closed='left').agg({
        'open_ask': 'first', 'high_ask': 'max', 'high_bid': 'max',
        'low_ask': 'min', 'low_bid': 'min',
        'close_ask': 'last', 'close_bid': 'last',
        'volume': 'sum', 'spread': 'mean',
    })
    df['close'] = df['close_ask']; df['open'] = df['open_ask']
    return df.dropna()

def compute_15m_indicators(df):
    df['atr14'] = (df['high_ask'] - df['low_ask']).rolling(14).mean()
    df['ema50_slope'] = df['close'].ewm(50).mean().diff(3)
    # FVG on 15m
    df['fvg_bull'] = df['low_ask'] > df['high_ask'].shift(2)
    df['fvg_bear'] = df['high_ask'] < df['low_ask'].shift(2)
    df['fvg_bull_top'] = df['low_ask']  # top of bull FVG is current low
    df['fvg_bull_bot'] = df['high_ask'].shift(2)  # bottom of bull FVG is prior high
    df['fvg_bear_top'] = df['low_ask'].shift(2)  # top of bear FVG is prior low
    df['fvg_bear_bot'] = df['high_ask']  # bottom of bear FVG is current high
    # Session flags
    df['hour_utc'] = df.index.hour
    df['is_session_open'] = df['hour_utc'].isin([7, 8, 12, 13, 16, 17])
    # Low-vol regime
    low_vol = df['atr14'] < df['atr14'].rolling(200).mean() * 0.8
    weak_trend = df['ema50_slope'].abs() < df['atr14'] * 2
    df['in_regime'] = (low_vol & weak_trend).fillna(False)
    return df

# =============================================================================
# 1-min engulfing detection
# =============================================================================

def detect_1m_engulf(df_1m, target_zone_low, target_zone_high, side='bull'):
    """Find 1-min engulfing candles within a price zone.

    target_zone_low/high: the FVG price zone
    side: 'bull' for bullish engulf (long entry), 'bear' for bearish engulf (short entry)

    Returns index of first 1m bar meeting criteria, or None.
    """
    if side == 'bull':
        # Bullish engulf: prior bar bearish (close < open), current bar bullish (close > open)
        # Current close >= prior open (engulfs), price within zone
        for i in range(1, len(df_1m)):
            idx = df_1m.index[i]
            price = df_1m.loc[idx, 'close_ask']
            if price < target_zone_low or price > target_zone_high:
                continue
            prev = df_1m.iloc[i-1]
            curr = df_1m.iloc[i]
            # Bearish prior + bullish current + engulf
            if (prev['close_ask'] < prev['open_ask'] and
                curr['close_ask'] > curr['open_ask'] and
                curr['close_ask'] >= prev['open_ask'] and
                curr['open_ask'] <= prev['close_ask']):
                return idx
        return None
    else:  # bear
        for i in range(1, len(df_1m)):
            idx = df_1m.index[i]
            price = df_1m.loc[idx, 'close_bid']
            if price < target_zone_low or price > target_zone_high:
                continue
            prev = df_1m.iloc[i-1]
            curr = df_1m.iloc[i]
            # Bullish prior + bearish current + engulf
            if (prev['close_bid'] > prev['open_ask'] and
                curr['close_bid'] < curr['open_ask'] and
                curr['close_bid'] <= prev['open_ask'] and
                curr['open_ask'] >= prev['close_bid']):
                return idx
        return None

# =============================================================================
# Three-step signal: Breakout → Retrace to FVG → 1m Engulf
# =============================================================================

def gen_breakout_retrace_fvg(df_15, df_1m, range_lookback=5, fvg_max_bars=5):
    """Generate signals using 3-step logic on 15m bars.

    Step 1: Identify 15m breakout bar (closes beyond N-bar range)
    Step 2: Within next fvg_max_bars bars, check if price retraces back toward an FVG
    Step 3: At the FVG zone, check 1m for engulfing confirmation

    Returns (signals_1m_df, signal_15m_df)
    """
    sig_15 = pd.DataFrame(index=df_15.index)
    sig_15['long_signal'] = False  # bear trap → long
    sig_15['short_signal'] = False  # bull trap → short
    sig_15['entry_bar_15'] = None   # which 15m bar the 1m entry fires on

    sig_1m = pd.DataFrame(index=df_1m.index)
    sig_1m['long'] = False
    sig_1m['short'] = False

    # Prior N-bar range
    range_high = df_15['high_ask'].shift(1).rolling(range_lookback).max()
    range_low = df_15['low_ask'].shift(1).rolling(range_lookback).min()

    for i in range(range_lookback + 1, len(df_15) - fvg_max_bars):
        bar = df_15.iloc[i]
        if not df_15['in_regime'].iloc[i]:
            continue

        # --- Step 1: Breakout ---
        broke_up = bar['close'] > range_high.iloc[i] and bar['close'] > bar['open']
        broke_down = bar['close'] < range_low.iloc[i] and bar['close'] < bar['open']

        if not broke_up and not broke_down:
            continue

        # --- Step 2: Retrace into FVG (within next fvg_max_bars) ---
        for j in range(i+1, min(i+1+fvg_max_bars, len(df_15))):
            retrace_bar = df_15.iloc[j]

            if broke_up:
                # Bull trap: broke above range → now reversing down
                # Look for bear FVG (gap below) that price is retracing into
                if (df_15['fvg_bear'].iloc[j] and
                    retrace_bar['close'] < bar['close'] and  # retracing down
                    retrace_bar['close'] < range_high.iloc[i]):  # back inside range
                    fvg_low = df_15['fvg_bear_bot'].iloc[j]
                    fvg_high = df_15['fvg_bear_top'].iloc[j]

                    # Find 1m bars within this 15m candle + next 15m
                    start_1m = df_15.index[j]
                    end_1m = start_1m + pd.Timedelta(minutes=30)
                    sub_1m = df_1m[(df_1m.index >= start_1m) & (df_1m.index < end_1m)]

                    if len(sub_1m) >= 2:
                        entry_idx = detect_1m_engulf(sub_1m, fvg_low, fvg_high, side='bear')
                        if entry_idx is not None:
                            sig_1m.loc[entry_idx, 'short'] = True
                            sig_15.iloc[i, sig_15.columns.get_loc('short_signal')] = True
                            sig_15.iloc[i, sig_15.columns.get_loc('entry_bar_15')] = df_15.index[j]
                    break  # found retrace, stop looking

            else:  # broke_down
                # Bear trap: broke below range → now reversing up
                # Look for bull FVG (gap above) that price is retracing into
                if (df_15['fvg_bull'].iloc[j] and
                    retrace_bar['close'] > bar['close'] and  # retracing up
                    retrace_bar['close'] > range_low.iloc[i]):  # back inside range
                    fvg_low = df_15['fvg_bull_bot'].iloc[j]
                    fvg_high = df_15['fvg_bull_top'].iloc[j]

                    start_1m = df_15.index[j]
                    end_1m = start_1m + pd.Timedelta(minutes=30)
                    sub_1m = df_1m[(df_1m.index >= start_1m) & (df_1m.index < end_1m)]

                    if len(sub_1m) >= 2:
                        entry_idx = detect_1m_engulf(sub_1m, fvg_low, fvg_high, side='bull')
                        if entry_idx is not None:
                            sig_1m.loc[entry_idx, 'long'] = True
                            sig_15.iloc[i, sig_15.columns.get_loc('long_signal')] = True
                            sig_15.iloc[i, sig_15.columns.get_loc('entry_bar_15')] = df_15.index[j]
                    break

    return sig_1m, sig_15

# =============================================================================
# Vectorized exit (same as v8/v10)
# =============================================================================

MAX_BARS = 45

def _build_forward_arrays(df, max_bars=45):
    n = len(df); N = max_bars
    def _s(col):
        out = np.full((n, N), np.nan, dtype=np.float64)
        vals = df[col].values
        for k in range(N): shift = k+1; out[:n-shift, k] = vals[shift:]
        return out
    return {'fwd_high_bid': _s('high_bid'), 'fwd_low_bid': _s('low_bid'),
            'fwd_high_ask': _s('high_ask'), 'fwd_low_ask': _s('low_ask'),
            'fwd_close_bid': _s('close_bid'), 'fwd_close_ask': _s('close_ask')}

def vec_long(ei, ep, fwd, tp, sl):
    n = len(ei); pnls = np.full(n, np.nan)
    if n == 0: return pnls
    st, tg = ep - sl, ep + tp
    fl = fwd['fwd_low_bid'][ei]; fh = fwd['fwd_high_bid'][ei]; fc = fwd['fwd_close_bid'][ei]
    sh = fl <= st[:, None]; th = fh >= tg[:, None]
    sb = np.argmax(sh, axis=1); tb = np.argmax(th, axis=1); N = fl.shape[1]
    sb[~sh.any(axis=1)] = N; tb[~th.any(axis=1)] = N
    for i in range(n):
        if sb[i] == N and tb[i] == N: lc = fc[i, -1]; pnls[i] = (lc-ep[i]) if not np.isnan(lc) else 0.0
        elif sb[i] <= tb[i]: pnls[i] = st[i] - ep[i]
        else: pnls[i] = tg[i] - ep[i]
    return pnls

def vec_short(ei, ep, fwd, tp, sl):
    n = len(ei); pnls = np.full(n, np.nan)
    if n == 0: return pnls
    st, tg = ep + sl, ep - tp
    fh = fwd['fwd_high_ask'][ei]; fl = fwd['fwd_low_ask'][ei]; fc = fwd['fwd_close_ask'][ei]
    sh = fh >= st[:, None]; th = fl <= tg[:, None]
    sb = np.argmax(sh, axis=1); tb = np.argmax(th, axis=1); N = fl.shape[1]
    sb[~sh.any(axis=1)] = N; tb[~th.any(axis=1)] = N
    for i in range(n):
        if sb[i] == N and tb[i] == N: lc = fc[i, -1]; pnls[i] = (ep[i]-lc) if not np.isnan(lc) else 0.0
        elif sb[i] <= tb[i]: pnls[i] = ep[i] - st[i]
        else: pnls[i] = ep[i] - tg[i]
    return pnls

# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 72)
    print("v13 Breakout Retrace FVG")
    print("=" * 72)

    print("\n[1/3] Loading & resampling...")
    df_1m = load_askbid_data()
    df_15 = resample_15m(df_1m)
    df_15 = compute_15m_indicators(df_15)
    df_15 = df_15.dropna(subset=['atr14', 'ema50_slope'])
    df_1m = df_1m.dropna(subset=['close_ask', 'close_bid'])
    cs = max(df_15.index[0], df_1m.index[0]); ce = min(df_15.index[-1], df_1m.index[-1])
    df_15 = df_15[(df_15.index >= cs) & (df_15.index <= ce)]
    df_1m = df_1m[(df_1m.index >= cs) & (df_1m.index <= ce)]
    print(f"  15m: {len(df_15)} bars, 1m: {len(df_1m)} bars")
    print(f"  In-regime: {df_15['in_regime'].sum():,} 15m bars")

    # --- Sweep range lookback + FVG bars ---
    print("\n[2/3] Sweeping range_lookback × fvg_max_bars...")
    lookbacks = [3, 5, 10]
    fvg_bars = [3, 5]

    print(f"{'LB':>4s} {'FVG':>4s} {'Signals':>7s} {'L':>5s} {'S':>5s}")
    print("-" * 35)
    sig_results = []
    for lb in lookbacks:
        for fb in fvg_bars:
            sig_1m, sig_15 = gen_breakout_retrace_fvg(df_15, df_1m, range_lookback=lb, fvg_max_bars=fb)
            nl = sig_1m['long'].sum(); ns = sig_1m['short'].sum()
            print(f"  {lb:>4d} {fb:>4d} {nl+ns:>7d} {nl:>5d} {ns:>5d}")
            sig_results.append({'lb': lb, 'fb': fb, 'nl': nl, 'ns': ns, 'sig_1m': sig_1m})

    # Pick best: most signals
    best = max(sig_results, key=lambda r: r['nl'] + r['ns'])
    if best['nl'] + best['ns'] == 0:
        print("\nZERO signals — pattern too strict. Trying without session filter...")
        # Try without in_regime gate
        df_15['in_regime'] = True
        sig_1m, sig_15 = gen_breakout_retrace_fvg(df_15, df_1m, range_lookback=5, fvg_max_bars=5)
        print(f"  Without regime gate: {sig_1m['long'].sum()}L, {sig_1m['short'].sum()}S")
        if sig_1m['long'].sum() + sig_1m['short'].sum() == 0:
            print("STILL ZERO — pattern doesn't fire in entire dataset. Exiting.")
            return
    else:
        sig_1m = best['sig_1m']
        print(f"\n  Using lookback={best['lb']}, fvg_bars={best['fb']}: {best['nl']}L, {best['ns']}S")

    # --- TP/SL Sweep ---
    print(f"\n[3/3] TP/SL sweep...")
    fwd_1m = _build_forward_arrays(df_1m, MAX_BARS)
    tp_grid = [10, 15, 20, 25, 30]
    sl_grid = [8, 10, 12, 15, 20]

    print(f"{'TP':>4s} {'SL':>4s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'Avg':>7s} {'L':>5s} {'S':>5s}")
    print("-" * 70)
    all_results = []
    for tp, sl in product(tp_grid, sl_grid):
        pnls = []; nl = 0; ns = 0
        lm = sig_1m['long']
        if lm.any():
            li = np.array([df_1m.index.get_loc(i) for i in sig_1m.index[lm]], dtype=np.int64)
            lp = df_1m['close_ask'].values[li]
            lpnls = vec_long(li, lp, fwd_1m, tp, sl); lpnls = lpnls[~np.isnan(lpnls)]
            pnls.extend(lpnls.tolist()); nl = len(lpnls)
        sm = sig_1m['short']
        if sm.any():
            si = np.array([df_1m.index.get_loc(i) for i in sig_1m.index[sm]], dtype=np.int64)
            sp = df_1m['close_bid'].values[si]
            spnls = vec_short(si, sp, fwd_1m, tp, sl); spnls = spnls[~np.isnan(spnls)]
            pnls.extend(spnls.tolist()); ns = len(spnls)

        if len(pnls) < 3: continue
        a = np.array(pnls); n = len(a); total = a.sum(); wr = (a > 0).mean() * 100
        pos = a[a > 0].sum(); neg = abs(a[a < 0].sum()); pf = pos/neg if neg > 0 else 99
        print(f"  {tp:>4.0f} {sl:>4.0f} {n:>7d} {total:>+10.1f} {wr:>6.1f}% {pf:>5.2f} {total/n:>+7.2f} {nl:>5d} {ns:>5d}")
        all_results.append({'tp': tp, 'sl': sl, 'trades': n, 'pnl': round(total, 1),
                           'wr': round(wr, 1), 'pf': round(pf, 2), 'avg': round(total/n, 2),
                           'longs': nl, 'shorts': ns})

    if all_results:
        RD = pd.DataFrame(all_results)
        best_tpsl = RD.loc[RD['pnl'].idxmax()]
        print(f"\n  Best: TP={best_tpsl['tp']:.0f} SL={best_tpsl['sl']:.0f} → {best_tpsl['trades']:.0f}t, {best_tpsl['pnl']:+.1f}pt, {best_tpsl['wr']:.1f}% WR, PF={best_tpsl['pf']:.2f}")

    # Signal diagnostics
    print(f"\n  Signal diagnostics:")
    mj_1m = (df_1m.index >= '2026-05-01') & (df_1m.index <= '2026-06-09')
    sig_mj = sig_1m.loc[sig_1m.index.isin(df_1m.index[mj_1m])]
    print(f"    May-June 2026: {sig_mj['long'].sum()}L, {sig_mj['short'].sum()}S")
    sig_full = sig_1m
    print(f"    Full period: {sig_full['long'].sum()}L, {sig_full['short'].sum()}S")

    # Count how many 15m breakouts were detected
    n_breakouts = ((df_15['close'] > df_15['high_ask'].shift(1).rolling(5).max()) |
                   (df_15['close'] < df_15['low_ask'].shift(1).rolling(5).min())).sum()
    n_fvg = (df_15['fvg_bull'] | df_15['fvg_bear']).sum()
    print(f"    15m breakouts: {n_breakouts}, FVGs: {n_fvg}")

    print(f"\nDONE.")


if __name__ == '__main__':
    main()
