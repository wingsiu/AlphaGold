#!/usr/bin/env python3
"""
v14 15-Min Long Wick Reversal — Production Research Script
============================================================
Classic candlestick reversal pattern on 15-min bars:
  - Long LOWER wick (hammer): buyers rejected lows → LONG at close
  - Long UPPER wick (shooting star): sellers rejected highs → SHORT at close

Wick thresholds: wick > X% of candle range, body < Y% of range.
Sweeps: wick_ratio, body_ratio, range position, TP/SL grid.
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
    df['candle_range'] = df['high_ask'] - df['low_ask']
    df['body'] = abs(df['close'] - df['open'])
    df['body_ratio'] = df['body'] / (df['candle_range'] + 0.01)

    # Wick ratios
    # Lower wick: min(open, close) - low
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low_ask']
    df['upper_wick'] = df['high_ask'] - df[['open', 'close']].max(axis=1)
    df['lower_wick_pct'] = df['lower_wick'] / (df['candle_range'] + 0.01) * 100
    df['upper_wick_pct'] = df['upper_wick'] / (df['candle_range'] + 0.01) * 100

    # Trend / position
    df['ema20'] = df['close'].ewm(20).mean()
    df['ema50'] = df['close'].ewm(50).mean()
    df['atr14'] = (df['high_ask'] - df['low_ask']).rolling(14).mean()

    df['range_high'] = df['high_ask'].rolling(20, min_periods=5).max()
    df['range_low'] = df['low_ask'].rolling(20, min_periods=5).min()
    df['pos_in_range'] = ((df['close'] - df['range_low']) /
                          (df['range_high'] - df['range_low'] + 0.01))

    # Volume context
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # Low-vol regime
    df['ema50_slope'] = df['ema50'].diff(3)
    low_vol = df['atr14'] < df['atr14'].rolling(200).mean() * 0.8
    weak_trend = df['ema50_slope'].abs() < df['atr14'] * 2
    df['in_regime'] = (low_vol & weak_trend).fillna(False)
    return df


def gen_wick_signals(df, min_wick_pct=60, max_body_pct=30, at_range_extreme=True, use_regime=True):
    """Generate long wick reversal signals.

    min_wick_pct: minimum wick as % of candle range
    max_body_pct: maximum body as % of candle range (small body = indecision)
    at_range_extreme: only at range bottom (for hammer) or top (for shooting star)
    """
    s = pd.DataFrame(index=df.index)
    s['long'] = False; s['short'] = False

    has_long_lower = df['lower_wick_pct'] >= min_wick_pct
    has_long_upper = df['upper_wick_pct'] >= min_wick_pct
    has_small_body = df['body_ratio'] * 100 <= max_body_pct

    regime = df['in_regime'] if use_regime else pd.Series(True, index=df.index)

    # Hammer: long lower wick + small body
    long_cond = has_long_lower & has_small_body & regime
    # Shooting star: long upper wick + small body
    short_cond = has_long_upper & has_small_body & regime

    if at_range_extreme:
        long_cond = long_cond & (df['pos_in_range'] < 0.30)  # near range low
        short_cond = short_cond & (df['pos_in_range'] > 0.70)  # near range high

    s['long'] = long_cond
    s['short'] = short_cond
    return s


# === Map 15m signal to 1m entry + vectorized exit ===

MAX_BARS_1M = 45

def map_to_1m(df_15, signal_15, df_1m):
    sig_1m = pd.DataFrame(index=df_1m.index)
    sig_1m['long'] = False; sig_1m['short'] = False
    for idx_15 in signal_15.index:
        if not signal_15.loc[idx_15, 'long'] and not signal_15.loc[idx_15, 'short']:
            continue
        next_start = idx_15 + pd.Timedelta(minutes=15)
        candidates = df_1m.index[(df_1m.index >= next_start) &
                                  (df_1m.index < next_start + pd.Timedelta(minutes=1))]
        if len(candidates) > 0:
            sig_1m.loc[candidates[0], 'long'] = signal_15.loc[idx_15, 'long']
            sig_1m.loc[candidates[0], 'short'] = signal_15.loc[idx_15, 'short']
    return sig_1m

def _build_fwd(df, max_bars=45):
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
    print("v14 15-Min Long Wick Reversal")
    print("=" * 72)

    print("\n[1/3] Loading...")
    df_1m = load_data()
    df_15 = resample_15m(df_1m)
    df_15 = compute_15m_indicators(df_15)
    df_15 = df_15.dropna(subset=['atr14', 'lower_wick_pct', 'upper_wick_pct'])
    cs = max(df_15.index[0], df_1m.index[0]); ce = min(df_15.index[-1], df_1m.index[-1])
    df_15 = df_15[(df_15.index >= cs) & (df_15.index <= ce)]
    df_1m = df_1m[(df_1m.index >= cs) & (df_1m.index <= ce)]
    print(f"  15m: {len(df_15)} bars ({df_15['in_regime'].sum():,} in-regime)")

    fwd = _build_fwd(df_1m, MAX_BARS_1M)

    # Sweep wick thresholds
    print("\n[2/3] Sweeping wick thresholds...")
    wick_pcts = [50, 60, 70]
    body_pcts = [25, 35, 50]
    extremes = [True, False]

    best_cfg = {'pnl': -999, 'wick': 0, 'body': 0, 'extreme': False, 'tp': 0, 'sl': 0, 'trades': 0}

    for wick_pct in wick_pcts:
        for body_pct in body_pcts:
            for at_ext in extremes:
                sig_15 = gen_wick_signals(df_15, min_wick_pct=wick_pct, max_body_pct=body_pct,
                                         at_range_extreme=at_ext, use_regime=True)
                sig_1m = map_to_1m(df_15, sig_15, df_1m)
                nl = sig_1m['long'].sum(); ns = sig_1m['short'].sum()
                total = nl + ns
                if total < 5:
                    continue

                # Find best TP/SL
                for tp in [8, 10, 12, 15, 20, 25]:
                    for sl in [8, 10, 12, 15, 20, 25]:
                        pnls = []
                        if nl:
                            li = np.array([df_1m.index.get_loc(i) for i in sig_1m.index[sig_1m['long']]], dtype=np.int64)
                            lp = df_1m['close_ask'].values[li]
                            lr = v_long(li, lp, fwd, tp, sl); lr = lr[~np.isnan(lr)]; pnls.extend(lr.tolist())
                        if ns:
                            si = np.array([df_1m.index.get_loc(i) for i in sig_1m.index[sig_1m['short']]], dtype=np.int64)
                            sp = df_1m['close_bid'].values[si]
                            sr = v_short(si, sp, fwd, tp, sl); sr = sr[~np.isnan(sr)]; pnls.extend(sr.tolist())
                        if len(pnls) < 5: continue
                        a = np.array(pnls); total_pnl = a.sum()
                        if total_pnl > best_cfg['pnl']:
                            best_cfg = {'pnl': total_pnl, 'wick': wick_pct, 'body': body_pct,
                                       'extreme': at_ext, 'tp': tp, 'sl': sl,
                                       'trades': len(a), 'nl': nl, 'ns': ns, 'sig': sig_1m}

                # Print summary for this threshold combo
                best_local_pnl = -999
                best_local_tp = 0; best_local_sl = 0; best_local_t = 0
                for tp, sl in product([8, 12, 15, 20, 25], repeat=2):
                    pnls = []
                    if nl:
                        li = np.array([df_1m.index.get_loc(i) for i in sig_1m.index[sig_1m['long']]], dtype=np.int64)
                        lp = df_1m['close_ask'].values[li]; lr = v_long(li, lp, fwd, tp, sl); lr = lr[~np.isnan(lr)]; pnls.extend(lr.tolist())
                    if ns:
                        si = np.array([df_1m.index.get_loc(i) for i in sig_1m.index[sig_1m['short']]], dtype=np.int64)
                        sp = df_1m['close_bid'].values[si]; sr = v_short(si, sp, fwd, tp, sl); sr = sr[~np.isnan(sr)]; pnls.extend(sr.tolist())
                    if len(pnls) < 5: continue
                    a = np.array(pnls)
                    if a.sum() > best_local_pnl:
                        best_local_pnl = a.sum(); best_local_tp = tp; best_local_sl = sl; best_local_t = len(a)

                mj_mask = (sig_1m.index >= '2026-05-01') & (sig_1m.index <= '2026-06-09')
                mj_count = sig_1m.loc[mj_mask, 'long'].sum() + sig_1m.loc[mj_mask, 'short'].sum()
                print(f"  w={wick_pct}% b<={body_pct}% ext={str(at_ext):5s} → {nl:>4d}L/{ns:>4d}S "
                      f"best={best_local_tp}/{best_local_sl}→{best_local_t}t,{best_local_pnl:+.1f}pt  MJ:{mj_count}sig")

    print(f"\n[3/3] Best config: wick>{best_cfg['wick']}%, body<{best_cfg['body']}%, "
          f"extreme={best_cfg['extreme']}, TP={best_cfg['tp']}, SL={best_cfg['sl']}")

    # Full TP/SL table for best config
    sig = best_cfg['sig']
    nl = sig['long'].sum(); ns = sig['short'].sum()
    print(f"\n  {nl}L / {ns}S signals — Full TP/SL grid:")
    print(f"  {'TP':>4s} {'SL':>4s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'Avg':>7s}")
    print(f"  {'-'*45}")
    for tp, sl in product([8, 10, 12, 15, 20, 25], repeat=2):
        pnls = []
        if nl:
            li = np.array([df_1m.index.get_loc(i) for i in sig.index[sig['long']]], dtype=np.int64)
            lp = df_1m['close_ask'].values[li]; lr = v_long(li, lp, fwd, tp, sl); lr = lr[~np.isnan(lr)]; pnls.extend(lr.tolist())
        if ns:
            si = np.array([df_1m.index.get_loc(i) for i in sig.index[sig['short']]], dtype=np.int64)
            sp = df_1m['close_bid'].values[si]; sr = v_short(si, sp, fwd, tp, sl); sr = sr[~np.isnan(sr)]; pnls.extend(sr.tolist())
        if len(pnls) < 5: continue
        a = np.array(pnls); n = len(a); total = a.sum(); wr = (a>0).mean()*100
        pos = a[a>0].sum(); neg = abs(a[a<0].sum()); pf = pos/neg if neg>0 else 99
        print(f"  {tp:>4.0f} {sl:>4.0f} {n:>7d} {total:>+10.1f} {wr:>6.1f}% {pf:>5.2f} {total/n:>+7.2f}")

    # May-June
    mj_mask = (df_1m.index >= '2026-05-01') & (df_1m.index <= '2026-06-09')
    mj_sig = sig.loc[mj_mask]
    mj_nl = mj_sig['long'].sum(); mj_ns = mj_sig['short'].sum()
    if mj_nl + mj_ns > 0:
        tp = best_cfg['tp']; sl = best_cfg['sl']
        pnls = []
        if mj_nl:
            li = np.array([df_1m.index.get_loc(i) for i in mj_sig.index[mj_sig['long']]], dtype=np.int64)
            lp = df_1m['close_ask'].values[li]; lr = v_long(li, lp, fwd, tp, sl); pnls.extend(lr[~np.isnan(lr)].tolist())
        if mj_ns:
            si = np.array([df_1m.index.get_loc(i) for i in mj_sig.index[mj_sig['short']]], dtype=np.int64)
            sp = df_1m['close_bid'].values[si]; sr = v_short(si, sp, fwd, tp, sl); pnls.extend(sr[~np.isnan(sr)].tolist())
        if pnls:
            mj_a = np.array(pnls)
            print(f"\n  May-June 2026: {len(mj_a)} trades, {mj_a.sum():+.1f}pt, {(mj_a>0).mean()*100:.1f}% WR")
        else:
            print(f"\n  May-June 2026: {mj_nl+mj_ns} signals but 0 valid exits")
    else:
        print(f"\n  May-June 2026: ZERO signals")

    # Wick statistics
    print(f"\n  Wick distribution (15m bars):")
    print(f"    Lower wick >50%: {(df_15['lower_wick_pct']>50).sum()}, >60%: {(df_15['lower_wick_pct']>60).sum()}, >70%: {(df_15['lower_wick_pct']>70).sum()}")
    print(f"    Upper wick >50%: {(df_15['upper_wick_pct']>50).sum()}, >60%: {(df_15['upper_wick_pct']>60).sum()}, >70%: {(df_15['upper_wick_pct']>70).sum()}")

    print(f"\nDONE.")


if __name__ == '__main__':
    main()
