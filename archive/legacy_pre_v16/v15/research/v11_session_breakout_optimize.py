#!/usr/bin/env python3
"""
v11 Session Breakout Optimization
==================================
Focused optimization of the winning v10 session_breakout archetype.

Sweeps:
  1. Range lookback: test 3, 5, 10, 15, 20 bar prior range for breakout threshold
  2. Session-specific: NY open, Asia open, London open, vs combined
  3. FVG confirmation: only breakouts toward unfilled FVG, vs no FVG filter
  4. Fine TP/SL grid: TP=[20,22,25,28,30], SL=[8,10,12,15]

Monthly walk-forward breakdown for stability check.
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
# Data (reused from v10)
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

def resample_15m(df_1m):
    df_15 = df_1m.resample('15min', label='left', closed='left').agg({
        'open_ask': 'first', 'high_ask': 'max', 'high_bid': 'max',
        'low_ask': 'min', 'low_bid': 'min',
        'close_ask': 'last', 'close_bid': 'last',
        'volume': 'sum', 'spread': 'mean',
    })
    df_15['close'] = df_15['close_ask']; df_15['open'] = df_15['open_ask']
    return df_15.dropna()

def compute_15m_indicators(df):
    df['ema20'] = df['close'].ewm(20).mean()
    df['ema50'] = df['close'].ewm(50).mean()
    df['ema50_slope'] = df['ema50'].diff(3)
    tr = pd.concat([
        df['high_ask'] - df['low_ask'],
        abs(df['high_ask'] - df['close_ask'].shift()),
        abs(df['low_ask'] - df['close_ask'].shift())
    ], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()

    # FVG
    df['fvg_bull'] = df['low_ask'] > df['high_ask'].shift(2)
    df['fvg_bear'] = df['high_ask'] < df['low_ask'].shift(2)

    # Session flags
    df['hour_utc'] = df.index.hour
    df['is_ny'] = df['hour_utc'].isin([12, 13])
    df['is_asia'] = df['hour_utc'].isin([16, 17])
    df['is_london'] = df['hour_utc'].isin([7, 8])
    df['session_start'] = (df['hour_utc'].diff() != 0) & (df['is_ny'] | df['is_asia'] | df['is_london'])

    # Low-vol regime
    low_vol = df['atr14'] < df['atr14'].rolling(200).mean() * 0.8
    weak_trend = df['ema50_slope'].abs() < df['atr14'] * 2
    df['in_regime'] = (low_vol & weak_trend).fillna(False)
    return df

# =============================================================================
# Signal generator with configurable lookback & FVG gate
# =============================================================================

def gen_session_breakout(df, range_lookback=5, require_fvg=False, session_mask=None):
    """Generate session breakout signals with configurable parameters.

    Args:
        range_lookback: number of bars for prior range (default 5)
        require_fvg: only take breakouts toward an unfilled FVG
        session_mask: boolean Series or None. If None, all sessions used.
    """
    s = pd.DataFrame(index=df.index)
    s['long'] = False; s['short'] = False

    at_session = df['session_start']
    if session_mask is not None:
        at_session = at_session & session_mask

    prior_high = df['high_ask'].shift(1).rolling(range_lookback).max()
    prior_low = df['low_ask'].shift(1).rolling(range_lookback).min()

    long_cond = (at_session & (df['close'] > prior_high) &
                 (df['close'] > df['open']) & df['in_regime'])
    short_cond = (at_session & (df['close'] < prior_low) &
                  (df['close'] < df['open']) & df['in_regime'])

    if require_fvg:
        # Only breakouts that move toward filling an FVG
        # Long: bear FVG present (gap below, price breaking up into it)
        # Short: bull FVG present (gap above, price breaking down into it)
        long_cond = long_cond & df['fvg_bear'].shift(1)
        short_cond = short_cond & df['fvg_bull'].shift(1)

    s['long'] = long_cond
    s['short'] = short_cond
    return s

# =============================================================================
# 15m -> 1m mapping & vectorized exit (same as v10)
# =============================================================================

MAX_BARS_1M = 45

def map_15m_to_1m(df_15, signal_15, df_1m):
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

def vec_long(entry_indices, entry_prices, fwd, tp, sl):
    n = len(entry_indices); pnls = np.full(n, np.nan)
    if n == 0: return pnls
    st, tg = entry_prices - sl, entry_prices + tp
    fl = fwd['fwd_low_bid'][entry_indices]; fh = fwd['fwd_high_bid'][entry_indices]
    fc = fwd['fwd_close_bid'][entry_indices]
    sh = fl <= st[:, None]; th = fh >= tg[:, None]
    sb = np.argmax(sh, axis=1); tb = np.argmax(th, axis=1)
    N = fl.shape[1]; sb[~sh.any(axis=1)] = N; tb[~th.any(axis=1)] = N
    for i in range(n):
        if sb[i] == N and tb[i] == N: lc = fc[i, -1]; pnls[i] = (lc - entry_prices[i]) if not np.isnan(lc) else 0.0
        elif sb[i] <= tb[i]: pnls[i] = st[i] - entry_prices[i]
        else: pnls[i] = tg[i] - entry_prices[i]
    return pnls

def vec_short(entry_indices, entry_prices, fwd, tp, sl):
    n = len(entry_indices); pnls = np.full(n, np.nan)
    if n == 0: return pnls
    st, tg = entry_prices + sl, entry_prices - tp
    fh = fwd['fwd_high_ask'][entry_indices]; fl = fwd['fwd_low_ask'][entry_indices]
    fc = fwd['fwd_close_ask'][entry_indices]
    sh = fh >= st[:, None]; th = fl <= tg[:, None]
    sb = np.argmax(sh, axis=1); tb = np.argmax(th, axis=1)
    N = fl.shape[1]; sb[~sh.any(axis=1)] = N; tb[~th.any(axis=1)] = N
    for i in range(n):
        if sb[i] == N and tb[i] == N: lc = fc[i, -1]; pnls[i] = (entry_prices[i] - lc) if not np.isnan(lc) else 0.0
        elif sb[i] <= tb[i]: pnls[i] = entry_prices[i] - st[i]
        else: pnls[i] = entry_prices[i] - tg[i]
    return pnls

# =============================================================================
# Monthly evaluation helper
# =============================================================================

def eval_config(signal_15, df_15, df_1m, fwd_1m, tp, sl, period_masks_1m):
    signals_1m = map_15m_to_1m(df_15, signal_15, df_1m)
    per_period = defaultdict(lambda: {'pnls': [], 'longs': 0, 'shorts': 0})
    for pname, pmask in period_masks_1m.items():
        pmask_idx = df_1m.index[pmask]
        sig_sub = signals_1m.loc[signals_1m.index.isin(pmask_idx)]
        if len(sig_sub) == 0: continue
        lm = sig_sub['long']
        if lm.any():
            li = np.array([df_1m.index.get_loc(i) for i in sig_sub.index[lm]], dtype=np.int64)
            lp = df_1m['close_ask'].values[li]
            lpnls = vec_long(li, lp, fwd_1m, tp, sl); lpnls = lpnls[~np.isnan(lpnls)]
            per_period[pname]['pnls'].extend(lpnls.tolist()); per_period[pname]['longs'] += len(lpnls)
        sm = sig_sub['short']
        if sm.any():
            si = np.array([df_1m.index.get_loc(i) for i in sig_sub.index[sm]], dtype=np.int64)
            sp = df_1m['close_bid'].values[si]
            spnls = vec_short(si, sp, fwd_1m, tp, sl); spnls = spnls[~np.isnan(spnls)]
            per_period[pname]['pnls'].extend(spnls.tolist()); per_period[pname]['shorts'] += len(spnls)
    return per_period

def summarize_period(pnls_dict):
    all_pnls = []; tl = ts = 0
    for st in pnls_dict.values():
        all_pnls.extend(st['pnls']); tl += st['longs']; ts += st['shorts']
    if len(all_pnls) < 3: return None
    a = np.array(all_pnls); n = len(a); total = a.sum(); wr = (a > 0).mean() * 100
    pos = a[a > 0].sum(); neg = abs(a[a < 0].sum()); pf = pos/neg if neg > 0 else 99
    mj_p = pnls_dict.get('may_jun_2026', {}).get('pnls', [])
    return {'trades': n, 'pnl': round(total, 1), 'wr': round(wr, 1), 'pf': round(pf, 2),
            'avg': round(total/n, 2), 'longs': tl, 'shorts': ts,
            'mj_trades': len(mj_p), 'mj_pnl': round(sum(mj_p) if mj_p else 0, 1)}

# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 72)
    print("v11 Session Breakout Optimization")
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

    print("[2/3] Building forward arrays...")
    fwd_1m = _build_forward_arrays(df_1m, MAX_BARS_1M)

    period_masks_1m = {
        'full': df_1m.index >= '2025-09-01',
        'pre_may_2026': (df_1m.index >= '2025-09-01') & (df_1m.index < '2026-05-01'),
        'may_jun_2026': (df_1m.index >= '2026-05-01') & (df_1m.index <= '2026-06-09'),
    }
    months = pd.date_range('2025-09-01', '2026-07-01', freq='MS', tz='UTC')
    month_masks_1m = {}
    for ms in months:
        me = ms + pd.offsets.MonthEnd(1)
        month_masks_1m[ms.strftime('%Y-%m')] = (df_1m.index >= ms) & (df_1m.index <= me)
    month_masks_1m['2026-06-p'] = (df_1m.index >= '2026-06-01') & (df_1m.index <= '2026-06-09')

    # Combine period masks with monthly masks
    all_period_masks = {**period_masks_1m, **month_masks_1m}

    # =========================================================================
    # SWEEP 1: Range lookback
    # =========================================================================
    print("\n[3/3] SWEEP 1: Range Lookback")
    print(f"{'Lookback':<12s} {'Signals':>7s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'MJ_T':>5s} {'MJ_PnL':>8s}")
    print("-" * 75)

    lookback_results = []
    for lb in [3, 5, 10, 15, 20]:
        sig = gen_session_breakout(df_15, range_lookback=lb, require_fvg=False)
        n_sig = sig['long'].sum() + sig['short'].sum()
        pp = eval_config(sig, df_15, df_1m, fwd_1m, 25, 12, all_period_masks)
        s = summarize_period(pp)
        if s:
            print(f"  {lb:<12d} {n_sig:>7d} {s['trades']:>7d} {s['pnl']:>+10.1f} {s['wr']:>6.1f}% {s['pf']:>5.2f} {s['mj_trades']:>5d} {s['mj_pnl']:>+8.1f}")
            s['lookback'] = lb; s['signals'] = n_sig; lookback_results.append(s)
        else:
            print(f"  {lb:<12d} {n_sig:>7d} {'< 3 trades':>17s}")

    # Monthly breakdown for best lookback
    if lookback_results:
        best_lb = max(lookback_results, key=lambda r: r['pnl'])
        print(f"\n  Best lookback={best_lb['lookback']} — monthly breakdown:")
        sig = gen_session_breakout(df_15, range_lookback=best_lb['lookback'], require_fvg=False)
        print(f"  {'Month':<12s} {'Trades':>6s} {'PnL':>10s} {'WR':>7s}")
        for mname in sorted(month_masks_1m.keys()):
            pp = eval_config(sig, df_15, df_1m, fwd_1m, 25, 12, {mname: month_masks_1m[mname]})
            s = summarize_period(pp)
            if s: print(f"  {mname:<12s} {s['trades']:>6d} {s['pnl']:>+10.1f} {s['wr']:>6.1f}%")

    # =========================================================================
    # SWEEP 2: Session-specific
    # =========================================================================
    print(f"\n{'='*75}")
    print("SWEEP 2: Session-Specific Performance (lookback=5, TP=25, SL=12)")
    print(f"{'Session':<20s} {'Signals':>7s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'MJ_T':>5s} {'MJ_PnL':>8s}")
    print("-" * 75)

    sessions = {
        'all': None,
        'ny_only': df_15['is_ny'],
        'asia_only': df_15['is_asia'],
        'london_only': df_15['is_london'],
        'ny+asia': df_15['is_ny'] | df_15['is_asia'],
        'ny+london': df_15['is_ny'] | df_15['is_london'],
    }
    session_results = []
    for sname, smask in sessions.items():
        sig = gen_session_breakout(df_15, range_lookback=5, require_fvg=False, session_mask=smask)
        n_sig = sig['long'].sum() + sig['short'].sum()
        pp = eval_config(sig, df_15, df_1m, fwd_1m, 25, 12, all_period_masks)
        s = summarize_period(pp)
        if s:
            print(f"  {sname:<20s} {n_sig:>7d} {s['trades']:>7d} {s['pnl']:>+10.1f} {s['wr']:>6.1f}% {s['pf']:>5.2f} {s['mj_trades']:>5d} {s['mj_pnl']:>+8.1f}")
            s['session'] = sname; s['signals'] = n_sig; session_results.append(s)
        else:
            print(f"  {sname:<20s} {n_sig:>7d} {'< 3 trades':>17s}")

    # =========================================================================
    # SWEEP 3: FVG confirmation
    # =========================================================================
    print(f"\n{'='*75}")
    print("SWEEP 3: FVG Confirmation Gate (lookback=5, TP=25, SL=12)")
    print(f"{'FVG Gate':<12s} {'Signals':>7s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'MJ_T':>5s} {'MJ_PnL':>8s}")
    print("-" * 75)

    fvg_results = []
    for use_fvg in [False, True]:
        sig = gen_session_breakout(df_15, range_lookback=5, require_fvg=use_fvg)
        n_sig = sig['long'].sum() + sig['short'].sum()
        pp = eval_config(sig, df_15, df_1m, fwd_1m, 25, 12, all_period_masks)
        s = summarize_period(pp)
        label = 'fvg_required' if use_fvg else 'no_fvg'
        if s:
            print(f"  {label:<12s} {n_sig:>7d} {s['trades']:>7d} {s['pnl']:>+10.1f} {s['wr']:>6.1f}% {s['pf']:>5.2f} {s['mj_trades']:>5d} {s['mj_pnl']:>+8.1f}")
            s['fvg_gate'] = label; s['signals'] = n_sig; fvg_results.append(s)
        else:
            print(f"  {label:<12s} {n_sig:>7d} {'< 3 trades':>17s}")

    # =========================================================================
    # SWEEP 4: Fine TP/SL grid
    # =========================================================================
    print(f"\n{'='*75}")
    print("SWEEP 4: Fine TP/SL Grid (lookback=5, no FVG, all sessions)")
    print(f"{'TP':>4s} {'SL':>4s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'MJ_T':>5s} {'MJ_PnL':>8s}")
    print("-" * 75)

    base_sig = gen_session_breakout(df_15, range_lookback=5, require_fvg=False)
    finetp = [20, 22, 25, 28, 30]
    finesl = [8, 10, 12, 15]
    fine_results = []
    for tp, sl in product(finetp, finesl):
        pp = eval_config(base_sig, df_15, df_1m, fwd_1m, tp, sl, all_period_masks)
        s = summarize_period(pp)
        if s:
            print(f"  {tp:>4.0f} {sl:>4.0f} {s['trades']:>7d} {s['pnl']:>+10.1f} {s['wr']:>6.1f}% {s['pf']:>5.2f} {s['mj_trades']:>5d} {s['mj_pnl']:>+8.1f}")
            s['tp'] = tp; s['sl'] = sl; fine_results.append(s)

    # =========================================================================
    # SWEEP 5: Best combo (best lookback + best session + best FVG + best TP/SL)
    # =========================================================================
    print(f"\n{'='*75}")
    print("SWEEP 5: Best combo configuration")
    print(f"{'='*75}")

    # Determine best from sweeps
    best_lb = max(lookback_results, key=lambda r: r['pnl'])['lookback'] if lookback_results else 5
    best_sess = max(session_results, key=lambda r: r['pnl']) if session_results else None
    best_fvg = max(fvg_results, key=lambda r: r['pnl']) if fvg_results else None
    best_tpsl = max(fine_results, key=lambda r: r['pnl']) if fine_results else {'tp': 25, 'sl': 12}

    use_fvg_best = best_fvg['fvg_gate'] == 'fvg_required' if best_fvg else False
    sess_mask = None
    sess_name = 'all'
    if best_sess and best_sess['session'] != 'all':
        sess_name = best_sess['session']
        sess_map = {'ny_only': df_15['is_ny'], 'asia_only': df_15['is_asia'],
                    'london_only': df_15['is_london'], 'ny+asia': df_15['is_ny'] | df_15['is_asia'],
                    'ny+london': df_15['is_ny'] | df_15['is_london']}
        sess_mask = sess_map.get(sess_name)

    opt_sig = gen_session_breakout(df_15, range_lookback=best_lb, require_fvg=use_fvg_best, session_mask=sess_mask)
    opt_pp = eval_config(opt_sig, df_15, df_1m, fwd_1m, best_tpsl['tp'], best_tpsl['sl'], all_period_masks)

    print(f"  Config: lookback={best_lb}, session={sess_name}, FVG={use_fvg_best}, TP={best_tpsl['tp']}, SL={best_tpsl['sl']}")
    s = summarize_period(opt_pp)
    if s:
        print(f"  Full:  {s['trades']:>5d} trades, {s['pnl']:>+8.1f} pts, {s['wr']:>5.1f}% WR, PF={s['pf']:.2f}, avg={s['avg']:>+.2f}/trade")
        print(f"  MJ:    {s['mj_trades']:>5d} trades, {s['mj_pnl']:>+8.1f} pts")

        # Detailed monthly
        print(f"\n  Monthly breakdown:")
        print(f"  {'Month':<12s} {'Trades':>6s} {'PnL':>10s} {'WR':>7s}")
        for mname in sorted(month_masks_1m.keys()):
            mpp = eval_config(opt_sig, df_15, df_1m, fwd_1m, best_tpsl['tp'], best_tpsl['sl'], {mname: month_masks_1m[mname]})
            ms = summarize_period(mpp)
            if ms: print(f"  {mname:<12s} {ms['trades']:>6d} {ms['pnl']:>+10.1f} {ms['wr']:>6.1f}%")
    else:
        print("  No trades — combination too restrictive")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*75}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*75}")
    print(f"  Best lookback: {best_lb}")
    print(f"  Best session:  {sess_name}")
    print(f"  Best FVG gate: {'ON' if use_fvg_best else 'OFF'}")
    print(f"  Best TP/SL:    {best_tpsl['tp']}/{best_tpsl['sl']}")
    if s:
        print(f"  Result: {s['trades']} trades, {s['pnl']:+.1f} pts, {s['wr']:.1f}% WR, PF={s['pf']:.2f}")
    print(f"\nDONE.")


if __name__ == '__main__':
    main()
