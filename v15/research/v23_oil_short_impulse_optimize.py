#!/usr/bin/env python3
"""
v23 Oil Short Impulse Optimization
====================================
Sweeps each parameter independently + adds enhancements to v22 baseline.
Optimizations:
  1. change_max sweep (-10 to -20)
  2. Volume threshold sweep (800 to 1600)
  3. Session filter: US-only vs UK-only vs both
  4. Add fullness filter (fullness < -0.2, from v20)
  5. Add bar direction confirmation (next bar also down)
  6. Time-of-day filter (first 2 hours vs rest of session)
  7. Combined best params + TP/SL re-optimize
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from data.data_loader import DataLoader


def load_oil_data(start_date="2024-01-01", end_date="2026-05-22"):
    loader = DataLoader()
    raw = loader.load_data(table_name="prices", start_date=start_date, end_date=end_date)
    raw.index = pd.to_datetime(raw['timestamp'], unit='ms')
    df = pd.DataFrame(index=raw.index)
    df['open_ask'] = raw['openPrice_ask'].astype(float)
    df['high_bid'] = raw['highPrice_bid'].astype(float)
    df['low_bid'] = raw['lowPrice_bid'].astype(float)
    df['high_ask'] = raw['highPrice_ask'].astype(float)
    df['low_ask'] = raw['lowPrice_ask'].astype(float)
    df['close_ask'] = raw['closePrice_ask'].astype(float)
    df['close_bid'] = raw['closePrice_bid'].astype(float)
    df['volume'] = raw['lastTradedVolume'].astype(float)
    df['spread'] = df['close_ask'] - df['close_bid']
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    return df


def compute_features(df_1m):
    df = df_1m.copy()
    df['change'] = df['close_ask'] - df['open_ask']
    df['prev_change'] = df['change'].shift(1)
    df['prev2_change'] = df['change'].shift(2)
    df['prev_lower_wick'] = (df['close_ask'].shift(1) - df['low_ask'].shift(1))
    df['prev_volume'] = df['volume'].shift(1)

    tr = pd.concat([df['high_ask']-df['low_ask'],
                    abs(df['high_ask']-df['close_ask'].shift()),
                    abs(df['low_ask']-df['close_ask'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    daily_high = df['high_ask'].resample('D').max()
    df['day_high'] = np.nan
    for day_idx in daily_high.index:
        mask = df.index.date == day_idx.date()
        df.loc[mask, 'day_high'] = daily_high.loc[day_idx]
    df['dist_day_high'] = df['day_high'] - df['close_ask']

    # Daily range for fullness
    daily_range = df.resample('D').agg({'high_ask': 'max', 'low_ask': 'min', 'open_ask': 'first', 'close_ask': 'last'}).dropna()
    daily_range['range'] = daily_range['high_ask'] - daily_range['low_ask']
    daily_range['avg_range_5d'] = daily_range['range'].rolling(5, min_periods=3).mean()
    df['day_open'] = np.nan; df['avg_range_5d'] = np.nan
    for day_idx in daily_range.index:
        mask = df.index.date == day_idx.date()
        df.loc[mask, 'day_open'] = daily_range.loc[day_idx, 'open_ask']
        df.loc[mask, 'avg_range_5d'] = daily_range.loc[day_idx, 'avg_range_5d']
    df['fullness'] = (df['close_ask'] - df['day_open']) / (df['avg_range_5d'] + 0.01)

    df_15 = df.resample('15min', label='right', closed='right').agg({'open_ask': 'first', 'close_ask': 'last'}).dropna()
    df_15['up'] = 0
    df_15.loc[df_15['close_ask'] > df_15['open_ask'], 'up'] = 1
    df_15.loc[df_15['close_ask'] < df_15['open_ask'], 'up'] = -1
    df_15['up_count3'] = df_15['up'].rolling(3, min_periods=1).sum()
    df['up_count3_15min'] = np.nan
    for idx_15 in df_15.index:
        next_start = idx_15 + pd.Timedelta(minutes=15)
        mask = (df.index >= idx_15) & (df.index < next_start)
        df.loc[mask, 'up_count3_15min'] = df_15.loc[idx_15, 'up_count3']

    df['is_us'] = df.index.hour.isin([12,13,14,15,16,17,18,19,20])
    df['is_uk'] = df.index.hour.isin([7,8,9,10,11,12,13,14,15,16])
    df['in_session'] = df['is_us'] | df['is_uk']
    # First 2 hours of US session (12-13 UTC)
    df['us_open_2h'] = df.index.hour.isin([12,13])
    # First 2 hours of UK session (7-8 UTC)
    df['uk_open_2h'] = df.index.hour.isin([7,8])
    # US 7-13 UTC (London morning + NY open overlap)
    df['us_7_13'] = df.index.hour.isin([7,8,9,10,11,12])

    return df


def build_mask(df, **cfg):
    """Build signal mask from config dict."""
    m = (
        (df['prev_change'] < cfg['change_max'])
        & (df['prev2_change'] < cfg['prev2_max'])
        & (df['prev2_change'] > cfg['prev2_min'])
        & (df['prev_lower_wick'] < cfg['lower_wick_max'])
        & (df['prev_volume'] > cfg['volume_min'])
        & (df['up_count3_15min'] != -3)
        & (df['dist_day_high'] < cfg['dist_high_max'])
    )
    if cfg.get('in_session', True):
        if cfg.get('us_7_13_only'):
            m &= df['us_7_13']
        elif cfg.get('us_only'):
            m &= df['is_us']
        elif cfg.get('uk_only'):
            m &= df['is_uk']
        else:
            m &= df['in_session']
    if cfg.get('fullness_max') is not None:
        m &= df['fullness'] < cfg['fullness_max']
    if cfg.get('us_open_only'):
        m &= df['us_open_2h']
    if cfg.get('uk_open_only'):
        m &= df['uk_open_2h']
    if cfg.get('confirm_bar_down'):
        # Also requires current bar to be bearish (reinforcing the impulse)
        m &= df['change'] < 0
    return m


def sim_short(ei, ep, df, tp, sl, max_bars=60):
    stop, target = ep + sl, ep - tp
    horizon = min(max_bars, len(df) - ei - 1)
    for i in range(1, horizon + 1):
        b = df.iloc[ei + i]
        if b['high_ask'] >= stop: return stop, i, 'sl'
        if b['low_ask'] <= target: return target, i, 'tp'
    return df.iloc[ei + horizon]['close_ask'], horizon, 'timeout'


def evaluate(mask, df, tp, sl, max_bars=60):
    trades = []
    for sig_idx in df.index[mask]:
        ei = df.index.get_loc(sig_idx)
        if ei + max_bars >= len(df): continue
        ep = df.iloc[ei]['close_bid']
        ex, bars, r = sim_short(ei, ep, df, tp, sl, max_bars)
        trades.append({'pnl': ep - ex, 'reason': r})
    return trades


def summarize(trades):
    if not trades: return None
    pnls = [t['pnl'] for t in trades]
    n = len(pnls); total = sum(pnls); wr = sum(1 for p in pnls if p > 0)/n*100
    p = sum(x for x in pnls if x > 0); neg = abs(sum(x for x in pnls if x < 0))
    pf = p/neg if neg > 0 else 99
    return {'trades': n, 'pnl': total, 'wr': wr, 'pf': pf, 'avg': total/n}


def print_row(label, s, nsig=0):
    if s:
        print(f"  {label:<25s} {nsig:>6,d} {s['trades']:>6d} {s['pnl']:>+9.1f} {s['wr']:>6.1f}% {s['pf']:>5.2f} {s['avg']:>+7.2f}")
    else:
        print(f"  {label:<25s} {nsig:>6,d} {'-':>6s}")


TP, SL = 70, 40
BASE = {'change_max': -14, 'prev2_max': 10, 'prev2_min': -14,
        'lower_wick_max': 35, 'volume_min': 1000, 'dist_high_max': 180,
        'in_session': True, 'us_only': False, 'uk_only': False}


def main():
    print("=" * 72)
    print("v23 Oil Short Impulse Optimization")
    print(f"  Base TP={TP}, SL={SL}")
    print("=" * 72)

    print("\n[1/2] Loading & features...")
    df = load_oil_data()
    df = compute_features(df)
    df = df.dropna(subset=['ATR', 'day_high', 'up_count3_15min', 'prev_change', 'fullness'])
    print(f"  {len(df):,} bars ready")

    # =========================================================================
    # SWEEP 1: change_max (the bearish drop threshold)
    # =========================================================================
    print(f"\n{'='*65}")
    print("SWEEP 1: change_max (prev bar change threshold)")
    print(f"  {'change_max':<10s} {'Signals':>7s} {'Trades':>6s} {'PnL':>9s} {'WR':>6s} {'PF':>5s} {'Avg':>7s}")
    print(f"  {'-'*55}")
    sweep1 = []
    for cm in [-10, -12, -14, -16, -18, -20]:
        cfg = {**BASE, 'change_max': cm}
        mask = build_mask(df, **cfg)
        trades = evaluate(mask, df, TP, SL)
        s = summarize(trades)
        if s:
            print_row(f"change<{cm}", s, mask.sum())
            sweep1.append({'change_max': cm, **s, 'signals': mask.sum()})

    best_cm = max(sweep1, key=lambda x: x['pnl'])['change_max']

    # =========================================================================
    # SWEEP 2: Volume threshold
    # =========================================================================
    print(f"\n{'='*65}")
    print(f"SWEEP 2: Volume threshold (change_max={best_cm})")
    print(f"  {'vol_min':<10s} {'Signals':>7s} {'Trades':>6s} {'PnL':>9s} {'WR':>6s} {'PF':>5s} {'Avg':>7s}")
    print(f"  {'-'*55}")
    sweep2 = []
    for vm in [800, 900, 1000, 1100, 1200, 1400, 1600]:
        cfg = {**BASE, 'change_max': best_cm, 'volume_min': vm}
        mask = build_mask(df, **cfg)
        trades = evaluate(mask, df, TP, SL)
        s = summarize(trades)
        if s and s['trades'] > 10:
            print_row(f"vol>{vm}", s, mask.sum())
            sweep2.append({'volume_min': vm, **s, 'signals': mask.sum()})

    best_vm = max(sweep2, key=lambda x: x['pnl'])['volume_min'] if sweep2 else 1000

    # =========================================================================
    # SWEEP 3: Session filter
    # =========================================================================
    print(f"\n{'='*65}")
    print(f"SWEEP 3: Session (change<{best_cm}, vol>{best_vm})")
    print(f"  {'Session':<15s} {'Signals':>7s} {'Trades':>6s} {'PnL':>9s} {'WR':>6s} {'PF':>5s} {'Avg':>7s}")
    print(f"  {'-'*55}")
    sweep3 = []
    for sess_name, sess_kwargs in [
        ('both', {'us_only': False, 'uk_only': False}),
        ('US only', {'us_only': True, 'uk_only': False}),
        ('UK only', {'us_only': False, 'uk_only': True}),
        ('US open 2h', {'us_only': False, 'uk_only': False, 'us_open_only': True}),
        ('UK open 2h', {'us_only': False, 'uk_only': False, 'uk_open_only': True}),
        ('US 7-13', {'us_only': False, 'uk_only': False, 'us_7_13_only': True}),
    ]:
        cfg = {**BASE, 'change_max': best_cm, 'volume_min': best_vm, **sess_kwargs}
        mask = build_mask(df, **cfg)
        trades = evaluate(mask, df, TP, SL)
        s = summarize(trades)
        if s and s['trades'] > 10:
            print_row(sess_name, s, mask.sum())
            sweep3.append({'session': sess_name, **s, 'signals': mask.sum(), **sess_kwargs})

    best_sess = max(sweep3, key=lambda x: x['pnl'])
    sess_kwargs = {k: best_sess.get(k, False) for k in ['us_only', 'uk_only', 'us_open_only', 'uk_open_only']}

    # =========================================================================
    # SWEEP 4: Add fullness filter (from v20 insight)
    # =========================================================================
    print(f"\n{'='*65}")
    print(f"SWEEP 4: Fullness cap (bearish day filter)")
    print(f"  {'fullness<':<12s} {'Signals':>7s} {'Trades':>6s} {'PnL':>9s} {'WR':>6s} {'PF':>5s} {'Avg':>7s}")
    print(f"  {'-'*55}")
    sweep4 = []
    for fmax in [None, -0.1, -0.2, -0.3]:
        cfg = {**BASE, 'change_max': best_cm, 'volume_min': best_vm, **sess_kwargs, 'fullness_max': fmax}
        mask = build_mask(df, **cfg)
        trades = evaluate(mask, df, TP, SL)
        s = summarize(trades)
        label = f"full<{fmax}" if fmax else "no filter"
        if s and s['trades'] > 10:
            print_row(label, s, mask.sum())
            sweep4.append({'fullness_max': fmax, **s, 'signals': mask.sum()})

    best_f = max(sweep4, key=lambda x: x['pnl'])['fullness_max']

    # =========================================================================
    # SWEEP 5: Confirm bar also down
    # =========================================================================
    print(f"\n{'='*65}")
    print(f"SWEEP 5: Confirm bar also bearish")
    print(f"  {'Confirm':<12s} {'Signals':>7s} {'Trades':>6s} {'PnL':>9s} {'WR':>6s} {'PF':>5s} {'Avg':>7s}")
    print(f"  {'-'*55}")
    sweep5 = []
    for confirm in [False, True]:
        cfg = {**BASE, 'change_max': best_cm, 'volume_min': best_vm, **sess_kwargs,
               'fullness_max': best_f, 'confirm_bar_down': confirm}
        mask = build_mask(df, **cfg)
        trades = evaluate(mask, df, TP, SL)
        s = summarize(trades)
        label = 'yes' if confirm else 'no'
        if s and s['trades'] > 10:
            print_row(label, s, mask.sum())
            sweep5.append({'confirm': confirm, **s, 'signals': mask.sum()})

    best_confirm = max(sweep5, key=lambda x: x['pnl'])['confirm']

    # =========================================================================
    # Best combined config
    # =========================================================================
    BEST = {**BASE, 'change_max': best_cm, 'volume_min': best_vm, **sess_kwargs,
            'fullness_max': best_f, 'confirm_bar_down': best_confirm}

    print(f"\n{'='*65}")
    print("BEST COMBINED CONFIG")
    print(f"  change<{best_cm}, vol>{best_vm}, fullness<{best_f}, confirm_down={best_confirm}")
    sess_desc = 'US' if sess_kwargs.get('us_only') else 'UK' if sess_kwargs.get('uk_only') else 'both'
    if sess_kwargs.get('us_open_only'): sess_desc += ' open-2h'
    if sess_kwargs.get('uk_open_only'): sess_desc += ' open-2h'
    print(f"  session={sess_desc}")

    mask_best = build_mask(df, **BEST)
    trades_best = evaluate(mask_best, df, TP, SL)
    s_best = summarize(trades_best)
    if s_best:
        print(f"  Base TP={TP}/SL={SL}: {s_best['trades']}t, {s_best['pnl']:+.0f}pts, "
              f"{s_best['wr']:.1f}% WR, PF={s_best['pf']:.2f}, avg={s_best['avg']:+.2f}/trade")

    # Re-baseline comparison
    mask_base = build_mask(df, **BASE)
    trades_base = evaluate(mask_base, df, TP, SL)
    s_base = summarize(trades_base)
    if s_base:
        print(f"  Baseline: {s_base['trades']}t, {s_base['pnl']:+.0f}pts, "
              f"{s_base['wr']:.1f}% WR, PF={s_base['pf']:.2f}, avg={s_base['avg']:+.2f}/trade")
        if s_best:
            improvement = s_best['pnl'] - s_base['pnl']
            print(f"  Improvement: {improvement:+.0f} pts")

    # =========================================================================
    # TP/SL sweep on best config
    # =========================================================================
    print(f"\n{'='*65}")
    print("TP/SL SWEEP (best config)")
    print(f"  {'TP':>4s} {'SL':>4s} {'Trades':>6s} {'PnL':>9s} {'WR':>6s} {'PF':>5s} {'Avg':>7s}")
    print(f"  {'-'*48}")
    best_overall = {'pnl': -99999}
    for tp in [50, 60, 70, 80, 90]:
        for sl in [30, 40, 50, 60]:
            trades = evaluate(mask_best, df, tp, sl)
            s = summarize(trades)
            if s and s['trades'] > 10:
                print(f"  {tp:>4.0f} {sl:>4.0f} {s['trades']:>6d} {s['pnl']:>+9.1f} {s['wr']:>6.1f}% {s['pf']:>5.2f} {s['avg']:>+7.2f}")
                if s['pnl'] > best_overall['pnl']:
                    best_overall = {**s, 'tp': tp, 'sl': sl}

    print(f"\n  Best overall: TP={best_overall.get('tp')}/{best_overall.get('sl')} → "
          f"{best_overall.get('trades')}t, {best_overall.get('pnl'):+.0f}pts, "
          f"{best_overall.get('wr'):.1f}% WR, PF={best_overall.get('pf'):.2f}")

    # =========================================================================
    # Monthly with best config + best TP/SL
    # =========================================================================
    final_tp = best_overall.get('tp', TP)
    final_sl = best_overall.get('sl', SL)
    print(f"\n{'='*65}")
    print(f"MONTHLY (best config, TP={final_tp}/SL={final_sl})")
    print(f"  {'Month':<10} {'Trades':>6} {'PnL':>9} {'WR':>6} {'SL':>5} {'TP':>5} {'TO':>5}")
    print(f"  {'-'*50}")

    months = pd.date_range("2024-02-01", "2026-06-01", freq="MS", tz="UTC")
    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        m_mask = mask_best & (df.index >= m_start) & (df.index <= m_end)
        if m_mask.sum() == 0: continue
        trades = evaluate(m_mask, df, final_tp, final_sl)
        if not trades: continue
        n = len(trades); pnls = [t['pnl'] for t in trades]
        reasons = {'sl': sum(1 for t in trades if t['reason']=='sl'),
                   'tp': sum(1 for t in trades if t['reason']=='tp'),
                   'timeout': sum(1 for t in trades if t['reason']=='timeout')}
        total = sum(pnls); wr = sum(1 for p in pnls if p > 0)/n*100
        print(f"  {str(m_start.date())[:7]:<10} {n:>6} {total:>+9.1f} {wr:>5.1f}% {reasons['sl']:>5} {reasons['tp']:>5} {reasons['timeout']:>5}")

    # By year
    print(f"\n  By year:")
    for year in [2024, 2025, 2026]:
        y_mask = mask_best & (df.index.year == year)
        trades = evaluate(y_mask, df, final_tp, final_sl)
        s = summarize(trades)
        if s:
            print(f"    {year}: {s['trades']}t, {s['pnl']:+.0f}pts, {s['wr']:.1f}% WR, PF={s['pf']:.2f}")

    # May-June 2026
    mj_mask = mask_best & (df.index >= '2026-05-01') & (df.index <= '2026-06-09')
    trades = evaluate(mj_mask, df, final_tp, final_sl)
    s = summarize(trades)
    if s:
        print(f"\n  May-June 2026: {s['trades']}t, {s['pnl']:+.0f}pts, {s['wr']:.1f}% WR")
    else:
        print(f"\n  May-June 2026: 0 trades")

    print(f"\nDONE.")


if __name__ == '__main__':
    main()
