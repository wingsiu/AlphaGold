#!/usr/bin/env python3
"""
v21 Oil Native Patterns — Three Oil-Specific Setups
=====================================================
Tests three oil-native patterns found in oil_trader exploration:

  Pattern A — Volume Breakout (Long):
    Trigger: 1m bar with volume >= 1500
    Volume Profile: after the bar, check if subsequent 5 bars stay above
    the trigger bar's open (bullish bias = breakout, not exhaustion)
    If bullish bias → LONG at next bar close, TP=60, SL=40

  Pattern B — 15m 2-Up Retrace (Long):
    When last 2 of 3 15m bars are up (up_count3_15min >= 2),
    AND current 1m close < current 15m bar open (intra-15m retrace)
    → LONG, TP=60, SL=40

  Pattern C — 15m Volume Drop (Short):
    15m bar with volume > 10000 AND close < open
    → SHORT at next 15m open, TP=60, SL=40
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
    df['close'] = df['close_ask']

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def compute_15m_features(df_1m):
    """Build 15m bars with features needed for patterns B and C."""
    df_15 = df_1m.resample('15min', label='right', closed='right').agg({
        'open_ask': 'first',
        'close_ask': 'last',
        'high_ask': 'max',
        'low_ask': 'min',
        'high_bid': 'max',
        'low_bid': 'min',
        'close_bid': 'last',
        'volume': 'sum',
    }).dropna()

    df_15['up'] = 0
    df_15.loc[df_15['close_ask'] > df_15['open_ask'], 'up'] = 1
    df_15.loc[df_15['close_ask'] < df_15['open_ask'], 'up'] = -1
    df_15['up_count3'] = df_15['up'].rolling(3, min_periods=1).sum()

    # Map 15m features back to 1m bars (use previous completed 15m bar)
    m15_feats = df_15[['open_ask', 'close_ask', 'high_ask', 'low_ask', 'high_bid', 'low_bid',
                        'close_bid', 'volume', 'up', 'up_count3']]
    m15_feats.columns = ['open_15', 'close_15', 'high_15_ask', 'low_15_ask',
                          'high_15_bid', 'low_15_bid', 'close_15_bid',
                          'volume_15', 'up_15', 'up_count3_15']

    df_1m = df_1m.copy()
    for col in m15_feats.columns:
        df_1m[col] = np.nan

    # For each 1m bar, fill with the previous completed 15m bar's values
    for idx_15 in m15_feats.index:
        # The 15m bar that ends at idx_15 covers bars after idx_15 - 15min
        next_15_start = idx_15 + pd.Timedelta(minutes=15)
        # Find 1m bars in the next 15m window that should reference this 15m bar
        mask = (df_1m.index >= idx_15) & (df_1m.index < next_15_start)
        for col in m15_feats.columns:
            df_1m.loc[mask, col] = m15_feats.loc[idx_15, col]

    return df_1m, df_15


def sim_long(ei, ep, df, tp, sl, max_bars=60):
    stop, target = ep - sl, ep + tp
    horizon = min(max_bars, len(df) - ei - 1)
    for i in range(1, horizon + 1):
        b = df.iloc[ei + i]
        if b['low_bid'] <= stop: return stop, i, 'sl'
        if b['high_bid'] >= target: return target, i, 'tp'
    return df.iloc[ei + horizon]['close_bid'], horizon, 'timeout'


def sim_short(ei, ep, df, tp, sl, max_bars=60):
    stop, target = ep + sl, ep - tp
    horizon = min(max_bars, len(df) - ei - 1)
    for i in range(1, horizon + 1):
        b = df.iloc[ei + i]
        if b['high_ask'] >= stop: return stop, i, 'sl'
        if b['low_ask'] <= target: return target, i, 'tp'
    return df.iloc[ei + horizon]['close_ask'], horizon, 'timeout'


def test_pattern(name, side, df, entry_indices, entry_prices, tp, sl, max_bars=60):
    """Test a pattern and return trades + summary."""
    trades = []
    for idx, ep in zip(entry_indices, entry_prices):
        ei = df.index.get_loc(idx)
        if ei + max_bars >= len(df):
            continue
        if side == 'long':
            ex, bars, r = sim_long(ei, ep, df, tp, sl, max_bars)
            pnl = ex - ep
        else:
            ex, bars, r = sim_short(ei, ep, df, tp, sl, max_bars)
            pnl = ep - ex
        trades.append({'pattern': name, 'pnl': pnl, 'reason': r})

    if not trades:
        return [], {'trades': 0, 'pnl': 0.0, 'wr': 0.0, 'pf': 0.0}

    a = np.array([t['pnl'] for t in trades])
    n = len(a); total = a.sum(); wr = (a > 0).mean() * 100
    pos = a[a > 0].sum(); neg = abs(a[a < 0].sum())
    pf = pos / neg if neg > 0 else 99
    return trades, {'trades': n, 'pnl': total, 'wr': wr, 'pf': pf, 'avg': total / n}


def main():
    TP, SL, MAX_BARS = 60, 40, 60
    print("=" * 72)
    print("v21 Oil Native Patterns")
    print("=" * 72)

    print("\n[1/3] Loading data...")
    df_1m = load_oil_data()
    print(f"  {len(df_1m):,} 1m bars, {df_1m.index[0]} -> {df_1m.index[-1]}")

    print("[2/3] Building 15m features...")
    df_1m, df_15 = compute_15m_features(df_1m)
    df_1m = df_1m.dropna(subset=['open_15', 'volume_15', 'up_count3_15'])
    print(f"  After merge: {len(df_1m):,} 1m bars, {len(df_15):,} 15m bars")

    # =========================================================================
    # Pattern A: Volume Breakout (vol>=1500 + follow-through check)
    # =========================================================================
    print(f"\n{'='*60}")
    print("PATTERN A: Volume Breakout (vol>=1500 + bullish follow-through)")
    print(f"{'='*60}")

    # Find bars with volume >= 1500
    vol_spikes = df_1m[df_1m['volume'] >= 1500]
    print(f"  Bars with vol>=1500: {len(vol_spikes):,}")

    # For each vol spike, check if the next 5 bars' prices stay above spike bar's open
    # (bullish = true breakout, not exhaustion)
    long_entries = []
    for sig_idx in vol_spikes.index:
        sig_pos = df_1m.index.get_loc(sig_idx)
        if sig_pos + 6 >= len(df_1m):
            continue
        spike_open = df_1m.iloc[sig_pos]['open_ask']
        next_5 = df_1m.iloc[sig_pos + 1:sig_pos + 6]
        # Check: all 5 bars' closes are above spike open (bullish follow-through)
        if (next_5['close_ask'] > spike_open).all():
            # Entry at the 6th bar (after confirmation)
            entry_idx = df_1m.index[sig_pos + 5]
            long_entries.append((entry_idx, df_1m.loc[entry_idx, 'close_ask']))

    print(f"  Bullish follow-through (all 5 bars > spike open): {len(long_entries)}")
    _, stats_a = test_pattern('vol_breakout', 'long', df_1m,
                              [e[0] for e in long_entries],
                              [e[1] for e in long_entries], TP, SL, MAX_BARS)
    print(f"  TP={TP}, SL={SL}: {stats_a['trades']} trades, {stats_a['pnl']:+.0f} pts, "
          f"{stats_a['wr']:.1f}% WR, PF={stats_a['pf']:.2f}, avg={stats_a['avg']:+.2f}/trade")

    # Also test without follow-through check (just vol >= 1500 → long)
    simple_entries = [(idx, df_1m.loc[idx, 'close_ask']) for idx in vol_spikes.index]
    _, stats_a_simple = test_pattern('vol_spike_simple', 'long', df_1m,
                                     [e[0] for e in simple_entries],
                                     [e[1] for e in simple_entries], TP, SL, MAX_BARS)
    print(f"  (Without follow-through): {stats_a_simple['trades']} trades, {stats_a_simple['pnl']:+.0f} pts, "
          f"{stats_a_simple['wr']:.1f}% WR, PF={stats_a_simple['pf']:.2f}")

    # =========================================================================
    # Pattern B: 15m 2-Up + 1m Retrace
    # =========================================================================
    print(f"\n{'='*60}")
    print("PATTERN B: 15m 2-Up + 1m Close < 15m Open (Retrace Long)")
    print(f"{'='*60}")

    # up_count3_15 >= 2 means at least 2 of last 3 15m bars are up
    # AND current 1m close < current 15m bar's open (intra-bar retrace)
    df_1m['intra_retrace'] = df_1m['close_ask'] < df_1m['open_15']
    b_mask = (df_1m['up_count3_15'] >= 2) & df_1m['intra_retrace']

    print(f"  up_count3>=2 & close<15m_open: {b_mask.sum():,} bars")

    b_indices = df_1m.index[b_mask]
    b_prices = [df_1m.loc[idx, 'close_ask'] for idx in b_indices]
    _, stats_b = test_pattern('_15m_2up_retrace', 'long', df_1m, b_indices, b_prices, TP, SL, MAX_BARS)
    print(f"  TP={TP}, SL={SL}: {stats_b['trades']} trades, {stats_b['pnl']:+.0f} pts, "
          f"{stats_b['wr']:.1f}% WR, PF={stats_b['pf']:.2f}, avg={stats_b['avg']:+.2f}/trade")

    # Also test: when exactly 2 up (not 3)
    b2_mask = (df_1m['up_count3_15'] == 2) & df_1m['intra_retrace']
    b2_indices = df_1m.index[b2_mask]
    b2_prices = [df_1m.loc[idx, 'close_ask'] for idx in b2_indices]
    _, stats_b2 = test_pattern('_15m_2up_exact', 'long', df_1m, b2_indices, b2_prices, TP, SL, MAX_BARS)
    print(f"  (up_count3==2 exactly): {stats_b2['trades']} trades, {stats_b2['pnl']:+.0f} pts, "
          f"{stats_b2['wr']:.1f}% WR, PF={stats_b2['pf']:.2f}")

    # When all 3 are up + retrace
    b3_mask = (df_1m['up_count3_15'] == 3) & df_1m['intra_retrace']
    b3_indices = df_1m.index[b3_mask]
    b3_prices = [df_1m.loc[idx, 'close_ask'] for idx in b3_indices]
    _, stats_b3 = test_pattern('_15m_3up_retrace', 'long', df_1m, b3_indices, b3_prices, TP, SL, MAX_BARS)
    print(f"  (up_count3==3 all up): {stats_b3['trades']} trades, {stats_b3['pnl']:+.0f} pts, "
          f"{stats_b3['wr']:.1f}% WR, PF={stats_b3['pf']:.2f}")

    # =========================================================================
    # Pattern C: 15m High Volume Bearish Bar → Short
    # =========================================================================
    print(f"\n{'='*60}")
    print("PATTERN C: 15m Volume > 10000 + Close < Open (Bearish Drop)")
    print(f"{'='*60}")

    # Find 15m bars with volume > 10000 and close < open
    c_15m = df_15[(df_15['volume'] > 10000) & (df_15['close_ask'] < df_15['open_ask'])]
    print(f"  15m bars vol>10000 & bearish: {len(c_15m):,}/{len(df_15):,}")

    # Entry: short at the NEXT 15m bar's open (first 1m bar of next 15m)
    c_entries = []
    for idx_15 in c_15m.index:
        next_15_start = idx_15 + pd.Timedelta(minutes=15)
        entry_candidates = df_1m.index[(df_1m.index >= next_15_start) &
                                        (df_1m.index < next_15_start + pd.Timedelta(minutes=1))]
        if len(entry_candidates) > 0:
            entry_idx = entry_candidates[0]
            c_entries.append((entry_idx, df_1m.loc[entry_idx, 'close_bid']))

    _, stats_c = test_pattern('_15m_vol_drop', 'short', df_1m,
                              [e[0] for e in c_entries],
                              [e[1] for e in c_entries], TP, SL, MAX_BARS)
    print(f"  TP={TP}, SL={SL}: {stats_c['trades']} trades, {stats_c['pnl']:+.0f} pts, "
          f"{stats_c['wr']:.1f}% WR, PF={stats_c['pf']:.2f}, avg={stats_c['avg']:+.2f}/trade")

    # Also test higher volume threshold
    for vol_thresh in [12000, 15000, 20000]:
        c_sub = df_15[(df_15['volume'] > vol_thresh) & (df_15['close_ask'] < df_15['open_ask'])]
        entries = []
        for idx_15 in c_sub.index:
            next_15_start = idx_15 + pd.Timedelta(minutes=15)
            ec = df_1m.index[(df_1m.index >= next_15_start) &
                              (df_1m.index < next_15_start + pd.Timedelta(minutes=1))]
            if len(ec) > 0:
                entries.append((ec[0], df_1m.loc[ec[0], 'close_bid']))
        _, s = test_pattern(f'vol>{vol_thresh}', 'short', df_1m,
                           [e[0] for e in entries], [e[1] for e in entries], TP, SL, MAX_BARS)
        print(f"  (vol>{vol_thresh}): {s['trades']}t, {s['pnl']:+.0f}pts, {s['wr']:.1f}% WR, PF={s['pf']:.2f}")

    # =========================================================================
    # Monthly breakdown for best pattern
    # =========================================================================
    print(f"\n{'='*60}")
    print("MONTHLY BREAKDOWN (Pattern A: vol breakout, TP=60/SL=40)")
    print(f"{'='*60}")

    months = pd.date_range("2024-02-01", "2026-06-01", freq="MS", tz="UTC")
    print(f"  {'Month':<10} {'Trades':>6} {'PnL':>10} {'WR':>7}")

    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        entries_m = [(idx, ep) for (idx, ep) in long_entries
                     if m_start <= idx <= m_end]
        if not entries_m:
            continue
        _, s = test_pattern('vol_breakout', 'long', df_1m,
                           [e[0] for e in entries_m],
                           [e[1] for e in entries_m], TP, SL, MAX_BARS)
        if s['trades'] > 0:
            print(f"  {str(m_start.date())[:7]:<10} {s['trades']:>6} {s['pnl']:>+10.1f} {s['wr']:>6.1f}%")

    # Monthly for Pattern B (up_count3>=2)
    print(f"\n{'='*60}")
    print("MONTHLY BREAKDOWN (Pattern B: 15m 2-up retrace, TP=60/SL=40)")
    print(f"{'='*60}")
    print(f"  {'Month':<10} {'Trades':>6} {'PnL':>10} {'WR':>7}")

    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        bm = b_mask & (df_1m.index >= m_start) & (df_1m.index <= m_end)
        if bm.sum() == 0: continue
        idxs = df_1m.index[bm]
        prs = [df_1m.loc[i, 'close_ask'] for i in idxs]
        _, s = test_pattern('_15m_2up_retrace', 'long', df_1m, idxs, prs, TP, SL, MAX_BARS)
        if s['trades'] > 0:
            print(f"  {str(m_start.date())[:7]:<10} {s['trades']:>6} {s['pnl']:>+10.1f} {s['wr']:>6.1f}%")

    # Monthly for Pattern C
    print(f"\n{'='*60}")
    print("MONTHLY BREAKDOWN (Pattern C: 15m vol>10k drop, TP=60/SL=40)")
    print(f"{'='*60}")
    print(f"  {'Month':<10} {'Trades':>6} {'PnL':>10} {'WR':>7}")

    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        cm = [(idx, ep) for (idx, ep) in c_entries
              if m_start <= idx <= m_end]
        if not cm: continue
        _, s = test_pattern('_15m_vol_drop', 'short', df_1m,
                           [e[0] for e in cm], [e[1] for e in cm], TP, SL, MAX_BARS)
        if s['trades'] > 0:
            print(f"  {str(m_start.date())[:7]:<10} {s['trades']:>6} {s['pnl']:>+10.1f} {s['wr']:>6.1f}%")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, s in [('A: Vol Breakout', stats_a), ('A simple (no filter)', stats_a_simple),
                     ('B: 15m 2-Up Retrace', stats_b), ('C: 15m Vol Drop', stats_c)]:
        spot = s['pnl'] / 100
        print(f"  {name:<25s}: {s['trades']:>4d}t, {s['pnl']:>+8.0f}pts (${spot:>+7.2f}), "
              f"{s['wr']:>5.1f}% WR, PF={s['pf']:.2f}, avg={s['avg']:>+.2f}/trade")

    print(f"\nDONE.")


if __name__ == '__main__':
    main()
