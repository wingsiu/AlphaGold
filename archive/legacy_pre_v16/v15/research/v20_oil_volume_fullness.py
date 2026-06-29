#!/usr/bin/env python3
"""
v20 Oil Volume + Fullness Pattern
===================================
Oil-native pattern based on:
  1. Volume spike: bar with volume >= 1500
  2. Fullness: (close - day_open) / avg_5day_range < -0.2 (bearish day)
     BUT close - day_open > -80 (not TOO bearish — limit to controlled pullback)
  3. Long entry: after 1-min bar meets both conditions
  4. TP=60, SL=40

Theory: high-volume pullback that's significant relative to recent range
but not crashing → reversal long.
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

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def compute_signals(df):
    """Compute daily ranges and fullness signal."""
    # Daily OHLC using ask
    daily = df.resample('D').agg({
        'open_ask': 'first',
        'high_ask': 'max',
        'low_ask': 'min',
        'close_ask': 'last',
    }).dropna()

    daily['range'] = daily['high_ask'] - daily['low_ask']
    daily['avg_range_5d'] = daily['range'].rolling(5, min_periods=3).mean()

    # Map daily metrics back to 1-min bars
    df['day_open'] = np.nan
    df['day_avg_range_5d'] = np.nan
    for day_idx in daily.index:
        mask = df.index.date == day_idx.date()
        df.loc[mask, 'day_open'] = daily.loc[day_idx, 'open_ask']
        df.loc[mask, 'day_avg_range_5d'] = daily.loc[day_idx, 'avg_range_5d']

    df['close_minus_open'] = df['close_ask'] - df['day_open']
    df['fullness'] = df['close_minus_open'] / (df['day_avg_range_5d'] + 0.01)

    # Signal: volume >= 1500 AND fullness < -0.2 AND close-close_open > -80
    df['vol_spike'] = df['volume'] >= 1500
    df['fullness_signal'] = (df['fullness'] < -0.2) & (df['close_minus_open'] > -80)
    df['signal'] = df['vol_spike'] & df['fullness_signal']

    return df, daily


def sim_long(entry_idx, entry_price, df, tp, sl, max_bars=60):
    """Bar-by-bar LONG: entry at close_ask, exit on bid."""
    stop, target = entry_price - sl, entry_price + tp
    horizon = min(max_bars, len(df) - entry_idx - 1)
    for i in range(1, horizon + 1):
        bar = df.iloc[entry_idx + i]
        if bar['low_bid'] <= stop:    return stop, i, 'sl'
        if bar['high_bid'] >= target: return target, i, 'tp'
    return df.iloc[entry_idx + horizon]['close_bid'], horizon, 'timeout'


def main():
    TP, SL, MAX_BARS = 60, 40, 60
    print("=" * 72)
    print("v20 Oil Volume + Fullness Pattern")
    print(f"  Rule: volume>=1500 + fullness< -0.2 + close-open > -80")
    print(f"  Entry: LONG at close_ask, TP={TP}, SL={SL}, MaxBars={MAX_BARS}")
    print("=" * 72)

    print("\n[1/3] Loading data...")
    df = load_oil_data()
    print(f"  {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")

    print("[2/3] Computing signals...")
    df, daily = compute_signals(df)

    n_vol = df['vol_spike'].sum()
    n_fullness = df['fullness_signal'].sum()
    n_signal = df['signal'].sum()
    print(f"  Volume >= 1500: {n_vol:,} bars ({n_vol/len(df)*100:.2f}%)")
    print(f"  Fullness signal: {n_fullness:,} bars")
    print(f"  Combined signals: {n_signal:,} bars")

    # Daily range stats
    print(f"  Daily range (5d avg): mean={daily['avg_range_5d'].mean():.0f}, "
          f"median={daily['avg_range_5d'].median():.0f}, "
          f"min={daily['avg_range_5d'].min():.0f}, max={daily['avg_range_5d'].max():.0f}")

    # =========================================================================
    # [3/3] Walk-forward by month
    # =========================================================================
    print(f"\n[3/3] Walk-forward evaluation")
    print(f"\n{'Month':<10} {'Signals':>7} {'Trades':>7} {'PnL':>10} {'WR':>7} {'SL':>5} {'TP':>5} {'TO':>5}")
    print("-" * 65)

    months = pd.date_range("2024-02-01", "2026-06-01", freq="MS", tz="UTC")
    all_trades = []
    total_signals = 0

    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        mask = (df.index >= m_start) & (df.index <= m_end) & df['signal']
        n_sig = mask.sum()
        total_signals += n_sig
        if n_sig == 0:
            continue

        # Take all signals (no model — this is a rule-based pattern)
        signal_indices = df.index[mask]
        pnls = []; reasons = {'sl': 0, 'tp': 0, 'timeout': 0}

        for sig_time in signal_indices:
            entry_idx = df.index.get_loc(sig_time)
            if entry_idx + MAX_BARS >= len(df):
                continue
            entry_price = df.iloc[entry_idx]['close_ask']
            exit_price, bars, reason = sim_long(entry_idx, entry_price, df, TP, SL, MAX_BARS)
            pnl = exit_price - entry_price
            pnls.append(pnl)
            reasons[reason] += 1
            all_trades.append({
                'month': str(m_start.date())[:7],
                'pnl': pnl,
                'reason': reason,
                'volume': df.iloc[entry_idx]['volume'],
                'fullness': df.iloc[entry_idx]['fullness'],
                'close_minus_open': df.iloc[entry_idx]['close_minus_open'],
            })

        if pnls:
            n = len(pnls)
            wr = sum(1 for p in pnls if p > 0) / n * 100
            total = sum(pnls)
            print(f"  {str(m_start.date())[:7]:<10} {n_sig:>7} {n:>7} {total:>+10.1f} {wr:>6.1f}% {reasons['sl']:>5} {reasons['tp']:>5} {reasons['timeout']:>5}")

    # =========================================================================
    # Results
    # =========================================================================
    TD = pd.DataFrame(all_trades)

    print(f"\n{'='*65}")
    print("FINAL RESULTS")
    print(f"{'='*65}")

    if len(TD) == 0:
        print("NO TRADES.")
        return

    tot = len(TD); tot_pnl = TD['pnl'].sum()
    wr = (TD['pnl'] > 0).mean() * 100
    pos = TD[TD['pnl'] > 0]['pnl'].sum()
    neg = abs(TD[TD['pnl'] < 0]['pnl'].sum())
    pf = pos / neg if neg > 0 else 99
    avg_pnl = tot_pnl / tot

    print(f"  Total signals: {total_signals:,}")
    print(f"  Total trades: {tot}")
    print(f"  PnL: {tot_pnl:+.0f} pts (~${tot_pnl/100:+.2f}/contract)")
    print(f"  Win rate: {wr:.1f}%")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Avg per trade: {avg_pnl:+.2f} pts")

    # Reason breakdown
    for r in ['tp', 'sl', 'timeout']:
        cnt = (TD['reason'] == r).sum()
        if cnt:
            sub = TD[TD['reason'] == r]
            print(f"  {r}: {cnt}t ({cnt/tot*100:.0f}%), PnL={sub['pnl'].sum():+.0f}, "
                  f"WR={(sub['pnl']>0).mean()*100:.0f}%")

    # Monthly breakdown
    print(f"\n  Monthly:")
    for m in sorted(TD['month'].unique()):
        ms = TD[TD['month'] == m]
        if len(ms):
            print(f"    {m}: {len(ms):>3}t, {ms['pnl'].sum():>+10.0f}, {(ms['pnl']>0).mean()*100:>5.0f}% WR")

    # =========================================================================
    # Parameter sweep: test adjacent thresholds
    # =========================================================================
    print(f"\n{'='*65}")
    print("THRESHOLD SENSITIVITY (quick sweep on full period)")
    print(f"{'='*65}")
    print(f"  Varying fullness and close-open thresholds:")
    print(f"  {'Fullness':<12s} {'Close-Open':>12s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s}")

    df['signal_all'] = df['volume'] >= 1500
    signal_mask = df['signal_all']
    signal_indices_all = df.index[signal_mask]

    for full_thresh in [-0.15, -0.20, -0.25, -0.30]:
        for co_thresh in [-60, -70, -80, -90, -100]:
            pnls = []
            for sig_time in signal_indices_all:
                fullness = df.loc[sig_time, 'fullness']
                co = df.loc[sig_time, 'close_minus_open']
                if not (fullness < full_thresh and co > co_thresh):
                    continue
                entry_idx = df.index.get_loc(sig_time)
                if entry_idx + MAX_BARS >= len(df):
                    continue
                ep = df.iloc[entry_idx]['close_ask']
                ex, _, _ = sim_long(entry_idx, ep, df, TP, SL, MAX_BARS)
                pnls.append(ex - ep)

            if len(pnls) > 3:
                a = np.array(pnls)
                n = len(a); t = a.sum(); w = (a > 0).mean() * 100
                print(f"  full<{full_thresh:+.2f}      co>{co_thresh:>4}       {n:>7d} {t:>+10.1f} {w:>6.1f}%")

    # =========================================================================
    # TP/SL sweep
    # =========================================================================
    print(f"\n{'='*65}")
    print("TP/SL SWEEP")
    print(f"{'='*65}")
    print(f"  {'TP':>4s} {'SL':>4s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'Avg':>8s}")

    # Use all combined signals
    comb_mask = df['signal']
    comb_indices = df.index[comb_mask]

    tps = [40, 50, 60, 70, 80]
    sls = [30, 40, 50, 60]
    best = {'pnl': -99999}

    for tp in tps:
        for sl in sls:
            pnls = []
            for sig_time in comb_indices:
                ei = df.index.get_loc(sig_time)
                if ei + MAX_BARS >= len(df): continue
                ep = df.iloc[ei]['close_ask']
                ex, _, _ = sim_long(ei, ep, df, tp, sl, MAX_BARS)
                pnls.append(ex - ep)

            if len(pnls) > 3:
                a = np.array(pnls)
                n = len(a); t = a.sum(); w = (a > 0).mean() * 100
                p = a[a > 0].sum(); negsum = abs(a[a < 0].sum())
                pf_val = p / negsum if negsum > 0 else 99
                print(f"  {tp:>4.0f} {sl:>4.0f} {n:>7d} {t:>+10.1f} {w:>6.1f}% {pf_val:>5.2f} {t/n:>+8.2f}")
                if t > best['pnl']:
                    best = {'tp': tp, 'sl': sl, 'pnl': t, 'trades': n, 'wr': w, 'pf': pf_val, 'avg': t/n}

    print(f"\n  Best TP/SL: {best['tp']}/{best['sl']} -> {best['trades']}t, {best['pnl']:+.0f}pts, "
          f"{best['wr']:.1f}% WR, PF={best['pf']:.2f}, avg={best['avg']:+.2f}/trade")

    # =========================================================================
    # May-June 2026 check
    # =========================================================================
    mj = TD[TD['month'].isin(['2026-05', '2026-06'])]
    if len(mj):
        mj_pnl = mj['pnl'].sum()
        mj_wr = (mj['pnl'] > 0).mean() * 100
        print(f"\n  May-June 2026: {len(mj)}t, {mj_pnl:+.0f} pts, {mj_wr:.0f}% WR")
    else:
        print(f"\n  May-June 2026: 0 trades")

    print(f"\nDONE.")


if __name__ == '__main__':
    main()
