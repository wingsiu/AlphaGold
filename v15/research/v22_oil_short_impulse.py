#!/usr/bin/env python3
"""
v22 Oil Short Impulse — Port oil_trader's best pattern to AlphaGold
=====================================================================
Ported from oil_trader/short_impulse_signals.py + backtester.py

Uses prev bar's change (last completed bar) instead of current bar
to avoid lookahead bias.

Rules (v2 IMPROVED config):
  - prev bar change < -16 (last completed bar had big bearish drop)
  - 2nd-prev bar change between -14 and +10
  - prev bar lower wick < 35
  - prev bar volume > 1100
  - up_count3_15min != -3 (not ALL 3 prior 15m bars down)
  - dist from day high < 180
  - spread <= 4.25, ATR <= 8.0
  - <= 8 short impulse signals in last 60 bars
  - US or UK session
  - Entry: SHORT at close_bid, TP=70, SL=40
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
    df['close'] = df['close_ask']
    df['volume'] = raw['lastTradedVolume'].astype(float)
    df['spread'] = df['close_ask'] - df['close_bid']

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def compute_features(df_1m):
    """Compute all short_impulse features using PREV bar data."""
    df = df_1m.copy()

    df['change'] = df['close_ask'] - df['open_ask']
    df['prev_change'] = df['change'].shift(1)
    df['prev2_change'] = df['change'].shift(2)
    df['prev_lower_wick'] = (df['close_ask'].shift(1) - df['low_ask'].shift(1))
    df['prev_volume'] = df['volume'].shift(1)

    # ATR
    tr = pd.concat([
        df['high_ask'] - df['low_ask'],
        abs(df['high_ask'] - df['close_ask'].shift()),
        abs(df['low_ask'] - df['close_ask'].shift())
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # Day high
    daily_high = df['high_ask'].resample('D').max()
    df['day_high'] = np.nan
    for day_idx in daily_high.index:
        mask = df.index.date == day_idx.date()
        df.loc[mask, 'day_high'] = daily_high.loc[day_idx]
    df['dist_day_high'] = df['day_high'] - df['close_ask']

    # 15m up_count3
    df_15 = df.resample('15min', label='right', closed='right').agg({
        'open_ask': 'first', 'close_ask': 'last',
    }).dropna()
    df_15['up'] = 0
    df_15.loc[df_15['close_ask'] > df_15['open_ask'], 'up'] = 1
    df_15.loc[df_15['close_ask'] < df_15['open_ask'], 'up'] = -1
    df_15['up_count3'] = df_15['up'].rolling(3, min_periods=1).sum()

    df['up_count3_15min'] = np.nan
    for idx_15 in df_15.index:
        next_start = idx_15 + pd.Timedelta(minutes=15)
        mask = (df.index >= idx_15) & (df.index < next_start)
        df.loc[mask, 'up_count3_15min'] = df_15.loc[idx_15, 'up_count3']

    # Sessions
    df['is_us'] = df.index.hour.isin([12, 13, 14, 15, 16, 17, 18, 19, 20])
    df['is_uk'] = df.index.hour.isin([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    df['in_session'] = df['is_us'] | df['is_uk']

    return df


def generate_signals(df, cfg):
    """Short impulse using PREV bar's completed data."""
    mask = (
        (df['prev_change'] < cfg['change_max'])
        & (df['prev2_change'] < cfg['prev_change_max'])
        & (df['prev2_change'] > cfg['prev_change_min'])
        & (df['prev_lower_wick'] < cfg['lower_wick_max'])
        & (df['prev_volume'] > cfg['volume_min'])
        & (df['up_count3_15min'] != -3)
        & (df['dist_day_high'] < cfg['dist_high_max'])
    )

    if cfg.get('spread_max') is not None:
        mask &= df['spread'] <= cfg['spread_max']
    if cfg.get('atr_max') is not None:
        mask &= df['ATR'] <= cfg['atr_max']

    if cfg.get('impulse_lookback_bars') and cfg.get('impulse_lookback_max') is not None:
        recent = mask.astype(int).rolling(cfg['impulse_lookback_bars'], min_periods=1).sum()
        mask &= recent.shift(1).fillna(0) <= cfg['impulse_lookback_max']

    mask &= df['in_session']
    return mask


def sim_short(ei, ep, df, tp, sl, max_bars=60):
    stop, target = ep + sl, ep - tp
    horizon = min(max_bars, len(df) - ei - 1)
    for i in range(1, horizon + 1):
        b = df.iloc[ei + i]
        if b['high_ask'] >= stop: return stop, i, 'sl'
        if b['low_ask'] <= target: return target, i, 'tp'
    return df.iloc[ei + horizon]['close_ask'], horizon, 'timeout'


def evaluate(signal_mask, df, tp, sl, max_bars=60):
    trades = []
    signal_indices = df.index[signal_mask]
    for sig_idx in signal_indices:
        ei = df.index.get_loc(sig_idx)
        if ei + max_bars >= len(df):
            continue
        ep = df.iloc[ei]['close_bid']
        ex, bars, r = sim_short(ei, ep, df, tp, sl, max_bars)
        pnl = ep - ex
        trades.append({'pnl': pnl, 'reason': r})
    return trades


def main():
    TP, SL, MAX_BARS = 70, 40, 60
    print("=" * 72)
    print("v22 Oil Short Impulse (prev-bar change, no lookahead)")
    print(f"  TP={TP}, SL={SL}")
    print("=" * 72)

    print("\n[1/2] Loading & features...")
    df = load_oil_data()
    df = compute_features(df)
    df = df.dropna(subset=['ATR', 'day_high', 'up_count3_15min', 'prev_change', 'spread'])
    print(f"  {len(df):,} bars ready")

    BASELINE = {
        'change_max': -14.0, 'prev_change_max': 10.0, 'prev_change_min': -14.0,
        'lower_wick_max': 35.0, 'volume_min': 1000.0, 'dist_high_max': 180.0,
        'spread_max': None, 'atr_max': None,
        'impulse_lookback_bars': None, 'impulse_lookback_max': None,
    }
    V2 = {
        'change_max': -16.0, 'prev_change_max': 10.0, 'prev_change_min': -14.0,
        'lower_wick_max': 35.0, 'volume_min': 1100.0, 'dist_high_max': 180.0,
        'spread_max': 4.25, 'atr_max': 8.0,
        'impulse_lookback_bars': 60, 'impulse_lookback_max': 8,
    }
    V3 = {
        'change_max': -16.0, 'prev_change_max': 10.0, 'prev_change_min': -14.0,
        'lower_wick_max': 35.0, 'volume_min': 1200.0, 'dist_high_max': 180.0,
        'spread_max': 4.00, 'atr_max': 6.5,
        'impulse_lookback_bars': 45, 'impulse_lookback_max': 5,
    }

    print(f"\n[2/2] Testing configs...")
    print(f"  {'Config':<12s} {'Signals':>7s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'Avg':>8s}")
    print(f"  {'-'*65}")

    best_name, best_cfg, best_pnl = None, None, -99999
    all_sigs = {}
    for name, cfg in [('baseline', BASELINE), ('v2', V2), ('v3', V3)]:
        sig = generate_signals(df, cfg)
        all_sigs[name] = sig
        trades = evaluate(sig, df, TP, SL, MAX_BARS)
        pnls = [t['pnl'] for t in trades]
        if len(pnls) > 3:
            n = len(pnls); total = sum(pnls); wr = sum(1 for p in pnls if p > 0) / n * 100
            p_sum = sum(p for p in pnls if p > 0)
            n_sum = abs(sum(p for p in pnls if p < 0))
            pf = p_sum / n_sum if n_sum > 0 else 99
            print(f"  {name:<12s} {sig.sum():>7,d} {n:>7d} {total:>+10.1f} {wr:>6.1f}% {pf:>5.2f} {total/n:>+8.2f}")
            if total > best_pnl:
                best_pnl = total; best_name = name; best_cfg = cfg
        else:
            print(f"  {name:<12s} {sig.sum():>7,d} <5 trades")

    if not best_name:
        print("No config produced trades.")
        return

    sig = all_sigs[best_name]

    # Monthly
    print(f"\n  Monthly ({best_name}):")
    months = pd.date_range("2024-02-01", "2026-06-01", freq="MS", tz="UTC")
    print(f"  {'Month':<10} {'Trades':>6} {'PnL':>10} {'WR':>7} {'SL':>5} {'TP':>5} {'TO':>5}")
    print(f"  {'-'*55}")

    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        m_mask = sig & (df.index >= m_start) & (df.index <= m_end)
        if m_mask.sum() == 0: continue
        trades = evaluate(m_mask, df, TP, SL, MAX_BARS)
        if not trades: continue
        n = len(trades); pnls = [t['pnl'] for t in trades]
        reasons = {'sl': sum(1 for t in trades if t['reason']=='sl'),
                   'tp': sum(1 for t in trades if t['reason']=='tp'),
                   'timeout': sum(1 for t in trades if t['reason']=='timeout')}
        total = sum(pnls); wr = sum(1 for p in pnls if p > 0) / n * 100
        print(f"  {str(m_start.date())[:7]:<10} {n:>6} {total:>+10.1f} {wr:>6.1f}% {reasons['sl']:>5} {reasons['tp']:>5} {reasons['timeout']:>5}")

    # TP/SL sweep
    print(f"\n  TP/SL sweep ({best_name}):")
    print(f"  {'TP':>4s} {'SL':>4s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'Avg':>8s}")
    print(f"  {'-'*55}")
    for tp in [50, 60, 70, 80, 90]:
        for sl in [30, 40, 50, 60]:
            trades = evaluate(sig, df, tp, sl, MAX_BARS)
            pnls = [t['pnl'] for t in trades]
            if len(pnls) < 5: continue
            n = len(pnls); total = sum(pnls); wr = sum(1 for p in pnls if p > 0) / n * 100
            p_sum = sum(p for p in pnls if p > 0)
            n_sum = abs(sum(p for p in pnls if p < 0))
            pf = p_sum / n_sum if n_sum > 0 else 99
            print(f"  {tp:>4.0f} {sl:>4.0f} {n:>7d} {total:>+10.1f} {wr:>6.1f}% {pf:>5.2f} {total/n:>+8.2f}")

    # By year
    print(f"\n  By year ({best_name}, TP=70/SL=40):")
    for year in [2024, 2025, 2026]:
        y_mask = sig & (df.index.year == year)
        trades = evaluate(y_mask, df, TP, SL, MAX_BARS)
        pnls = [t['pnl'] for t in trades]
        if pnls:
            n = len(pnls); total = sum(pnls); wr = sum(1 for p in pnls if p > 0) / n * 100
            print(f"    {year}: {n}t, {total:+.0f}pts (${total/100:+.2f}), {wr:.1f}% WR")

    # May-June 2026
    mj_mask = sig & (df.index >= '2026-05-01') & (df.index <= '2026-06-09')
    trades = evaluate(mj_mask, df, TP, SL, MAX_BARS)
    if trades:
        pnls = [t['pnl'] for t in trades]
        n = len(pnls); total = sum(pnls); wr = sum(1 for p in pnls if p > 0) / n * 100
        print(f"\n  May-June 2026: {n}t, {total:+.0f}pts (${total/100:+.2f}), {wr:.1f}% WR")
    else:
        print(f"\n  May-June 2026: 0 trades")

    print(f"\nDONE.")


if __name__ == '__main__':
    main()
