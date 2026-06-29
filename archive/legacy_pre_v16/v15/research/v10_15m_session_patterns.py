#!/usr/bin/env python3
"""
v10 15-Min Session Patterns — Archetype Scanner
================================================
Tests patterns on 15-min resampled bars, executed on 1-min ask/bid.

New ideas (not in v8/v9):
  1. 15-min bar patterns: micro_momentum, range_fade, doji_reversal on 15m bars
  2. Session open breakout with FVG (Fair Value Gap) confirmation
  3. Engulfing reversal at session opens (NY 12:00 UTC, Asia 16:00 UTC)

Signal logic runs on 15m bars; entries on next 1m bar; exit on 1m ask/bid.
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
# Data Loading
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
# 15-min Resampling
# =============================================================================

def resample_15m(df_1m):
    """Resample 1-min ask/bid bars to 15-min for signal generation.

    Returns DataFrame with 15-min OHLC (using close_ask as 'close').
    Also keeps high_bid/low_bid/high_ask/low_ask for proper exit simulation context.
    """
    df_15 = df_1m.resample('15min', label='left', closed='left').agg({
        'open_ask': 'first',
        'high_ask': 'max',
        'high_bid': 'max',
        'low_ask': 'min',
        'low_bid': 'min',
        'close_ask': 'last',
        'close_bid': 'last',
        'volume': 'sum',
        'spread': 'mean',
    })
    df_15['close'] = df_15['close_ask']
    df_15['open'] = df_15['open_ask']
    df_15 = df_15.dropna()
    return df_15


def compute_15m_indicators(df):
    """Indicators on 15-min bars."""
    df['ema20'] = df['close'].ewm(20).mean()
    df['ema50'] = df['close'].ewm(50).mean()
    df['ema200'] = df['close'].ewm(200).mean()
    df['ema50_slope'] = df['ema50'].diff(3)  # 3 bars on 15m = ~45min lookback

    tr = pd.concat([
        df['high_ask'] - df['low_ask'],
        abs(df['high_ask'] - df['close_ask'].shift()),
        abs(df['low_ask'] - df['close_ask'].shift())
    ], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()

    def _rsi(s, p):
        d = s.diff(); g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - 100 / (1 + g / l.replace(0, 1))
    for p in [5, 14]:
        df[f'rsi_{p}'] = _rsi(df['close'], p)

    df['range_high'] = df['high_ask'].rolling(20, min_periods=5).max()
    df['range_low'] = df['low_ask'].rolling(20, min_periods=5).min()
    df['pos_in_range'] = ((df['close'] - df['range_low']) /
                          (df['range_high'] - df['range_low'] + 0.001))

    df['body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high_ask'] - df['low_ask']
    df['body_ratio'] = df['body'] / (df['candle_range'] + 0.001)

    for n in [1, 3]:
        df[f'ret_{n}'] = df['close'].pct_change(n) * 100

    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # FVG: gap between current high and bar+2 low (bearish FVG) or current low and bar+2 high (bullish FVG)
    # An FVG exists when a gap forms between non-overlapping candles 2 bars apart
    df['fvg_bull'] = df['low_ask'] > df['high_ask'].shift(2)  # gap up: price jumped, left unfilled space
    df['fvg_bear'] = df['high_ask'] < df['low_ask'].shift(2)  # gap down
    df['fvg_bull_size'] = (df['low_ask'] - df['high_ask'].shift(2)) / (df['atr14'] + 0.01)
    df['fvg_bear_size'] = (df['low_ask'].shift(2) - df['high_ask']) / (df['atr14'] + 0.01)

    # Engulfing patterns
    po, ph, pl, pc = df['open'].shift(1), df['high_ask'].shift(1), df['low_ask'].shift(1), df['close'].shift(1)
    df['bull_engulf'] = (df['close'] > df['open']) & (pc < po) & (df['close'] >= po) & (df['open'] <= pc)
    df['bear_engulf'] = (df['close'] < df['open']) & (pc > po) & (df['close'] <= po) & (df['open'] >= pc)

    # Session flags (UTC hours)
    df['hour_utc'] = df.index.hour
    # NY open: 12:00-14:00 UTC (8:00-10:00 ET)
    df['is_ny_open'] = df['hour_utc'].isin([12, 13])
    # Asia open: 00:00-02:00 HKT = 16:00-18:00 UTC prior day
    df['is_asia_open'] = df['hour_utc'].isin([16, 17])
    # London open: 07:00-09:00 UTC
    df['is_london_open'] = df['hour_utc'].isin([7, 8])

    # Is this the first bar of the session?
    df['session_start'] = (df['hour_utc'].diff() != 0) & (
        df['is_ny_open'] | df['is_asia_open'] | df['is_london_open']
    )

    # Low-vol regime (same concept as v8, on 15m scale)
    low_vol = df['atr14'] < df['atr14'].rolling(200).mean() * 0.8  # vol < 80% of 200-bar avg
    weak_trend = df['ema50_slope'].abs() < df['atr14'] * 2
    df['in_regime'] = (low_vol & weak_trend).fillna(False)

    return df


# =============================================================================
# Signal Generators (on 15m bars)
# =============================================================================

def sig_15m_micro_momentum(df):
    """1-bar continuation on 15m (equivalent to 15-min micro trend)."""
    s = pd.DataFrame(index=df.index)
    s['long'] = (df['ret_1'] > 0.10) & (df['close'] > df['ema20']) & df['in_regime']
    s['short'] = (df['ret_1'] < -0.10) & (df['close'] < df['ema20']) & df['in_regime']
    return s


def sig_15m_range_fade(df):
    """20-bar range fade on 15m bars."""
    s = pd.DataFrame(index=df.index)
    s['long'] = (df['pos_in_range'] < 0.20) & df['in_regime']
    s['short'] = (df['pos_in_range'] > 0.80) & df['in_regime']
    return s


def sig_fvg_reversal(df):
    """FVG fill: when a fair value gap exists, trade toward filling it.

    Bullish FVG: low > high_2bars_ago => price should come back down to fill gap => SHORT at FVG top
    Bearish FVG: high < low_2bars_ago => price should come back up to fill gap => LONG at FVG bottom

    Entry only when price has started to reverse into the gap.
    """
    s = pd.DataFrame(index=df.index)
    # FVG detected 2 bars ago, now price moving back into gap
    fvg_bull_active = df['fvg_bull'].shift(1)  # gap exists above
    fvg_bear_active = df['fvg_bear'].shift(1)  # gap exists below

    # Enter short when bearish candle appears near bull FVG (fading the gap up)
    s['short'] = (fvg_bull_active & (df['close'] < df['open']) &
                  (df['close'] < df['high_ask'].shift(2)))  # started filling down

    # Enter long when bullish candle appears near bear FVG (fading the gap down)
    s['long'] = (fvg_bear_active & (df['close'] > df['open']) &
                 (df['close'] > df['low_ask'].shift(2)))  # started filling up

    return s


def sig_session_engulf(df):
    """Engulfing reversal at session opens (NY/Asia/London).

    At session open: engulfing candle in opposite direction of prior trend.
    Bullish engulf at session start => long.
    Bearish engulf at session start => short.
    """
    s = pd.DataFrame(index=df.index)
    at_session = df['is_ny_open'] | df['is_asia_open'] | df['is_london_open']
    s['long'] = at_session & df['bull_engulf'] & (df['close'] > df['ema20'])
    s['short'] = at_session & df['bear_engulf'] & (df['close'] < df['ema20'])
    return s


def sig_session_breakout(df):
    """Session open breakout: first bar of session breaks prior range.

    If NY opens above prior 5-bar high + bullish candle => long.
    If NY opens below prior 5-bar low + bearish candle => short.
    """
    s = pd.DataFrame(index=df.index)
    at_session_start = df['session_start']
    prior_high = df['high_ask'].shift(1).rolling(5).max()
    prior_low = df['low_ask'].shift(1).rolling(5).min()

    s['long'] = (at_session_start & (df['close'] > prior_high) &
                 (df['close'] > df['open']) & df['in_regime'])
    s['short'] = (at_session_start & (df['close'] < prior_low) &
                  (df['close'] < df['open']) & df['in_regime'])
    return s


def sig_fvg_engulf_combo(df):
    """FVG + engulfing combo at session opens (highest quality setup).

    Bullish: bear FVG present + bullish engulf at session open + ret_1 > 0
    Bearish: bull FVG present + bearish engulf at session open + ret_1 < 0
    """
    s = pd.DataFrame(index=df.index)
    at_session = df['is_ny_open'] | df['is_asia_open'] | df['is_london_open']
    s['long'] = (at_session & df['fvg_bear'].shift(1) & df['bull_engulf'] &
                 (df['ret_1'] > 0))
    s['short'] = (at_session & df['fvg_bull'].shift(1) & df['bear_engulf'] &
                  (df['ret_1'] < 0))
    return s


# =============================================================================
# Signal-to-1min mapping & Exit
# =============================================================================

MAX_BARS_1M = 45  # 15 bars * 3 on 1m for timeout (~45 min on 1m from 3x15m)

def map_15m_to_1m_signals(df_15, signal_15, df_1m):
    """Map 15-min bar signals to the first 1-min bar of the NEXT 15-min candle.

    Returns 1-min DataFrame with long/short columns.
    """
    sig_1m = pd.DataFrame(index=df_1m.index)
    sig_1m['long'] = False
    sig_1m['short'] = False

    for idx_15 in signal_15.index:
        if not signal_15.loc[idx_15, 'long'] and not signal_15.loc[idx_15, 'short']:
            continue
        # Find the first 1-min bar AFTER this 15-min candle closes
        next_15_start = idx_15 + pd.Timedelta(minutes=15)
        # Take first 1m bar in the next 15m window
        candidates = df_1m.index[(df_1m.index >= next_15_start) &
                                  (df_1m.index < next_15_start + pd.Timedelta(minutes=1))]
        if len(candidates) > 0:
            entry_idx = candidates[0]
            sig_1m.loc[entry_idx, 'long'] = signal_15.loc[idx_15, 'long']
            sig_1m.loc[entry_idx, 'short'] = signal_15.loc[idx_15, 'short']

    return sig_1m


def _build_forward_arrays(df, max_bars=45):
    n = len(df); N = max_bars
    def _s(col):
        out = np.full((n, N), np.nan, dtype=np.float64)
        vals = df[col].values
        for k in range(N):
            shift = k + 1
            out[:n-shift, k] = vals[shift:]
        return out
    return {
        'fwd_high_bid': _s('high_bid'), 'fwd_low_bid': _s('low_bid'),
        'fwd_high_ask': _s('high_ask'), 'fwd_low_ask': _s('low_ask'),
        'fwd_close_bid': _s('close_bid'), 'fwd_close_ask': _s('close_ask'),
    }


def vectorized_long_exit(entry_indices, entry_prices, fwd, tp, sl):
    n = len(entry_indices)
    pnls = np.full(n, np.nan, dtype=np.float64)
    if n == 0: return pnls
    stops, targets = entry_prices - sl, entry_prices + tp
    fwd_low = fwd['fwd_low_bid'][entry_indices]
    fwd_high = fwd['fwd_high_bid'][entry_indices]
    fwd_close = fwd['fwd_close_bid'][entry_indices]
    sl_hit = fwd_low <= stops[:, None]; tp_hit = fwd_high >= targets[:, None]
    sl_bar = np.argmax(sl_hit, axis=1); tp_bar = np.argmax(tp_hit, axis=1)
    N = fwd_low.shape[1]
    sl_bar[~sl_hit.any(axis=1)] = N; tp_bar[~tp_hit.any(axis=1)] = N
    for i in range(n):
        sbi, tbi = sl_bar[i], tp_bar[i]
        if sbi == N and tbi == N:
            lc = fwd_close[i, -1]; pnls[i] = lc - entry_prices[i] if not np.isnan(lc) else 0.0
        elif sbi <= tbi: pnls[i] = stops[i] - entry_prices[i]
        else: pnls[i] = targets[i] - entry_prices[i]
    return pnls


def vectorized_short_exit(entry_indices, entry_prices, fwd, tp, sl):
    n = len(entry_indices)
    pnls = np.full(n, np.nan, dtype=np.float64)
    if n == 0: return pnls
    stops, targets = entry_prices + sl, entry_prices - tp
    fwd_high = fwd['fwd_high_ask'][entry_indices]
    fwd_low = fwd['fwd_low_ask'][entry_indices]
    fwd_close = fwd['fwd_close_ask'][entry_indices]
    sl_hit = fwd_high >= stops[:, None]; tp_hit = fwd_low <= targets[:, None]
    sl_bar = np.argmax(sl_hit, axis=1); tp_bar = np.argmax(tp_hit, axis=1)
    N = fwd_low.shape[1]
    sl_bar[~sl_hit.any(axis=1)] = N; tp_bar[~tp_hit.any(axis=1)] = N
    for i in range(n):
        sbi, tbi = sl_bar[i], tp_bar[i]
        if sbi == N and tbi == N:
            lc = fwd_close[i, -1]; pnls[i] = entry_prices[i] - lc if not np.isnan(lc) else 0.0
        elif sbi <= tbi: pnls[i] = entry_prices[i] - stops[i]
        else: pnls[i] = entry_prices[i] - targets[i]
    return pnls


# =============================================================================
# Runner
# =============================================================================

ARCHETYPES = {
    '15m_micro_momentum': sig_15m_micro_momentum,
    '15m_range_fade': sig_15m_range_fade,
    'fvg_reversal': sig_fvg_reversal,
    'session_engulf': sig_session_engulf,
    'session_breakout': sig_session_breakout,
    'fvg_engulf_combo': sig_fvg_engulf_combo,
}

TP_GRID = [8, 12, 15, 20, 25, 30]
SL_GRID = [8, 12, 15, 20, 25, 30]


def test_archetype(name, signal_fn, df_15, df_1m, fwd_1m, period_masks_1m):
    signals_15 = signal_fn(df_15)
    signals_1m = map_15m_to_1m_signals(df_15, signals_15, df_1m)
    results = []

    for tp, sl in product(TP_GRID, SL_GRID):
        per_period = defaultdict(lambda: {'pnls': [], 'longs': 0, 'shorts': 0})
        for pname, pmask in period_masks_1m.items():
            pmask_idx = df_1m.index[pmask]
            sig_sub = signals_1m.loc[signals_1m.index.isin(pmask_idx)]
            if len(sig_sub) == 0: continue

            long_mask = sig_sub['long']
            if long_mask.any():
                li = np.array([df_1m.index.get_loc(i) for i in sig_sub.index[long_mask]], dtype=np.int64)
                lp = df_1m['close_ask'].values[li]
                lpnls = vectorized_long_exit(li, lp, fwd_1m, tp, sl)
                lpnls = lpnls[~np.isnan(lpnls)]
                per_period[pname]['pnls'].extend(lpnls.tolist())
                per_period[pname]['longs'] += len(lpnls)

            short_mask = sig_sub['short']
            if short_mask.any():
                si = np.array([df_1m.index.get_loc(i) for i in sig_sub.index[short_mask]], dtype=np.int64)
                sp = df_1m['close_bid'].values[si]
                spnls = vectorized_short_exit(si, sp, fwd_1m, tp, sl)
                spnls = spnls[~np.isnan(spnls)]
                per_period[pname]['pnls'].extend(spnls.tolist())
                per_period[pname]['shorts'] += len(spnls)

        all_pnls, tl, ts = [], 0, 0
        for st in per_period.values():
            all_pnls.extend(st['pnls']); tl += st['longs']; ts += st['shorts']
        if len(all_pnls) < 3: continue

        a = np.array(all_pnls); n = len(a); total = a.sum(); wr = (a > 0).mean() * 100
        pos_sum = a[a > 0].sum(); neg_sum = abs(a[a < 0].sum())
        pf = pos_sum / neg_sum if neg_sum > 0 else 99
        mj_pnls = per_period.get('may_jun_2026', {}).get('pnls', [])
        mj_total = sum(mj_pnls) if mj_pnls else 0
        results.append({
            'archetype': name, 'tp': tp, 'sl': sl,
            'trades': n, 'pnl': round(total, 1), 'wr': round(wr, 1), 'pf': round(pf, 2),
            'avg_pts': round(total / n, 2), 'longs': tl, 'shorts': ts,
            'mj_trades': len(mj_pnls), 'mj_pnl': round(mj_total, 1),
        })
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 72)
    print("v10 15-Min Session Pattern Scanner")
    print("=" * 72)

    print("\n[1/4] Loading 1-min data...")
    df_1m = load_askbid_data()
    print(f"  {len(df_1m):,} bars")

    print("[2/4] Resampling to 15-min & computing indicators...")
    df_15 = resample_15m(df_1m)
    df_15 = compute_15m_indicators(df_15)
    df_15 = df_15.dropna(subset=['atr14', 'ema50_slope', 'rsi_5', 'body_ratio'])
    df_1m = df_1m.dropna(subset=['close_ask', 'close_bid'])  # ensure 1m is clean
    print(f"  15-min: {len(df_15):,} bars, {df_15['in_regime'].sum():,} in-regime")

    # Match date ranges
    common_start = max(df_15.index[0], df_1m.index[0])
    common_end = min(df_15.index[-1], df_1m.index[-1])
    df_15 = df_15[(df_15.index >= common_start) & (df_15.index <= common_end)]
    df_1m = df_1m[(df_1m.index >= common_start) & (df_1m.index <= common_end)]

    # Period masks on 1-min index
    period_masks_1m = {
        'full': df_1m.index >= '2025-09-01',
        'pre_may_2026': (df_1m.index >= '2025-09-01') & (df_1m.index < '2026-05-01'),
        'may_jun_2026': (df_1m.index >= '2026-05-01') & (df_1m.index <= '2026-06-09'),
    }
    for nm, mk in period_masks_1m.items():
        n = mk.sum()
        print(f"  {nm}: {n:,} 1-min bars")

    print("[3/4] Building forward arrays...")
    fwd_1m = _build_forward_arrays(df_1m, MAX_BARS_1M)

    print(f"\n[4/4] Testing {len(ARCHETYPES)} archetypes ({len(TP_GRID)*len(SL_GRID)} TPxSL combos each)...")
    all_results = []
    for arch_name, sig_fn in ARCHETYPES.items():
        print(f"  {arch_name}...", end=" ", flush=True)
        res = test_archetype(arch_name, sig_fn, df_15, df_1m, fwd_1m, period_masks_1m)
        all_results.extend(res)
        if res:
            best = max(res, key=lambda r: r['pnl'])
            print(f"{len(res)} cfg, best={best['tp']}/{best['sl']} -> {best['trades']}t, {best['pnl']:+.1f}pt")
        else:
            print("NO TRADES")

    if not all_results:
        print("\nNo trades. Signals too sparse or thresholds too strict.")
        return

    RD = pd.DataFrame(all_results)

    # Top 20
    print(f"\n{'='*90}")
    print("TOP 20 CONFIGURATIONS")
    print(f"{'='*90}")
    top = RD.nlargest(20, 'pnl')
    print(f"{'Archetype':<24s} {'TP':>4s} {'SL':>4s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'Avg':>7s} {'L':>5s} {'S':>5s} {'MJ_T':>5s} {'MJ_PnL':>8s}")
    print("-" * 90)
    for _, r in top.iterrows():
        print(f"{r['archetype']:<24s} {r['tp']:>4.0f} {r['sl']:>4.0f} "
              f"{r['trades']:>7.0f} {r['pnl']:>+10.1f} {r['wr']:>6.1f}% "
              f"{r['pf']:>5.2f} {r['avg_pts']:>+7.2f} "
              f"{r['longs']:>5.0f} {r['shorts']:>5.0f} "
              f"{r['mj_trades']:>5.0f} {r['mj_pnl']:>+8.1f}")

    # Best per archetype
    print(f"\n{'='*90}")
    print("BEST PER ARCHETYPE")
    print(f"{'='*90}")
    for arch in ARCHETYPES:
        sub = RD[RD['archetype'] == arch]
        if len(sub) == 0:
            print(f"  {arch:<24s}: NO TRADES")
            continue
        b = sub.loc[sub['pnl'].idxmax()]
        print(f"  {arch:<24s}: TP={b['tp']:>4.0f} SL={b['sl']:>4.0f} "
              f"{b['trades']:>5.0f}t, {b['pnl']:>+8.1f}pt, {b['wr']:>5.1f}% WR, "
              f"PF={b['pf']:.2f}, avg={b['avg_pts']:>+.2f}/t, "
              f"L:{b['longs']:.0f} S:{b['shorts']:.0f}, MJ:{b['mj_trades']:.0f}t/{b['mj_pnl']:+.1f}pt")

    # May-June focus
    print(f"\n{'='*90}")
    print("TOP 10 BY MAY-JUNE PNL")
    print(f"{'='*90}")
    mj_valid = RD[RD['mj_trades'] > 0]
    if len(mj_valid):
        for _, r in mj_valid.nlargest(10, 'mj_pnl').iterrows():
            print(f"  {r['archetype']:<24s} TP={r['tp']:>4.0f} SL={r['sl']:>4.0f} "
                  f"MJ: {r['mj_trades']:>4.0f}t {r['mj_pnl']:>+8.1f}pt | "
                  f"All: {r['trades']:>5.0f}t {r['pnl']:>+8.1f}pt, {r['wr']:>5.1f}% WR")

    # Signal frequency on 15m bars
    print(f"\n{'='*90}")
    print("15-MIN SIGNAL FREQUENCY (full period)")
    print(f"{'='*90}")
    for arch, fn in ARCHETYPES.items():
        sig = fn(df_15)
        l = sig['long'].sum(); s = sig['short'].sum()
        print(f"  {arch:<24s}: {l:>5}L, {s:>5}S = {l+s:>5} signals on 15m (→ {1 if l+s>0 else 0} on 1m)")

    # Session distribution
    print(f"\n{'='*90}")
    print("SESSION DISTRIBUTION (15m bars)")
    print(f"{'='*90}")
    for session in ['is_ny_open', 'is_asia_open', 'is_london_open']:
        n = df_15[session].sum()
        print(f"  {session}: {n} bars")
    n_sess_start = df_15['session_start'].sum()
    print(f"  session_start: {n_sess_start} bars")
    n_fvg_bull = df_15['fvg_bull'].sum()
    n_fvg_bear = df_15['fvg_bear'].sum()
    n_engulf_bull = df_15['bull_engulf'].sum()
    n_engulf_bear = df_15['bear_engulf'].sum()
    print(f"  FVG bull: {n_fvg_bull}, FVG bear: {n_fvg_bear}")
    print(f"  Bull engulf: {n_engulf_bull}, Bear engulf: {n_engulf_bear}")

    print(f"\n{'='*90}")
    print("DONE.")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
