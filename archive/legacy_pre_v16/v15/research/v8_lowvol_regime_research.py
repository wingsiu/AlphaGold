#!/usr/bin/env python3
"""
v8 Low-Vol Regime Archetype Scanner (v15 Edition)
===================================================
Tests simple rule-based pattern archetypes on low-volatility regimes
where v5/v6 retrace patterns don't fire (May-June 2026).

Uses v15 infrastructure: RegimeDetector for low-vol gate, v15 config.
NO XGBoost — pure rule-based patterns, bar-by-bar ask/bid simulation.

Archetypes:
  1. Mean Reversion (RSI extremes)
  2. Range Fade (position-in-range)
  3. Bollinger Band Touch
  4. Inside Bar Breakout
  5. Micro Momentum (3-bar continuation)
  6. Doji Reversal

TPxSL grid sweep on each. Results ranked for v9 model build.
"""
import sys
from pathlib import Path

# v15/_paths.py path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
from data.data_loader import DataLoader


# =============================================================================
# Data Loading (ask/bid aware — same pattern as v5/v6)
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
    df['close'] = df['close_ask']  # indicator calc convenience
    df['volume'] = raw['lastTradedVolume'].astype(float)
    df['spread'] = df['close_ask'] - df['close_bid']
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


# =============================================================================
# Indicators (single pass for speed)
# =============================================================================

def compute_indicators(df):
    df['ema20'] = df['close'].ewm(20).mean()
    df['ema50'] = df['close'].ewm(50).mean()
    df['ema200'] = df['close'].ewm(200).mean()
    df['ema50_slope'] = df['ema50'].diff(10)

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

    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['ema20'] + 2 * df['bb_std']
    df['bb_lower'] = df['ema20'] - 2 * df['bb_std']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.001)

    df['range_high'] = df['high_ask'].rolling(50, min_periods=10).max()
    df['range_low'] = df['low_ask'].rolling(50, min_periods=10).min()
    df['pos_in_range'] = ((df['close'] - df['range_low']) /
                          (df['range_high'] - df['range_low'] + 0.001))

    df['body'] = abs(df['close_ask'] - df['open_ask'])
    df['candle_range'] = df['high_ask'] - df['low_ask']
    df['body_ratio'] = df['body'] / (df['candle_range'] + 0.001)

    for n in [3, 5]:
        df[f'ret_{n}'] = df['close'].pct_change(n) * 100

    df['vol_ratio'] = df['volume'] / df['volume'].rolling(50).mean()

    df['high20'] = df['high_ask'].rolling(20, min_periods=5).max()
    df['low20'] = df['low_bid'].rolling(20, min_periods=5).min()
    df['dip_pct'] = (df['high20'] - df['close']) / df['high20'] * 100
    df['rally_pct'] = (df['close'] - df['low20']) / df['low20'] * 100
    return df


# =============================================================================
# Regime Gate — using v15-style approach (ATR + weak trend for low-vol)
# =============================================================================

def define_regime(df):
    """Low-vol regime: ATR < 3.5 + weak trend + v5/v6 setups scarce."""
    low_vol = df['atr14'] < 3.5
    weak_trend = df['ema50_slope'].abs() < 0.15
    up_setup = ((df['ema50_slope'] > 0) & (df['close'] > df['ema200']) &
                (df['dip_pct'] >= 0.15) & (df['dip_pct'] <= 3.0))
    dn_setup = ((df['ema50_slope'] < -0.1) & (df['close'] < df['ema200']) &
                (df['rally_pct'] >= 0.2) & (df['rally_pct'] <= 3.0))
    retrace_count = (up_setup | dn_setup).rolling(60, min_periods=1).sum()
    df['in_regime'] = (low_vol & weak_trend & (retrace_count < 10)).fillna(False)
    df['retrace_count'] = retrace_count
    return df


# =============================================================================
# Exit Simulation — VECTORIZED
# =============================================================================
# Precompute rolling max/min arrays for fast exit determination.
# For each entry at bar i: we need the forward high_bid, low_bid, high_ask,
# low_ask, close_bid, close_ask over [i+1, i+max_bars].
# We precompute these as shifted columns to avoid per-trade Python loops.

def _build_forward_arrays(df, max_bars=30):
    """Precompute forward OHLC arrays for vectorized exit.

    Returns arrays of shape (len(df), max_bars):
      fwd_high_bid[i, k] = high_bid at bar i+k+1 (0-indexed forward)
      fwd_low_bid[i, k]  = low_bid at bar i+k+1
      fwd_high_ask[i, k] = high_ask at bar i+k+1
      fwd_low_ask[i, k]  = low_ask at bar i+k+1
      fwd_close_bid[i, k] = close_bid at bar i+k+1
      fwd_close_ask[i, k] = close_ask at bar i+k+1
    The last max_bars rows are NaN-padded.
    """
    n = len(df)
    N = max_bars

    def _shifted(col):
        # shape (n, N): col shifted back by 1..N
        out = np.full((n, N), np.nan, dtype=np.float64)
        vals = df[col].values
        for k in range(N):
            shift = k + 1
            out[:n-shift, k] = vals[shift:]
        return out

    return {
        'fwd_high_bid': _shifted('high_bid'),
        'fwd_low_bid': _shifted('low_bid'),
        'fwd_high_ask': _shifted('high_ask'),
        'fwd_low_ask': _shifted('low_ask'),
        'fwd_close_bid': _shifted('close_bid'),
        'fwd_close_ask': _shifted('close_ask'),
    }


def vectorized_long_exit(entry_indices, entry_prices, fwd, tp, sl):
    """Vectorized LONG exit: entry at ask, SL/TP on bid.

    entry_indices: np.array of integer bar positions
    entry_prices: np.array of entry ask prices
    fwd: dict from _build_forward_arrays
    tp, sl: floats

    Returns: np.array of PnLs (exit_price - entry_ask)
    """
    n = len(entry_indices)
    pnls = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return pnls

    stops = entry_prices - sl
    targets = entry_prices + tp

    # Gather forward arrays for all entries at once
    fwd_low = fwd['fwd_low_bid'][entry_indices]   # (n, max_bars)
    fwd_high = fwd['fwd_high_bid'][entry_indices]  # (n, max_bars)
    fwd_close = fwd['fwd_close_bid'][entry_indices]  # (n, max_bars)

    # Find first bar where SL hits, TP hits, or both
    sl_hit = fwd_low <= stops[:, None]    # (n, max_bars) bool
    tp_hit = fwd_high >= targets[:, None]

    # First bar index where condition is true (max_bars if never)
    sl_bar = np.argmax(sl_hit, axis=1)    # (n,)
    tp_bar = np.argmax(tp_hit, axis=1)

    # argmax returns 0 if true at first position OR if never true
    # Fix: rows where never hit → set to max_bars
    sl_never = ~sl_hit.any(axis=1)
    tp_never = ~tp_hit.any(axis=1)
    sl_bar[sl_never] = fwd_low.shape[1]
    tp_bar[tp_never] = fwd_low.shape[1]

    max_bar = fwd_low.shape[1] - 1  # last valid index
    timeout_bar = max_bar

    # Determine which triggers first
    for i in range(n):
        sbi, tbi = sl_bar[i], tp_bar[i]
        if sbi == fwd_low.shape[1] and tbi == fwd_low.shape[1]:
            # Neither hit — timeout at close_bid of last valid bar
            last_close = fwd_close[i, timeout_bar]
            pnls[i] = last_close - entry_prices[i] if not np.isnan(last_close) else 0.0
        elif sbi <= tbi:
            # SL hit first (or same bar, SL takes priority)
            pnls[i] = stops[i] - entry_prices[i]
        else:
            pnls[i] = targets[i] - entry_prices[i]

    return pnls


def vectorized_short_exit(entry_indices, entry_prices, fwd, tp, sl):
    """Vectorized SHORT exit: entry at bid, SL/TP on ask.

    entry_indices: np.array of integer bar positions
    entry_prices: np.array of entry bid prices
    fwd: dict from _build_forward_arrays
    tp, sl: floats

    Returns: np.array of PnLs (entry_bid - exit_ask)
    """
    n = len(entry_indices)
    pnls = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return pnls

    stops = entry_prices + sl
    targets = entry_prices - tp

    fwd_high = fwd['fwd_high_ask'][entry_indices]
    fwd_low = fwd['fwd_low_ask'][entry_indices]
    fwd_close = fwd['fwd_close_ask'][entry_indices]

    sl_hit = fwd_high >= stops[:, None]
    tp_hit = fwd_low <= targets[:, None]

    sl_bar = np.argmax(sl_hit, axis=1)
    tp_bar = np.argmax(tp_hit, axis=1)

    sl_never = ~sl_hit.any(axis=1)
    tp_never = ~tp_hit.any(axis=1)
    sl_bar[sl_never] = fwd_low.shape[1]
    tp_bar[tp_never] = fwd_low.shape[1]

    timeout_bar = fwd_low.shape[1] - 1

    for i in range(n):
        sbi, tbi = sl_bar[i], tp_bar[i]
        if sbi == fwd_low.shape[1] and tbi == fwd_low.shape[1]:
            last_close = fwd_close[i, timeout_bar]
            pnls[i] = entry_prices[i] - last_close if not np.isnan(last_close) else 0.0
        elif sbi <= tbi:
            pnls[i] = entry_prices[i] - stops[i]
        else:
            pnls[i] = entry_prices[i] - targets[i]

    return pnls


# =============================================================================
# Archetype Signal Functions
# =============================================================================

def sig_mean_reversion(df):
    s = pd.DataFrame(index=df.index)
    s['long'] = (df['rsi_5'] < 25) & df['in_regime']
    s['short'] = (df['rsi_5'] > 75) & df['in_regime']
    return s


def sig_range_fade(df):
    s = pd.DataFrame(index=df.index)
    s['long'] = (df['pos_in_range'] < 0.20) & df['in_regime']
    s['short'] = (df['pos_in_range'] > 0.80) & df['in_regime']
    return s


def sig_bb_touch(df):
    s = pd.DataFrame(index=df.index)
    s['long'] = (df['close'] <= df['bb_lower']) & df['in_regime']
    s['short'] = (df['close'] >= df['bb_upper']) & df['in_regime']
    return s


def sig_inside_bar_breakout(df):
    s = pd.DataFrame(index=df.index)
    ph, pl = df['high_ask'].shift(1), df['low_ask'].shift(1)
    pc, po = df['close_ask'].shift(1), df['open_ask'].shift(1)
    inside = (df['high_ask'] < ph) & (df['low_ask'] > pl)
    s['long'] = (inside & (pc > po) & (df['close'] > ph) & df['in_regime'])
    s['short'] = (inside & (pc < po) & (df['close'] < pl) & df['in_regime'])
    return s


def sig_micro_momentum(df):
    s = pd.DataFrame(index=df.index)
    s['long'] = ((df['ret_3'] > 0.15) & (df['close'] > df['ema20']) & df['in_regime'])
    s['short'] = ((df['ret_3'] < -0.15) & (df['close'] < df['ema20']) & df['in_regime'])
    return s


def sig_doji_reversal(df):
    s = pd.DataFrame(index=df.index)
    doji = df['body_ratio'] < 0.25
    s['long'] = (doji & (df['pos_in_range'] < 0.25) & df['in_regime'])
    s['short'] = (doji & (df['pos_in_range'] > 0.75) & df['in_regime'])
    return s


# =============================================================================
# Runner
# =============================================================================

ARCHETYPES = {
    'mean_reversion': sig_mean_reversion,
    'range_fade': sig_range_fade,
    'bb_touch': sig_bb_touch,
    'inside_bar_breakout': sig_inside_bar_breakout,
    'micro_momentum': sig_micro_momentum,
    'doji_reversal': sig_doji_reversal,
}

TP_GRID = [5, 8, 10, 12, 15, 20]
SL_GRID = [5, 8, 10, 12, 15, 20]
MAX_BARS = 30


def _signal_indices_and_prices(signals, df, pmask):
    """Extract integer indices and entry prices for long/short signals in a period.

    Returns (long_indices, long_prices, short_indices, short_prices)
    all as numpy int/float arrays.
    """
    pmask_idx = df.index[pmask]
    sig_sub = signals.loc[signals.index.isin(pmask_idx)]

    long_mask = sig_sub['long'] if 'long' in sig_sub.columns else pd.Series(False, index=sig_sub.index)
    short_mask = sig_sub['short'] if 'short' in sig_sub.columns else pd.Series(False, index=sig_sub.index)

    long_idx_list = sig_sub.index[long_mask]
    short_idx_list = sig_sub.index[short_mask]

    long_indices = np.array([df.index.get_loc(i) for i in long_idx_list], dtype=np.int64)
    short_indices = np.array([df.index.get_loc(i) for i in short_idx_list], dtype=np.int64)

    long_prices = df['close_ask'].values[long_indices] if len(long_indices) else np.array([])
    short_prices = df['close_bid'].values[short_indices] if len(short_indices) else np.array([])

    return long_indices, long_prices, short_indices, short_prices


def test_archetype(name, signal_fn, df, period_masks):
    """Sweep TPxSL grid. Uses vectorized exit for speed.

    Pre-builds forward arrays once, then for each TP/SL combo and each period mask,
    computes all exits in a single vectorized call.
    """
    signals = signal_fn(df)
    fwd = _build_forward_arrays(df, MAX_BARS)
    results = []

    # Pre-extract signal indices/prices per period (same for all TP/SL combos)
    period_data = {}
    for pname, pmask in period_masks.items():
        li, lp, si, sp = _signal_indices_and_prices(signals, df, pmask)
        if len(li) > 0 or len(si) > 0:
            period_data[pname] = (li, lp, si, sp)

    for tp, sl in product(TP_GRID, SL_GRID):
        per_period = defaultdict(lambda: {'pnls': [], 'longs': 0, 'shorts': 0})
        for pname, (li, lp, si, sp) in period_data.items():
            # Vectorized long exits
            if len(li) > 0:
                long_pnls = vectorized_long_exit(li, lp, fwd, tp, sl)
                long_pnls = long_pnls[~np.isnan(long_pnls)]
                per_period[pname]['pnls'].extend(long_pnls.tolist())
                per_period[pname]['longs'] += len(long_pnls)
            # Vectorized short exits
            if len(si) > 0:
                short_pnls = vectorized_short_exit(si, sp, fwd, tp, sl)
                short_pnls = short_pnls[~np.isnan(short_pnls)]
                per_period[pname]['pnls'].extend(short_pnls.tolist())
                per_period[pname]['shorts'] += len(short_pnls)

        all_pnls, tl, ts = [], 0, 0
        for st in per_period.values():
            all_pnls.extend(st['pnls']); tl += st['longs']; ts += st['shorts']
        if len(all_pnls) < 5:
            continue
        a = np.array(all_pnls)
        n = len(a); total = a.sum(); wr = (a > 0).mean() * 100
        pos_sum = a[a > 0].sum(); neg_sum = abs(a[a < 0].sum())
        pf = pos_sum / neg_sum if neg_sum > 0 else 99
        mj_pnls = per_period.get('may_jun_2026', {}).get('pnls', [])
        mj_total = sum(mj_pnls) if mj_pnls else 0
        results.append({
            'archetype': name, 'tp': tp, 'sl': sl,
            'trades': n, 'pnl': round(total, 1),
            'wr': round(wr, 1), 'pf': round(pf, 2),
            'avg_pts': round(total / n, 2),
            'longs': tl, 'shorts': ts,
            'mj_trades': len(mj_pnls), 'mj_pnl': round(mj_total, 1),
        })
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 72)
    print("v8 Low-Vol Regime Archetype Scanner (v15 Edition)")
    print("=" * 72)

    print("\n[1/4] Loading ask/bid data...")
    df = load_askbid_data()
    print(f"  {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")

    print("[2/4] Computing indicators & regime gate...")
    df = compute_indicators(df)
    df = define_regime(df)
    df = df.dropna(subset=['atr14', 'ema50_slope', 'rsi_5', 'bb_pct', 'pos_in_range'])
    print(f"  After NaN drop: {len(df):,} bars, {df['in_regime'].sum():,} in-regime")

    print("[3/4] Defining period masks...")
    period_masks = {
        'full': df.index >= '2025-09-01',
        'pre_may_2026': (df.index >= '2025-09-01') & (df.index < '2026-05-01'),
        'may_jun_2026': (df.index >= '2026-05-01') & (df.index <= '2026-06-09'),
        '2026_05': (df.index >= '2026-05-01') & (df.index <= '2026-05-31'),
        '2026_06': (df.index >= '2026-06-01') & (df.index <= '2026-06-09'),
    }
    for nm, mk in period_masks.items():
        sub = df.loc[mk]
        reg = sub['in_regime'].sum()
        pct = reg / len(sub) * 100 if len(sub) else 0
        print(f"  {nm:<18s}: {len(sub):>6,} bars, {reg:>5,} regime ({pct:.1f}%), "
              f"ATR={sub['atr14'].mean():.2f}, slope={sub['ema50_slope'].mean():+.3f}")

    n_cfg = len(TP_GRID) * len(SL_GRID)
    print(f"\n[4/4] Testing archetypes ({n_cfg} TPxSL combos each)...")
    all_results = []
    for arch_name, sig_fn in ARCHETYPES.items():
        print(f"  {arch_name}...", end=" ", flush=True)
        res = test_archetype(arch_name, sig_fn, df, period_masks)
        all_results.extend(res)
        if res:
            best = max(res, key=lambda r: r['pnl'])
            print(f"{len(res)} cfg, best={best['tp']}/{best['sl']} -> "
                  f"{best['trades']}t, {best['pnl']:+.1f}pt")
        else:
            print("NO TRADES")

    if not all_results:
        print("\nFATAL: No trades. Regime gate too strict or signal thresholds too extreme.")
        return

    RD = pd.DataFrame(all_results)

    # -- Top 25 --
    print(f"\n{'=' * 95}")
    print("TOP 25 CONFIGURATIONS (by Total PnL)")
    print(f"{'=' * 95}")
    top = RD.nlargest(25, 'pnl')
    print(f"{'Archetype':<22s} {'TP':>4s} {'SL':>4s} {'Trades':>7s} {'PnL':>10s} "
          f"{'WR':>7s} {'PF':>6s} {'Avg':>7s} {'L':>5s} {'S':>5s} {'MJ_T':>5s} {'MJ_PnL':>8s}")
    print("-" * 95)
    for _, r in top.iterrows():
        print(f"{r['archetype']:<22s} {r['tp']:>4.0f} {r['sl']:>4.0f} "
              f"{r['trades']:>7.0f} {r['pnl']:>+10.1f} {r['wr']:>6.1f}% "
              f"{r['pf']:>5.2f} {r['avg_pts']:>+7.2f} "
              f"{r['longs']:>5.0f} {r['shorts']:>5.0f} "
              f"{r['mj_trades']:>5.0f} {r['mj_pnl']:>+8.1f}")

    # -- Best per Archetype --
    print(f"\n{'=' * 95}")
    print("BEST CONFIG BY ARCHETYPE")
    print(f"{'=' * 95}")
    for arch in ARCHETYPES:
        sub = RD[RD['archetype'] == arch]
        if len(sub) == 0:
            print(f"  {arch:<22s}: NO DATA")
            continue
        b = sub.loc[sub['pnl'].idxmax()]
        print(f"  {arch:<22s}: TP={b['tp']:>4.0f} SL={b['sl']:>4.0f} "
              f"{b['trades']:>5.0f}t, {b['pnl']:>+8.1f}pt, {b['wr']:>5.1f}% WR, "
              f"PF={b['pf']:.2f}, avg={b['avg_pts']:>+.2f}/trade, "
              f"L:{b['longs']:.0f} S:{b['shorts']:.0f}, MJ:{b['mj_trades']:.0f}t/{b['mj_pnl']:+.1f}pt")

    # -- Top by May-June PnL --
    print(f"\n{'=' * 95}")
    print("TOP 15 CONFIGURATIONS (by May-June 2026 PnL)")
    print(f"{'=' * 95}")
    mj_valid = RD[RD['mj_trades'] > 0]
    if len(mj_valid):
        for _, r in mj_valid.nlargest(15, 'mj_pnl').iterrows():
            print(f"  {r['archetype']:<22s} TP={r['tp']:>4.0f} SL={r['sl']:>4.0f} "
                  f"MJ: {r['mj_trades']:>4.0f}t {r['mj_pnl']:>+8.1f}pt | "
                  f"All: {r['trades']:>5.0f}t {r['pnl']:>+8.1f}pt, {r['wr']:>5.1f}% WR")
    else:
        print("  NO trades during May-June 2026. Low-vol lacks edge for rule-based patterns.")

    # -- Signal Frequency --
    print(f"\n{'=' * 95}")
    print("SIGNAL FREQUENCY (May-June 2026)")
    print(f"{'=' * 95}")
    mj_idx = period_masks['may_jun_2026']
    mj_idx = df.index[mj_idx]
    mj_df = df.loc[mj_idx]
    print(f"  Total bars: {len(mj_df):,}, In-regime: {mj_df['in_regime'].sum():,}")
    for arch, fn in ARCHETYPES.items():
        sig = fn(df)
        sig_mj = sig.loc[sig.index.isin(mj_idx)]
        l = sig_mj['long'].sum(); s = sig_mj['short'].sum()
        print(f"  {arch:<22s}: {l:>5}L, {s:>5}S = {l+s:>5} signals")

    # -- Regime Characteristics --
    print(f"\n{'=' * 95}")
    print("REGIME CHARACTERISTICS BY PERIOD")
    print(f"{'=' * 95}")
    for nm, mk in period_masks.items():
        sub = df.loc[mk]
        reg = sub[sub['in_regime']]
        if len(reg):
            print(f"  {nm:<18s}: n={len(reg):>5,}, ATR={reg['atr14'].mean():.2f}, "
                  f"slope={reg['ema50_slope'].mean():+.3f}, RSI5={reg['rsi_5'].mean():.1f}, "
                  f"pos_range={reg['pos_in_range'].mean():.2f}, retrace_ct={reg['retrace_count'].mean():.1f}")

    print(f"\n{'=' * 95}")
    print("DONE. Best archetype + TP/SL above. Build v9 as focused optimization.")
    print(f"{'=' * 95}")


if __name__ == '__main__':
    main()
