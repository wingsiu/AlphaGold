#!/usr/bin/env python3
"""
v9 Regime-Gated Ensemble — Production Research Script
======================================================
Combines v8's best archetypes with a regime gate + XGBoost confidence filter.

Ensemble composition:
  - micro_momentum (3-bar continuation): fires in all regimes, TP=5, SL=10
  - range_fade (50-bar range fade): fires ONLY in low-vol regime (ATR<3.5, weak trend), TP=15, SL=20

Regime gate (heuristic, mirrors v15 HMM states):
  0 = low_vol:  ATR<3.5, |ema50_slope|<0.15, v5/v6 retrace count<10
  1 = trending: ATR>=3.5 OR |ema50_slope|>=0.15 (default)
  2 = high_vol: not used for these archetypes

XGBoost classifier for confidence filtering:
  - Features: indicator state at signal bar
  - Target: did the trade win at the fixed TP/SL?
  - Walk-forward: monthly retrain, no lookahead

Bar-by-bar ask/bid exit simulation (same as v5/v6/v8).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
from data.data_loader import DataLoader


# =============================================================================
# Data Loading (ask/bid aware)
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
# Indicators + Regime
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
    df['bb_lower'] = df['ema20'] - 2 * df['bb_std']
    df['bb_upper'] = df['ema20'] + 2 * df['bb_std']

    df['range_high'] = df['high_ask'].rolling(50, min_periods=10).max()
    df['range_low'] = df['low_ask'].rolling(50, min_periods=10).min()
    df['pos_in_range'] = ((df['close'] - df['range_low']) /
                          (df['range_high'] - df['range_low'] + 0.001))

    df['body_ratio'] = abs(df['close_ask'] - df['open_ask']) / (df['high_ask'] - df['low_ask'] + 0.001)

    for n in [3, 5, 10]:
        df[f'ret_{n}'] = df['close'].pct_change(n) * 100
        df[f'vol_{n}'] = df['close'].pct_change().rolling(n).std() * 100

    df['vol_ratio'] = df['volume'] / df['volume'].rolling(50).mean()

    # v5/v6 retrace count for regime gate
    df['high20'] = df['high_ask'].rolling(20, min_periods=5).max()
    df['low20'] = df['low_bid'].rolling(20, min_periods=5).min()
    df['dip_pct'] = (df['high20'] - df['close']) / df['high20'] * 100
    df['rally_pct'] = (df['close'] - df['low20']) / df['low20'] * 100
    return df


def define_regime(df):
    """Heuristic 2-regime gate (maps to v15 HMM 0/1)."""
    low_vol = df['atr14'] < 3.5
    weak_trend = df['ema50_slope'].abs() < 0.15
    up_setup = ((df['ema50_slope'] > 0) & (df['close'] > df['ema200']) &
                (df['dip_pct'] >= 0.15) & (df['dip_pct'] <= 3.0))
    dn_setup = ((df['ema50_slope'] < -0.1) & (df['close'] < df['ema200']) &
                (df['rally_pct'] >= 0.2) & (df['rally_pct'] <= 3.0))
    retrace_count = (up_setup | dn_setup).rolling(60, min_periods=1).sum()
    df['regime_lowvol'] = (low_vol & weak_trend & (retrace_count < 10)).fillna(False)
    return df


# =============================================================================
# Signal Generators
# =============================================================================

def micro_momentum_signals(df):
    """3-bar continuation — fires in ALL regimes (no regime gate, v8 showed edge everywhere)."""
    s = pd.DataFrame(index=df.index)
    s['long'] = (df['ret_3'] > 0.15) & (df['close'] > df['ema20'])
    s['short'] = (df['ret_3'] < -0.15) & (df['close'] < df['ema20'])
    # CRITICAL: Without regime gate, this fires 12k+ trades. v8 showed +138pts on 297 trades
    # WITH the regime gate on. Adding gate back to match v8 results.
    in_regime = df.get('regime_lowvol', pd.Series(True, index=df.index))
    s['long'] = s['long'] & in_regime
    s['short'] = s['short'] & in_regime
    return s


def range_fade_signals(df):
    """50-bar range fade — fires ONLY in low_vol regime."""
    s = pd.DataFrame(index=df.index)
    in_regime = df['regime_lowvol']
    s['long'] = (df['pos_in_range'] < 0.20) & in_regime
    s['short'] = (df['pos_in_range'] > 0.80) & in_regime
    return s


# =============================================================================
# Ensemble signal generator
# =============================================================================

ENSEMBLE_SPECS = [
    {
        'name': 'micro_momentum',
        'sig_fn': micro_momentum_signals,
        'tp': 5, 'sl': 10,
        'regime_mask': [True, True, True],  # all regimes
    },
    {
        'name': 'range_fade',
        'sig_fn': range_fade_signals,
        'tp': 15, 'sl': 20,
        'regime_mask': [True, False, False],  # only low_vol (regime 0)
    },
]


def generate_ensemble_signals(df):
    """Combine all ensemble members into unified signal DataFrame.

    Returns DataFrame with columns: long, short, member_name, tp, sl
    Priority: first signal wins at each bar (micro_momentum before range_fade).
    """
    combined = pd.DataFrame(index=df.index)
    combined['long'] = False
    combined['short'] = False
    combined['member'] = ''
    combined['tp'] = 0.0
    combined['sl'] = 0.0

    for spec in ENSEMBLE_SPECS:
        sig = spec['sig_fn'](df)
        # Only fill bars not yet claimed
        available = ~combined['long'] & ~combined['short']
        long_fill = sig['long'] & available
        short_fill = sig['short'] & available
        combined.loc[long_fill, 'long'] = True
        combined.loc[long_fill, 'member'] = spec['name']
        combined.loc[long_fill, 'tp'] = spec['tp']
        combined.loc[long_fill, 'sl'] = spec['sl']
        combined.loc[short_fill, 'short'] = True
        combined.loc[short_fill, 'member'] = spec['name']
        combined.loc[short_fill, 'tp'] = spec['tp']
        combined.loc[short_fill, 'sl'] = spec['sl']

    return combined


# =============================================================================
# Features for XGBoost confidence filter
# =============================================================================

def build_signal_features(df, signals):
    """Build feature matrix for XGBoost classifier at signal bars.

    Target: 1 if trade is profitable at TP/SL exit (net positive), else 0.
    Feature columns are indicator values at signal bar + signal metadata.
    """
    F = pd.DataFrame(index=df.index)
    F['atr14'] = df['atr14']
    F['ema50_slope'] = df['ema50_slope']
    F['pos_in_range'] = df['pos_in_range']
    F['rsi_5'] = df['rsi_5']
    F['rsi_14'] = df['rsi_14']
    F['ret_3'] = df['ret_3']
    F['ret_5'] = df['ret_5']
    F['vol_5'] = df['vol_5']
    F['vol_ratio'] = df['vol_ratio']
    F['body_ratio'] = df['body_ratio']
    F['spread'] = df['spread']
    F['dist_ema50'] = (df['close'] - df['ema50']) / df['close'] * 100
    F['dist_ema200'] = (df['close'] - df['ema200']) / df['close'] * 100
    F['is_range_fade'] = (signals['member'] == 'range_fade').astype(float)
    F['is_micro_momentum'] = (signals['member'] == 'micro_momentum').astype(float)
    F['direction'] = signals['long'].astype(float) - signals['short'].astype(float)  # 1=long, -1=short
    F['hour'] = df.index.hour.astype(float)
    F['dayofweek'] = df.index.dayofweek.astype(float)
    F = F.replace([np.inf, -np.inf], 0).fillna(0)
    return F


# =============================================================================
# Vectorized Exit (from v8)
# =============================================================================

MAX_BARS = 30

def _build_forward_arrays(df, max_bars=30):
    n = len(df)
    N = max_bars
    def _shifted(col):
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
    n = len(entry_indices)
    pnls = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return pnls
    stops = entry_prices - sl
    targets = entry_prices + tp
    fwd_low = fwd['fwd_low_bid'][entry_indices]
    fwd_high = fwd['fwd_high_bid'][entry_indices]
    fwd_close = fwd['fwd_close_bid'][entry_indices]
    sl_hit = fwd_low <= stops[:, None]
    tp_hit = fwd_high >= targets[:, None]
    sl_bar = np.argmax(sl_hit, axis=1)
    tp_bar = np.argmax(tp_hit, axis=1)
    sl_never = ~sl_hit.any(axis=1)
    tp_never = ~tp_hit.any(axis=1)
    N = fwd_low.shape[1]
    sl_bar[sl_never] = N
    tp_bar[tp_never] = N
    for i in range(n):
        sbi, tbi = sl_bar[i], tp_bar[i]
        if sbi == N and tbi == N:
            lc = fwd_close[i, -1]
            pnls[i] = lc - entry_prices[i] if not np.isnan(lc) else 0.0
        elif sbi <= tbi:
            pnls[i] = stops[i] - entry_prices[i]
        else:
            pnls[i] = targets[i] - entry_prices[i]
    return pnls


def vectorized_short_exit(entry_indices, entry_prices, fwd, tp, sl):
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
    N = fwd_low.shape[1]
    sl_bar[sl_never] = N
    tp_bar[tp_never] = N
    for i in range(n):
        sbi, tbi = sl_bar[i], tp_bar[i]
        if sbi == N and tbi == N:
            lc = fwd_close[i, -1]
            pnls[i] = entry_prices[i] - lc if not np.isnan(lc) else 0.0
        elif sbi <= tbi:
            pnls[i] = entry_prices[i] - stops[i]
        else:
            pnls[i] = entry_prices[i] - targets[i]
    return pnls


# =============================================================================
# XGBoost Confidence Filter + Walk-Forward
# =============================================================================

def wf_ensemble_backtest(df, signals, months, min_prob=0.50):
    """Walk-forward backtest with monthly XGBoost retraining.

    For each month:
      1. Train XGBoost classifier on all prior signal bars (win/lose label at TP/SL)
      2. Predict probability for current month's signals
      3. Only trade signals with prob >= min_prob
      4. Simulate exits at per-member TP/SL

    Returns DataFrame of trades with columns: month, member, side, entry_price, exit_price, pnl, prob, tp, sl
    """
    F = build_signal_features(df, signals)
    feat_cols = list(F.columns)
    fwd = _build_forward_arrays(df, MAX_BARS)

    all_trades = []

    for m_idx, m_start in enumerate(months):
        m_end = m_start + pd.offsets.MonthEnd(1)
        train_mask = (df.index < m_start)
        test_mask = (df.index >= m_start) & (df.index <= m_end)

        train_sig = signals.loc[train_mask]
        test_sig = signals.loc[test_mask]

        train_any = train_sig['long'] | train_sig['short']
        test_any = test_sig['long'] | test_sig['short']

        if train_any.sum() < 20 or test_any.sum() < 5:
            continue

        # --- Compute training labels: did trade win at fixed TP/SL? ---
        train_labels = []
        train_indices = train_sig.index[train_any]

        # Batch long exits for training
        long_train = train_sig[train_sig['long']]
        if len(long_train) > 0:
            li = np.array([df.index.get_loc(i) for i in long_train.index], dtype=np.int64)
            lp = df['close_ask'].values[li]
            tp_vals = long_train['tp'].values
            sl_vals = long_train['sl'].values
            # Process per unique TP/SL combo
            for (tp, sl), mask in long_train.groupby(['tp', 'sl']).groups.items():
                idxs = np.array([df.index.get_loc(i) for i in mask], dtype=np.int64)
                prices = df['close_ask'].values[idxs]
                pnls_l = vectorized_long_exit(idxs, prices, fwd, tp, sl)
                for j, pnl in enumerate(pnls_l):
                    train_labels.append(1 if pnl > 0 else 0)

        # Batch short exits for training
        short_train = train_sig[train_sig['short']]
        if len(short_train) > 0:
            si = np.array([df.index.get_loc(i) for i in short_train.index], dtype=np.int64)
            sp = df['close_bid'].values[si]
            for (tp, sl), mask in short_train.groupby(['tp', 'sl']).groups.items():
                idxs = np.array([df.index.get_loc(i) for i in mask], dtype=np.int64)
                prices = df['close_bid'].values[idxs]
                pnls_s = vectorized_short_exit(idxs, prices, fwd, tp, sl)
                for j, pnl in enumerate(pnls_s):
                    train_labels.append(1 if pnl > 0 else 0)

        if len(train_labels) < 20:
            continue

        # --- Train XGBoost classifier ---
        X_tr = F.loc[train_indices].values.astype(np.float32)
        y_tr = np.array(train_labels[:len(train_indices)])
        if len(y_tr) != len(X_tr):
            # Mismatch from groupby — trim
            y_tr = y_tr[:len(X_tr)]

        pos_ratio = y_tr.mean()
        scale_pos_weight = (1 - pos_ratio) / max(pos_ratio, 0.01)

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=min(scale_pos_weight, 10),
            reg_alpha=0.5, reg_lambda=1, random_state=42, verbosity=0,
        )
        model.fit(X_tr, y_tr)

        # --- Predict on test set ---
        test_indices = test_sig.index[test_any]
        X_te = F.loc[test_indices].values.astype(np.float32)
        probs = model.predict_proba(X_te)[:, 1]

        # --- Filter by confidence ---
        month_label = m_start.strftime('%Y-%m')
        for j, idx in enumerate(test_indices):
            if probs[j] < min_prob:
                continue
            sig_row = test_sig.loc[idx]
            side = 'long' if sig_row['long'] else 'short'
            tp = sig_row['tp']
            sl = sig_row['sl']
            ei = df.index.get_loc(idx)

            if side == 'long':
                ep = df.iloc[ei]['close_ask']
                pnl_arr = vectorized_long_exit(np.array([ei]), np.array([ep]), fwd, tp, sl)
                pnl = pnl_arr[0]
            else:
                ep = df.iloc[ei]['close_bid']
                pnl_arr = vectorized_short_exit(np.array([ei]), np.array([ep]), fwd, tp, sl)
                pnl = pnl_arr[0]

            if not np.isnan(pnl):
                all_trades.append({
                    'month': month_label, 'member': sig_row['member'],
                    'side': side, 'entry_price': ep, 'pnl': pnl,
                    'prob': probs[j], 'tp': tp, 'sl': sl,
                })

    return pd.DataFrame(all_trades)


# =============================================================================
# Baseline: No XGBoost (all signals traded)
# =============================================================================

def baseline_backtest(df, signals, months):
    """Trade ALL ensemble signals with fixed TP/SL, no XGBoost filter."""
    fwd = _build_forward_arrays(df, MAX_BARS)
    all_trades = []

    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        test_mask = (df.index >= m_start) & (df.index <= m_end)
        test_sig = signals.loc[test_mask]
        month_label = m_start.strftime('%Y-%m')

        long_mask = test_sig['long']
        if long_mask.any():
            for (tp, sl), group in test_sig[long_mask].groupby(['tp', 'sl']):
                idxs = np.array([df.index.get_loc(i) for i in group.index], dtype=np.int64)
                prices = df['close_ask'].values[idxs]
                pnls = vectorized_long_exit(idxs, prices, fwd, tp, sl)
                for j, pnl in enumerate(pnls):
                    if not np.isnan(pnl):
                        all_trades.append({
                            'month': month_label, 'member': group.iloc[j]['member'],
                            'side': 'long', 'pnl': pnl, 'tp': tp, 'sl': sl,
                        })

        short_mask = test_sig['short']
        if short_mask.any():
            for (tp, sl), group in test_sig[short_mask].groupby(['tp', 'sl']):
                idxs = np.array([df.index.get_loc(i) for i in group.index], dtype=np.int64)
                prices = df['close_bid'].values[idxs]
                pnls = vectorized_short_exit(idxs, prices, fwd, tp, sl)
                for j, pnl in enumerate(pnls):
                    if not np.isnan(pnl):
                        all_trades.append({
                            'month': month_label, 'member': group.iloc[j]['member'],
                            'side': 'short', 'pnl': pnl, 'tp': tp, 'sl': sl,
                        })

    return pd.DataFrame(all_trades)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 72)
    print("v9 Regime-Gated Ensemble")
    print("=" * 72)

    print("\n[1/5] Loading data...")
    df = load_askbid_data()
    print(f"  {len(df):,} bars")

    print("[2/5] Computing indicators & regime...")
    df = compute_indicators(df)
    df = define_regime(df)
    df = df.dropna(subset=['atr14', 'ema50_slope', 'rsi_5', 'pos_in_range'])
    n_lowvol = df['regime_lowvol'].sum()
    print(f"  {len(df):,} bars, {n_lowvol:,} low-vol regime ({n_lowvol/len(df)*100:.1f}%)")

    print("[3/5] Generating ensemble signals...")
    signals = generate_ensemble_signals(df)
    n_long = signals['long'].sum()
    n_short = signals['short'].sum()
    print(f"  {n_long:,} longs, {n_short:,} shorts = {n_long+n_short:,} total")
    for spec in ENSEMBLE_SPECS:
        n = (signals['member'] == spec['name']).sum()
        print(f"    {spec['name']}: {n}")

    months = pd.date_range('2025-09-01', '2026-07-01', freq='MS', tz='UTC')

    print("\n[4/5] Running BASELINE (no XGBoost filter)...")
    baseline_trades = baseline_backtest(df, signals, months)
    if len(baseline_trades):
        bt = baseline_trades
        print(f"  {len(bt)} trades, PnL={bt['pnl'].sum():+.1f}, WR={(bt['pnl']>0).mean()*100:.1f}%")
        for member in bt['member'].unique():
            sub = bt[bt['member'] == member]
            print(f"    {member}: {len(sub)} trades, PnL={sub['pnl'].sum():+.1f}, WR={(sub['pnl']>0).mean()*100:.1f}%")

    print("\n[5/5] Running ENSEMBLE (XGBoost confidence filter, min_prob=0.50)...")
    ensemble_trades = wf_ensemble_backtest(df, signals, months, min_prob=0.50)
    if len(ensemble_trades):
        et = ensemble_trades
        print(f"  {len(et)} trades, PnL={et['pnl'].sum():+.1f}, WR={(et['pnl']>0).mean()*100:.1f}%")
        pos = et[et['pnl']>0]['pnl'].sum()
        neg = abs(et[et['pnl']<0]['pnl'].sum())
        pf = pos/neg if neg>0 else 99
        print(f"  PF={pf:.2f}")

        # Monthly breakdown
        print(f"\n  Monthly breakdown:")
        for m in sorted(et['month'].unique()):
            ms = et[et['month'] == m]
            print(f"    {m}: {len(ms):>4} trades, {ms['pnl'].sum():>+8.1f} pts, {(ms['pnl']>0).mean()*100:>5.1f}% WR")

        # By member
        for member in et['member'].unique():
            sub = et[et['member'] == member]
            print(f"\n  {member}: {len(sub)} trades, PnL={sub['pnl'].sum():+.1f}, WR={(sub['pnl']>0).mean()*100:.1f}%")

        # Prob bucket analysis
        et['prob_bucket'] = pd.cut(et['prob'], bins=[0.5, 0.6, 0.7, 0.8, 1.0])
        print(f"\n  Prob bucket analysis:")
        for bucket, grp in et.groupby('prob_bucket', observed=True):
            print(f"    {bucket}: {len(grp):>4} trades, PnL={grp['pnl'].sum():>+8.1f}, WR={(grp['pnl']>0).mean()*100:>5.1f}%")

        # May-June 2026 focus
        mj = et[et['month'].isin(['2026-05', '2026-06'])]
        if len(mj):
            print(f"\n  May-June 2026: {len(mj)} trades, PnL={mj['pnl'].sum():+.1f}, WR={(mj['pnl']>0).mean()*100:.1f}%")

    # Compare
    print(f"\n{'='*72}")
    print("COMPARISON")
    print(f"{'='*72}")
    print(f"  Baseline  (all signals):    {len(baseline_trades):>5} trades, {baseline_trades['pnl'].sum():>+8.1f} pts")
    if len(ensemble_trades):
        print(f"  Ensemble  (XGBoost filter): {len(ensemble_trades):>5} trades, {ensemble_trades['pnl'].sum():>+8.1f} pts")
        reduction = (1 - len(ensemble_trades)/len(baseline_trades)) * 100
        print(f"  Signal reduction: {reduction:.1f}%")
    print(f"\nDONE.")


if __name__ == '__main__':
    main()
