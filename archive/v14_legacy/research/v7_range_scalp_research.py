Pattern for low-volatility, range-bound markets where v5/v6 retrace models fail.

Designed for regimes like May-Jun 2026:
  - ATR(14) < 3.5 pts (low vol), weak trend, price oscillating around EMAs
  - v5/v6 produce few trades because retrace dips are too shallow
  - This model scalps small directional moves within ranges

Architecture:
  1. Regime gate: ATR(14) < 3.5, retrace setups < threshold (low signal count = ranging)
  2. XGBoost classifier: predict direction (UP/DOWN) for next 10 bars
  3. Entry: high-confidence (prob > 0.6) directional calls on ask/bid
  4. Exit: tight TP (8-12 pts) + tight SL (8-12 pts) + short timeout (20 bars)
  5. Bar-by-bar OHLC with ask/bid

Key difference vs v5/v6: smaller bar horizon (10 vs 30), tighter SL/TP, classifier not regressor.
"""
#!/usr/bin/env python3
"""
v8 Low-Vol Regime Model — Production Research Script
XGBoost REGRESSOR (not classifier — avoids short-bias) with ATR-adaptive TP/SL.

For low-volatility regimes (ATR < 3.5) where v5/v6 retrace patterns don't fire.
10-bar forward return prediction, balanced percentile entry, tight ATR-scaled stops.

Key fixes from failed v7 classifier:
  - Regressor instead of classifier = no directional bias
  - Percentile thresholds (p10/p90) = naturally balanced long/short
  - ATR-adaptive TP/SL = tight stops in low vol, wider in moderate vol
"""
==================================================
Pattern for low-volatility, range-bound markets where v5/v6 retrace models fail.

Designed for regimes like May-Jun 2026:
  - ATR(14) < 3.5 pts (low vol), weak trend, price oscillating around EMAs
  - v5/v6 produce few trades because retrace dips are too shallow
  - This model scalps small directional moves within ranges

Architecture:
  1. Regime gate: ATR(14) < 3.5, retrace setups < threshold (low signal count = ranging)
  2. XGBoost classifier: predict direction (UP/DOWN) for next 10 bars
  3. Entry: high-confidence (prob > 0.6) directional calls on ask/bid
  4. Exit: tight TP (8-12 pts) + tight SL (8-12 pts) + short timeout (20 bars)
  5. Bar-by-bar OHLC with ask/bid

Key difference vs v5/v6: smaller bar horizon (10 vs 30), tighter SL/TP, classifier not regressor.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import xgboost as xgb
from data.data_loader import DataLoader


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
    df['close'] = df['close_bid']
    df['volume'] = raw['lastTradedVolume'].astype(float)
    df['spread'] = df['close_ask'] - df['close_bid']
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def define_indicators(df):
    """Core indicators for regime + features."""
    df['ema50'] = df['close'].ewm(50).mean()
    df['ema200'] = df['close'].ewm(200).mean()
    df['ema50_slope'] = df['ema50'].diff(10)
    df['ema20'] = df['close'].ewm(20).mean()
    
    # ATR
    tr = pd.concat([df['high_ask'] - df['low_ask'],
                    abs(df['high_ask'] - df['close_ask'].shift()),
                    abs(df['low_ask'] - df['close_ask'].shift())], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()
    
    # Range detection
    df['range_high'] = df['high_ask'].rolling(50).max()
    df['range_low'] = df['low_ask'].rolling(50, min_periods=5).min()
    df['range_width'] = (df['range_high'] - df['range_low']) / df['close'] * 100
    df['pos_in_range'] = (df['close'] - df['range_low']) / (df['range_high'] - df['range_low'] + 0.01)
    
    # Retrace counts (for regime gate)
    df['low20'] = df['low_bid'].rolling(20, min_periods=5).min()
    df['high20'] = df['high_ask'].rolling(20, min_periods=5).max()
    df['dip_pct'] = (df['high20'] - df['close']) / df['high20'] * 100
    df['rally_pct'] = (df['close'] - df['low20']) / df['low20'] * 100
    
    return df


def build_features(df):
    """Feature matrix for directional classifier."""
    def rsi(s, p):
        d = s.diff(); g = d.clip(0).rolling(p).mean(); l = (-d.clip(0)).rolling(p).mean()
        return 100 - 100 / (1 + g / l.replace(0, 1))

    F = pd.DataFrame(index=df.index)
    F['atr14'] = df['atr14']
    F['pos_in_range'] = df['pos_in_range']
    F['range_width'] = df['range_width']
    F['ema50_slope'] = df['ema50_slope']
    F['dist_ema50'] = (df['close'] - df['ema50']) / df['close'] * 100
    F['dist_ema200'] = (df['close'] - df['ema200']) / df['close'] * 100
    F['dip_pct'] = df['dip_pct']
    F['rally_pct'] = df['rally_pct']

    for n in [3, 5, 10, 20]:
        F[f'ret_{n}'] = df['close'].pct_change(n).fillna(0) * 100
        F[f'rsi_{n}'] = rsi(df['close'], n).fillna(50)
        F[f'ret_std_{n}'] = df['close'].pct_change().rolling(n).std().fillna(0) * 100

    F['vol_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
    F['body'] = abs(df['close_bid'] - df['open_ask']) / (df['high_ask'] - df['low_ask'] + 0.01)
    F['upper_wick'] = (df['high_ask'] - df[['open_ask', 'close_bid']].max(axis=1)) / (df['high_ask'] - df['low_ask'] + 0.01)
    F['lower_wick'] = (df[['open_ask', 'close_bid']].min(axis=1) - df['low_ask']) / (df['high_ask'] - df['low_ask'] + 0.01)

    for lag in [1, 2]:
        F[f'rsi_5_lag{lag}'] = F['rsi_5'].shift(lag).fillna(50)
    F['hour'] = df.index.hour.astype(float)
    F['dayofweek'] = df.index.dayofweek.astype(float)
    F['spread_pct'] = df['spread'] / df['close'] * 100
    F = F.replace([np.inf, -np.inf], 0).fillna(0)
    return F


def simulate_long_exit(entry_idx, entry_price_ask, df, tp, sl, max_bars=20):
    """Bar-by-bar exit for LONG: entry at ask, exit at bid."""
    stop = entry_price_ask - sl
    target = entry_price_ask + tp
    horizon = min(max_bars, len(df) - entry_idx - 1)
    for i in range(1, horizon + 1):
        bar = df.iloc[entry_idx + i]
        if bar['low_bid'] <= stop:
            return stop, i, 'sl'
        if bar['high_bid'] >= target:
            return target, i, 'tp'
    return df.iloc[entry_idx + horizon]['close_bid'], horizon, 'timeout'


def simulate_short_exit(entry_idx, entry_price_bid, df, tp, sl, max_bars=20):
    """Bar-by-bar exit for SHORT: entry at bid, exit at ask."""
    stop = entry_price_bid + sl
    target = entry_price_bid - tp
    horizon = min(max_bars, len(df) - entry_idx - 1)
    for i in range(1, horizon + 1):
        bar = df.iloc[entry_idx + i]
        if bar['high_ask'] >= stop:
            return stop, i, 'sl'
        if bar['low_ask'] <= target:
            return target, i, 'tp'
    return df.iloc[entry_idx + horizon]['close_ask'], horizon, 'timeout'


def main():
    HORIZON = 10        # short horizon for scalp
    SL = 10             # tight stop
    TP = 10             # tight target
    MIN_PROB = 0.60     # confidence threshold
    MAX_BARS = 20       # short timeout

    print("Loading ask/bid data...")
    df = load_askbid_data()
    df = define_indicators(df)

    # Regime gate: low-vol range where retrace patterns are scarce
    # We gate on ATR < threshold AND not too many retrace setups
    F = build_features(df)

    # Target: directional label (1=UP, 0=DOWN) over HORIZON bars
    fwd_return = df['close_bid'].shift(-HORIZON) - df['close_ask']
    df['target_up'] = (fwd_return > 0).astype(int)
    df = df.dropna(subset=['target_up'])
    F = F.loc[df.index]
    feat_cols = list(F.columns)

    # Regime definition: low ATR + weak trend + mid-range frequency of setups
    is_low_vol = df['atr14'] < 4.0
    is_weak_trend = df['ema50_slope'].abs() < df['atr14'] * 3  # trend < 3 ATR units
    uptrend_setup = (df['ema50_slope'] > 0.1) & (df['close'] > df['ema200']) & (df['dip_pct'] >= 0.15) & (df['dip_pct'] <= 3.0)
    downtrend_setup = (df['ema50_slope'] < -0.1) & (df['close'] < df['ema200']) & (df['rally_pct'] >= 0.15) & (df['rally_pct'] <= 3.0)
    few_retraces = (uptrend_setup | downtrend_setup).rolling(60).sum().fillna(0) < 15
    in_regime = is_low_vol & is_weak_trend & few_retraces

    print(f"Data: {len(df)} bars, in_regime: {in_regime.sum()}")

    # Walk-forward
    months = pd.date_range("2025-09-01", "2026-06-01", freq="MS", tz="UTC")
    all_trades = []

    print(f"\n{'Month':<10} {'RegBars':>7} {'Trades':>7} {'PnL':>10} {'WR':>7} {'SL':>5} {'TP':>5} {'TO':>5}")
    print("-" * 55)

    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        train_mask = (df.index < m_start) & in_regime
        test_mask = (df.index >= m_start) & (df.index <= m_end) & in_regime

        # Check if this month even HAS enough regime bars for any pattern
        reg_bars = test_mask.sum()
        if reg_bars < 50 or train_mask.sum() < 300:
            continue

        X_tr = F.loc[train_mask].values.astype(np.float32)
        y_tr = df.loc[train_mask, 'target_up'].values
        X_te = F.loc[test_mask].values.astype(np.float32)

        model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.7,
                                  reg_alpha=1, reg_lambda=2, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]

        test_indices = df.loc[test_mask].index
        pnls = []; reasons = {'sl': 0, 'tp': 0, 'timeout': 0}

        for j in range(len(probs)):
            prob = probs[j]
            if prob < MIN_PROB and prob > (1 - MIN_PROB):
                continue  # not confident either way

            entry_idx = df.index.get_loc(test_indices[j])
            if prob >= MIN_PROB:
                # Long at ask
                entry_price = df.iloc[entry_idx]['close_ask']
                exit_price, bars, reason = simulate_long_exit(entry_idx, entry_price, df, tp=TP, sl=SL, max_bars=MAX_BARS)
                pnl = exit_price - entry_price
            else:
                # Short at bid
                entry_price = df.iloc[entry_idx]['close_bid']
                exit_price, bars, reason = simulate_short_exit(entry_idx, entry_price, df, tp=TP, sl=SL, max_bars=MAX_BARS)
                pnl = entry_price - exit_price

            pnls.append(pnl); reasons[reason] += 1
            all_trades.append({'month': str(m_start.date())[:7], 'pnl': pnl, 'prob': prob, 'reason': reason})

        n_t = len(pnls)
        if n_t == 0:
            continue
        wins = sum(1 for p in pnls if p > 0); total = sum(pnls)
        print(f"{str(m_start.date())[:7]:<10} {reg_bars:>7} {n_t:>7} {total:>+10.1f} {wins/n_t*100:>6.1f}% {reasons['sl']:>5} {reasons['tp']:>5} {reasons['timeout']:>5}")

    if not all_trades:
        print("\nNo trades — regime gate may be too strict.")
        return

    TD = pd.DataFrame(all_trades)
    tot = len(TD); tot_pnl = TD['pnl'].sum()
    wr = (TD['pnl'] > 0).mean() * 100
    pos = TD[TD['pnl'] > 0]['pnl'].sum()
    neg = abs(TD[TD['pnl'] < 0]['pnl'].sum()) if TD[TD['pnl'] < 0]['pnl'].sum() != 0 else 0.01
    pf = pos / neg

    print(f"\n{'='*68}")
    print(f"RESULTS: {tot} trades, +{tot_pnl:.1f} pts, {wr:.1f}% WR, PF={pf:.2f}")
    print(f"  Avg PnL/trade: {tot_pnl/tot:+.2f} pts")

    # By month
    print(f"\nMonthly breakdown:")
    for m in TD['month'].unique():
        ms = TD[TD['month'] == m]
        if len(ms):
            print(f"  {m}: {len(ms):>3} trades, {ms['pnl'].sum():>+8.1f} pts, {(ms['pnl']>0).mean()*100:>5.1f}% WR")

    # Direction split
    long_trades = TD[TD['prob'] >= MIN_PROB]
    short_trades = TD[TD['prob'] <= (1 - MIN_PROB)]
    if len(long_trades):
        print(f"\n  Longs:  {len(long_trades):>3} trades, {long_trades['pnl'].sum():>+8.1f} pts, {(long_trades['pnl']>0).mean()*100:>5.1f}% WR")
    if len(short_trades):
        print(f"  Shorts: {len(short_trades):>3} trades, {short_trades['pnl'].sum():>+8.1f} pts, {(short_trades['pnl']>0).mean()*100:>5.1f}% WR")


if __name__ == '__main__':
    main()
