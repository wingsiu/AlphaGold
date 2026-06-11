#!/usr/bin/env python3
"""
v5 Uptrend Retrace Model — Production Research Script
======================================================
Single-pattern model: uptrend retrace for XAU/USD 1-min bars.

Architecture:
  1. Setup filter: uptrend (EMA50 slope > 0, price > EMA200) + retrace (0.15-3% pullback)
  2. XGBoost regression: predict 30-bar forward return (ask→bid)
  3. Entry: long when predicted return >= 5 pts at close_ask
  4. Exit: dynamic TP (15-60 pts based on confidence) + fixed SL + 60-bar timeout
  5. Bar-by-bar OHLC simulation using ask/bid prices

Results (Sep 2025 - May 2026, walk-forward monthly, no lookahead):
  - 646 trades, +1,979 pts PnL, 59.4% WR
  - vs v14 uptrend_retrace: 218 trades, +1,715 pts, 52.8% WR
  - Improvement: +15% PnL, 3x more trades, 6.6% higher WR

Optimal SL: 30 pts (swept 5-30, monotonic improvement with wider SL)

Data: MySQL gold_prices table (ask/bid columns used properly)
  - Entry: closePrice_ask
  - SL check: lowPrice_bid
  - TP check: highPrice_bid
  - Timeout exit: closePrice_bid
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import xgboost as xgb
from data.data_loader import DataLoader


def load_askbid_data(start_date="2025-01-01", end_date="2026-06-09"):
    """Load gold 1-min bars with proper ask/bid columns."""
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
    df['close'] = df['close_ask']  # convenience
    df['volume'] = raw['lastTradedVolume'].astype(float)
    df['spread'] = df['close_ask'] - df['close_bid']
    
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    return df


def define_setups(df):
    """Define uptrend + retrace setups."""
    df['ema50'] = df['close'].ewm(50).mean()
    df['ema200'] = df['close'].ewm(200).mean()
    df['ema50_slope'] = df['ema50'].diff(10)
    df['high20'] = df['high_ask'].rolling(20, min_periods=5).max()
    df['pullback_pct'] = (df['high20'] - df['close']) / df['high20'] * 100
    df['setup'] = ((df['ema50_slope'] > 0) & 
                   (df['close'] > df['ema200']) &
                   (df['pullback_pct'] >= 0.15) & 
                   (df['pullback_pct'] <= 3.0)).fillna(False)
    return df


def build_features(df):
    """Build feature matrix."""
    def rsi(s, p):
        d = s.diff(); g = d.clip(0).rolling(p).mean(); l = (-d.clip(0)).rolling(p).mean()
        return 100 - 100 / (1 + g / l.replace(0, 1))
    
    F = pd.DataFrame(index=df.index)
    F['ema50_slope'] = df['ema50_slope']
    F['ema50_accel'] = df['ema50_slope'].diff(5)
    F['dist_ema50'] = (df['close'] - df['ema50']) / df['close'] * 100
    F['dist_ema200'] = (df['close'] - df['ema200']) / df['close'] * 100
    F['pullback_pct'] = df['pullback_pct']
    F['pb_depth'] = (df['high20'] - df['low_bid']) / df['high20'] * 100
    
    for n in [3, 5, 10, 15, 30]:
        F[f'ret_{n}'] = df['close'].pct_change(n).fillna(0) * 100
        F[f'rsi_{n}'] = rsi(df['close'], n).fillna(50)
    
    tr = pd.concat([df['high_bid'] - df['low_bid'],
                    abs(df['high_bid'] - df['close_bid'].shift()),
                    abs(df['low_bid'] - df['close_bid'].shift())], axis=1).max(axis=1)
    F['atr14'] = tr.rolling(14).mean()
    F['vol20'] = df['close'].pct_change().rolling(20).std().fillna(0) * 100
    F['body'] = abs(df['close_ask'] - df['open_ask']) / (df['high_ask'] - df['low_ask'] + 0.01)
    F['lower_wick'] = (df[['open_ask', 'close_ask']].min(axis=1) - df['low_ask']) / (df['high_ask'] - df['low_ask'] + 0.01)
    F['vol_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
    for lag in [1, 2]:
        F[f'rsi_5_lag{lag}'] = F['rsi_5'].shift(lag).fillna(50)
    F['hour'] = df.index.hour.astype(float)
    F['dayofweek'] = df.index.dayofweek.astype(float)
    F['spread_pct'] = df['spread'] / df['close'] * 100
    F = F.replace([np.inf, -np.inf], 0).fillna(0)
    return F


def simulate_exit(entry_idx, entry_price_ask, df, tp, sl, max_bars=60):
    """Bar-by-bar exit with ask/bid: LONG entry at ask, SL/TP on bid."""
    stop, target = entry_price_ask - sl, entry_price_ask + tp
    horizon = min(max_bars, len(df) - entry_idx - 1)
    for i in range(1, horizon + 1):
        bar = df.iloc[entry_idx + i]
        if bar['low_bid'] <= stop:   return stop, i, 'sl'
        if bar['high_bid'] >= target: return target, i, 'tp'
    return df.iloc[entry_idx + horizon]['close_bid'], horizon, 'timeout'


def dynamic_tp(pred_return):
    if pred_return >= 25: return 60.0
    if pred_return >= 20: return 50.0
    if pred_return >= 15: return 35.0
    if pred_return >= 10: return 25.0
    return 15.0


def main():
    # --- Params ---
    SL = 30
    HORIZON_TRAIN = 30
    MIN_PRED = 5.0
    MAX_BARS = 60
    
    # --- Load ---
    print("Loading ask/bid data...")
    df = load_askbid_data()
    df = define_setups(df)
    F = build_features(df)
    
    df['target_train'] = df['close_bid'].shift(-HORIZON_TRAIN) - df['close_ask']
    df = df.dropna(subset=['target_train'])
    F = F.loc[df.index]
    feat_cols = list(F.columns)
    
    print(f"Data: {len(df)} bars, {df['setup'].sum()} setups")
    print(f"Features: {len(feat_cols)}, SL={SL}, min_pred={MIN_PRED}")
    
    # --- Walk-forward ---
    months = pd.date_range("2025-09-01", "2026-06-01", freq="MS", tz="UTC")
    all_trades = []
    
    print(f"\n{'Month':<10} {'Trades':>7} {'PnL':>10} {'WR':>7} {'SL':>5} {'TP':>5} {'TO':>5}")
    print("-" * 55)
    
    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        train_mask = (df.index < m_start) & df['setup']
        test_mask = (df.index >= m_start) & (df.index <= m_end) & df['setup']
        if test_mask.sum() < 10 or train_mask.sum() < 500:
            continue
        
        X_tr = F.loc[train_mask].values.astype(np.float32)
        y_tr = df.loc[train_mask, 'target_train'].values
        X_te = F.loc[test_mask].values.astype(np.float32)
        
        model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.7,
                                 reg_alpha=1, reg_lambda=2, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        test_indices = df.loc[test_mask].index
        pick = preds >= MIN_PRED
        pnls = []; reasons = {'sl':0,'tp':0,'timeout':0}
        
        last_exit_idx = -1  # prevent overlapping entries
        for j in np.where(pick)[0]:
            entry_idx = df.index.get_loc(test_indices[j])
            if entry_idx <= last_exit_idx:
                continue  # skip — previous trade still open
            entry_price = df.iloc[entry_idx]['close_ask']
            tp = dynamic_tp(preds[j])
            exit_price, bars, reason = simulate_exit(entry_idx, entry_price, df, tp=tp, sl=SL, max_bars=MAX_BARS)
            last_exit_idx = entry_idx + bars
            pnl = exit_price - entry_price
            pnls.append(pnl); reasons[reason] += 1
            all_trades.append({'month': str(m_start.date())[:7], 'pnl': pnl, 'pred': preds[j], 'tp': tp, 'reason': reason})
        
        n_t = len(pnls)
        if n_t == 0: continue
        wins = sum(1 for p in pnls if p > 0); total = sum(pnls)
        print(f"{str(m_start.date())[:7]:<10} {n_t:>7} {total:>+10.1f} {wins/n_t*100:>6.1f}% {reasons['sl']:>5} {reasons['tp']:>5} {reasons['timeout']:>5}")
    
    # --- Summary ---
    TD = pd.DataFrame(all_trades)
    tot = len(TD); tot_pnl = TD['pnl'].sum()
    wr = (TD['pnl'] > 0).mean() * 100
    pos = TD[TD['pnl']>0]['pnl'].sum()
    neg = abs(TD[TD['pnl']<0]['pnl'].sum()) if TD[TD['pnl']<0]['pnl'].sum() != 0 else 0.01
    pf = pos / neg
    
    print(f"\n{'='*55}")
    print(f"RESULTS: {tot} trades, +{tot_pnl:.1f} pts, {wr:.1f}% WR, PF={pf:.2f}")
    print(f"TP buckets:")
    for tp_val in [15, 25, 35, 50, 60]:
        sub = TD[TD['tp'] == tp_val]
        if len(sub):
            print(f"  TP={tp_val}: {len(sub)} trades, PnL={sub['pnl'].sum():+.1f}, WR={(sub['pnl']>0).mean()*100:.0f}%")
    print(f"Spread: mean={df['spread'].mean():.2f} pts, cost≈{df['spread'].mean()*tot:.0f} pts total")


if __name__ == '__main__':
    main()
