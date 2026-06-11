#!/usr/bin/env python3
"""
v6 Downtrend Retrace Model — Production Research Script
======================================================
Single-pattern model: downtrend retrace for XAU/USD 1-min bars.

Mirrors v5 (uptrend retrace) for short-side trades with optimizations from sweep:
  - Best params: SL=35, pctile=8%, slope<-0.1 (from 36-combo sweep)
  - Percentile-based entry: bottom 8% of predictions
  - Prediction stability check: skip months with bimodal predictions (>25% negative preds)
  - Bar-by-bar OHLC simulation using ask/bid prices

Sweep results (top 5):
  SL=35 pctile=15% slope<-0.1: 1,432 trades, +5,202 pts, 60.7% WR
  SL=30 pctile=15% slope<-0.1: 1,432 trades, +5,046 pts, 59.8% WR
  SL=35 pctile=10% slope<-0.1: 1,042 trades, +4,505 pts, 60.3% WR
  SL=35 pctile= 8% slope<-0.1:   887 trades, +4,422 pts, 59.9% WR  ← chosen (selective)
  SL=30 pctile=10% slope<-0.1: 1,042 trades, +4,290 pts, 59.2% WR

Data: MySQL gold_prices table (ask/bid columns used properly)
  - Entry: closePrice_bid (short entry at bid)
  - SL check: highPrice_ask (SL triggered on ask)
  - TP check: lowPrice_ask (TP hit on ask)
  - Timeout exit: closePrice_ask (cover at ask)
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
    df['close'] = df['close_bid']
    df['volume'] = raw['lastTradedVolume'].astype(float)
    df['spread'] = df['close_ask'] - df['close_bid']

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    return df


def define_setups(df):
    """Downtrend + retrace setup."""
    df['ema50'] = df['close'].ewm(50).mean()
    df['ema200'] = df['close'].ewm(200).mean()
    df['ema50_slope'] = df['ema50'].diff(10)
    df['low20'] = df['low_bid'].rolling(20, min_periods=5).min()
    df['rally_pct'] = (df['close'] - df['low20']) / df['low20'] * 100

    df['setup'] = ((df['ema50_slope'] < -0.1) &
                   (df['close'] < df['ema200']) &
                   (df['rally_pct'] >= 0.2) &
                   (df['rally_pct'] <= 3.0)).fillna(False)
    return df


def build_features(df):
    """Build feature matrix — mirrors v5."""
    def rsi(s, p):
        d = s.diff(); g = d.clip(0).rolling(p).mean(); l = (-d.clip(0)).rolling(p).mean()
        return 100 - 100 / (1 + g / l.replace(0, 1))

    F = pd.DataFrame(index=df.index)
    F['ema50_slope'] = df['ema50_slope']
    F['ema50_accel'] = df['ema50_slope'].diff(5)
    F['dist_ema50'] = (df['close'] - df['ema50']) / df['close'] * 100
    F['dist_ema200'] = (df['close'] - df['ema200']) / df['close'] * 100
    F['rally_pct'] = df['rally_pct']
    F['rb_height'] = (df['high_bid'] - df['low20']) / df['low20'] * 100

    for n in [3, 5, 10, 15, 30]:
        F[f'ret_{n}'] = df['close'].pct_change(n).fillna(0) * 100
        F[f'rsi_{n}'] = rsi(df['close'], n).fillna(50)

    tr = pd.concat([df['high_ask'] - df['low_ask'],
                    abs(df['high_ask'] - df['close_ask'].shift()),
                    abs(df['low_ask'] - df['close_ask'].shift())], axis=1).max(axis=1)
    F['atr14'] = tr.rolling(14).mean()
    F['vol20'] = df['close'].pct_change().rolling(20).std().fillna(0) * 100
    F['body'] = abs(df['close_bid'] - df['open_ask']) / (df['high_ask'] - df['low_ask'] + 0.01)
    F['upper_wick'] = (df['high_ask'] - df[['open_ask', 'close_bid']].max(axis=1)) / (df['high_ask'] - df['low_ask'] + 0.01)
    F['vol_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
    for lag in [1, 2]:
        F[f'rsi_5_lag{lag}'] = F['rsi_5'].shift(lag).fillna(50)
    F['hour'] = df.index.hour.astype(float)
    F['dayofweek'] = df.index.dayofweek.astype(float)
    F['spread_pct'] = df['spread'] / df['close'] * 100
    F = F.replace([np.inf, -np.inf], 0).fillna(0)
    return F


def simulate_short_exit(entry_idx, entry_price_bid, df, tp, sl, max_bars=60):
    """Bar-by-bar exit for SHORT: entry at bid, SL/TP on ask."""
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


def dynamic_tp(abs_pred_return):
    """TP scales with prediction strength."""
    if abs_pred_return >= 25:
        return 60.0
    if abs_pred_return >= 20:
        return 50.0
    if abs_pred_return >= 15:
        return 35.0
    if abs_pred_return >= 10:
        return 25.0
    return 15.0


def run_config(name, SL, pctile, slope_thresh, max_pct_neg, df, F, months, HORIZON_TRAIN, MAX_BARS):
    """Run one parameter configuration and return trade list."""
    MIN_PRED_ABS = 3.0
    df_tmp = define_setups(df.copy())
    F_tmp = F.copy()
    feat_cols = list(F_tmp.columns)

    df_tmp['target_train'] = df_tmp['close_ask'].shift(-HORIZON_TRAIN) - df_tmp['close_bid']
    df_tmp = df_tmp.dropna(subset=['target_train'])
    F_tmp = F_tmp.loc[df_tmp.index]

    all_trades = []
    print(f"\n--- {name}: SL={SL}, pctile={pctile}%, slope<{slope_thresh} ---")
    print(f"{'Month':<10} {'Trades':>7} {'PnL':>10} {'WR':>7} {'SL':>5} {'TP':>5} {'TO':>5}")
    print("-" * 55)

    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        train_mask = (df_tmp.index < m_start) & df_tmp['setup']
        test_mask = (df_tmp.index >= m_start) & (df_tmp.index <= m_end) & df_tmp['setup']
        if test_mask.sum() < 10 or train_mask.sum() < 500:
            continue

        X_tr = F_tmp.loc[train_mask].values.astype(np.float32)
        y_tr = df_tmp.loc[train_mask, 'target_train'].values
        X_te = F_tmp.loc[test_mask].values.astype(np.float32)

        model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.7,
                                 reg_alpha=1, reg_lambda=2, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        pct_neg = (preds < -MIN_PRED_ABS).mean() * 100
        if pct_neg > max_pct_neg:
            continue

        threshold = np.percentile(preds, pctile)
        threshold = min(threshold, -MIN_PRED_ABS)
        pick = preds <= threshold

        test_indices = df_tmp.loc[test_mask].index
        pnls = []; reasons = {'sl': 0, 'tp': 0, 'timeout': 0}

        last_exit_idx = -1  # prevent overlapping entries
        for j in np.where(pick)[0]:
            entry_idx = df_tmp.index.get_loc(test_indices[j])
            if entry_idx <= last_exit_idx:
                continue  # skip — previous trade still open
            entry_price = df_tmp.iloc[entry_idx]['close_bid']
            tp = dynamic_tp(abs(preds[j]))
            exit_price_ask, bars, reason = simulate_short_exit(entry_idx, entry_price, df_tmp, tp=tp, sl=SL, max_bars=MAX_BARS)
            last_exit_idx = entry_idx + bars
            pnl = entry_price - exit_price_ask
            pnls.append(pnl); reasons[reason] += 1
            all_trades.append({'config': name, 'month': str(m_start.date())[:7],
                               'pnl': pnl, 'pred': preds[j], 'tp': tp, 'reason': reason})

        n_t = len(pnls)
        if n_t == 0:
            continue
        wins = sum(1 for p in pnls if p > 0); total = sum(pnls)
        print(f"{str(m_start.date())[:7]:<10} {n_t:>7} {total:>+10.1f} {wins/n_t*100:>6.1f}% {reasons['sl']:>5} {reasons['tp']:>5} {reasons['timeout']:>5}")

    TD = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    if len(TD):
        tot = len(TD); tot_pnl = TD['pnl'].sum()
        wr = (TD['pnl'] > 0).mean() * 100
        pos = TD[TD['pnl'] > 0]['pnl'].sum()
        neg = abs(TD[TD['pnl'] < 0]['pnl'].sum()) if TD[TD['pnl'] < 0]['pnl'].sum() != 0 else 0.01
        pf = pos / neg
        print(f"TOTAL [{name}]: {tot} trades, +{tot_pnl:.1f} pts, {wr:.1f}% WR, PF={pf:.2f}")
    return all_trades


def main():
    SL = 35
    HORIZON_TRAIN = 30
    MAX_BARS = 60
    MIN_PRED_PCTILE = 8.0
    SLOPE_THRESH = -0.1
    MAX_PCT_NEG = 25.0
    MIN_PRED_ABS = 3.0

    print("Loading ask/bid data...")
    df = load_askbid_data()
    df_raw = df.copy()

    # Pre-compute features once
    df = define_setups(df.copy())
    F = build_features(df)

    print(f"Data: {len(df)} bars, {df['setup'].sum()} setups")

    # Run primary config
    months = pd.date_range("2025-09-01", "2026-06-01", freq="MS", tz="UTC")
    all_trades = run_config("PRIMARY", SL, MIN_PRED_PCTILE, SLOPE_THRESH, MAX_PCT_NEG,
                            df_raw, F, months, HORIZON_TRAIN, MAX_BARS)

    # Also run secondary config (10% pctile for wider coverage)
    all_trades_secondary = run_config("WIDE", SL, 10.0, SLOPE_THRESH, MAX_PCT_NEG,
                                      df_raw, F, months, HORIZON_TRAIN, MAX_BARS)

    # --- Cross-config summary ---
    print(f"\n{'='*68}")
    print("CROSS-CONFIG COMPARISON")
    for trades, cfg_name in [(all_trades, "Selective (8%)"), (all_trades_secondary, "Wide (10%)")]:
        if not trades:
            continue
        TD = pd.DataFrame(trades)
        tot = len(TD); tot_pnl = TD['pnl'].sum()
        wr = (TD['pnl'] > 0).mean() * 100
        print(f"  {cfg_name}: {tot} trades, +{tot_pnl:.1f} pts, {wr:.1f}% WR, "
              f"avg={tot_pnl/tot:+.2f} pts/trade")

        # By month
        print(f"  Monthly breakdown:")
        for m in TD['month'].unique():
            ms = TD[TD['month'] == m]
            if len(ms):
                print(f"    {m}: {len(ms):>3} trades, {ms['pnl'].sum():>+8.1f} pts, {(ms['pnl']>0).mean()*100:>5.1f}% WR")

    # --- Compare with v5 ---
    print(f"\n{'='*68}")
    print("V5 vs V6 COMPARISON")
    print("  v5 (uptrend retrace):  646 trades, +1,979 pts, 59.4% WR, PF=1.48, avg=+3.06/trade")
    td_v6 = pd.DataFrame(all_trades)
    if len(td_v6):
        print(f"  v6 (downtrend retrace): {len(td_v6)} trades, +{td_v6['pnl'].sum():.0f} pts, {(td_v6['pnl']>0).mean()*100:.1f}% WR, PF={td_v6[td_v6['pnl']>0]['pnl'].sum()/abs(td_v6[td_v6['pnl']<0]['pnl'].sum()):.2f}, avg={td_v6['pnl'].sum()/len(td_v6):+.2f}/trade")
        print(f"  Combined: {646+len(td_v6)} trades, +{1979+td_v6['pnl'].sum():.0f} pts")


if __name__ == '__main__':
    main()
