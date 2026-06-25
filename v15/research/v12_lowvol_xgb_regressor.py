#!/usr/bin/env python3
"""
v12 Low-Vol XGBoost Regressor — Production Research Script
============================================================
Follows v5/v6 architecture exactly (XGBoost regressor, walk-forward,
dynamic TP, bar-by-bar exit), but with a LOW-VOL setup filter instead
of trend+retrace.

Architecture (mirrors v5/v6):
  1. Setup filter: low-vol regime (ATR<3.5, weak trend, v5/v6 setups scarce)
  2. XGBoost regression: predict 30-bar forward return
  3. Entry: LONG on pred >= MIN_PRED, SHORT on pred <= -MIN_PRED
  4. Exit: dynamic TP + fixed SL + 60-bar timeout
  5. Bar-by-bar OHLC simulation with proper ask/bid

Config sweep: SL=[20,30] x MIN_PRED=[5,10] = 4 configs
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    df['close'] = df['close_ask']
    df['volume'] = raw['lastTradedVolume'].astype(float)
    df['spread'] = df['close_ask'] - df['close_bid']
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def define_setups(df):
    df['ema50'] = df['close'].ewm(50).mean()
    df['ema200'] = df['close'].ewm(200).mean()
    df['ema50_slope'] = df['ema50'].diff(10)
    tr = pd.concat([
        df['high_ask'] - df['low_ask'],
        abs(df['high_ask'] - df['close_ask'].shift()),
        abs(df['low_ask'] - df['close_ask'].shift())
    ], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()
    df['range_high'] = df['high_ask'].rolling(50, min_periods=10).max()
    df['range_low'] = df['low_ask'].rolling(50, min_periods=10).min()
    df['pos_in_range'] = ((df['close'] - df['range_low']) /
                          (df['range_high'] - df['range_low'] + 0.001))
    df['high20'] = df['high_ask'].rolling(20, min_periods=5).max()
    df['low20'] = df['low_bid'].rolling(20, min_periods=5).min()
    df['dip_pct'] = (df['high20'] - df['close']) / df['high20'] * 100
    df['rally_pct'] = (df['close'] - df['low20']) / df['low20'] * 100
    up_retrace = ((df['ema50_slope'] > 0) & (df['close'] > df['ema200']) &
                  (df['dip_pct'] >= 0.15) & (df['dip_pct'] <= 3.0))
    dn_retrace = ((df['ema50_slope'] < -0.1) & (df['close'] < df['ema200']) &
                  (df['rally_pct'] >= 0.2) & (df['rally_pct'] <= 3.0))
    retrace_count = (up_retrace | dn_retrace).rolling(60, min_periods=1).sum()
    low_vol = df['atr14'] < 3.5
    weak_trend = df['ema50_slope'].abs() < 0.15
    scarce = retrace_count < 10
    in_lowvol = (low_vol & weak_trend & scarce).fillna(False)
    df['setup_long'] = in_lowvol & (df['pos_in_range'] < 0.35)
    df['setup_short'] = in_lowvol & (df['pos_in_range'] > 0.65)
    df['setup'] = df['setup_long'] | df['setup_short']
    return df


def build_features(df):
    def rsi(s, p):
        d = s.diff(); g = d.clip(0).rolling(p).mean()
        l = (-d.clip(0)).rolling(p).mean()
        return 100 - 100 / (1 + g / l.replace(0, 1))
    F = pd.DataFrame(index=df.index)
    F['ema50_slope'] = df['ema50_slope']
    F['ema50_accel'] = df['ema50_slope'].diff(5)
    F['dist_ema50'] = (df['close'] - df['ema50']) / df['close'] * 100
    F['dist_ema200'] = (df['close'] - df['ema200']) / df['close'] * 100
    F['pos_in_range'] = df['pos_in_range']
    F['range_width'] = (df['range_high'] - df['range_low']) / df['close'] * 100
    for n in [3, 5, 10, 15, 30]:
        F[f'ret_{n}'] = df['close'].pct_change(n).fillna(0) * 100
        F[f'rsi_{n}'] = rsi(df['close'], n).fillna(50)
    tr = pd.concat([
        df['high_bid'] - df['low_bid'],
        abs(df['high_bid'] - df['close_bid'].shift()),
        abs(df['low_bid'] - df['close_bid'].shift())
    ], axis=1).max(axis=1)
    F['atr14'] = tr.rolling(14).mean()
    F['atr_ratio'] = F['atr14'] / tr.rolling(200).mean()
    F['vol20'] = df['close'].pct_change().rolling(20).std().fillna(0) * 100
    F['body'] = abs(df['close_ask'] - df['open_ask']) / (df['high_ask'] - df['low_ask'] + 0.01)
    F['lower_wick'] = (df[['open_ask', 'close_ask']].min(axis=1) - df['low_ask']) / (df['high_ask'] - df['low_ask'] + 0.01)
    F['upper_wick'] = (df['high_ask'] - df[['open_ask', 'close_ask']].max(axis=1)) / (df['high_ask'] - df['low_ask'] + 0.01)
    F['vol_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
    for lag in [1, 2]:
        F[f'rsi_5_lag{lag}'] = F['rsi_5'].shift(lag).fillna(50)
        F[f'ret_3_lag{lag}'] = F['ret_3'].shift(lag).fillna(0)
    F['hour'] = df.index.hour.astype(float)
    F['dayofweek'] = df.index.dayofweek.astype(float)
    F['spread_pct'] = df['spread'] / df['close'] * 100
    F = F.replace([np.inf, -np.inf], 0).fillna(0)
    return F


def simulate_long_exit(entry_idx, entry_price_ask, df, tp, sl, max_bars=60):
    stop, target = entry_price_ask - sl, entry_price_ask + tp
    horizon = min(max_bars, len(df) - entry_idx - 1)
    for i in range(1, horizon + 1):
        bar = df.iloc[entry_idx + i]
        if bar['low_bid'] <= stop:
            return stop, i, 'sl'
        if bar['high_bid'] >= target:
            return target, i, 'tp'
    return df.iloc[entry_idx + horizon]['close_bid'], horizon, 'timeout'


def simulate_short_exit(entry_idx, entry_price_bid, df, tp, sl, max_bars=60):
    stop, target = entry_price_bid + sl, entry_price_bid - tp
    horizon = min(max_bars, len(df) - entry_idx - 1)
    for i in range(1, horizon + 1):
        bar = df.iloc[entry_idx + i]
        if bar['high_ask'] >= stop:
            return stop, i, 'sl'
        if bar['low_ask'] <= target:
            return target, i, 'tp'
    return df.iloc[entry_idx + horizon]['close_ask'], horizon, 'timeout'


def dynamic_tp(abs_pred_return):
    if abs_pred_return >= 25:
        return 60.0
    if abs_pred_return >= 20:
        return 50.0
    if abs_pred_return >= 15:
        return 35.0
    if abs_pred_return >= 10:
        return 25.0
    if abs_pred_return >= 7:
        return 20.0
    return 12.0


def summarize_trades(trades, label):
    if not trades:
        return None
    TD = pd.DataFrame(trades)
    tot = len(TD); tot_pnl = TD['pnl'].sum(); wr = (TD['pnl'] > 0).mean() * 100
    pos = TD[TD['pnl'] > 0]['pnl'].sum(); neg = abs(TD[TD['pnl'] < 0]['pnl'].sum())
    pf = pos / neg if neg > 0 else 99
    n_long = (TD['side'] == 'long').sum(); n_short = (TD['side'] == 'short').sum()
    mj = TD[TD['month'].isin(['2026-05', '2026-06'])]
    return {
        'label': label, 'trades': tot, 'pnl': round(tot_pnl, 1), 'wr': round(wr, 1),
        'pf': round(pf, 2), 'avg': round(tot_pnl / tot, 2),
        'longs': n_long, 'shorts': n_short,
        'mj_trades': len(mj), 'mj_pnl': round(mj['pnl'].sum(), 1),
    }


def run_config(sl, min_pred, df, F, months, horizon, max_bars):
    """Run one SL/MIN_PRED config, return trades list."""
    all_trades = []
    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        train_mask = (df.index < m_start) & df['setup']
        test_mask_long = (df.index >= m_start) & (df.index <= m_end) & df['setup_long']
        test_mask_short = (df.index >= m_start) & (df.index <= m_end) & df['setup_short']
        total_setups = test_mask_long.sum() + test_mask_short.sum()
        if total_setups < 20 or train_mask.sum() < 500:
            continue

        X_tr = F.loc[train_mask].values.astype(np.float32)
        y_tr = df.loc[train_mask, 'target'].values

        model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.7,
                                 reg_alpha=1, reg_lambda=2, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)

        # Long side
        if test_mask_long.sum() > 0:
            X_te_l = F.loc[test_mask_long].values.astype(np.float32)
            preds_l = model.predict(X_te_l)
            test_idx_l = df.loc[test_mask_long].index
            for j in np.where(preds_l >= min_pred)[0]:
                entry_idx = df.index.get_loc(test_idx_l[j])
                ep = df.iloc[entry_idx]['close_ask']
                tp = dynamic_tp(preds_l[j])
                ex, bars, reason = simulate_long_exit(entry_idx, ep, df, tp=tp, sl=sl, max_bars=max_bars)
                all_trades.append({'month': str(m_start.date())[:7], 'pnl': ex - ep,
                                   'pred': preds_l[j], 'tp': tp, 'reason': reason, 'side': 'long'})

        # Short side
        if test_mask_short.sum() > 0:
            X_te_s = F.loc[test_mask_short].values.astype(np.float32)
            preds_s = model.predict(X_te_s)
            test_idx_s = df.loc[test_mask_short].index
            for j in np.where(preds_s <= -min_pred)[0]:
                entry_idx = df.index.get_loc(test_idx_s[j])
                ep = df.iloc[entry_idx]['close_bid']
                tp = dynamic_tp(abs(preds_s[j]))
                ex, bars, reason = simulate_short_exit(entry_idx, ep, df, tp=tp, sl=sl, max_bars=max_bars)
                all_trades.append({'month': str(m_start.date())[:7], 'pnl': ep - ex,
                                   'pred': abs(preds_s[j]), 'tp': tp, 'reason': reason, 'side': 'short'})

    return all_trades


def main():
    HORIZON = 30
    MAX_BARS = 60
    configs = [
        (20, 5.0, 'SL=20 MP=5'),
        (20, 10.0, 'SL=20 MP=10'),
        (30, 5.0, 'SL=30 MP=5'),
        (30, 10.0, 'SL=30 MP=10'),
    ]

    print("=" * 72)
    print("v12 Low-Vol XGBoost Regressor — Config Sweep")
    print(f"  Horizon={HORIZON}, MaxBars={MAX_BARS}")
    print("=" * 72)

    print("\n[1/3] Loading & computing...")
    df = load_askbid_data()
    df = define_setups(df)
    F = build_features(df)
    df['target'] = df['close_bid'].shift(-HORIZON) - df['close_ask']
    df = df.dropna(subset=['target'])
    F = F.loc[df.index]
    print(f"  {len(df):,} bars, {df['setup'].sum():,} setups ({df['setup_long'].sum():,}L/{df['setup_short'].sum():,}S)")

    print("\n[2/3] Config sweep...")
    months = pd.date_range("2025-09-01", "2026-06-01", freq="MS", tz="UTC")
    all_summaries = []

    for sl, min_pred, label in configs:
        print(f"  {label}...", end=" ", flush=True)
        trades = run_config(sl, min_pred, df, F, months, HORIZON, MAX_BARS)
        summary = summarize_trades(trades, label)
        if summary:
            all_summaries.append(summary)
            print(f"{summary['trades']}t, {summary['pnl']:+.1f}pt, {summary['wr']:.1f}% WR, PF={summary['pf']:.2f}, MJ: {summary['mj_trades']}t/{summary['mj_pnl']:+.1f}pt")
        else:
            print("NO TRADES")

    if not all_summaries:
        print("\nNo trades in any config.")
        return

    # --- Cross-config comparison ---
    print(f"\n[3/3] Config comparison:")
    print(f"{'Config':<18s} {'Trades':>6s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} {'Avg':>7s} {'L':>5s} {'S':>5s} {'MJ_T':>6s} {'MJ_PnL':>8s}")
    print("-" * 85)
    for s in all_summaries:
        print(f"{s['label']:<18s} {s['trades']:>6d} {s['pnl']:>+10.1f} {s['wr']:>6.1f}% {s['pf']:>5.2f} {s['avg']:>+7.2f} {s['longs']:>5d} {s['shorts']:>5d} {s['mj_trades']:>6d} {s['mj_pnl']:>+8.1f}")

    # --- Best config monthly detail ---
    best = max(all_summaries, key=lambda s: s['pnl'])
    best_sl = int(best['label'].split('=')[1].split()[0])
    best_mp = float(best['label'].split('=')[2])
    print(f"\nBest config: {best['label']}")

    trades = run_config(best_sl, best_mp, df, F, months, HORIZON, MAX_BARS)
    TD = pd.DataFrame(trades)

    print(f"\n  Monthly breakdown:")
    for m in sorted(TD['month'].unique()):
        ms = TD[TD['month'] == m]
        m_wr = (ms['pnl'] > 0).mean() * 100
        print(f"    {m}: {len(ms):>3} trades, {ms['pnl'].sum():>+8.1f} pts, {m_wr:>5.1f}% WR")

    print(f"\n  By side:")
    for side in ['long', 'short']:
        ss = TD[TD['side'] == side]
        if len(ss):
            s_wr = (ss['pnl'] > 0).mean() * 100
            print(f"    {side}: {len(ss):>3} trades, {ss['pnl'].sum():>+8.1f} pts, {s_wr:>5.1f}% WR")

    # TP buckets
    print(f"\n  TP buckets:")
    for tp_val in [12, 20, 25, 35, 50, 60]:
        sub = TD[TD['tp'] == tp_val]
        if len(sub) > 0:
            s_wr = (sub['pnl'] > 0).mean() * 100
            print(f"    TP={tp_val:>3.0f}: {len(sub):>3} trades, {sub['pnl'].sum():>+8.1f}, {s_wr:>5.0f}% WR")

    # Pred strength
    TD['pred_bucket'] = pd.cut(TD['pred'], bins=[5, 10, 15, 20, 100])
    print(f"\n  Pred strength:")
    for bucket, grp in TD.groupby('pred_bucket', observed=True):
        b_wr = (grp['pnl'] > 0).mean() * 100
        print(f"    Pred {bucket}: {len(grp):>3} trades, {grp['pnl'].sum():>+8.1f}, {b_wr:>5.0f}% WR")

    mj = TD[TD['month'].isin(['2026-05', '2026-06'])]
    if len(mj):
        print(f"\n  May-June 2026: {len(mj)} trades, {mj['pnl'].sum():+.1f} pts, {(mj['pnl']>0).mean()*100:.1f}% WR")
    else:
        print(f"\n  May-June 2026: NO TRADES")

    print(f"\nDONE.")


if __name__ == '__main__':
    main()
