#!/usr/bin/env python3
"""
v17 Oil Retrace Model — Port of v5/v6 XGBoost Regressor to Oil
================================================================
Tests whether the gold v5/v6 retrace architecture works on oil 1-min bars.

Oil data: `prices` table, *_ask/*_bid columns (same schema as gold_prices).
DB units / 100 = spot $.

Architecture:
  - Uptrend retrace (long): EMA50 slope>0, price>EMA200, 0.15-3% pullback
  - Downtrend retrace (short): EMA50 slope<-0.1, price<EMA200, 0.2-3% rally
  - XGBoost regressor, walk-forward monthly, dynamic TP, bar-by-bar exit

Oil-specific scaling:
  - Gold ~3000 pts/oz, Oil ~7000 pts/bbl → oil moves ~2.3x gold in DB units
  - Gold ATR ~5-8 pts, Oil ATR ~3-5 pts (in DB units)
  - TP/SL grid calibrated for oil: TP=[100,150,200,250,300], SL=[60,80,100,120]
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


def load_oil_data(start_date="2024-01-01", end_date="2026-06-09"):
    """Load oil 1-min bars with proper ask/bid columns."""
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


def define_setups(df):
    """Define trend + retrace setups (both long and short)."""
    df['ema50'] = df['close'].ewm(50).mean()
    df['ema200'] = df['close'].ewm(200).mean()
    df['ema50_slope'] = df['ema50'].diff(10)
    df['high20'] = df['high_ask'].rolling(20, min_periods=5).max()
    df['low20'] = df['low_bid'].rolling(20, min_periods=5).min()
    df['pullback_pct'] = (df['high20'] - df['close']) / df['high20'] * 100
    df['rally_pct'] = (df['close'] - df['low20']) / df['low20'] * 100

    # Uptrend retrace (long)
    df['setup_long'] = ((df['ema50_slope'] > 0) & (df['close'] > df['ema200']) &
                        (df['pullback_pct'] >= 0.15) & (df['pullback_pct'] <= 3.0)).fillna(False)
    # Downtrend retrace (short)
    df['setup_short'] = ((df['ema50_slope'] < -0.1) & (df['close'] < df['ema200']) &
                         (df['rally_pct'] >= 0.2) & (df['rally_pct'] <= 3.0)).fillna(False)
    df['setup'] = df['setup_long'] | df['setup_short']
    return df


def build_features(df):
    """Build feature matrix — mirrors v5/v6."""
    def rsi(s, p):
        d = s.diff(); g = d.clip(0).rolling(p).mean()
        l = (-d.clip(0)).rolling(p).mean()
        return 100 - 100 / (1 + g / l.replace(0, 1))

    F = pd.DataFrame(index=df.index)
    F['ema50_slope'] = df['ema50_slope']
    F['ema50_accel'] = df['ema50_slope'].diff(5)
    F['dist_ema50'] = (df['close'] - df['ema50']) / df['close'] * 100
    F['dist_ema200'] = (df['close'] - df['ema200']) / df['close'] * 100
    F['pullback_pct'] = df['pullback_pct']
    F['rally_pct'] = df['rally_pct']
    F['pb_depth'] = (df['high20'] - df['low_bid']) / df['high20'] * 100
    F['rb_height'] = (df['high_bid'] - df['low20']) / df['low20'] * 100

    for n in [3, 5, 10, 15, 30]:
        F[f'ret_{n}'] = df['close'].pct_change(n).fillna(0) * 100
        F[f'rsi_{n}'] = rsi(df['close'], n).fillna(50)

    tr = pd.concat([df['high_bid'] - df['low_bid'],
                    abs(df['high_bid'] - df['close_bid'].shift()),
                    abs(df['low_bid'] - df['close_bid'].shift())], axis=1).max(axis=1)
    F['atr14'] = tr.rolling(14).mean()
    F['atr_ratio'] = F['atr14'] / tr.rolling(200).mean()
    F['vol20'] = df['close'].pct_change().rolling(20).std().fillna(0) * 100
    F['body'] = abs(df['close_ask'] - df['open_ask']) / (df['high_ask'] - df['low_ask'] + 0.01)
    F['lower_wick'] = (df[['open_ask', 'close_ask']].min(axis=1) - df['low_ask']) / (df['high_ask'] - df['low_ask'] + 0.01)
    F['upper_wick'] = (df['high_ask'] - df[['open_ask', 'close_ask']].max(axis=1)) / (df['high_ask'] - df['low_ask'] + 0.01)
    F['vol_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
    for lag in [1, 2]:
        F[f'rsi_5_lag{lag}'] = F['rsi_5'].shift(lag).fillna(50)
    F['hour'] = df.index.hour.astype(float)
    F['dayofweek'] = df.index.dayofweek.astype(float)
    F['spread_pct'] = df['spread'] / df['close'] * 100
    F = F.replace([np.inf, -np.inf], 0).fillna(0)
    return F


def sim_long(entry_idx, entry_price_ask, df, tp, sl, max_bars=60):
    """Bar-by-bar LONG exit: entry at ask, SL/TP on bid."""
    stop, target = entry_price_ask - sl, entry_price_ask + tp
    horizon = min(max_bars, len(df) - entry_idx - 1)
    for i in range(1, horizon + 1):
        bar = df.iloc[entry_idx + i]
        if bar['low_bid'] <= stop:    return stop, i, 'sl'
        if bar['high_bid'] >= target: return target, i, 'tp'
    return df.iloc[entry_idx + horizon]['close_bid'], horizon, 'timeout'


def sim_short(entry_idx, entry_price_bid, df, tp, sl, max_bars=60):
    """Bar-by-bar SHORT exit: entry at bid, SL/TP on ask."""
    stop, target = entry_price_bid + sl, entry_price_bid - tp
    horizon = min(max_bars, len(df) - entry_idx - 1)
    for i in range(1, horizon + 1):
        bar = df.iloc[entry_idx + i]
        if bar['high_ask'] >= stop: return stop, i, 'sl'
        if bar['low_ask'] <= target: return target, i, 'tp'
    return df.iloc[entry_idx + horizon]['close_ask'], horizon, 'timeout'


def dynamic_tp(abs_pred):
    """Dynamic TP scaled for oil DB units (oil moves ~2.3x gold in pts)."""
    if abs_pred >= 400: return 500
    if abs_pred >= 300: return 400
    if abs_pred >= 200: return 300
    if abs_pred >= 150: return 250
    if abs_pred >= 100: return 200
    if abs_pred >= 50:  return 150
    return 100


def run_side(name, side, df, F, months, SL, HORIZON, MIN_PRED, MAX_BARS):
    """Walk-forward run for one side (long or short)."""
    all_trades = []

    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        setup_col = f'setup_{side}'
        train_mask = (df.index < m_start) & df[setup_col]
        test_mask = (df.index >= m_start) & (df.index <= m_end) & df[setup_col]

        if test_mask.sum() < 5 or train_mask.sum() < 500:
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
        reasons = {'sl': 0, 'tp': 0, 'timeout': 0}
        month_pnls = []

        if side == 'long':
            pick = preds >= MIN_PRED
            for j in np.where(pick)[0]:
                entry_idx = df.index.get_loc(test_indices[j])
                entry_price = df.iloc[entry_idx]['close_ask']
                tpv = dynamic_tp(preds[j])
                ex, bars, r = sim_long(entry_idx, entry_price, df, tpv, SL, MAX_BARS)
                pnl = ex - entry_price
                month_pnls.append(pnl); reasons[r] += 1
                all_trades.append({'month': str(m_start.date())[:7], 'pnl': pnl,
                                   'pred': preds[j], 'tp': tpv, 'reason': r, 'side': side})
        else:
            pick = preds <= -MIN_PRED
            for j in np.where(pick)[0]:
                entry_idx = df.index.get_loc(test_indices[j])
                if entry_idx + MAX_BARS >= len(df):
                    continue
                entry_price = df.iloc[entry_idx]['close_bid']
                tpv = dynamic_tp(abs(preds[j]))
                ex, bars, r = sim_short(entry_idx, entry_price, df, tpv, SL, MAX_BARS)
                pnl = entry_price - ex
                month_pnls.append(pnl); reasons[r] += 1
                all_trades.append({'month': str(m_start.date())[:7], 'pnl': pnl,
                                   'pred': preds[j], 'tp': tpv, 'reason': r, 'side': side})

        if month_pnls:
            n = len(month_pnls)
            wr = sum(1 for p in month_pnls if p > 0) / n * 100
            print(f"  {str(m_start.date())[:7]:<10} {n:>5}t {sum(month_pnls):>+10.1f} {wr:>6.1f}% WR  SL:{reasons['sl']:>3} TP:{reasons['tp']:>3} TO:{reasons['timeout']:>3}")

    return all_trades


# =============================================================================
# TP/SL Sweep
# =============================================================================
def sweep_tp_sl(df, F, months, HORIZON, MIN_PRED, MAX_BARS):
    """Grid-sweep TP/SL to find optimal oil-specific parameters."""
    tps = [100, 150, 200, 250, 300]
    sls = [60, 80, 100, 120, 150]

    print(f"\n{'='*80}")
    print("TP/SL SWEEP for Oil")
    print(f"{'='*80}")
    print(f"{'TP':>5s} {'SL':>5s} {'Trades':>7s} {'PnL':>12s} {'WR':>7s} {'PF':>6s} {'Avg':>8s}")
    print("-" * 65)

    sweep_results = []

    for tp, sl in [(t, s) for t in tps for s in sls]:
        # Quick eval: use last 3 months as validation
        val_start = months[-4]
        train_mask = (df.index < val_start) & df['setup']
        test_mask = (df.index >= val_start) & df['setup']

        if test_mask.sum() < 10 or train_mask.sum() < 1000:
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

        pnls = []
        # Longs
        long_pick = preds >= MIN_PRED
        for j in np.where(long_pick)[0]:
            ei = df.index.get_loc(test_indices[j])
            if ei + MAX_BARS >= len(df): continue
            ep = df.iloc[ei]['close_ask']
            ex, _, _ = sim_long(ei, ep, df, tp, sl, MAX_BARS)
            pnls.append(ex - ep)

        # Shorts
        short_pick = preds <= -MIN_PRED
        for j in np.where(short_pick)[0]:
            ei = df.index.get_loc(test_indices[j])
            if ei + MAX_BARS >= len(df): continue
            ep = df.iloc[ei]['close_bid']
            ex, _, _ = sim_short(ei, ep, df, tp, sl, MAX_BARS)
            pnls.append(ep - ex)

        if len(pnls) < 5: continue
        a = np.array(pnls)
        tot = len(a); total = a.sum(); wr = (a > 0).mean() * 100
        pos = a[a > 0].sum(); neg = abs(a[a < 0].sum())
        pf = pos / neg if neg > 0 else 99
        avg = total / tot

        print(f"  {tp:>5.0f} {sl:>5.0f} {tot:>7d} {total:>+12.1f} {wr:>6.1f}% {pf:>5.2f} {avg:>+8.2f}")
        sweep_results.append({'tp': tp, 'sl': sl, 'trades': tot, 'pnl': total,
                              'wr': wr, 'pf': pf, 'avg': avg})

    if sweep_results:
        best = max(sweep_results, key=lambda r: r['pnl'])
        print(f"\n  Best: TP={best['tp']}, SL={best['sl']} → {best['trades']}t, {best['pnl']:+.0f}pts, {best['wr']:.1f}% WR, PF={best['pf']:.2f}")
        return best
    return {'tp': 200, 'sl': 100}


def main():
    HORIZON, MIN_PRED, MAX_BARS = 30, 40, 60
    print("=" * 72)
    print("v17 Oil Retrace Model (v5/v6 architecture with ask/bid)")
    print(f"  SL swept, Horizon={HORIZON}, MinPred={MIN_PRED}")
    print("=" * 72)

    print("\n[1/4] Loading oil data with ask/bid...")
    df = load_oil_data()
    print(f"  {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")
    print(f"  Price range: {df['close'].min():.0f} -> {df['close'].max():.0f} DB units (~${df['close'].min()/100:.1f} -> ${df['close'].max()/100:.1f})")
    print(f"  Spread: mean={df['spread'].mean():.2f} pts, max={df['spread'].max():.2f} pts")

    print("[2/4] Setups & features...")
    df = define_setups(df)
    F = build_features(df)
    df['target_train'] = df['close_bid'].shift(-HORIZON) - df['close_ask']
    df = df.dropna(subset=['target_train'])
    F = F.loc[df.index]
    print(f"  {len(df):,} bars, setups={df['setup'].sum():,} ({df['setup_long'].sum():,}L/{df['setup_short'].sum():,}S)")

    months = pd.date_range("2025-01-01", "2026-06-01", freq="MS", tz="UTC")

    # Sweep TP/SL on validation period
    print("[3/4] Sweeping TP/SL...")
    best_tpsl = sweep_tp_sl(df, F, months, HORIZON, MIN_PRED, MAX_BARS)
    SL = best_tpsl['sl']

    # Full walk-forward with best SL
    print(f"\n[4/4] Full walk-forward (SL={SL})...")
    print(f"\n  {'='*60}")
    print(f"  LONG (uptrend retrace)")
    print(f"  {'Month':<10} {'Trades':>5} {'PnL':>10} {'WR':>7}")
    print(f"  {'-'*40}")
    long_trades = run_side('long', 'long', df, F, months, SL, HORIZON, MIN_PRED, MAX_BARS)

    print(f"\n  {'='*60}")
    print(f"  SHORT (downtrend retrace)")
    print(f"  {'Month':<10} {'Trades':>5} {'PnL':>10} {'WR':>7}")
    print(f"  {'-'*40}")
    short_trades = run_side('short', 'short', df, F, months, SL, HORIZON, MIN_PRED, MAX_BARS)

    all_trades = long_trades + short_trades
    TD = pd.DataFrame(all_trades)

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    if len(TD) == 0:
        print("NO TRADES.")
        return

    tot = len(TD); tot_pnl = TD['pnl'].sum()
    wr = (TD['pnl'] > 0).mean() * 100
    pos = TD[TD['pnl'] > 0]['pnl'].sum()
    neg = abs(TD[TD['pnl'] < 0]['pnl'].sum())
    pf = pos / neg if neg > 0 else 99

    print(f"  Combined: {tot}t, {tot_pnl:+.0f} pts, {wr:.1f}% WR, PF={pf:.2f}, avg={tot_pnl/tot:+.1f} pts/trade")
    print(f"  Spot $: ~${tot_pnl/100:+.2f}/contract")

    for side in ['long', 'short']:
        ss = TD[TD['side'] == side]
        if len(ss):
            s_pf = ss[ss['pnl'] > 0]['pnl'].sum() / abs(ss[ss['pnl'] < 0]['pnl'].sum()) if len(ss[ss['pnl'] < 0]) > 0 else 99
            print(f"  {side}: {len(ss)}t, {ss['pnl'].sum():+.0f} pts, {(ss['pnl']>0).mean()*100:.0f}% WR, PF={s_pf:.2f}")

    print(f"\n  Monthly:")
    for m in sorted(TD['month'].unique()):
        ms = TD[TD['month'] == m]
        if len(ms):
            print(f"    {m}: {len(ms):>3}t, {ms['pnl'].sum():>+10.0f}, {(ms['pnl']>0).mean()*100:>5.0f}% WR")

    print(f"\n  TP buckets:")
    for tp in sorted(TD['tp'].unique()):
        sub = TD[TD['tp'] == tp]
        print(f"    TP={tp:>5.0f}: {len(sub):>3}t, {sub['pnl'].sum():>+10.0f}, {(sub['pnl']>0).mean()*100:>5.0f}% WR")

    mj = TD[TD['month'].isin(['2026-05', '2026-06'])]
    if len(mj):
        print(f"\n  May-June 2026: {len(mj)}t, {mj['pnl'].sum():+.0f}, {(mj['pnl']>0).mean()*100:.0f}% WR")
    else:
        print(f"\n  May-June 2026: 0 trades")

    print(f"\nDONE.")


if __name__ == '__main__':
    main()
