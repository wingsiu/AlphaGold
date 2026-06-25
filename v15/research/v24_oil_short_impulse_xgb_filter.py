#!/usr/bin/env python3
"""
v24 Oil Short Impulse + XGBoost Probability Filter — TP/SL Sweep
==================================================================
Extends v22/v23 winning short impulse pattern with an XGBoost win/loss
classifier, following gold's pattern_training.py approach.

Workflow:
  1. Load data & compute features (once)
  2. For each TP/SL combo: generate signals, sim trades, build XGBoost features,
     walk-forward train, sweep prob thresholds
  3. Report best per TP/SL
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from data.data_loader import DataLoader

# =============================================================================
# Data Loading
# =============================================================================


def load_oil_data(start_date="2024-01-01", end_date="2026-05-22"):
    loader = DataLoader()
    raw = loader.load_data(table_name="prices", start_date=start_date, end_date=end_date)
    raw.index = pd.to_datetime(raw["timestamp"], unit="ms")

    df = pd.DataFrame(index=raw.index)
    df["open_ask"] = raw["openPrice_ask"].astype(float)
    df["high_bid"] = raw["highPrice_bid"].astype(float)
    df["low_bid"] = raw["lowPrice_bid"].astype(float)
    df["high_ask"] = raw["highPrice_ask"].astype(float)
    df["low_ask"] = raw["lowPrice_ask"].astype(float)
    df["close_ask"] = raw["closePrice_ask"].astype(float)
    df["close_bid"] = raw["closePrice_bid"].astype(float)
    df["close"] = df["close_ask"]
    df["volume"] = raw["lastTradedVolume"].astype(float)
    df["spread"] = df["close_ask"] - df["close_bid"]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


# =============================================================================
# Feature Engineering
# =============================================================================


def compute_features(df_1m):
    """Compute short impulse features + extras for XGBoost prediction."""
    df = df_1m.copy()

    df["change"] = df["close_ask"] - df["open_ask"]
    df["prev_change"] = df["change"].shift(1)
    df["prev2_change"] = df["change"].shift(2)
    df["prev_lower_wick"] = df["close_ask"].shift(1) - df["low_ask"].shift(1)
    df["prev_upper_wick"] = df["high_ask"].shift(1) - df["close_ask"].shift(1)
    df["prev_volume"] = df["volume"].shift(1)
    df["prev_range"] = df["high_ask"].shift(1) - df["low_ask"].shift(1)
    df["prev_spread"] = df["spread"].shift(1)

    tr = pd.concat(
        [df["high_ask"] - df["low_ask"],
         abs(df["high_ask"] - df["close_ask"].shift()),
         abs(df["low_ask"] - df["close_ask"].shift())], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_ratio"] = df["prev_range"] / (df["ATR"] + 0.01)

    daily_high = df["high_ask"].resample("D").max()
    df["day_high"] = np.nan
    for day_idx in daily_high.index:
        mask = df.index.date == day_idx.date()
        df.loc[mask, "day_high"] = daily_high.loc[day_idx]
    df["dist_day_high"] = df["day_high"] - df["close_ask"]

    daily_range = (
        df.resample("D")
        .agg({"high_ask": "max", "low_ask": "min", "open_ask": "first", "close_ask": "last"})
        .dropna())
    daily_range["range"] = daily_range["high_ask"] - daily_range["low_ask"]
    daily_range["avg_range_5d"] = daily_range["range"].rolling(5, min_periods=3).mean()
    df["day_open"] = np.nan
    df["avg_range_5d"] = np.nan
    for day_idx in daily_range.index:
        mask = df.index.date == day_idx.date()
        df.loc[mask, "day_open"] = daily_range.loc[day_idx, "open_ask"]
        df.loc[mask, "avg_range_5d"] = daily_range.loc[day_idx, "avg_range_5d"]
    df["fullness"] = (df["close_ask"] - df["day_open"]) / (df["avg_range_5d"] + 0.01)

    df_15 = (df.resample("15min", label="right", closed="right")
             .agg({"open_ask": "first", "close_ask": "last"}).dropna())
    df_15["up"] = 0
    df_15.loc[df_15["close_ask"] > df_15["open_ask"], "up"] = 1
    df_15.loc[df_15["close_ask"] < df_15["open_ask"], "up"] = -1
    df_15["up_count3"] = df_15["up"].rolling(3, min_periods=1).sum()
    df["up_count3_15min"] = np.nan
    for idx_15 in df_15.index:
        next_start = idx_15 + pd.Timedelta(minutes=15)
        mask = (df.index >= idx_15) & (df.index < next_start)
        df.loc[mask, "up_count3_15min"] = df_15.loc[idx_15, "up_count3"]

    df_15_ext = (df.resample("15min", label="right", closed="right")
                 .agg({"open_ask": "first", "close_ask": "last", "high_ask": "max",
                       "low_ask": "min", "volume": "sum"}).dropna())
    df_15_ext["ret"] = df_15_ext["close_ask"].pct_change()
    df_15_ext["ret_3"] = df_15_ext["ret"].rolling(3, min_periods=1).sum()
    df_15_ext["ret_5"] = df_15_ext["ret"].rolling(5, min_periods=1).sum()
    df["ret_3_15m"] = np.nan
    df["ret_5_15m"] = np.nan
    for idx_15 in df_15_ext.index:
        next_start = idx_15 + pd.Timedelta(minutes=15)
        mask = (df.index >= idx_15) & (df.index < next_start)
        df.loc[mask, "ret_3_15m"] = df_15_ext.loc[idx_15, "ret_3"]
        df.loc[mask, "ret_5_15m"] = df_15_ext.loc[idx_15, "ret_5"]

    df["is_us"] = df.index.hour.isin([12, 13, 14, 15, 16, 17, 18, 19, 20])
    df["is_uk"] = df.index.hour.isin([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    df["in_session"] = df["is_us"] | df["is_uk"]
    df["uk_7_16"] = df.index.hour.isin([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    df["us_7_13"] = df.index.hour.isin([7, 8, 9, 10, 11, 12])
    df["vol_ma_20"] = df["volume"].rolling(20, min_periods=5).mean()
    df["vol_ratio_20"] = df["prev_volume"] / (df["vol_ma_20"] + 0.01)
    df["ret_1m"] = df["close_ask"].pct_change()
    df["ret_3m"] = df["ret_1m"].rolling(3, min_periods=1).sum()
    df["ret_5m"] = df["ret_1m"].rolling(5, min_periods=1).sum()

    return df


# =============================================================================
# Signal Generation
# =============================================================================

SIGNAL_CFG = {
    "change_max": -14.0,
    "prev2_max": 10.0,
    "prev2_min": -14.0,
    "lower_wick_max": 35.0,
    "volume_min": 800.0,
    "dist_high_max": 180.0,
    "in_session": True,
    "uk_only": True,
}


def generate_signals(df, cfg=None):
    if cfg is None:
        cfg = SIGNAL_CFG
    mask = (
        (df["prev_change"] < cfg["change_max"])
        & (df["prev2_change"] < cfg["prev2_max"])
        & (df["prev2_change"] > cfg["prev2_min"])
        & (df["prev_lower_wick"] < cfg["lower_wick_max"])
        & (df["prev_volume"] > cfg["volume_min"])
        & (df["up_count3_15min"] != -3)
        & (df["dist_day_high"] < cfg["dist_high_max"]))
    if cfg.get("uk_only"):
        mask &= df["is_uk"]
    elif cfg.get("us_only"):
        mask &= df["is_us"]
    elif cfg.get("in_session", True):
        mask &= df["in_session"]
    return mask


# =============================================================================
# Forward Simulation
# =============================================================================

MAX_BARS = 60


def sim_short(ei, ep, df, tp, sl, max_bars=MAX_BARS):
    stop, target = ep + sl, ep - tp
    horizon = min(max_bars, len(df) - ei - 1)
    for i in range(1, horizon + 1):
        b = df.iloc[ei + i]
        if b["high_ask"] >= stop:
            return stop, i, "sl"
        if b["low_ask"] <= target:
            return target, i, "tp"
    return df.iloc[ei + horizon]["close_ask"], horizon, "timeout"


def evaluate(mask, df, tp, sl, max_bars=MAX_BARS):
    trades = []
    records = []
    for sig_idx in df.index[mask]:
        ei = df.index.get_loc(sig_idx)
        if ei + max_bars >= len(df):
            continue
        ep = df.iloc[ei]["close_bid"]
        ex, bars_held, reason = sim_short(ei, ep, df, tp, sl, max_bars)
        pnl = ep - ex
        trades.append({"pnl": pnl, "reason": reason, "entry_idx": sig_idx, "entry_pos": ei})
        records.append({"entry_idx": sig_idx, "entry_pos": ei, "entry_price": ep,
                        "exit_price": ex, "pnl": pnl, "reason": reason, "bars_held": bars_held})
    return trades, records


# =============================================================================
# XGBoost Features
# =============================================================================

XGB_FEATURES = [
    "prev_change", "prev2_change", "prev_lower_wick", "prev_upper_wick",
    "prev_volume", "prev_range", "prev_spread", "ATR", "ATR_ratio",
    "dist_day_high", "fullness", "up_count3_15min", "ret_3_15m", "ret_5_15m",
    "ret_1m", "ret_3m", "ret_5m", "vol_ratio_20", "is_us", "hour",
]


def extract_signal_features(df, records):
    features = []
    for i, rec in enumerate(records):
        idx = rec["entry_idx"]
        row = df.loc[idx]
        feat = {}
        for col in XGB_FEATURES:
            if col == "hour":
                feat[col] = idx.hour
            elif col == "is_us":
                feat[col] = int(row.get(col, 0))
            else:
                feat[col] = float(row.get(col, np.nan))
        feat["signal_index"] = i
        feat["entry_idx"] = idx
        features.append(feat)
    X = pd.DataFrame(features)
    valid = X[XGB_FEATURES].notna().all(axis=1)
    return X[valid].reset_index(drop=True)


# =============================================================================
# Walk-Forward Training
# =============================================================================


def train_wf_and_evaluate(df, signal_mask, records, X, prob_threshold=0.55):
    y = np.array([1.0 if r["pnl"] > 0 else 0.0 for r in records])
    n = len(X)
    if n < 20:
        return None

    all_months = sorted(set(X["entry_idx"].dt.to_period("M")))
    test_months = [m for m in all_months if m >= pd.Period("2024-07", freq="M")]
    if not test_months:
        return None

    X_all = X[XGB_FEATURES].astype(float).values
    probas = np.zeros(n)
    trained_in_test = np.zeros(n, dtype=bool)

    for test_month in test_months:
        train_months = [m for m in all_months if m < test_month]
        if not train_months:
            continue
        train_mask = X["entry_idx"].dt.to_period("M").isin(train_months)
        test_mask = X["entry_idx"].dt.to_period("M") == test_month
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) < 20 or len(test_idx) < 3:
            continue

        train_y = y[train_idx]
        win_idx = train_idx[train_y == 1]
        loss_idx = train_idx[train_y == 0]
        if len(loss_idx) > len(win_idx) and len(win_idx) > 0:
            rng = np.random.RandomState(42 + test_month.ordinal)
            loss_idx = rng.choice(loss_idx, len(win_idx), replace=False)
            train_idx = np.concatenate([win_idx, loss_idx])

        X_train = X_all[train_idx]
        y_train = y[train_idx]
        X_test = X_all[test_idx]
        scale = max(1.0, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1))

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
            random_state=42, eval_metric="logloss")
        model.fit(X_train, y_train)
        probas[test_idx] = model.predict_proba(X_test)[:, 1]
        trained_in_test[test_idx] = True

    test_mask = trained_in_test
    n_test = test_mask.sum()
    if n_test == 0:
        return None

    test_indices = X.loc[test_mask, "signal_index"].values
    return {"probas": probas[test_mask], "labels": y[test_mask], "test_indices": test_indices}


# =============================================================================
# Stats helper
# =============================================================================

def stats(pnls):
    if not pnls:
        return {"trades": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0}
    n = len(pnls); total = sum(pnls)
    wr = sum(1 for p in pnls if p > 0) / n * 100 if n else 0
    ps = sum(p for p in pnls if p > 0)
    ns = abs(sum(p for p in pnls if p < 0))
    pf = ps / ns if ns > 0 else 99
    return {"trades": n, "pnl": total, "wr": wr, "pf": pf}


# =============================================================================
# Main — TP/SL Sweep
# =============================================================================


def main():
    print("=" * 72)
    print("v24 Oil Short Impulse + XGBoost — TP/SL Sweep")
    print(f"  vol>800, UK 7-16")
    print("=" * 72)

    print("\n[1/3] Loading & computing features (once)...")
    df = load_oil_data()
    df = compute_features(df)
    df = df.dropna(subset=["ATR", "day_high", "up_count3_15min", "prev_change",
                           "fullness", "ret_1m", "vol_ratio_20", "ret_3_15m"])
    print(f"  {len(df):,} bars ready")
    print(f"  {generate_signals(df).sum():,} UK signals at vol>800")

    tpsl_combos = [(60, 40), (70, 40), (80, 40), (70, 50), (80, 50), (80, 60), (90, 60)]
    print(f"\n[2/3] Sweeping {len(tpsl_combos)} TP/SL combos...")
    print(f"  {'TP/SL':<12s} {'BaseTrd':>8s} {'PnL':>10s} {'WR':>7s} {'PF':>6s} │ "
          f"{'Prob':>5s} {'FiltTrd':>8s} {'FiltPnL':>10s} {'FW':>7s} {'FPF':>6s} {'AUC':>6s} {'dPnl':>8s}")
    print(f"  {'─' * 120}")

    summary = []
    for tp, sl in tpsl_combos:
        mask = generate_signals(df)
        trades, records = evaluate(mask, df, tp=tp, sl=sl)
        pnls = [t["pnl"] for t in trades]
        b = stats(pnls)

        X = extract_signal_features(df, records)
        result = train_wf_and_evaluate(df, mask, records, X, prob_threshold=0.50) if len(X) >= 20 else None

        best = None
        if result is not None:
            ti = result["test_indices"]
            for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
                passed = result["probas"] >= thresh
                fp = [records[i]["pnl"] for i in ti[passed]]
                if not fp:
                    continue
                fs = stats(fp)
                if best is None or (fs["trades"] > 20 and fs["pf"] > best["pf"]):
                    try:
                        auc = roc_auc_score(result["labels"].astype(int), result["probas"])
                    except Exception:
                        auc = None
                    best = {"thresh": thresh, **fs, "auc": auc}

        if best:
            delta = best["pnl"] - b["pnl"]
            print(f"  TP={tp}/SL={sl:<6} {b['trades']:>8d} {b['pnl']:>+10.1f} "
                  f"{b['wr']:>6.1f}% {b['pf']:>5.2f} │ "
                  f"{best['thresh']:>4.2f} {best['trades']:>8d} {best['pnl']:>+10.1f} "
                  f"{best['wr']:>6.1f}% {best['pf']:>5.2f} {best['auc'] or 0:>5.3f} "
                  f"{delta:>+8.1f}")
            summary.append({"tp": tp, "sl": sl, "base": b, "filt": best, "delta": delta})
        else:
            print(f"  TP={tp}/SL={sl:<6} {b['trades']:>8d} {b['pnl']:>+10.1f} "
                  f"{b['wr']:>6.1f}% {b['pf']:>5.2f} │  (no WF predictions)")

    print(f"\n[3/3] Summary (sorted by XGB PnL)")
    print(f"  {'TP/SL':<12s} {'Base PnL':>10s} {'Base PF':>7s} │ {'XGB PnL':>10s} {'XGB PF':>7s} {'d':>8s} {'Thr':>5s}")
    print(f"  {'─' * 72}")
    for r in sorted(summary, key=lambda r: r["filt"]["pnl"], reverse=True):
        print(f"  TP={r['tp']}/SL={r['sl']:<6} {r['base']['pnl']:>+10.1f} {r['base']['pf']:>6.2f} │ "
              f"{r['filt']['pnl']:>+10.1f} {r['filt']['pf']:>6.2f} {r['delta']:>+8.1f} {r['filt']['thresh']:>4.2f}")

    print(f"\nDONE.")


if __name__ == "__main__":
    main()
