#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from datetime import timedelta

from zoneinfo import ZoneInfo
# Add project root to sys.path
from V13._paths import PROJECT_ROOT

from xgboost_filter_model.train_filter_1min import load_price_data, prepare_features as prepare_base_features
from xgboost_filter_model.train_filter_v10 import add_liquidity_indicators, redefine_target_v10
from config.v13_config import FILTER_CONFIG, TARGET_CONFIG, MODEL_CONFIG, WF_CONFIG

def prepare_data_v13(start_date="2020-01-01", end_date="2026-05-07", use_cache=True):
    """
    Loads and prepares the full dataset with v10 features and targets.
    """
    cache_dir = PROJECT_ROOT / "runtime" / "_tmp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"data_v13_{start_date}_{end_date}.joblib"

    # If end_date is today or in the future, the underlying DB is still being
    # appended to. A stale joblib cache from earlier in the day would silently
    # drop the freshly-arrived bars (e.g. signals after the cache was written
    # would be invisible to the backtest). Force a fresh load in that case.
    from datetime import date as _date
    try:
        _end_d = pd.to_datetime(end_date).date()
        if _end_d >= _date.today():
            use_cache = False
    except Exception:
        pass

    if use_cache and cache_path.exists():
        print(f"Loading cached data from {cache_path}...")
        return joblib.load(cache_path)

    print(f"Loading data from {start_date} to {end_date}...")
    df = load_price_data(start_date=start_date, end_date=end_date)

    # Base Feature Engineering
    df = prepare_base_features(df, move_threshold=10, er_threshold=0.3, future_window=45)

    # Add Liquidity Zone Indicators
    df = add_liquidity_indicators(df)
    df = add_image_model_predictions(df)

    # Redefine Target v10
    df = redefine_target_v10(df, horizon=TARGET_CONFIG["horizon"])

    # Pre-filtering for energetic segments (consistent with v10)
    df['bar_move'] = (df['close'] - df['open']).abs()
    df_filtered = df[(df['bar_move'] > FILTER_CONFIG["min_bar_move"]) & (df['volume'] > FILTER_CONFIG["min_volume"])].copy()

    df_filtered = df_filtered.dropna()

    if use_cache:
        print(f"Caching data to {cache_path}...")
        joblib.dump(df_filtered, cache_path)

    return df_filtered

def add_image_model_predictions(df: pd.DataFrame) -> pd.DataFrame:
    print("Adding image model prediction features...")
    model_path = PROJECT_ROOT / "training" / "image_trend_model.joblib"
    if not model_path.exists():
        print(f"Warning: Image model not found at {model_path}. Skipping.")
        return df
    bundle = joblib.load(model_path)
    s1_model = bundle["stage1"]
    cfg = bundle["config"]
    window_1m = cfg.get("window", 150)
    df = df.copy()
    def _session_info(ts, timezone, start_h, start_m, end_h, end_m):
        local_ts = ts.tz_convert(timezone)
        minute_of_day = local_ts.hour * 60 + local_ts.minute
        s = start_h * 60 + start_m
        e = end_h * 60 + end_m
        if s <= minute_of_day < e:
            return 1.0, (minute_of_day - s) / (e - s)
        return 0.0, 0.0
    HK_TZ = ZoneInfo("Asia/Hong_Kong")
    LONDON_TZ = ZoneInfo("Europe/London")
    NY_TZ = ZoneInfo("America/New_York")
    day_start_offset = pd.Timedelta(hours=2)
    df["day_utc2"] = (df.index + day_start_offset).floor("D")
    df["day_open"] = df.groupby("day_utc2")["open"].transform("first")
    df["day_high_rolling"] = df.groupby("day_utc2")["high"].cummax()
    df["day_low_rolling"] = df.groupby("day_utc2")["low"].cummin()
    df["Dchange_utc2_rel"] = (df["close"] - df["day_open"]) / df["day_open"]
    df["Dupper_wick_utc2_rel"] = (df["day_high_rolling"] - df[["day_open", "close"]].max(axis=1)) / df["day_open"]
    df["Dlower_wick_utc2_rel"] = (df[["day_open", "close"]].min(axis=1) - df["day_low_rolling"]) / df["day_open"]
    df["bar_move"] = (df["close"] - df["open"]).abs()
    mask = (df["bar_move"] > 3) & (df["volume"] > 250)
    indices = np.where(mask)[0]
    s1_probs = np.full(len(df), np.nan)
    print(f"Calculating image model predictions for {len(indices)} samples...")
    for idx, i in enumerate(indices):
        if i < window_1m - 1:
            continue
        if idx % 2000 == 0:
            print(f"  Progress: {idx}/{len(indices)}")
        w = df.iloc[i - window_1m + 1 : i + 1]
        c0 = float(w["close"].iloc[0]) or 1.0
        open_rel  = w["open"].to_numpy()  / c0 - 1.0
        high_rel  = w["high"].to_numpy()  / c0 - 1.0
        low_rel   = w["low"].to_numpy()   / c0 - 1.0
        close_rel = w["close"].to_numpy() / c0 - 1.0
        body_rel  = (w["close"].to_numpy() - w["open"].to_numpy()) / c0
        range_rel = (w["high"].to_numpy()  - w["low"].to_numpy())  / c0
        vol = w["volume"].to_numpy(dtype=float)
        vol_mean, vol_std = np.mean(vol), np.std(vol)
        vol_z = np.zeros_like(vol) if vol_std < 1e-9 else (vol - vol_mean) / vol_std
        v0 = float(vol[0])
        vol_rel = np.zeros_like(vol) if abs(v0) < 1e-9 else vol / v0 - 1.0
        vd = np.diff(vol, prepend=vol[0])
        vd_std = np.std(vd)
        vol_diff_norm = np.zeros_like(vd) if vd_std < 1e-9 else vd / vd_std
        img = np.stack([open_rel, high_rel, low_rel, close_rel, body_rel, range_rel, vol_z, vol_rel, vol_diff_norm], axis=0).flatten()
        ts = df.index[i]
        is_asia, asia_prog = _session_info(ts, HK_TZ, 8, 0, 16, 0)
        is_london, lon_prog = _session_info(ts, LONDON_TZ, 8, 0, 16, 30)
        is_ny, ny_prog = _session_info(ts, NY_TZ, 9, 30, 16, 0)
        extra = [
            df["Dchange_utc2_rel"].iloc[i],
            df["Dupper_wick_utc2_rel"].iloc[i],
            df["Dlower_wick_utc2_rel"].iloc[i],
            is_asia, asia_prog,
            is_london, lon_prog,
            is_ny, ny_prog
        ]
        full_input = np.concatenate([img, extra]).reshape(1, -1)
        if hasattr(s1_model, "predict_proba"):
             s1_probs[i] = s1_model.predict_proba(full_input)[0][1]
    df["image_s1_prob"] = s1_probs
    return df

def train_wf_v13():
    # 1. Config
    FULL_START = WF_CONFIG["full_start"]
    WF_START = WF_CONFIG["wf_start"]
    WF_END = WF_CONFIG["wf_end"]
    RETRAIN_DAYS = WF_CONFIG["retrain_days"]

    df = prepare_data_v13(start_date=FULL_START, end_date=WF_END)

    # Features selection (consistent with v10)
    exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
               'day_high_rolling', 'day_low_rolling', 'day_open',
               'Dchange_utc2_rel', 'Dupper_wick_utc2_rel', 'Dlower_wick_utc2_rel',
               'trend_label', 'target_v10', 'is_trend', 'atr', 'day_utc2',
               'future_max_move', 'future_min_move', 'future_er', 'atr_threshold',
               'bar_move', 'hour', 'day_id', 'day_high', 'day_low']

    exclude += ['day_high', 'day_low', 'day_open', 'high_90', 'low_90',
                'closePrice_ask', 'closePrice_bid', 'highPrice_ask', 'lowPrice_bid',
                'closePrice', 'lowPrice', 'open_price',
                'highPrice_bid', 'lowPrice_ask', 'openPrice_bid', 'openPrice_ask']

    features = [c for c in df.columns if c not in exclude]

    current_test_start = pd.to_datetime(WF_START)
    end_dt = pd.to_datetime(WF_END)
    # Robust timezone handling: localize if naive, convert if tz-aware
    if current_test_start.tzinfo is None:
        current_test_start = current_test_start.tz_localize('UTC')
    else:
        current_test_start = current_test_start.tz_convert('UTC')
    if end_dt.tzinfo is None:
        end_dt = end_dt.tz_localize('UTC')
    else:
        end_dt = end_dt.tz_convert('UTC')

    all_oos_preds = []
    all_oos_actuals = []

    cycle = 1

    while current_test_start < end_dt:
        current_test_end = current_test_start + timedelta(days=RETRAIN_DAYS)
        print(f"\n>>> WF Cycle {cycle}: Testing {current_test_start.date()} to {min(current_test_end.date(), end_dt.date())}")

        # Save model for this cycle to avoid future retraining
        model_dir = PROJECT_ROOT / "runtime" / "bot_assets" / "wf_models_v13"
        model_dir.mkdir(parents=True, exist_ok=True)
        cycle_model_path = model_dir / f"filter_v13_cycle_{cycle}_{current_test_start.date()}.joblib"

        # Split into Train and Test Sets
        df_train_all = df[df.index < current_test_start].copy()
        df_test = df[(df.index >= current_test_start) & (df.index < current_test_end)].copy()

        if df_test.empty:
            print("No test data for this cycle, skipping.")
            current_test_start = current_test_end
            cycle += 1
            continue

        # Balancing classes on Training Set
        df_trend = df_train_all[df_train_all['is_trend'] == 1]
        df_flat = df_train_all[df_train_all['is_trend'] == 0]

        if len(df_trend) == 0 or len(df_flat) == 0:
            print("Not enough trend/flat samples to train, skipping.")
            current_test_start = current_test_end
            cycle += 1
            continue

        df_flat_sampled = df_flat.sample(n=min(len(df_flat), len(df_trend)), random_state=42)
        df_balanced = pd.concat([df_trend, df_flat_sampled]).sort_index()

        X_train = df_balanced[features]
        y_train = df_balanced['is_trend']

        X_test = df_test[features]
        y_test = df_test['is_trend']

        # Train XGBoost
        clf = XGBClassifier(
            **MODEL_CONFIG["s1"],
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # Save the model for this specific cycle
        joblib.dump(clf, cycle_model_path)
        print(f"  Cycle {cycle} model saved to {cycle_model_path}")

        # Predict on Test Set
        y_pred = clf.predict(X_test)

        all_oos_preds.append(pd.Series(y_pred, index=df_test.index))
        all_oos_actuals.append(y_test)

        # Evaluate cycle
        print(f"Cycle {cycle} OOS Balanced Accuracy: {(y_pred == y_test).mean():.4f}")

        # Advance
        current_test_start = current_test_end
        cycle += 1

    # Final Evaluation
    print("\n" + "="*50)
    print("FINISHED WALK-FORWARD VALIDATION")
    print("="*50)

    if all_oos_preds:
        y_pred_all = pd.concat(all_oos_preds)
        y_test_all = pd.concat(all_oos_actuals)

        print("\n--- Aggregate OOS Walk-Forward Performance ---")
        print(classification_report(y_test_all, y_pred_all))

        # Save the cumulative walk-forward predictions for analysis
        df_results = pd.DataFrame({'actual': y_test_all, 'predicted': y_pred_all})
        results_path = PROJECT_ROOT / "xgboost_filter_model" / "wf_v13_results.csv"
        df_results.to_csv(results_path)
        print(f"Results saved to {results_path}")

        # Train final model on most recent data to save
        print("\nTraining final production model on last window...")
        # (Assuming the last trained 'clf' is the most recent production-ready model)
        model_path = PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v13_wf_image.joblib"
        joblib.dump(clf, model_path)
        print(f"Final model saved to {model_path}")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    train_wf_v13()

