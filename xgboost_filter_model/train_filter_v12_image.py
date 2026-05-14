#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from zoneinfo import ZoneInfo
# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xgboost_filter_model.train_filter_1min import load_price_data, prepare_features as prepare_base_features


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

def add_liquidity_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds indicators for liquidity zones and reversal triggers.
    - Zone Hit: Price near Asian High/Low
    - Reversal Trigger: Standard ER and Net Move conditions
    - Equal Highs/Lows: Detecting "Double Tops/Bottoms" for liquidity.
    """
    print("Adding liquidity zone, recovery, and equal high/low indicators...")
    df = df.copy()

    # 1. Asian Session Levels (00:00 - 08:00 UTC+2 approx)
    df['hour'] = df.index.hour

    # 2. Daily Session Levels (Day-to-Date High/Low resetting at midnight)
    df['day_id'] = df.index.date
    df['day_high'] = df.groupby('day_id')['high'].cummax()
    df['day_low'] = df.groupby('day_id')['low'].cummin()

    # 3. Equal Highs and Equal Lows (Liquidity levels)
    # Define "Equal" as within $2 (standard for Gold noise)
    EQ_THRESHOLD = 2.0

    # Check current high/low against previous peaks/troughs in the day
    df['is_eq_high'] = np.where((df['high'] >= df['day_high'].shift(1) - EQ_THRESHOLD) & (df['high'] <= df['day_high'].shift(1) + EQ_THRESHOLD), 1.0, 0.0)
    df['is_eq_low'] = np.where((df['low'] >= df['day_low'].shift(1) - EQ_THRESHOLD) & (df['low'] <= df['day_low'].shift(1) + EQ_THRESHOLD), 1.0, 0.0)

    # Zone Indicators (Proximity to current day's extremes)
    df['near_high_zone'] = np.where(df['high'] >= df['day_high'] - 5, 1.0, 0.0)
    df['near_low_zone'] = np.where(df['low'] <= df['day_low'] + 5, 1.0, 0.0)

    # 4. Leg 2 Recovery Indicators (Net move > 10 after hitting a zone)
    df['recovery_long'] = np.where((df['close'] - df['low'].rolling(15).min() > 10) & (df['near_low_zone'] > 0), 1.0, 0.0)
    df['recovery_short'] = np.where((df['high'].rolling(15).max() - df['close'] > 10) & (df['near_high_zone'] > 0), 1.0, 0.0)

    return df

def redefine_target_v10(df: pd.DataFrame, horizon: int = 45) -> pd.DataFrame:
    """
    Redefines target:
    Up Trend: future_high - close > 20 AND close - future_low < 10
    Down Trend: close - future_low > 20 AND future_high - close < 10
    Trend: Up OR Down
    Otherwise: Flat
    """
    print(f"Redefining target with Horizon={horizon}, TP=20, SL=10...")
    df = df.copy()

    future_high = df['high'].shift(-horizon).rolling(window=horizon, min_periods=1).max()
    future_low = df['low'].shift(-horizon).rolling(window=horizon, min_periods=1).min()

    up_move = future_high - df['close']
    down_move = df['close'] - future_low

    is_up = (up_move > 20) & (down_move < 10)
    is_down = (down_move > 20) & (up_move < 10)

    df['target_v10'] = 0 # Flat
    df.loc[is_up, 'target_v10'] = 1 # Up
    df.loc[is_down, 'target_v10'] = -1 # Down

    # Combined Trend label for Stage 1 prediction
    df['is_trend'] = np.where(df['target_v10'] != 0, 1, 0)

    return df

def train_v12_filter():
    # 1. Load Data with standardized range
    # Train/Validation: Until 2026-04-10
    # OOS: After 2026-04-10
    FULL_START = "2020-01-01"
    OOS_START = "2026-04-10"
    df = load_price_data(start_date=FULL_START, end_date="2026-05-10")

    # 2. Base Feature Engineering
    df = prepare_base_features(df, move_threshold=10, er_threshold=0.1, future_window=45)

    # 3. Add Liquidity Zone Indicators
    df = add_liquidity_indicators(df)
    df = add_image_model_predictions(df)

    # 4. Redefine Target v10
    df = redefine_target_v10(df, horizon=45)

    # 5. Filter for Energetic segments
    df['bar_move'] = (df['close'] - df['open']).abs()
    df = df[(df['bar_move'] > 3) & (df['volume'] > 250)]

    # Drop rows with NaN
    df = df.dropna()

    # Split into Train and OOS Sets based on Date
    df_train_all = df[df.index < OOS_START].copy()
    df_oos = df[df.index >= OOS_START].copy()

    # Features selection
    exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
               'day_high_rolling', 'day_low_rolling', 'day_open',
               'Dchange_utc2_rel', 'Dupper_wick_utc2_rel', 'Dlower_wick_utc2_rel',
               'trend_label', 'target_v10', 'is_trend', 'atr', 'day_utc2',
               'future_max_move', 'future_min_move', 'future_er', 'atr_threshold',
               'bar_move', 'hour', 'day_id', 'day_high', 'day_low']

    # Also exclude raw prices to prevent drift learning
    exclude += ['day_high', 'day_low', 'day_open', 'high_90', 'low_90',
                'closePrice_ask', 'closePrice_bid', 'highPrice_ask', 'lowPrice_bid',
                'closePrice', 'lowPrice', 'open_price',
                'highPrice_bid', 'lowPrice_ask', 'openPrice_bid', 'openPrice_ask']

    features = [c for c in df.columns if c not in exclude]

    # 5. Class Balancing (Only on Training data)
    print(f"\nBalancing classes on Training Set ({FULL_START} to {OOS_START})...")
    print(f"Original training counts: {df_train_all['is_trend'].value_counts().to_dict()}")

    df_trend = df_train_all[df_train_all['is_trend'] == 1]
    df_flat = df_train_all[df_train_all['is_trend'] == 0].sample(n=len(df_trend), random_state=42)
    df_balanced = pd.concat([df_trend, df_flat]).sort_index()

    X_train = df_balanced[features]
    y_train = df_balanced['is_trend']

    X_oos = df_oos[features]
    y_oos = df_oos['is_trend']

    # 6. Train XGBoost
    print(f"\nTraining Stage 1 Filter with {len(X_train)} samples...")
    clf = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 7. Evaluate on OOS (Out of Sample)
    y_pred_oos = clf.predict(X_oos)
    print(f"\n--- Stage 1 Filter OOS Evaluation ({OOS_START} onwards) ---")
    if not y_oos.empty:
        print(classification_report(y_oos, y_pred_oos))
    else:
        print("No OOS data available for evaluation.")

    # Feature Importance
    importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop 10 Features:")
    print(importances.head(10))

    # 8. Save
    model_path = PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v12_image.joblib"
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_v12_filter()
