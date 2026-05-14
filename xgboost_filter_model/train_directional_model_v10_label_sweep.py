#!/usr/bin/env python3
"""
Performs a comprehensive sweep across different target, stop, and horizon definitions
to find the most predictable risk-reward profile for the Stage 2 model.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xgboost_filter_model.train_filter_1min import load_price_data, prepare_features as prepare_base_features
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgboost_filter_model.train_directional_model_v9 import add_momentum_features

def define_custom_target(df: pd.DataFrame, target: float, stop: float, horizon: int) -> pd.Series:
    """
    Redefines the target based on specific target, stop, and horizon.
    """
    future_highs = df['high'].shift(-horizon).rolling(window=horizon).max()
    future_lows = df['low'].shift(-horizon).rolling(window=horizon).min()

    up_move = future_highs - df['close']
    down_move = df['close'] - future_lows

    is_uptrend = (up_move > target) & (down_move < stop)
    is_downtrend = (down_move > target) & (up_move < stop)

    labels = pd.Series(0, index=df.index)
    labels.loc[is_uptrend] = 1
    labels.loc[is_downtrend] = -1
    # Tie-break
    labels.loc[is_uptrend & is_downtrend] = 1
    return labels

def run_label_sweep():
    print("--- Starting V10: Label Definition Sweep (Target/Stop/Horizon) ---")

    # 1. Load data once
    df = load_price_data()
    # Pre-calculate ALL base features (using max horizon for base prep)
    df_featured = prepare_base_features(df, move_threshold=15, er_threshold=0.25, future_window=120)
    df_featured = add_directional_features(df_featured)
    df_featured = add_ma_features(df_featured)
    df_featured = add_momentum_features(df_featured)
    df_featured.dropna(inplace=True)

    # 2. Load Stage 1 filter model
    filter_model = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "filter_model.joblib")

    # 3. Define feature lists
    base_features = [
        'adx', 'adx_slope', 'volatility', 'fractal_dimension', 'er_15', 'er_30', 'er_90',
        'wr_15', 'wr_30', 'wr_90', 'change_15', 'change_30', 'change_90',
        'upper_wick_15', 'upper_wick_30', 'upper_wick_90', 'lower_wick_15', 'lower_wick_30', 'lower_wick_90',
        'down_efficiency_ratio', 'up_efficiency_ratio', 'volume_price_corr', 'volume_trend', 'volume_osc',
        'day_progress', 'is_asia', 'asia_progress', 'is_london', 'london_progress', 'is_ny', 'ny_progress',
        'change', 'upper_wick', 'lower_wick', 'bar_change', 'bar_upper_wick', 'bar_lower_wick'
    ]
    dir_features_v2 = [f'directional_change_{w}' for w in [15, 30, 90]] + [f'wick_ratio_{w}' for w in [15, 30, 90]]
    ma_features = [f'price_vs_ma_{m}' for m in [10, 30, 90]] + ['ma_10_vs_30', 'ma_30_vs_90']
    momentum_features = [f'rsi_{w}' for w in [14, 30]] + ['macd', 'macd_signal', 'macd_diff'] + [f'roc_{w}' for w in [15, 30, 60]]
    all_features = base_features + dir_features_v2 + ma_features + momentum_features

    # 4. Filter once with Stage 1
    X_base = df_featured[base_features]
    trend_predictions = filter_model.predict(X_base)
    df_trend_only = df_featured[trend_predictions == 1].copy()
    print(f"Stage 1 filter identified {len(df_trend_only)} potential trend bars.")

    # 5. Sweep Parameters
    targets = [15, 20, 30]
    stops = [10, 15]
    horizons = [60, 90, 120]

    results = []
    total = len(targets) * len(stops) * len(horizons)
    count = 0

    for t in targets:
        for s in stops:
            for h in horizons:
                count += 1
                print(f"\n--- [{count}/{total}] Sweep: Target={t}, Stop={s}, Horizon={h} ---")

                # Create the target Series for this specific iteration
                y_labels = define_custom_target(df_trend_only, target=t, stop=s, horizon=h)

                # Check class counts
                counts = y_labels.value_counts()
                n_up = counts.get(1, 0)
                n_down = counts.get(-1, 0)
                n_flat = counts.get(0, 0)

                if n_up < 500 or n_down < 500:
                    print(f"Skipping: Not enough samples (Up: {n_up}, Down: {n_down})")
                    continue

                # Balance classes (under-sample flat)
                n_target = int((n_up + n_down) / 2)
                df_temp = df_trend_only.copy()
                df_temp['tmp_y'] = y_labels

                df_up = df_temp[df_temp['tmp_y'] == 1]
                df_down = df_temp[df_temp['tmp_y'] == -1]
                df_flat = df_temp[df_temp['tmp_y'] == 0]

                df_flat_sampled = resample(df_flat, replace=False, n_samples=min(len(df_flat), n_target), random_state=42)
                df_balanced = pd.concat([df_up, df_down, df_flat_sampled])

                X_bal = df_balanced[all_features]
                y_bal = df_balanced['tmp_y']

                # Train/Test Split
                le = LabelEncoder()
                y_enc = le.fit_transform(y_bal)
                X_train, X_test, y_train, y_test = train_test_split(X_bal, y_enc, test_size=0.3, shuffle=True, random_state=42)

                # Fast training for sweep
                model = XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    objective='multi:softmax', num_class=3, eval_metric='mlogloss'
                )
                model.fit(X_train, y_train, verbose=False)

                # Metrics
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)

                macro_f1 = report['macro avg']['f1-score']
                up_prec = report.get('2', {}).get('precision', 0) if '2' in report else report.get('1.0', {}).get('precision', 0) # le might index differently
                # Safer lookup
                labels_in_test = le.inverse_transform(np.unique(y_test))
                up_idx = str(np.where(le.classes_ == 1)[0][0])
                down_idx = str(np.where(le.classes_ == -1)[0][0])

                up_prec = report.get(up_idx, {}).get('precision', 0)
                down_prec = report.get(down_idx, {}).get('precision', 0)

                results.append({
                    'target': t, 'stop': s, 'horizon': h,
                    'macro_f1': macro_f1, 'up_prec': up_prec, 'down_prec': down_prec,
                    'n_up': n_up, 'n_down': n_down
                })
                print(f"Macro F1: {macro_f1:.4f}, Up Prec: {up_prec:.4f}, Down Prec: {down_prec:.4f}")

    print("\n\n--- LABEL SWEEP SUMMARY ---")
    summary = pd.DataFrame(results).sort_values('macro_f1', ascending=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_label_sweep()

