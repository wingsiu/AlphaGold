#!/usr/bin/env python3
"""
Trains a ninth-generation, second-stage XGBoost model with even more features.
Added: RSI, MACD, ROC.
Target: 30 Move, 15 Stop, 120 Horizon.
Stage 1 Filter is used for sampling.
Balanced classes via under-sampling.
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
from xgboost_filter_model.train_directional_model_v4 import redefine_directional_target

from xgboost_filter_model.momentum_features import add_momentum_features  # noqa: F401

def train_directional_model_v9():
    """
    Trains and evaluates the v9 directional model with momentum features.
    """
    print("--- Starting Stage 2 (V9): Multi-Class Directional Model with Momentum Features ---")

    # 1. Load and prepare data
    # TRAINING DATA CUTOFF: We train only on data BEFORE the predefined test start
    train_end_cutoff = "2025-11-25"
    print(f"Training using data up to {train_end_cutoff} ONLY.")

    df = load_price_data(end_date=train_end_cutoff)
    # Use best Stage 1 params
    df_featured = prepare_base_features(df, move_threshold=15, er_threshold=0.25, future_window=120)
    df_featured = add_directional_features(df_featured)
    df_featured = add_ma_features(df_featured)
    df_featured = add_momentum_features(df_featured)

    # Redefine target using 30/15 logic
    df_featured = redefine_directional_target(df_featured)
    df_featured.dropna(inplace=True)

    # 2. Load filter model
    filter_model_path = PROJECT_ROOT / "xgboost_filter_model" / "filter_model.joblib"
    if not filter_model_path.exists():
        print(f"Error: Filter model not found at {filter_model_path}")
        return
    filter_model = joblib.load(filter_model_path)

    # 3. Define features
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

    X = df_featured[all_features]

    # --- STAGE 1 FILTERING ---
    trend_predictions = filter_model.predict(X[base_features])
    df_trend_only = df_featured[trend_predictions == 1].copy()

    # --- BALANCING ---
    df_up = df_trend_only[df_trend_only['directional_target_v4'] == 1]
    df_down = df_trend_only[df_trend_only['directional_target_v4'] == -1]
    df_flat = df_trend_only[df_trend_only['directional_target_v4'] == 0]

    n_samples = int((len(df_up) + len(df_down)) / 2)
    df_flat_downsampled = resample(df_flat, replace=False, n_samples=n_samples, random_state=42)
    df_balanced = pd.concat([df_up, df_down, df_flat_downsampled])

    X_balanced = df_balanced[all_features]
    y_balanced = df_balanced['directional_target_v4']

    # 4. Train Model
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_balanced)

    print("\nBalanced class distribution for training:")
    print(pd.Series(y_encoded).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_encoded, test_size=0.3, shuffle=True, random_state=42)

    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss'
    )

    print("\nTraining Stage 2 multi-class model with momentum features...")
    model.fit(X_train, y_train, verbose=False)
    print("Training complete.")

    # 5. Evaluate
    print("\n--- Balanced Stage 2 Evaluation with Momentum ---")
    y_pred = model.predict(X_test)
    target_names = [f"Class {c}" for c in le.classes_]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Feature Importance
    print("\nTop 40 Feature Importances:")
    sorted_importances = sorted(zip(model.feature_importances_, all_features), reverse=True)
    for importance, name in sorted_importances[:40]:
        print(f"{name}: {importance:.4f}")

    # Save the model and label encoder
    model_path = PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v9.joblib"
    le_path = PROJECT_ROOT / "xgboost_filter_model" / "directional_le_v9.joblib"

    joblib.dump(model, model_path)
    joblib.dump(le, le_path)
    print(f"\nDirectional model saved to {model_path}")
    print(f"Label encoder saved to {le_path}")

if __name__ == "__main__":
    train_directional_model_v9()

