#!/usr/bin/env python3
"""
Trains a sixth-generation, multi-class second-stage XGBoost model.
This model predicts trend direction (Up, Down, or Flat) using the asymmetric
target definition, but only on the subset of data identified as a "Trend"
by the first-stage filter model.
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

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xgboost_filter_model.train_filter_1min import load_price_data, prepare_features as prepare_base_features
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgboost_filter_model.train_directional_model_v4 import redefine_directional_target

def train_directional_model_v6_multiclass():
    """
    Trains and evaluates the v6 multi-class directional model.
    """
    print("--- Starting Stage 2 (V6): Multi-Class Directional Model ---")

    # 1. Load and prepare data
    df = load_price_data()
    df_featured = prepare_base_features(df, move_threshold=15, er_threshold=0.25, future_window=120)
    df_featured = add_directional_features(df_featured)
    df_featured = add_ma_features(df_featured)
    df_featured = redefine_directional_target(df_featured)
    df_featured.dropna(inplace=True)

    # 2. Load filter model
    filter_model = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "filter_model.joblib")

    # 3. Define features and filter data
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
    all_features = base_features + dir_features_v2 + ma_features

    X = df_featured[all_features]
    trend_predictions = filter_model.predict(X[base_features])
    df_trend_only = df_featured[trend_predictions == 1].copy()

    print(f"\nStage 1 filtered the data down to {len(df_trend_only)} samples.")

    # 4. Prepare data for Stage 2 multi-class model
    X_stage2 = df_trend_only[all_features]
    y_stage2 = df_trend_only['directional_target_v4'] # This contains -1, 0, 1

    # --- Label Definitions ---
    print("\n--- Label Definitions for Stage 2 Model ---")
    print("The target is defined by the asymmetric risk-reward over the next 120 minutes:")
    print("Label '1' (Up-Trend): Future High > $30 AND Future Low > -$15")
    print("Label '-1' (Down-Trend): Future Low < -$30 AND Future High < $15")
    print("Label '0' (Flat): Anything else.")

    # Encode labels to be 0, 1, 2 for XGBoost
    le = LabelEncoder()
    y_stage2_encoded = le.fit_transform(y_stage2)

    print("\nLabels are encoded for the model as follows:")
    for i, label in enumerate(le.classes_):
        print(f"Original Label '{label}' -> Encoded as '{i}'")

    print("\nFinal class distribution for Stage 2 training:")
    print(y_stage2.value_counts())

    # 5. Train model
    X_train, X_test, y_train, y_test = train_test_split(X_stage2, y_stage2_encoded, test_size=0.3, shuffle=False)

    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss'
    )

    print("\nTraining Stage 2 multi-class model...")
    model.fit(X_train, y_train, verbose=False)
    print("Training complete.")

    # 6. Evaluate
    print("\n--- Stage 2 Multi-Class Model Evaluation ---")
    y_pred = model.predict(X_test)

    target_names = [f"Class {c}" for c in le.classes_]
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("\nFeature Importances (Stage 2 Multi-Class Model):")
    sorted_importances = sorted(zip(model.feature_importances_, all_features), reverse=True)
    for importance, name in sorted_importances:
        print(f"{name}: {importance:.4f}")

if __name__ == "__main__":
    train_directional_model_v6_multiclass()

