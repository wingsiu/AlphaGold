#!/usr/bin/env python3
"""
Trains a single, multi-class XGBoost model to predict one of three market regimes:
Up-Trend, Down-Trend, or No-Trend.
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

def train_multiclass_model():
    """
    Trains and evaluates a single multi-class model for trend prediction.
    """
    print("--- Starting Multi-Class Model Training ---")

    # 1. Load and prepare data with all features
    best_move = 15
    best_er = 0.25
    best_window = 120

    df = load_price_data()
    df_featured = prepare_base_features(df, move_threshold=best_move, er_threshold=best_er, future_window=best_window)
    df_featured = add_directional_features(df_featured)
    df_featured.dropna(inplace=True)

    # 2. Define features and target
    base_features = [
        'adx', 'adx_slope', 'volatility', 'fractal_dimension', 'er_15', 'er_30', 'er_90',
        'wr_15', 'wr_30', 'wr_90', 'change_15', 'change_30', 'change_90',
        'upper_wick_15', 'upper_wick_30', 'upper_wick_90', 'lower_wick_15', 'lower_wick_30', 'lower_wick_90',
        'down_efficiency_ratio', 'up_efficiency_ratio', 'volume_price_corr', 'volume_trend', 'volume_osc',
        'day_progress', 'is_asia', 'asia_progress', 'is_london', 'london_progress', 'is_ny', 'ny_progress',
        'change', 'upper_wick', 'lower_wick', 'bar_change', 'bar_upper_wick', 'bar_lower_wick'
    ]
    new_directional_features = [f'directional_change_{w}' for w in [15, 30, 90]] + [f'wick_ratio_{w}' for w in [15, 30, 90]]
    all_features = base_features + new_directional_features

    X = df_featured[all_features]
    y = df_featured['trend_label'] # Target is -1, 0, or 1

    # Encode labels to be 0, 1, 2 for XGBoost's multi-class objective
    # -1 (Down) -> 0
    #  0 (None) -> 1
    #  1 (Up)   -> 2
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("\nLabel distribution for multi-class model:")
    print(pd.Series(y_encoded).value_counts(normalize=True))

    # 3. Train the multi-class model
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, shuffle=False)

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # For multi-class, we don't use scale_pos_weight. Class imbalance is handled by the algorithm if needed,
    # but often the raw distribution is used.
    model = XGBClassifier(
        n_estimators=150, # Slightly more estimators for a more complex task
        max_depth=5,      # Deeper trees might be needed
        learning_rate=0.1,
        objective='multi:softmax', # Specify multi-class objective
        num_class=3,               # We have 3 classes
        eval_metric='mlogloss'
    )

    print("\nTraining multi-class model...")
    model.fit(X_train, y_train, verbose=False)
    print("Training complete.")

    # 4. Evaluate the model
    print("\n--- Multi-Class Model Evaluation ---")
    y_pred = model.predict(X_test)

    # Use the label encoder to get original labels for the report
    target_names = le.inverse_transform([0, 1, 2])
    target_names = [f'Down-Trend ({target_names[0]})', f'No-Trend ({target_names[1]})', f'Up-Trend ({target_names[2]})']

    print(classification_report(y_test, y_pred, target_names=target_names))

    # Feature Importance
    print("\nFeature Importances (Multi-Class Model):")
    sorted_importances = sorted(zip(model.feature_importances_, all_features), reverse=True)
    for importance, name in sorted_importances:
        print(f"{name}: {importance:.4f}")

    # Save the model
    model_path = PROJECT_ROOT / "xgboost_filter_model" / "multiclass_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nMulti-class model saved to {model_path}")


if __name__ == "__main__":
    train_multiclass_model()

