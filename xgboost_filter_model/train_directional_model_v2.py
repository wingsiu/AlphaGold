#!/usr/bin/env python3
"""
Trains a second-stage XGBoost model to predict the direction (Up/Down) of a trend,
using the output from the first-stage filter model and new directional features.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xgboost_filter_model.train_filter_1min import load_price_data, prepare_features as prepare_base_features

def add_directional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds new features specifically designed to be directional.
    """
    print("Adding new directional features...")
    df_new = df.copy()

    for w in [15, 30, 90]:
        rolling_high = df_new['high'].rolling(window=w).max()
        rolling_low = df_new['low'].rolling(window=w).min()
        rolling_open = df_new['open'].shift(w - 1)
        rolling_close = df_new['close']

        rolling_range = (rolling_high - rolling_low).replace(0, np.nan)

        # Net change over the window, normalized by range. Positive -> Up, Negative -> Down.
        df_new[f'directional_change_{w}'] = (rolling_close - rolling_open) / rolling_range

        # Ratio of total upper wick size to total lower wick size over the window.
        # > 1 suggests more selling pressure, < 1 suggests more buying pressure.
        upper_wicks = (df_new['high'] - df_new[['open', 'close']].max(axis=1)).rolling(window=w).sum()
        lower_wicks = (df_new[['open', 'close']].min(axis=1) - df_new['low']).rolling(window=w).sum()

        # To avoid division by zero, we add a small epsilon.
        df_new[f'wick_ratio_{w}'] = upper_wicks / (lower_wicks + 1e-6)

    return df_new

def train_new_directional_model():
    """
    Trains and evaluates the second-stage directional model with new features.
    """
    print("--- Starting Stage 2: New Directional Model Training ---")

    # 1. Load and prepare data
    best_move = 15
    best_er = 0.25
    best_window = 120

    df = load_price_data()
    # Prepare the same base features as before
    df_featured = prepare_base_features(df, move_threshold=best_move, er_threshold=best_er, future_window=best_window)
    # Add the new directional features
    df_featured = add_directional_features(df_featured)
    df_featured.dropna(inplace=True)

    # 2. Load the pre-trained filter model
    filter_model_path = PROJECT_ROOT / "xgboost_filter_model" / "filter_model.joblib"
    if not filter_model_path.exists():
        print(f"Error: Filter model not found at {filter_model_path}")
        return

    print(f"Loading filter model from {filter_model_path}...")
    filter_model = joblib.load(filter_model_path)

    # 3. Define features and filter the data
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

    trend_predictions = filter_model.predict(X[base_features])
    df_trend_only = df_featured[trend_predictions == 1].copy()

    if df_trend_only.empty:
        print("The filter model did not predict any trends.")
        return

    print(f"Filtered down to {len(df_trend_only)} samples predicted as 'Trend'.")

    # 4. Prepare data for the new directional model
    X_directional = df_trend_only[all_features]
    y_directional = df_trend_only['trend_label'].map({1: 1, -1: 0})

    # Remove any samples where the mapping resulted in NaN
    valid_indices = y_directional.notna()
    X_directional = X_directional[valid_indices]
    y_directional = y_directional[valid_indices]

    print("\nLabel distribution for new directional model:")
    print(y_directional.value_counts(normalize=True))

    # 5. Train the new directional model
    X_train, X_test, y_train, y_test = train_test_split(X_directional, y_directional, test_size=0.3, shuffle=False)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    new_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, scale_pos_weight=scale_pos_weight, eval_metric='logloss')

    print("\nTraining new directional model...")
    new_model.fit(X_train, y_train, verbose=False)
    print("Training complete.")

    # 6. Evaluate the new model
    print("\n--- New Directional Model Evaluation ---")
    y_pred = new_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Down-Trend (-1)', 'Up-Trend (1)']))

    # Feature Importance
    print("\nFeature Importances (New Directional Model):")
    sorted_importances = sorted(zip(new_model.feature_importances_, all_features), reverse=True)
    for importance, name in sorted_importances:
        print(f"{name}: {importance:.4f}")

if __name__ == "__main__":
    train_new_directional_model()

