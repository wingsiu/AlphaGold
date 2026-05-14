#!/usr/bin/env python3
"""
Trains a second-stage XGBoost model to predict the direction (Up/Down) of a trend,
using the output from the first-stage filter model.
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

from xgboost_filter_model.train_filter_1min import load_price_data, prepare_features

def visualize_directional_features(df: pd.DataFrame, features: list):
    """
    Visualizes the distributions of top features for the directional model.
    """
    print("\n--- Visualizing Top Features for Directional Model ---")

    # Select only the trend data
    df_trends = df[df['trend_label'] != 0].copy()
    if df_trends.empty:
        print("No trend data to visualize.")
        return

    # Get the top 6 features
    top_features = features[:6]

    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='trend_label', y=feature, data=df_trends)
        plt.title(f'Distribution of {feature} by Trend Direction')
        plt.xlabel('Trend Direction (1: Up, -1: Down)')
        plt.ylabel(feature)

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "xgboost_filter_model" / "directional_feature_distribution.png"
    plt.savefig(plot_path)
    print(f"Feature distribution plot saved to {plot_path}")
    plt.close()


def train_directional_model():
    """
    Trains and evaluates the second-stage directional model.
    """
    print("--- Starting Stage 2: Directional Model Training ---")

    # 1. Load and prepare data using the same functions as the filter model
    # Use the best parameters found from the previous sweep
    best_move = 15
    best_er = 0.25
    best_window = 120

    df = load_price_data()
    df_featured = prepare_features(df, move_threshold=best_move, er_threshold=best_er, future_window=best_window)

    # 2. Load the pre-trained filter model
    filter_model_path = PROJECT_ROOT / "xgboost_filter_model" / "filter_model.joblib"
    if not filter_model_path.exists():
        print(f"Error: Filter model not found at {filter_model_path}")
        print("Please run train_filter_1min.py first to train and save the filter model.")
        return

    print(f"Loading filter model from {filter_model_path}...")
    filter_model = joblib.load(filter_model_path)

    # 3. Define features and filter the data
    features = [
        'adx', 'adx_slope', 'volatility', 'fractal_dimension',
        'er_15', 'er_30', 'er_90',
        'wr_15', 'wr_30', 'wr_90',
        'change_15', 'change_30', 'change_90',
        'upper_wick_15', 'upper_wick_30', 'upper_wick_90',
        'lower_wick_15', 'lower_wick_30', 'lower_wick_90',
        'down_efficiency_ratio', 'up_efficiency_ratio',
        'volume_price_corr', 'volume_trend', 'volume_osc',
        'day_progress',
        'is_asia', 'asia_progress', 'is_london', 'london_progress', 'is_ny', 'ny_progress',
        'change', 'upper_wick', 'lower_wick',
        'bar_change', 'bar_upper_wick', 'bar_lower_wick'
    ]
    X = df_featured[features]

    # Use the filter model to predict which samples are part of a trend
    trend_predictions = filter_model.predict(X)

    # Select only the data points that the filter model identified as a "Trend"
    df_trend_only = df_featured[trend_predictions == 1].copy()

    if df_trend_only.empty:
        print("The filter model did not predict any trends. Cannot train the directional model.")
        return

    print(f"Filtered down to {len(df_trend_only)} samples predicted as 'Trend' by the stage-1 model.")

    # 4. Prepare data for the directional model
    X_directional = df_trend_only[features]
    y_directional = df_trend_only['trend_label'] # Target is now 1 (Up) or -1 (Down)

    # Map target to 0 and 1 for XGBoost
    y_directional_mapped = y_directional.map({1: 1, -1: 0})

    # Remove any samples where the trend label is 0 (should not happen after filtering, but as a safeguard)
    valid_indices = y_directional_mapped.notna()
    X_directional = X_directional[valid_indices]
    y_directional_mapped = y_directional_mapped[valid_indices]

    print("\nLabel distribution for directional model:")
    print(y_directional_mapped.value_counts(normalize=True))

    # 5. Train the directional model
    X_train, X_test, y_train, y_test = train_test_split(X_directional, y_directional_mapped, test_size=0.3, shuffle=False)

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    if len(y_train.unique()) < 2:
        print("Not enough class variety in the training set. Cannot train.")
        return

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    directional_model = XGBClassifier(
        n_estimators=100,
        max_depth=4, # Can use a slightly shallower model
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )

    print("\nTraining directional model...")
    directional_model.fit(X_train, y_train, verbose=False)
    print("Directional model training complete.")

    # 6. Evaluate the directional model
    print("\n--- Directional Model Evaluation ---")
    y_pred = directional_model.predict(X_test)

    print("Classification Report on Test Set (Directional Model):")
    print(classification_report(y_test, y_pred, target_names=['Down-Trend (-1)', 'Up-Trend (1)']))

    # Feature Importance
    print("\nFeature Importances (Directional Model):")
    sorted_importances = sorted(zip(directional_model.feature_importances_, features), reverse=True)
    for importance, name in sorted_importances:
        print(f"{name}: {importance:.4f}")

    # Visualize the top features
    top_feature_names = [name for _, name in sorted_importances]
    visualize_directional_features(X_directional.join(y_directional), top_feature_names)

    # Save the trained directional model
    directional_model_path = PROJECT_ROOT / "xgboost_filter_model" / "directional_model.joblib"
    joblib.dump(directional_model, directional_model_path)
    print(f"\nDirectional model saved to {directional_model_path}")


if __name__ == "__main__":
    train_directional_model()

