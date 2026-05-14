#!/usr/bin/env python3
"""
Trains a seventh-generation, second-stage XGBoost model to predict trend direction,
using a new asymmetric target definition (15/10) based on risk-reward.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xgboost_filter_model.train_filter_1min import load_price_data, prepare_features as prepare_base_features
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features

def redefine_directional_target_v7(df: pd.DataFrame) -> pd.DataFrame:
    """
    Redefines the target for the directional model based on a 15/10 risk-reward ratio.
    """
    print("Redefining directional target with 15/10 asymmetric risk-reward...")
    df_new = df.copy()
    horizon = 120

    future_highs = df_new['high'].shift(-horizon).rolling(window=horizon).max()
    future_lows = df_new['low'].shift(-horizon).rolling(window=horizon).min()

    up_move = future_highs - df_new['close']
    down_move = df_new['close'] - future_lows

    # Conditions for the new target: target > 15, stop < 10
    is_uptrend = (up_move > 15) & (down_move < 10)
    is_downtrend = (down_move > 15) & (up_move < 10)

    # Create the new target column: 1 for Up, -1 for Down, 0 for Flat
    df_new['directional_target_v7'] = 0
    df_new.loc[is_uptrend, 'directional_target_v7'] = 1
    df_new.loc[is_downtrend, 'directional_target_v7'] = -1

    df_new.loc[is_uptrend & is_downtrend, 'directional_target_v7'] = 1

    return df_new

def train_directional_model_v7():
    """
    Trains and evaluates the v7 directional model with the new target.
    """
    print("--- Starting Stage 2: V7 Directional Model Training (Target: 15, Stop: 10) ---")

    # 1. Load and prepare data
    df = load_price_data()
    df_featured = prepare_base_features(df, move_threshold=15, er_threshold=0.25, future_window=120)
    df_featured = add_directional_features(df_featured)
    df_featured = add_ma_features(df_featured)

    # Apply the new target definition
    df_featured = redefine_directional_target_v7(df_featured)
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

    # We only want to train on the Up and Down classes of our new target
    df_final = df_trend_only[df_trend_only['directional_target_v7'] != 0]

    if df_final.empty:
        print("No samples matched the new 15/10 Up/Down trend definitions.")
        return

    X_directional = df_final[all_features]
    y_directional = df_final['directional_target_v7'].map({1: 1, -1: 0})

    print("\nFinal class distribution for V7 model training:")
    print(y_directional.value_counts())

    # 5. Train model
    X_train, X_test, y_train, y_test = train_test_split(X_directional, y_directional, test_size=0.3, shuffle=False)

    if len(y_train.unique()) < 2:
        print("Training set has only one class. Cannot train.")
        return

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Using best params from previous sweep
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=scale_pos_weight, eval_metric='logloss')

    print("\nTraining V7 directional model...")
    model.fit(X_train, y_train, verbose=False)
    print("Training complete.")

    # 6. Evaluate
    print("\n--- V7 Directional Model Evaluation ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Down-Trend (-1)', 'Up-Trend (1)']))

    print("\nFeature Importances (V7 Directional Model):")
    sorted_importances = sorted(zip(model.feature_importances_, all_features), reverse=True)
    for importance, name in sorted_importances:
        print(f"{name}: {importance:.4f}")

if __name__ == "__main__":
    train_directional_model_v7()

