#!/usr/bin/env python3
"""
Performs a hyperparameter sweep for the V4 directional model to find the optimal
XGBoost parameters.
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
from xgboost_filter_model.train_directional_model_v4 import redefine_directional_target

def sweep_directional_model_v5():
    """
    Performs a hyperparameter sweep on the directional model.
    """
    print("--- Starting V5: Hyperparameter Sweep for Directional Model ---")

    # 1. Load and prepare data
    df = load_price_data()
    df_featured = prepare_base_features(df, move_threshold=15, er_threshold=0.25, future_window=120)
    df_featured = add_directional_features(df_featured)
    df_featured = add_ma_features(df_featured)
    df_featured = redefine_directional_target(df_featured)
    df_featured.dropna(inplace=True)

    print("\n--- Class Label Distributions ---")
    print("\n1. Distribution of new asymmetric target (v4) across all data:")
    print(df_featured['directional_target_v4'].value_counts())

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
    df_final = df_trend_only[df_trend_only['directional_target_v4'] != 0]

    X_directional = df_final[all_features]
    y_directional = df_final['directional_target_v4'].map({1: 1, -1: 0})

    print("\n2. Final distribution for Stage 2 model training (after Stage 1 filtering):")
    print(y_directional.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X_directional, y_directional, test_size=0.3, shuffle=False)

    # 4. Define hyperparameter grid (using only the best from last sweep for speed)
    param_grid = {
        'n_estimators': [100],
        'max_depth': [3],
        'learning_rate': [0.05]
    }

    results = []
    total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])
    current_combination = 0

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # 5. Run sweep
    for n in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                current_combination += 1
                print(f"\n--- Running [{current_combination}/{total_combinations}]: n_estimators={n}, max_depth={depth}, learning_rate={lr} ---")

                model = XGBClassifier(
                    n_estimators=n,
                    max_depth=depth,
                    learning_rate=lr,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss'
                )
                model.fit(X_train, y_train, verbose=False)
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)

                accuracy = report['accuracy']
                up_f1 = report.get('1', {}).get('f1-score', 0)
                down_f1 = report.get('0', {}).get('f1-score', 0)

                results.append({
                    'n_estimators': n,
                    'max_depth': depth,
                    'learning_rate': lr,
                    'accuracy': accuracy,
                    'up_f1': up_f1,
                    'down_f1': down_f1
                })
                print(f"Accuracy: {accuracy:.4f}, Up F1: {up_f1:.4f}, Down F1: {down_f1:.4f}")

    # 6. Print summary
    print("\n\n--- Hyperparameter Sweep Summary ---")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    sweep_directional_model_v5()

