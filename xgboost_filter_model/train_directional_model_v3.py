#!/usr/bin/env python3
"""
Trains a third-generation, second-stage XGBoost model to predict trend direction,
using moving average features.
"""
import sys
from pathlib import Path
import pandas as pd
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

def add_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features based on moving averages.
    """
    print("Adding moving average features...")
    df_new = df.copy()

    mas = [10, 30, 90]
    for m in mas:
        df_new[f'ma_{m}'] = df_new['close'].rolling(window=m).mean()

    # Price relative to MAs
    for m in mas:
        df_new[f'price_vs_ma_{m}'] = (df_new['close'] - df_new[f'ma_{m}']) / df_new[f'ma_{m}']

    # MA crossover features
    df_new['ma_10_vs_30'] = (df_new['ma_10'] - df_new['ma_30']) / df_new['ma_30']
    df_new['ma_30_vs_90'] = (df_new['ma_30'] - df_new['ma_90']) / df_new['ma_90']

    # Drop the MA columns themselves as they are just intermediate steps
    df_new.drop(columns=[f'ma_{m}' for m in mas], inplace=True)

    return df_new

def train_directional_model_v3():
    """
    Trains and evaluates the v3 directional model.
    """
    print("--- Starting Stage 2: V3 Directional Model Training ---")

    # 1. Load and prepare data
    best_move = 15
    best_er = 0.25
    best_window = 120

    df = load_price_data()
    df_featured = prepare_base_features(df, move_threshold=best_move, er_threshold=best_er, future_window=best_window)
    df_featured = add_directional_features(df_featured)
    df_featured = add_ma_features(df_featured)
    df_featured.dropna(inplace=True)

    # 2. Load filter model
    filter_model_path = PROJECT_ROOT / "xgboost_filter_model" / "filter_model.joblib"
    if not filter_model_path.exists():
        print(f"Error: Filter model not found at {filter_model_path}")
        return
    filter_model = joblib.load(filter_model_path)

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

    # Predict trends using only the base features the filter model was trained on
    trend_predictions = filter_model.predict(X[base_features])
    df_trend_only = df_featured[trend_predictions == 1].copy()

    if df_trend_only.empty:
        print("The filter model did not predict any trends.")
        return

    print(f"Filtered down to {len(df_trend_only)} samples predicted as 'Trend'.")

    # 4. Prepare data for directional model
    X_directional = df_trend_only[all_features]
    y_directional = df_trend_only['trend_label'].map({1: 1, -1: 0})

    valid_indices = y_directional.notna()
    X_directional = X_directional[valid_indices]
    y_directional = y_directional[valid_indices]

    print("\nLabel distribution for directional model:")
    print(y_directional.value_counts(normalize=True))

    # 5. Train model
    X_train, X_test, y_train, y_test = train_test_split(X_directional, y_directional, test_size=0.3, shuffle=False)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=scale_pos_weight, eval_metric='logloss')

    print("\nTraining V3 directional model...")
    model.fit(X_train, y_train, verbose=False)
    print("Training complete.")

    # 6. Evaluate
    print("\n--- V3 Directional Model Evaluation ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Down-Trend (-1)', 'Up-Trend (1)']))

    print("\nFeature Importances (V3 Directional Model):")
    sorted_importances = sorted(zip(model.feature_importances_, all_features), reverse=True)
    for importance, name in sorted_importances:
        print(f"{name}: {importance:.4f}")

if __name__ == "__main__":
    train_directional_model_v3()

