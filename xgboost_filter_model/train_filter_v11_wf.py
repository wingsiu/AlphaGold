#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from datetime import timedelta

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xgboost_filter_model.train_filter_1min import load_price_data, prepare_features as prepare_base_features
from xgboost_filter_model.train_filter_v10 import add_liquidity_indicators, redefine_target_v10

def prepare_data_v11(start_date="2020-01-01", end_date="2026-05-07"):
    """
    Loads and prepares the full dataset with v10 features and targets.
    """
    print(f"Loading data from {start_date} to {end_date}...")
    df = load_price_data(start_date=start_date, end_date=end_date)

    # Base Feature Engineering
    df = prepare_base_features(df, move_threshold=10, er_threshold=0.1, future_window=45)

    # Add Liquidity Zone Indicators
    df = add_liquidity_indicators(df)

    # Redefine Target v10
    df = redefine_target_v10(df, horizon=45)

    # Pre-filtering for energetic segments (consistent with v10)
    df['bar_move'] = (df['close'] - df['open']).abs()
    df_filtered = df[(df['bar_move'] > 3) & (df['volume'] > 250)].copy()

    df_filtered = df_filtered.dropna()
    return df_filtered

def train_wf_v11():
    # 1. Config
    FULL_START = "2020-01-01"
    WF_START = "2025-01-01"  # Start walk-forward from 2025
    WF_END = "2026-05-07"

    # RETRAIN_DAYS = 14 (Two week cycle)
    # INITIAL_TRAIN_DAYS = 365 * 4 (Starting with ~4 years of data)
    RETRAIN_DAYS = 14

    df = prepare_data_v11(start_date=FULL_START, end_date=WF_END)

    # Features selection (consistent with v10)
    exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
               'trend_label', 'target_v10', 'is_trend', 'atr', 'day_utc2',
               'future_max_move', 'future_min_move', 'future_er', 'atr_threshold',
               'bar_move', 'hour', 'day_id', 'day_high', 'day_low']

    exclude += ['day_high', 'day_low', 'day_open', 'high_90', 'low_90',
                'closePrice_ask', 'closePrice_bid', 'highPrice_ask', 'lowPrice_bid',
                'closePrice', 'lowPrice', 'open_price',
                'highPrice_bid', 'lowPrice_ask', 'openPrice_bid', 'openPrice_ask']

    features = [c for c in df.columns if c not in exclude]

    current_test_start = pd.to_datetime(WF_START).tz_localize('UTC')
    end_dt = pd.to_datetime(WF_END).tz_localize('UTC')

    all_oos_preds = []
    all_oos_actuals = []

    cycle = 1

    while current_test_start < end_dt:
        current_test_end = current_test_start + timedelta(days=RETRAIN_DAYS)
        print(f"\n>>> WF Cycle {cycle}: Testing {current_test_start.date()} to {min(current_test_end.date(), end_dt.date())}")

        # Split into Train and Test Sets
        df_train_all = df[df.index < current_test_start].copy()
        df_test = df[(df.index >= current_test_start) & (df.index < current_test_end)].copy()

        if df_test.empty:
            print("No test data for this cycle, skipping.")
            current_test_start = current_test_end
            cycle += 1
            continue

        # Balancing classes on Training Set
        df_trend = df_train_all[df_train_all['is_trend'] == 1]
        df_flat = df_train_all[df_train_all['is_trend'] == 0]

        if len(df_trend) == 0 or len(df_flat) == 0:
            print("Not enough trend/flat samples to train, skipping.")
            current_test_start = current_test_end
            cycle += 1
            continue

        df_flat_sampled = df_flat.sample(n=min(len(df_flat), len(df_trend)), random_state=42)
        df_balanced = pd.concat([df_trend, df_flat_sampled]).sort_index()

        X_train = df_balanced[features]
        y_train = df_balanced['is_trend']

        X_test = df_test[features]
        y_test = df_test['is_trend']

        # Train XGBoost
        clf = XGBClassifier(
            n_estimators=150, # Slightly reduced for faster WF cycles
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # Predict on Test Set
        y_pred = clf.predict(X_test)

        all_oos_preds.append(pd.Series(y_pred, index=df_test.index))
        all_oos_actuals.append(y_test)

        # Evaluate cycle
        print(f"Cycle {cycle} OOS Balanced Accuracy: {(y_pred == y_test).mean():.4f}")

        # Advance
        current_test_start = current_test_end
        cycle += 1

    # Final Evaluation
    print("\n" + "="*50)
    print("FINISHED WALK-FORWARD VALIDATION")
    print("="*50)

    if all_oos_preds:
        y_pred_all = pd.concat(all_oos_preds)
        y_test_all = pd.concat(all_oos_actuals)

        print("\n--- Aggregate OOS Walk-Forward Performance ---")
        print(classification_report(y_test_all, y_pred_all))

        # Save the cumulative walk-forward predictions for analysis
        df_results = pd.DataFrame({'actual': y_test_all, 'predicted': y_pred_all})
        results_path = PROJECT_ROOT / "xgboost_filter_model" / "wf_v11_results.csv"
        df_results.to_csv(results_path)
        print(f"Results saved to {results_path}")

        # Train final model on most recent data to save
        print("\nTraining final production model on last window...")
        # (Assuming the last trained 'clf' is the most recent production-ready model)
        model_path = PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v11_wf.joblib"
        joblib.dump(clf, model_path)
        print(f"Final model saved to {model_path}")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    train_wf_v11()

