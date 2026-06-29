#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from datetime import timedelta
from zoneinfo import ZoneInfo
from config.v13_config import WF_CONFIG

# Add project root to sys.path
from V13._paths import PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse logic from v13 Filter and previous directional models
from xgboost_filter_model.train_filter_v13_wf_image import load_price_data, prepare_data_v13
from xgboost_filter_model.train_directional_model_v9 import add_momentum_features
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features

def prepare_stage2_data(start_date="2020-01-01", end_date="2026-05-10"):
    """
    Loads and prepares the dataset for Stage 2.
    Specifically uses v13 features (Image model + Liquidity) and adds Directional/Momentum features.
    """
    print(f"Preparing Stage 2 data from {start_date} to {end_date}...")

    # 1. Load data using v13 logic (includes Image S1 Probs and Liquidity)
    df = prepare_data_v13(start_date=start_date, end_date=end_date)

    # 2. Add Directional specific features
    df = add_directional_features(df)
    df = add_ma_features(df)
    df = add_momentum_features(df)

    # 3. Handle Target: +1 (Up) vs -1 (Down), ignoring 0 (Flat) for Stage 2
    # The 'target_v10' from prepare_data_v13 (via redefine_target_v10) already has 1, -1, 0.
    # Stage 2 focuses only on the samples where target is NOT 0.

    df = df.dropna()
    return df

def train_stage2_wf():
    # 1. Config (match filter model exactly)
    FULL_START = WF_CONFIG["full_start"]
    WF_START = WF_CONFIG["wf_start"]
    WF_END = WF_CONFIG["wf_end"]
    RETRAIN_DAYS = WF_CONFIG["retrain_days"]

    # Load filter model (to simulate real-world usage where only Stage 1 Trends are passed to Stage 2)
    filter_model_path = PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v13_wf_image.joblib"
    if not filter_model_path.exists():
        print(f"Error: Stage 1 Filter model (v13) not found at {filter_model_path}. Please train it first.")
        return
    filter_model = joblib.load(filter_model_path)

    df = prepare_stage2_data(start_date=FULL_START, end_date=WF_END)

    # 2. Feature Selection
    # Exclude targets and raw metadata
    exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
               'day_high_rolling', 'day_low_rolling', 'day_open',
               'Dchange_utc2_rel', 'Dupper_wick_utc2_rel', 'Dlower_wick_utc2_rel',
               'trend_label', 'target_v10', 'is_trend', 'atr', 'day_utc2',
               'future_max_move', 'future_min_move', 'future_er', 'atr_threshold',
               'bar_move', 'hour', 'day_id', 'day_high', 'day_low']

    exclude += ['day_high', 'day_low', 'day_open', 'high_90', 'low_90',
                'closePrice_ask', 'closePrice_bid', 'highPrice_ask', 'lowPrice_bid',
                'closePrice', 'lowPrice', 'open_price',
                'highPrice_bid', 'lowPrice_ask', 'openPrice_bid', 'openPrice_ask']

    features = [c for c in df.columns if c not in exclude]

    # Filter features used by Stage 1 model to apply it correctly
    # Note: Stage 1 features are a subset of Stage 2 features
    s1_features = [f for f in features if f not in [
        'directional_change_15', 'directional_change_30', 'directional_change_90',
        'wick_ratio_15', 'wick_ratio_30', 'wick_ratio_90',
        'price_vs_ma_10', 'price_vs_ma_30', 'price_vs_ma_90', 'ma_10_vs_30', 'ma_30_vs_90',
        'rsi_14', 'rsi_30', 'macd', 'macd_signal', 'macd_diff', 'roc_15', 'roc_30', 'roc_60'
    ]]

    current_test_start = pd.to_datetime(WF_START)
    end_dt = pd.to_datetime(WF_END)
    # Robust timezone handling: localize if naive, convert if tz-aware
    if current_test_start.tzinfo is None:
        current_test_start = current_test_start.tz_localize('UTC')
    else:
        current_test_start = current_test_start.tz_convert('UTC')
    if end_dt.tzinfo is None:
        end_dt = end_dt.tz_localize('UTC')
    else:
        end_dt = end_dt.tz_convert('UTC')

    all_oos_preds = []
    all_oos_actuals = []

    cycle = 1

    while current_test_start < end_dt:
        current_test_end = current_test_start + timedelta(days=RETRAIN_DAYS)
        print(f"\n>>> Stage 2 WF Cycle {cycle}: Testing {current_test_start.date()} to {min(current_test_end.date(), end_dt.date())}")

        # Save model for this cycle to avoid future retraining
        model_dir = PROJECT_ROOT / "runtime" / "bot_assets" / "wf_models_v13"
        model_dir.mkdir(parents=True, exist_ok=True)
        cycle_model_path = model_dir / f"directional_v13_cycle_{cycle}_{current_test_start.date()}.joblib"

        # --- PRE-FILTERING SAMPLES ---
        # We ONLY train/test on samples where Stage 1 PREDICTS a Trend.
        # This aligns with the "Waterfall" architecture.

        # Split full DF for Stage 1 prediction
        df_train_all = df[df.index < current_test_start].copy()
        df_test_all = df[(df.index >= current_test_start) & (df.index < current_test_end)].copy()

        if df_test_all.empty:
            print("No test data for this cycle, skipping.")
            current_test_start = current_test_end
            cycle += 1
            continue

        # Predict Stage 1 trends
        df_train_all['s1_pred'] = filter_model.predict(df_train_all[s1_features])
        df_test_all['s1_pred'] = filter_model.predict(df_test_all[s1_features])

        # Directional Model Dataset: Predicted Trend=1 AND actual target is NOT 0 (since Stage 2 is pure direction)
        # However, for realistic testing, we only care about performance ON predicted trends.
        # If Stage 1 is wrong (predicts Trend on a Flat), Stage 2 will still try to guess a direction.

        df_train_s2 = df_train_all[(df_train_all['s1_pred'] == 1) & (df_train_all['target_v10'] != 0)].copy()
        df_test_s2 = df_test_all[df_test_all['s1_pred'] == 1].copy()

        if len(df_train_s2) < 50 or df_test_s2.empty:
            print(f"Insufficient samples (Train: {len(df_train_s2)}, Test: {len(df_test_s2)}). Skipping cycle.")
            current_test_start = current_test_end
            cycle += 1
            continue

        # Map targets to Binary: 1 = Up, 0 = Down (mapping -1 to 0)
        y_train = df_train_s2['target_v10'].map({1: 1, -1: 0})
        X_train = df_train_s2[features]

        # In test, we might have actual '0' (Flat) if Stage 1 failed.
        # But we must compare Stage 2's binary choice (+1 or -1) against reality.
        # If reality is 0, then ANY binary choice by Stage 2 is technicially 'wrong' in a system context,
        # but here we focus on Directional Accuracy on actual trends.
        y_test = df_test_s2['target_v10']
        X_test = df_test_s2[features]

        # Balancing Training Set
        df_up = df_train_s2[df_train_s2['target_v10'] == 1]
        df_down = df_train_s2[df_train_s2['target_v10'] == -1]
        min_samples = min(len(df_up), len(df_down))
        df_balanced = pd.concat([
            df_up.sample(min_samples, random_state=42),
            df_down.sample(min_samples, random_state=42)
        ]).sort_index()

        X_train_bal = df_balanced[features]
        y_train_bal = df_balanced['target_v10'].map({1: 1, -1: 0})

        # 3. Train XGBoost (Binary)
        clf = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        clf.fit(X_train_bal, y_train_bal)

        # Save the model for this specific cycle
        joblib.dump(clf, cycle_model_path)
        print(f"  Stage 2 Cycle {cycle} model saved to {cycle_model_path}")

        # 4. Predict
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        # Map 1 -> +1, 0 -> -1
        y_pred = np.where(y_pred_prob > 0.5, 1, -1)

        all_oos_preds.append(pd.Series(y_pred, index=df_test_s2.index))
        all_oos_actuals.append(y_test)

        # Accuracy calculation (only on samples where actual trend was +/- 1)
        valid_indices = y_test != 0
        if valid_indices.any():
            acc = (y_pred[valid_indices] == y_test[valid_indices]).mean()
            print(f"Cycle {cycle} Directional Accuracy (on actual Trends): {acc:.4f}")
        else:
            print(f"Cycle {cycle} had no actual Trends in S1-filtered test set.")

        # Advance
        current_test_start = current_test_end
        cycle += 1

    # Final Evaluation
    print("\n" + "="*50)
    print("FINISHED STAGE 2 WALK-FORWARD VALIDATION")
    print("="*50)

    if all_oos_preds:
        y_pred_all = pd.concat(all_oos_preds)
        y_test_all = pd.concat(all_oos_actuals)

        # For the classification report, we filter out 0 from actuals to see pure directional performance
        mask = y_test_all != 0
        print("\n--- Aggregate Stage 2 Directional Performance (OOS) ---")
        if mask.any():
            print(classification_report(y_test_all[mask], y_pred_all[mask], target_names=['Down (-1)', 'Up (+1)']))
        else:
            print("No actual trends found in filtered test set.")

        # Save results
        df_results = pd.DataFrame({'actual': y_test_all, 'predicted': y_pred_all})
        results_path = PROJECT_ROOT / "xgboost_filter_model" / "wf_v13_stage2_results.csv"
        df_results.to_csv(results_path)
        print(f"Results saved to {results_path}")

        # Train final production model on last window
        print("\nTraining final Stage 2 model on most recent data...")
        model_path = PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v13_wf.joblib"
        joblib.dump(clf, model_path)
        print(f"Final Stage 2 model saved to {model_path}")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    train_stage2_wf()

