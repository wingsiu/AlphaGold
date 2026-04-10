#!/usr/bin/env python3
"""
Train ML Model for Uptrend Recognition
======================================

This script trains machine learning models to recognize candlestick patterns
that precede uptrends in gold prices.

Usage:
    python train_uptrend_ml_model.py

Output:
    - Trained models saved to ml_models/
    - Performance metrics and visualizations
    - Feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data import DataLoader
from training.ml_candle_pattern_recognition import UptrendRecognitionSystem, plot_results
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)


def prepare_gold_data(df):
    """Prepare gold price data for ML training — handles both IG and legacy column names."""
    def pick(a, b):
        return df[a] if a in df.columns else df[b]

    df_prepared = pd.DataFrame({
        'open':  pick('openPrice',  'open'),
        'high':  pick('highPrice',  'high'),
        'low':   pick('lowPrice',   'low'),
        'close': pick('closePrice', 'close'),
    })

    if 'timestamp' in df.columns:
        df_prepared.index = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    elif 'snapshotTimeUTC' in df.columns:
        df_prepared.index = pd.to_datetime(df['snapshotTimeUTC'])

    return df_prepared


def analyze_prediction_accuracy(df, system, data, test_size=0.2):
    """Analyze when the model correctly predicts uptrends"""
    print("\n" + "="*80)
    print("PREDICTION ACCURACY ANALYSIS")
    print("="*80)

    from sklearn.model_selection import train_test_split

    # Split data
    X = data[system.extractor.feature_names].fillna(0)
    y = data['is_uptrend']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Get predictions for test set
    model = system.models['random_forest']
    y_pred_proba = model.predict_proba(X_test)

    # Analyze by confidence level
    test_data = X_test.copy()
    test_data['actual'] = y_test.values
    test_data['predicted_proba'] = y_pred_proba

    confidence_levels = [
        ('High Confidence (>70%)', 0.7, 1.0),
        ('Medium Confidence (50-70%)', 0.5, 0.7),
        ('Low Confidence (<50%)', 0.0, 0.5)
    ]

    print("\nAccuracy by Confidence Level:")
    print("-" * 60)

    for label, min_prob, max_prob in confidence_levels:
        mask = (test_data['predicted_proba'] >= min_prob) & (test_data['predicted_proba'] < max_prob)
        if mask.sum() > 0:
            accuracy = test_data[mask]['actual'].mean()
            count = mask.sum()
            print(f"{label:30} Count: {count:4}  Accuracy: {accuracy:.1%}")

    # ROC curve for both models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    from sklearn.metrics import roc_curve, auc

    for model_name in ['random_forest', 'gradient_boosting']:
        model = system.models[model_name]
        y_pred_proba_model = model.predict_proba(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_model)
        roc_auc = auc(fpr, tpr)

        ax1.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - Uptrend Detection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Probability distribution
    uptrend_probs = y_pred_proba[y_test == 1]
    no_uptrend_probs = y_pred_proba[y_test == 0]

    ax2.hist(no_uptrend_probs, bins=30, alpha=0.5, label='No Uptrend', color='red')
    ax2.hist(uptrend_probs, bins=30, alpha=0.5, label='Uptrend', color='green')
    ax2.axvline(x=0.5, color='k', linestyle='--', label='Threshold')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Probability Distribution by Actual Label')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('uptrend_ml_roc_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved ROC analysis to uptrend_ml_roc_analysis.png")

    return fig


def visualize_feature_patterns(system):
    """Visualize most important features"""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE VISUALIZATION")
    print("="*80)

    # Random Forest feature importance
    fig1 = system.models['random_forest'].plot_feature_importance(top_n=25)
    plt.savefig('uptrend_ml_feature_importance_rf.png', dpi=150, bbox_inches='tight')
    print("✅ Saved Random Forest feature importance to uptrend_ml_feature_importance_rf.png")

    # Gradient Boosting feature importance
    fig2 = system.models['gradient_boosting'].plot_feature_importance(top_n=25)
    plt.savefig('uptrend_ml_feature_importance_gb.png', dpi=150, bbox_inches='tight')
    print("✅ Saved Gradient Boosting feature importance to uptrend_ml_feature_importance_gb.png")

    # Print top features
    print("\nTop 10 Most Important Features (Random Forest):")
    print("-" * 60)
    top_features = system.models['random_forest'].feature_importance.head(10)
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']:30} {row['importance']:.4f}")

    return fig1, fig2


def test_on_recent_data(df, system, lookback=100):
    """Test model on recent price data"""
    print("\n" + "="*80)
    print("TESTING ON RECENT DATA")
    print("="*80)

    print(f"\nScanning last {lookback} candles for uptrend patterns...")

    predictions = system.scan_recent_patterns(df, model_type='random_forest', lookback=lookback)

    if len(predictions) > 0:
        print(f"\n✅ Found {len(predictions)} potential uptrend signals:")
        print("-" * 80)
        print(predictions.to_string(index=False))

        # Save predictions
        predictions.to_csv('recent_uptrend_predictions.csv', index=False)
        print("\n✅ Saved predictions to recent_uptrend_predictions.csv")

        # Visualize
        start_idx = max(0, len(df) - lookback - 50)
        df_recent = df.iloc[start_idx:].reset_index(drop=True)

        # Adjust prediction indices
        predictions_adjusted = predictions.copy()
        predictions_adjusted['idx'] = predictions_adjusted['idx'] - start_idx

        fig = plot_results(df_recent, predictions_adjusted,
                          title=f"Uptrend Predictions (Last {lookback} Candles)")
        plt.savefig('uptrend_ml_recent_predictions.png', dpi=150, bbox_inches='tight')
        print("✅ Saved visualization to uptrend_ml_recent_predictions.png")

    else:
        print("\n⚠️ No high-confidence uptrend signals detected in recent data")

    return predictions


def main():
    print("="*80)
    print("TRAIN ML MODEL FOR UPTREND RECOGNITION")
    print("="*80)
    print()

    # Configuration
    FORWARD_BARS = 15  # Look ahead 15 bars (~15 minutes for 1-min data)
    THRESHOLD_PCT = 0.15  # 0.15% price increase = uptrend ($3-4 on gold)

    print(f"Configuration:")
    print(f"  Forward lookback: {FORWARD_BARS} bars")
    print(f"  Uptrend threshold: {THRESHOLD_PCT}% price increase")
    print()

    # Load data
    print("Loading gold price data...")
    df_raw = DataLoader().load_data(
        table_name='gold_prices',
        start_date='2025-05-20',
        end_date='2026-02-05',
    )
    print()

    # Prepare data
    df = prepare_gold_data(df_raw)
    print(f"✅ Prepared {len(df)} candles for training")
    print()

    # Initialize ML system
    system = UptrendRecognitionSystem(
        forward_bars=FORWARD_BARS,
        threshold_pct=THRESHOLD_PCT
    )

    # Prepare features and labels
    data = system.prepare_data(df)

    # Train models
    X_train, X_test, y_train, y_test = system.train_models(data, test_size=0.2)

    # Analyze prediction accuracy
    analyze_prediction_accuracy(df, system, data)

    # Visualize features
    visualize_feature_patterns(system)

    # Test on recent data
    predictions = test_on_recent_data(df, system, lookback=200)

    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    system.save_models(directory='ml_models')
    print()

    # Summary
    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print()
    print("✅ Models trained and saved")
    print("✅ Performance metrics calculated")
    print("✅ Visualizations generated")
    print()
    print("Next steps:")
    print("  1. Review feature importance charts")
    print("  2. Check recent predictions")
    print("  3. Use predict_uptrend_live.py for real-time predictions")
    print()
    print("Files generated:")
    print("  - ml_models/random_forest_uptrend_model.pkl")
    print("  - ml_models/gradient_boosting_uptrend_model.pkl")
    print("  - uptrend_ml_feature_importance_rf.png")
    print("  - uptrend_ml_feature_importance_gb.png")
    print("  - uptrend_ml_roc_analysis.png")
    print("  - uptrend_ml_recent_predictions.png")
    print("  - recent_uptrend_predictions.csv")
    print()


if __name__ == "__main__":
    main()

