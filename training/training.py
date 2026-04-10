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

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data import DataLoader
try:
    from ml_alpha_model import UptrendRecognitionSystem, chronological_train_test_split, plot_results
except ModuleNotFoundError:
    # Support importing training.py as a module from project root.
    from training.ml_alpha_model import UptrendRecognitionSystem, chronological_train_test_split, plot_results
import warnings

warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parent.parent

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)


def prepare_gold_data(df):
    """Prepare gold price data for ML training"""
    def pick_column(candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    open_col = pick_column(['open', 'openPrice'])
    high_col = pick_column(['high', 'highPrice'])
    low_col = pick_column(['low', 'lowPrice'])
    close_col = pick_column(['close', 'closePrice'])
    volume_col = pick_column(['volume', 'lastTradedVolume'])
    timestamp_col = pick_column(['snapshotTimeUTC', 'timestamp'])

    missing = [name for name, col in {
        'open': open_col,
        'high': high_col,
        'low': low_col,
        'close': close_col,
    }.items() if col is None]
    if missing:
        raise ValueError(f"Missing required price columns: {', '.join(missing)}")

    df_prepared = pd.DataFrame({
        'open': df[open_col],
        'high': df[high_col],
        'low': df[low_col],
        'close': df[close_col],
        'volume': df[volume_col] if volume_col else 0.0,
    })

    if timestamp_col == 'timestamp':
        df_prepared.index = pd.to_datetime(df[timestamp_col], unit='ms', utc=True, errors='coerce')
    elif timestamp_col is not None:
        df_prepared.index = pd.to_datetime(df[timestamp_col], utc=True, errors='coerce')

    # Drop rows with invalid timestamps if index is datetime-based.
    if isinstance(df_prepared.index, pd.DatetimeIndex):
        df_prepared = df_prepared[~df_prepared.index.isna()]

    return df_prepared


def analyze_prediction_accuracy(df, system, data, test_size=0.2):
    """Analyze when the model correctly predicts uptrends"""
    print("\n" + "="*80)
    print("PREDICTION ACCURACY ANALYSIS")
    print("="*80)

    # Split data
    X = data[system.extractor.feature_names].fillna(0)
    y = data['is_uptrend']

    _, X_test, _, y_test = chronological_train_test_split(X, y, test_size=test_size)

    # Get predictions for test set
    model = system.models['random_forest']
    y_pred_proba = model.predict_proba(X_test)

    # Analyze by confidence level
    test_data = X_test.copy()
    test_data['actual'] = y_test.values
    test_data['predicted_proba'] = y_pred_proba

    confidence_levels = [
        ('High Confidence (>60%)', 0.6, 1.0),
        ('Medium Confidence (50-60%)', 0.5, 0.6),
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


def save_test_period_cache(df, data, system, test_size=0.2, out_path='training/test_period_signals.csv'):
    """Save chronological test-period features and model probabilities for direct backtest reuse."""

    X = data[system.extractor.feature_names].fillna(0)
    y = data['is_uptrend']

    _, X_test, _, y_test = chronological_train_test_split(X, y, test_size=test_size)

    rows = data.loc[X_test.index, ['idx', 'is_uptrend']].copy()
    rows['timestamp'] = df.index[rows['idx'].astype(int)].astype(str).values
    rows['actual'] = y_test.values

    rf_model = system.models['random_forest']
    gb_model = system.models['gradient_boosting']
    rows['rf_probability'] = rf_model.predict_proba(X_test)
    rows['gb_probability'] = gb_model.predict_proba(X_test)
    rows['rf_signal'] = rf_model.predict(X_test).astype(int)
    rows['gb_signal'] = gb_model.predict(X_test).astype(int)

    # Keep feature values for quick what-if analysis without recomputing extractor outputs.
    feature_snapshot = X_test.copy()
    feature_snapshot.columns = [f'feat_{c}' for c in feature_snapshot.columns]
    rows = pd.concat([rows.reset_index(drop=True), feature_snapshot.reset_index(drop=True)], axis=1)

    rows = rows.sort_values('idx').reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows.to_csv(out_path, index=False)
    print(
        f"✅ Saved chronological test-period cache to {out_path} "
        f"({len(rows)} rows, idx {rows['idx'].iloc[0]}->{rows['idx'].iloc[-1]})"
    )


def save_full_period_cache(df, data, system, out_path='training/full_period_signals_from_training.csv'):
    """Save full-period model probabilities from the same feature matrix used during training."""

    X_all = data[system.extractor.feature_names].fillna(0)
    rows = data[['idx', 'is_uptrend']].copy()
    rows['timestamp'] = df.index[rows['idx'].astype(int)].astype(str).values
    rows['actual'] = rows['is_uptrend'].astype(int)

    rf_model = system.models['random_forest']
    gb_model = system.models['gradient_boosting']
    rows['rf_probability'] = rf_model.predict_proba(X_all)
    rows['gb_probability'] = gb_model.predict_proba(X_all)
    rows['rf_signal'] = rf_model.predict(X_all).astype(int)
    rows['gb_signal'] = gb_model.predict(X_all).astype(int)

    rows = rows.sort_values('idx').reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows.to_csv(out_path, index=False)
    print(
        f"✅ Saved full-period signal cache to {out_path} "
        f"({len(rows)} rows, idx {rows['idx'].iloc[0]}->{rows['idx'].iloc[-1]})"
    )


def optimize_target_threshold_pct(df, forward_bars, candidates, target_positive_rate=0.02, max_adverse_low_pct=None):
    """Pick threshold_pct whose positive rate is closest to the target rate."""
    if len(df) <= forward_bars + 1:
        return candidates[0], []

    # Future max high over next N bars (exclude current bar).
    future_max_high = (
        df['high']
        .shift(-1)
        .rolling(window=forward_bars, min_periods=1)
        .max()
        .shift(-(forward_bars - 1))
    )
    future_min_low = (
        df['low']
        .shift(-1)
        .rolling(window=forward_bars, min_periods=1)
        .min()
        .shift(-(forward_bars - 1))
    )
    gain_pct = ((future_max_high - df['close']) / df['close']) * 100
    drawdown_pct = ((future_min_low - df['close']) / df['close']) * 100
    valid_mask = gain_pct.notna()
    if max_adverse_low_pct is not None:
        valid_mask &= drawdown_pct.notna()

    valid_gain = gain_pct[valid_mask]
    valid_drawdown = drawdown_pct[valid_mask]
    if len(valid_gain) == 0:
        return candidates[0], []

    stats = []
    for threshold in candidates:
        hit = valid_gain >= threshold
        if max_adverse_low_pct is not None:
            hit &= valid_drawdown >= -float(max_adverse_low_pct)
        pos_rate = float(hit.mean())
        stats.append((threshold, pos_rate, abs(pos_rate - target_positive_rate)))

    best = min(stats, key=lambda x: x[2])
    return best[0], stats


def build_parser():
    parser = argparse.ArgumentParser(description="Train ML models for uptrend recognition.")
    parser.add_argument("--table", default="gold_prices")
    parser.add_argument("--start-date", default="2025-05-20")
    parser.add_argument("--end-date", default="2026-02-04")
    parser.add_argument("--forward-bars", type=int, default=45)
    parser.add_argument("--threshold-pct", type=float, default=0.20)
    parser.add_argument("--max-adverse-low-pct", type=float, default=None)
    parser.add_argument("--auto-optimize-threshold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-positive-rate", type=float, default=0.05)
    parser.add_argument("--threshold-candidates", default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-dir", default="ml_models")
    return parser


def main():
    args = build_parser().parse_args()

    print("="*80)
    print("TRAIN ML MODEL FOR UPTREND RECOGNITION")
    print("="*80)
    print()

    # Configuration
    FORWARD_BARS = args.forward_bars
    THRESHOLD_PCT = args.threshold_pct
    MAX_ADVERSE_LOW_PCT = args.max_adverse_low_pct
    AUTO_OPTIMIZE_THRESHOLD = bool(args.auto_optimize_threshold)
    TARGET_POSITIVE_RATE = args.target_positive_rate

    print(f"Configuration:")
    print(f"  Forward lookback: {FORWARD_BARS} bars")
    print(f"  Uptrend threshold: {THRESHOLD_PCT}% price increase")
    if MAX_ADVERSE_LOW_PCT is not None:
        print(f"  Max adverse low: -{MAX_ADVERSE_LOW_PCT}% over the same window")
    print(f"  Auto optimize threshold: {AUTO_OPTIMIZE_THRESHOLD}")
    print(f"  Validation split: chronological 80/20")
    print()

    # Load data
    print("Loading gold price data...")
    data_loader = DataLoader()
    df_raw = data_loader.load_data(
        table_name=args.table,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print()

    # Prepare data
    df = prepare_gold_data(df_raw)
    print(f"✅ Prepared {len(df)} candles for training")
    print()

    if AUTO_OPTIMIZE_THRESHOLD:
        candidates = [float(x.strip()) for x in args.threshold_candidates.split(',') if x.strip()]
        best_threshold, stats = optimize_target_threshold_pct(
            df,
            forward_bars=FORWARD_BARS,
            candidates=candidates,
            target_positive_rate=TARGET_POSITIVE_RATE,
            max_adverse_low_pct=MAX_ADVERSE_LOW_PCT,
        )
        print("Threshold optimization (by target positive rate):")
        for threshold, pos_rate, _ in stats:
            print(f"  threshold={threshold:.2f}% -> positive_rate={pos_rate * 100:.2f}%")
        THRESHOLD_PCT = best_threshold
        print(f"✅ Selected THRESHOLD_PCT={THRESHOLD_PCT:.2f}% (target positive rate {TARGET_POSITIVE_RATE * 100:.2f}%)")
        print()

    # Initialize ML system
    system = UptrendRecognitionSystem(
        forward_bars=FORWARD_BARS,
        threshold_pct=THRESHOLD_PCT,
        max_adverse_low_pct=MAX_ADVERSE_LOW_PCT,
    )

    # Prepare features and labels
    data = system.prepare_data(df)

    # Train models
    X_train, X_test, y_train, y_test = system.train_models(data, test_size=args.test_size)

    # Analyze prediction accuracy
    analyze_prediction_accuracy(df, system, data)

    # Save reusable caches for direct backtest reuse.
    test_cache_path = ROOT_DIR / 'training' / 'test_period_signals.csv'
    full_cache_path = ROOT_DIR / 'training' / 'full_period_signals_from_training.csv'
    save_test_period_cache(df, data, system, test_size=args.test_size, out_path=str(test_cache_path))
    save_full_period_cache(df, data, system, out_path=str(full_cache_path))

    # Visualize features
    visualize_feature_patterns(system)

    # Test on recent data
    predictions = test_on_recent_data(df, system, lookback=200)

    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    model_dir = ROOT_DIR / args.model_dir
    system.save_models(directory=str(model_dir))
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
    print("  - ml_models/random_forest_alpha_model.pkl")
    print("  - ml_models/gradient_boosting_alpha_model.pkl")
    print("  - uptrend_ml_feature_importance_rf.png")
    print("  - uptrend_ml_feature_importance_gb.png")
    print("  - uptrend_ml_roc_analysis.png")
    print("  - uptrend_ml_recent_predictions.png")
    print("  - recent_uptrend_predictions.csv")
    print("  - training/test_period_signals.csv")
    print("  - training/full_period_signals_from_training.csv")
    print()


if __name__ == "__main__":
    main()

