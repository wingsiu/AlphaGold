"""
Machine Learning Candle Pattern Recognition for Uptrend Detection
===================================================================

This module uses machine learning to identify candlestick patterns that
precede uptrends in gold prices.

Features:
- Extracts 50+ candlestick pattern features
- Trains Random Forest and Gradient Boosting models
- Identifies uptrend probability with confidence scores
- Provides feature importance analysis
- Visualizes pattern recognition results

Author: AI Assistant
Date: February 2026
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import pickle
import os

warnings.filterwarnings('ignore')


class CandlePatternExtractor:
    """Extract candlestick pattern features for ML"""

    def __init__(self):
        self.feature_names = []

    def extract_single_candle_features(self, df, idx):
        """Extract features from a single candlestick"""
        features = {}

        o = df['open'].iloc[idx]
        h = df['high'].iloc[idx]
        l = df['low'].iloc[idx]
        c = df['close'].iloc[idx]

        # Basic measurements
        body = c - o
        body_abs = abs(body)
        range_val = h - l

        # Avoid division by zero
        if range_val == 0:
            range_val = 0.01

        # Body features
        features['body'] = body
        features['body_abs'] = body_abs
        features['body_pct'] = (body_abs / range_val) * 100
        features['is_bullish'] = 1 if body > 0 else 0
        features['is_bearish'] = 1 if body < 0 else 0

        # Wick features
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        features['upper_wick'] = upper_wick
        features['lower_wick'] = lower_wick
        features['upper_wick_pct'] = (upper_wick / range_val) * 100
        features['lower_wick_pct'] = (lower_wick / range_val) * 100

        # Pattern identifiers
        features['is_doji'] = 1 if (body_abs / range_val) < 0.1 else 0
        features['is_hammer'] = 1 if (lower_wick > 2 * body_abs and upper_wick < body_abs) else 0
        features['is_shooting_star'] = 1 if (upper_wick > 2 * body_abs and lower_wick < body_abs) else 0
        features['is_marubozu'] = 1 if (body_abs / range_val) > 0.9 else 0
        features['is_spinning_top'] = 1 if (0.3 < (body_abs / range_val) < 0.5) else 0

        # Range features
        features['range'] = range_val

        return features

    def extract_multi_candle_features(self, df, idx, lookback=5):
        """Extract features from multiple candles"""
        features = {}

        if idx < lookback:
            return features

        # Get recent candles
        window = df.iloc[idx - lookback:idx + 1]

        # Consecutive patterns
        consecutive_green = 0
        consecutive_red = 0

        for i in range(len(window)):
            candle_body = window['close'].iloc[i] - window['open'].iloc[i]
            if candle_body > 0:
                consecutive_green += 1
                consecutive_red = 0
            elif candle_body < 0:
                consecutive_red += 1
                consecutive_green = 0
            else:
                consecutive_green = 0
                consecutive_red = 0

        features['consecutive_green'] = consecutive_green
        features['consecutive_red'] = consecutive_red

        # Price momentum
        features['price_change_5'] = window['close'].iloc[-1] - window['close'].iloc[0]
        features['high_change_5'] = window['high'].iloc[-1] - window['high'].iloc[0]
        features['low_change_5'] = window['low'].iloc[-1] - window['low'].iloc[0]

        # Volatility
        features['range_std_5'] = window['high'].sub(window['low']).std()
        features['close_std_5'] = window['close'].std()

        # Trend strength
        features['higher_highs'] = sum(window['high'].iloc[i] > window['high'].iloc[i - 1]
                                       for i in range(1, len(window)))
        features['higher_lows'] = sum(window['low'].iloc[i] > window['low'].iloc[i - 1]
                                      for i in range(1, len(window)))
        features['lower_highs'] = sum(window['high'].iloc[i] < window['high'].iloc[i - 1]
                                      for i in range(1, len(window)))
        features['lower_lows'] = sum(window['low'].iloc[i] < window['low'].iloc[i - 1]
                                     for i in range(1, len(window)))

        # Average body size
        bodies = abs(window['close'] - window['open'])
        features['avg_body_size'] = bodies.mean()
        features['max_body_size'] = bodies.max()

        # Support/Resistance proximity
        recent_high = window['high'].max()
        recent_low = window['low'].min()
        current_close = window['close'].iloc[-1]

        if recent_high - recent_low > 0:
            features['price_position'] = (current_close - recent_low) / (recent_high - recent_low)
        else:
            features['price_position'] = 0.5

        return features

    def extract_technical_indicators(self, df, idx, lookback=20):
        """Extract technical indicator features"""
        features = {}

        if idx < lookback:
            return features

        window = df.iloc[idx - lookback:idx + 1]

        # Simple Moving Averages
        features['sma_5'] = window['close'].iloc[-5:].mean() if idx >= 5 else window['close'].mean()
        features['sma_10'] = window['close'].iloc[-10:].mean() if idx >= 10 else window['close'].mean()
        features['sma_20'] = window['close'].mean()

        current_price = df['close'].iloc[idx]
        features['price_vs_sma5'] = (current_price - features['sma_5']) / features['sma_5'] * 100
        features['price_vs_sma10'] = (current_price - features['sma_10']) / features['sma_10'] * 100
        features['price_vs_sma20'] = (current_price - features['sma_20']) / features['sma_20'] * 100

        # RSI-like momentum
        changes = window['close'].diff()
        gains = changes.clip(lower=0).mean()
        losses = -changes.clip(upper=0).mean()

        if losses != 0:
            rs = gains / losses
            features['rsi_approx'] = 100 - (100 / (1 + rs))
        else:
            features['rsi_approx'] = 100

        # Volume proxy (range as volume substitute)
        features['avg_range'] = (window['high'] - window['low']).mean()
        features['current_range_vs_avg'] = ((df['high'].iloc[idx] - df['low'].iloc[idx]) /
                                            features['avg_range']) if features['avg_range'] > 0 else 1

        return features

    def extract_all_features(self, df, idx):
        """Extract all features for a given candle index"""
        all_features = {}

        # Single candle features
        all_features.update(self.extract_single_candle_features(df, idx))

        # Multi-candle features
        all_features.update(self.extract_multi_candle_features(df, idx, lookback=5))

        # Technical indicators
        all_features.update(self.extract_technical_indicators(df, idx, lookback=20))

        return all_features

    def create_feature_dataframe(self, df, min_idx=20):
        """Create feature dataframe for entire dataset"""
        import sys

        # Validate we have enough data
        if len(df) < min_idx:
            print(f"⚠️  Not enough data for feature extraction!")
            print(f"   Need at least {min_idx} bars, but only have {len(df)} bars")
            return pd.DataFrame()  # Return empty dataframe

        feature_list = []
        total = len(df) - min_idx

        print(f"Extracting features from {total:,} candles...")
        sys.stdout.flush()

        for i, idx in enumerate(range(min_idx, len(df))):
            features = self.extract_all_features(df, idx)
            features['idx'] = idx
            feature_list.append(features)

            # Show progress every 10%
            if (i + 1) % max(1, total // 10) == 0 or i == 0:
                progress = (i + 1) / total * 100
                print(f"  Progress: {progress:.0f}% ({i + 1:,}/{total:,} candles)")
                sys.stdout.flush()

        feature_df = pd.DataFrame(feature_list)
        self.feature_names = [col for col in feature_df.columns if col != 'idx']

        print(f"✅ Extracted {len(self.feature_names)} features from {len(feature_df)} candles")

        return feature_df


class UptrendLabeler:
    """Label candles based on future price movement (uptrend detection)"""

    def __init__(self, forward_bars=10, threshold_pct=0.2):
        """
        Parameters:
        -----------
        forward_bars : int
            Number of bars to look ahead
        threshold_pct : float
            Minimum price increase percentage to label as uptrend
        """
        self.forward_bars = forward_bars
        self.threshold_pct = threshold_pct

    def create_labels(self, df, feature_df):
        """Create uptrend labels based on future price movement"""
        import sys

        # Check if feature_df is empty
        if len(feature_df) == 0:
            print("⚠️  Cannot create labels: feature dataframe is empty")
            return pd.DataFrame()

        # Check if 'idx' column exists
        if 'idx' not in feature_df.columns:
            print("⚠️  Cannot create labels: 'idx' column missing from feature dataframe")
            print(f"   Available columns: {list(feature_df.columns)}")
            return pd.DataFrame()

        labels = []
        total = len(feature_df)

        print(f"Creating labels for {total:,} samples...")
        sys.stdout.flush()

        for i, idx in enumerate(feature_df['idx'].values):
            # Look ahead
            future_end_idx = min(idx + self.forward_bars, len(df) - 1)

            current_price = df['close'].iloc[idx]
            future_high = df['high'].iloc[idx + 1:future_end_idx + 1].max() if future_end_idx > idx else current_price

            # Calculate price increase
            price_increase_pct = ((future_high - current_price) / current_price) * 100

            # Label as uptrend if price increases by threshold
            is_uptrend = 1 if price_increase_pct >= self.threshold_pct else 0

            labels.append({
                'idx': idx,
                'is_uptrend': is_uptrend,
                'future_gain_pct': price_increase_pct
            })

            # Show progress every 10%
            if (i + 1) % max(1, total // 10) == 0 or i == 0:
                progress = (i + 1) / total * 100
                print(f"  Progress: {progress:.0f}% ({i + 1:,}/{total:,} samples)")
                sys.stdout.flush()

        label_df = pd.DataFrame(labels)

        # Print label distribution
        if len(label_df) > 0:
            uptrends = label_df['is_uptrend'].sum()
            total_labels = len(label_df)
            print(f"✅ Created labels for {total_labels} samples")
            print(f"\nLabel distribution:")
            print(f"  Uptrends: {uptrends} ({uptrends / total_labels * 100:.1f}%)")
            print(f"  No uptrends: {total_labels - uptrends} ({(total_labels - uptrends) / total_labels * 100:.1f}%)")

        return label_df


class MLCandlePatternModel:
    """Machine Learning model for candle pattern recognition"""

    def __init__(self, model_type='random_forest'):
        """
        Parameters:
        -----------
        model_type : str
            'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.feature_names = None

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X_train, y_train, feature_names):
        """Train the model"""
        import sys

        print(f"Training {self.model_type} model...")
        print(f"Training samples: {len(X_train):,}")
        print(f"Positive class (uptrend): {y_train.sum():,} ({y_train.sum() / len(y_train) * 100:.1f}%)")
        sys.stdout.flush()

        # Store feature names
        self.feature_names = feature_names

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"Cross-validation ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        sys.stdout.flush()

        return self

    def predict(self, X):
        """Predict uptrend probability"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Predict uptrend probability scores"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_scaled = self.scaler.transform(X_test)

        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        print(f"\nTest samples: {len(X_test)}")
        print(f"Actual uptrends: {y_test.sum()} ({y_test.sum() / len(y_test) * 100:.1f}%)")
        print(f"Predicted uptrends: {y_pred.sum()} ({y_pred.sum() / len(y_pred) * 100:.1f}%)")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Uptrend', 'Uptrend']))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.3f}")

        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }

    def plot_feature_importance(self, top_n=20):
        """Plot top feature importances"""
        if self.feature_importance is None:
            print("No feature importance available")
            return

        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)

        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Most Important Features for Uptrend Detection')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        return plt.gcf()

    def save(self, filepath):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']

        print(f"Model loaded from {filepath}")
        return self


class UptrendRecognitionSystem:
    """Complete system for ML-based uptrend recognition"""

    def __init__(self, forward_bars=10, threshold_pct=0.2):
        """
        Parameters:
        -----------
        forward_bars : int
            Number of bars to look ahead for uptrend detection
        threshold_pct : float
            Minimum price increase percentage to label as uptrend
        """
        self.extractor = CandlePatternExtractor()
        self.labeler = UptrendLabeler(forward_bars=forward_bars, threshold_pct=threshold_pct)
        self.models = {}
        self.results = {}

    def prepare_data(self, df):
        """Prepare features and labels from price data"""
        print("\n" + "=" * 80)
        print("PREPARING DATA")
        print("=" * 80)

        print(f"\nExtracting candlestick features...")
        feature_df = self.extractor.create_feature_dataframe(df)
        print(f"✅ Extracted {len(self.extractor.feature_names)} features from {len(feature_df)} candles")

        print(f"\nCreating uptrend labels...")
        label_df = self.labeler.create_labels(df, feature_df)
        print(f"✅ Created labels for {len(label_df)} samples")

        # Merge features and labels
        data = feature_df.merge(label_df, on='idx')

        print(f"\nLabel distribution:")
        print(f"  Uptrends: {data['is_uptrend'].sum()} ({data['is_uptrend'].sum() / len(data) * 100:.1f}%)")
        print(
            f"  No uptrends: {(~data['is_uptrend'].astype(bool)).sum()} ({(~data['is_uptrend'].astype(bool)).sum() / len(data) * 100:.1f}%)")

        return data

    def train_models(self, data, test_size=0.2):
        """Train multiple ML models"""
        print("\n" + "=" * 80)
        print("TRAINING MODELS")
        print("=" * 80)

        # Prepare X and y
        X = data[self.extractor.feature_names].fillna(0)
        y = data['is_uptrend']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\nTrain size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")

        # Train Random Forest
        print("\n" + "-" * 80)
        rf_model = MLCandlePatternModel('random_forest')
        rf_model.train(X_train, y_train, self.extractor.feature_names)
        self.models['random_forest'] = rf_model

        # Train Gradient Boosting
        print("\n" + "-" * 80)
        gb_model = MLCandlePatternModel('gradient_boosting')
        gb_model.train(X_train, y_train, self.extractor.feature_names)
        self.models['gradient_boosting'] = gb_model

        # Evaluate both models
        print("\n" + "=" * 80)
        print("RANDOM FOREST RESULTS")
        self.results['random_forest'] = rf_model.evaluate(X_test, y_test)

        print("\n" + "=" * 80)
        print("GRADIENT BOOSTING RESULTS")
        self.results['gradient_boosting'] = gb_model.evaluate(X_test, y_test)

        return X_train, X_test, y_train, y_test

    def predict_uptrend(self, df, idx, model_type='random_forest'):
        """Predict if a candle is at the start of an uptrend"""
        features = self.extractor.extract_all_features(df, idx)
        X = pd.DataFrame([features])[self.extractor.feature_names].fillna(0)

        model = self.models[model_type]
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        return {
            'is_uptrend': bool(prediction),
            'uptrend_probability': probability,
            'confidence': 'High' if probability > 0.7 else 'Medium' if probability > 0.5 else 'Low'
        }

    def scan_recent_patterns(self, df, model_type='random_forest', lookback=20):
        """Scan recent candles for uptrend patterns"""
        results = []

        start_idx = max(20, len(df) - lookback)

        for idx in range(start_idx, len(df)):
            prediction = self.predict_uptrend(df, idx, model_type)

            if prediction['is_uptrend']:
                results.append({
                    'idx': idx,
                    'timestamp': df.index[idx] if hasattr(df, 'index') else idx,
                    'close': df['close'].iloc[idx],
                    'probability': prediction['uptrend_probability'],
                    'confidence': prediction['confidence']
                })

        return pd.DataFrame(results)

    def save_models(self, directory='ml_models'):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)

        for name, model in self.models.items():
            filepath = os.path.join(directory, f'{name}_uptrend_model.pkl')
            model.save(filepath)

    def load_models(self, directory='ml_models'):
        """Load saved models"""
        for model_type in ['random_forest', 'gradient_boosting']:
            filepath = os.path.join(directory, f'{model_type}_uptrend_model.pkl')
            if os.path.exists(filepath):
                model = MLCandlePatternModel(model_type)
                model.load(filepath)
                self.models[model_type] = model


def plot_results(df, predictions_df, title="Uptrend Predictions"):
    """Plot price chart with uptrend predictions"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])

    # Plot price
    ax1.plot(df['close'], label='Close Price', alpha=0.7)

    # Mark uptrend predictions
    for _, pred in predictions_df.iterrows():
        idx = pred['idx']
        color = 'green' if pred['confidence'] == 'High' else 'orange'
        ax1.axvline(x=idx, color=color, alpha=0.3, linestyle='--')
        ax1.scatter(idx, df['close'].iloc[idx], color=color, s=100, zorder=5)

    ax1.set_xlabel('Bar Index')
    ax1.set_ylabel('Price')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot probability
    ax2.scatter(predictions_df['idx'], predictions_df['probability'],
                c=predictions_df['probability'], cmap='RdYlGn', s=50)
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax2.set_xlabel('Bar Index')
    ax2.set_ylabel('Uptrend Probability')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("ML Candle Pattern Recognition System")
    print("This module should be imported and used by other scripts")
    print("\nExample usage:")
    print("""
    from ml_candle_pattern_recognition import UptrendRecognitionSystem
    from data_utils import load_standard_period

    # Load data
    df = load_standard_period()

    # Initialize system
    system = UptrendRecognitionSystem(forward_bars=10, threshold_pct=0.2)

    # Prepare and train
    data = system.prepare_data(df)
    system.train_models(data)

    # Predict on recent data
    predictions = system.scan_recent_patterns(df, lookback=50)
    print(predictions)

    # Save models
    system.save_models()
    """)

