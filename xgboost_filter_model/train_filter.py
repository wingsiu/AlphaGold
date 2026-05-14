#!/usr/bin/env python3
"""
Trains an XGBoost filter model to classify market regimes (Strong Trend, Consolidation, Noise)
for 1-minute gold price data, as part of a two-stage trading architecture.
"""
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import pandas_ta as pta
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add project root to sys.path to allow importing from other directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import DataLoader

def load_price_data(start_date: str = "2020-01-01", end_date: str = "2026-05-07") -> pd.DataFrame:
    """Loads 1-minute gold price data for the specified date range."""
    print(f"Loading data from {start_date} to {end_date}...")
    loader = DataLoader()
    df = loader.load_data(
        table_name='gold_prices',
        start_date=start_date,
        end_date=end_date
    )
    print(f"Data loaded successfully: {len(df)} rows.")
    # Ensure index is datetime
    df.index = pd.to_datetime(df['timestamp'], unit='ms')

    # Rename columns to match the expected format for feature engineering
    df = df.rename(columns={
        'openPrice': 'open',
        'highPrice': 'high',
        'lowPrice': 'low',
        'closePrice': 'close',
        'lastTradedVolume': 'volume'
    })

    # Ensure index is timezone-aware (UTC)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    return df

def _session_flag_and_progress(ts_utc: pd.Timestamp, spec: dict) -> tuple[float, float]:
    """Calculates if a timestamp is in a session and the progress through it."""
    local_ts = ts_utc.tz_convert(spec["timezone"])
    minute_of_day = local_ts.hour * 60 + local_ts.minute
    start_min = spec["start_hour"] * 60 + spec["start_minute"]
    end_min = spec["end_hour"] * 60 + spec["end_minute"]

    in_session = start_min <= minute_of_day < end_min
    if not in_session:
        return 0.0, 0.0

    duration = max(end_min - start_min, 1)
    progress = (minute_of_day - start_min) / duration
    return 1.0, np.clip(progress, 0.0, 1.0)

def fractal_dimension(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculates a simplified fractal dimension index.
    This measures the "jaggedness" of the price path.
    A value close to 1 indicates a smooth, trending market.
    A value close to 2 indicates a noisy, random market.
    """
    roll = df['close'].rolling(window=window)
    price_range = roll.max() - roll.min()

    # Sum of bar-to-bar distances (path length)
    path_length = abs(df['close'].diff()).rolling(window=window - 1).sum()

    # Avoid division by zero
    price_range_safe = price_range.replace(0, np.nan)

    # The dimension is related to how path length scales with range
    # A simplified index can be derived from their ratio
    fd = path_length / price_range_safe

    # Normalize to a more standard range if desired, but raw ratio is informative
    return fd


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the feature matrix for the XGBoost filter model. The entire process is now
    based on a 15-minute timeframe.
    """
    print("Resampling to 15-minute timeframe...")
    # 1. Resample to 15-minute bars
    df_15m = df.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    print(f"Resampled data shape: {df_15m.shape}")

    print("Preparing features on 15-minute data...")
    # 2. Basic price changes (Log Returns)
    df_15m['returns'] = np.log(df_15m['close'] / df_15m['close'].shift(1))

    # 3. Trend Strength: ADX and its slope
    adx_indicator = ta.trend.ADXIndicator(df_15m['high'], df_15m['low'], df_15m['close'], window=14)
    df_15m['adx'] = adx_indicator.adx()
    df_15m['adx_slope'] = df_15m['adx'].diff(3)

    # 4. Volatility Indicators: ATR and Realized Volatility
    df_15m['atr'] = ta.volatility.AverageTrueRange(df_15m['high'], df_15m['low'], df_15m['close'], window=14).average_true_range()
    df_15m['volatility'] = df_15m['returns'].rolling(window=30).std()

    # 5. Efficiency Ratio & Fractal Dimension
    net_change = abs(df_15m['close'] - df_15m['close'].shift(6))
    sum_of_changes = abs(df_15m['close'].diff()).rolling(window=6).sum()
    df_15m['efficiency_ratio'] = net_change / sum_of_changes
    df_15m['fractal_dimension'] = fractal_dimension(df_15m, window=14)

    # 6. Volume Features
    # Volume-Price Correlation
    df_15m['volume_price_corr'] = df_15m['returns'].rolling(window=30).corr(df_15m['volume'].pct_change())
    # Volume Trend (Ratio of moving averages)
    df_15m['volume_trend'] = df_15m['volume'].rolling(window=10).mean() / df_15m['volume'].rolling(window=30).mean()
    # Volume Oscillator
    pvo_df = pta.pvo(df_15m['volume'], fast=5, slow=20)
    df_15m['volume_osc'] = pvo_df.iloc[:, 0] # Use the main PVO line


    # 7. Time Features (with UTC+2 day start and session progress)

    # Define day based on UTC+2 offset
    day_start_offset = pd.Timedelta(hours=2)
    df_15m['day_utc2'] = (df_15m.index + day_start_offset).floor('D')

    # Calculate day-based features
    df_15m['day_open'] = df_15m.groupby('day_utc2')['open'].transform('first')
    df_15m['day_high'] = df_15m.groupby('day_utc2')['high'].cummax()
    df_15m['day_low'] = df_15m.groupby('day_utc2')['low'].cummin()

    day_range = df_15m['day_high'] - df_15m['day_low']
    day_range_safe = day_range.replace(0, np.nan) # Avoid division by zero

    df_15m['change'] = (df_15m['close'] - df_15m['day_open']) / day_range_safe
    df_15m['upper_wick'] = (df_15m['day_high'] - df_15m[['day_open', 'close']].max(axis=1)) / day_range_safe
    df_15m['lower_wick'] = (df_15m[['day_open', 'close']].min(axis=1) - df_15m['day_low']) / day_range_safe

    # Calculate current bar features
    bar_range = df_15m['high'] - df_15m['low']
    bar_range_safe = bar_range.replace(0, np.nan)
    df_15m['bar_change'] = (df_15m['close'] - df_15m['open']) / bar_range_safe
    df_15m['bar_upper_wick'] = (df_15m['high'] - df_15m[['open', 'close']].max(axis=1)) / bar_range_safe
    df_15m['bar_lower_wick'] = (df_15m[['open', 'close']].min(axis=1) - df_15m['low']) / bar_range_safe

    # Calculate time elapsed since day start
    day_start_time = df_15m.groupby('day_utc2')['day_utc2'].transform('first')
    df_15m['day_progress'] = (df_15m.index - day_start_time).dt.total_seconds() / (24 * 3600)

    # Define sessions with timezones
    sessions = {
        "asia": {"timezone": ZoneInfo("Asia/Hong_Kong"), "start_hour": 8, "start_minute": 0, "end_hour": 16, "end_minute": 0},
        "london": {"timezone": ZoneInfo("Europe/London"), "start_hour": 8, "start_minute": 0, "end_hour": 16, "end_minute": 30},
        "ny": {"timezone": ZoneInfo("America/New_York"), "start_hour": 9, "start_minute": 30, "end_hour": 16, "end_minute": 0},
    }

    # Calculate session features
    session_features = pd.DataFrame(index=df_15m.index)
    for name, spec in sessions.items():
        flags_progress = df_15m.index.to_series().apply(lambda ts: _session_flag_and_progress(ts, spec))
        session_features[f'is_{name}'] = [fp[0] for fp in flags_progress]
        session_features[f'{name}_progress'] = [fp[1] for fp in flags_progress]

    df_15m = df_15m.join(session_features)


    # 8. Define Label (Target) based on the next 4 bars (1 hour)
    future_window = 12

    # Calculate future values first
    future_highs = df_15m['high'].shift(-future_window).rolling(window=future_window).max()
    future_lows = df_15m['low'].shift(-future_window).rolling(window=future_window).min()
    future_er = df_15m['efficiency_ratio'].shift(-future_window).rolling(window=future_window).mean()

    df_15m['future_max_move'] = future_highs - df_15m['close']
    df_15m['future_min_move'] = future_lows - df_15m['close']
    df_15m['atr_threshold'] = 25 #df_15m['atr'] * 2
    df_15m['future_er'] = future_er # Keep future_er for analysis

    # Define the label before dropping NaNs from feature calculation
    is_large_move = (df_15m['future_max_move'] > df_15m['atr_threshold']) | (df_15m['future_min_move'].abs() > df_15m['atr_threshold'])
    is_efficient_future = future_er > 0.3

    df_15m['is_strong_trend'] = (is_large_move & is_efficient_future).astype(int)
    #df_15m['is_strong_trend'] = (is_large_move).astype(int)

    print("Features and labels prepared.")
    # Now drop all rows with any NaNs, which will include the rows at the end
    # where future values could not be calculated.
    return df_15m.dropna()

def visualize_features(df: pd.DataFrame):
    """Creates visualizations to analyze the relationship between features and price."""
    print("Generating feature visualizations...")

    # Limit to a smaller sample for clarity, e.g., a few months on the 15-min chart
    sample_df = df["2026-01-01":"2026-04-30"].copy()

    # 1. Plot Price vs. Efficiency Ratio and ADX Slope
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    ax1.set_title('Gold Price vs. Efficiency Ratio & ADX Slope')
    ax1.plot(sample_df.index, sample_df['close'], label='Close Price', color='blue')
    ax1.legend(loc='upper left')

    ax2.plot(sample_df.index, sample_df['efficiency_ratio'], label='Efficiency Ratio', color='green', alpha=0.7)
    ax2.axhline(0.5, color='gray', linestyle='--', label='ER=0.5')
    ax2.legend(loc='upper left')

    ax3.bar(sample_df.index, sample_df['adx_slope'], label='ADX Slope (3min)', color='red', alpha=0.5)
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'xgboost_filter_model' / 'feature_vs_price.png')
    print("Saved feature_vs_price.png")

    # 2. Boxplot of Efficiency Ratio during Trend vs. Non-Trend
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='is_strong_trend', y='efficiency_ratio', data=df)
    plt.title('Efficiency Ratio Distribution for Trend vs. Non-Trend')
    plt.xlabel('Is Strong Trend (Label)')
    plt.ylabel('Efficiency Ratio')
    plt.savefig(PROJECT_ROOT / 'xgboost_filter_model' / 'er_boxplot.png')
    print("Saved er_boxplot.png")

    # 3. Boxplot of ATR Threshold during Trend vs. Non-Trend
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='is_strong_trend', y='atr_threshold', data=df)
    plt.title('ATR Threshold Distribution for Trend vs. Non-Trend')
    plt.xlabel('Is Strong Trend (Label)')
    plt.ylabel('ATR Threshold')
    plt.savefig(PROJECT_ROOT / 'xgboost_filter_model' / 'atr_threshold_boxplot.png')
    print("Saved atr_threshold_boxplot.png")

    # Print percentile distribution of atr_threshold
    print("\nATR Threshold Percentile Distribution:")
    print(df['atr_threshold'].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

    # 5. Plot distribution of Efficiency Ratio and Future ER
    plt.figure(figsize=(12, 6))
    sns.histplot(df['efficiency_ratio'], color="blue", label='Efficiency Ratio (Current)', kde=True, stat="density", linewidth=0)
    sns.histplot(df['future_er'], color="orange", label='Future ER (Label Condition)', kde=True, stat="density", linewidth=0)
    plt.title('Distribution of Current vs. Future Efficiency Ratio')
    plt.legend()
    plt.savefig(PROJECT_ROOT / 'xgboost_filter_model' / 'er_distribution.png')
    print("Saved er_distribution.png")


    # 4. Distribution of Strong Trends by Hour (Removed as 'hour' is no longer a feature)
    # plt.figure(figsize=(10, 6))
    # sns.countplot(x='hour', hue='is_strong_trend', data=df)
    # plt.title('Distribution of Strong Trends by Hour of Day (UTC)')
    # plt.savefig('trend_by_hour.png')
    # print("Saved trend_by_hour.png")

def main():
    """Main function to run the training pipeline."""
    df = load_price_data()
    df_featured = prepare_features(df)

    # Check if the dataframe is empty after feature preparation
    if df_featured.empty:
        print("\nError: DataFrame is empty after feature preparation and dropping NaNs.")
        print("This is likely due to a large number of initial NaNs from indicator calculations.")
        print("Please ensure the loaded data range is sufficient for all indicator windows.")
        return

    print("\nFeature and Labeling Complete.")
    print(f"Data shape after processing: {df_featured.shape}")
    print("Label distribution:")
    print(df_featured['is_strong_trend'].value_counts(normalize=True))

    visualize_features(df_featured)

    # --- Train the XGBoost Filter Model ---
    print("\n--- Training XGBoost Filter Model ---")

    # 1. Define Features (X) and Target (y)
    features = [
        'adx', 'adx_slope', 'volatility', 'efficiency_ratio', 'fractal_dimension',
        'volume_price_corr', 'volume_trend', 'volume_osc',
        'day_progress',
        'is_asia', 'asia_progress', 'is_london', 'london_progress', 'is_ny', 'ny_progress',
        'change', 'upper_wick', 'lower_wick',
        'bar_change', 'bar_upper_wick', 'bar_lower_wick'
    ]
    X = df_featured[features]
    y = df_featured['is_strong_trend']

    # 2. Split data into training and testing sets (time-series split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # 3. Initialize and train the XGBoost Classifier
    # 'scale_pos_weight' is important for imbalanced datasets.
    # It's the ratio of negative class to positive class.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Make predictions and evaluate the model
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)

    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred, target_names=['Non-Trend (0)', 'Strong Trend (1)']))

    # Feature Importance
    print("\nFeature Importances:")
    for importance, name in sorted(zip(model.feature_importances_, features), reverse=True):
        print(f"{name}: {importance:.4f}")


if __name__ == "__main__":
    main()
