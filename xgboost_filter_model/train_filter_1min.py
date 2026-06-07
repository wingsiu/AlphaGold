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
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add project root to sys.path to allow importing from other directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import DataLoader

def load_price_data(
    start_date: str = "2020-01-01",
    end_date: str = "2026-05-07",
    *,
    table_name: str | None = None,
) -> pd.DataFrame:
    """Load 1-minute OHLCV from MySQL (default gold_prices; oil uses prices)."""
    import os

    table = table_name or os.environ.get("V14_PRICE_TABLE", "gold_prices")
    print(f"Loading {table} from {start_date} to {end_date}...")
    loader = DataLoader()
    df = loader.load_data(
        table_name=table,
        start_date=start_date,
        end_date=end_date,
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


def prepare_features(df: pd.DataFrame, move_threshold: float, er_threshold: float, future_window: int, for_live_inference: bool = False) -> pd.DataFrame:
    """
    Prepares the feature matrix for the XGBoost filter model using a 1-minute timeframe.
    When for_live_inference=True, skips future-label computation and only drops warm-up NaN rows
    (not the latest bars), so the most recent bar is preserved for live scoring.
    """
    df_1m = df.copy()
    # No print statements here to keep the sweep output clean

    # 2. Basic price changes (Log Returns)
    df_1m['returns'] = np.log(df_1m['close'] / df_1m['close'].shift(1))

    # 3. Trend Strength: ADX and its slope
    adx_indicator = ta.trend.ADXIndicator(df_1m['high'], df_1m['low'], df_1m['close'], window=14)
    df_1m['adx'] = adx_indicator.adx()
    df_1m['adx_slope'] = df_1m['adx'].diff(3)

    # 4. Volatility Indicators: ATR and Realized Volatility
    df_1m['atr'] = ta.volatility.AverageTrueRange(df_1m['high'], df_1m['low'], df_1m['close'], window=14).average_true_range()
    df_1m['volatility'] = df_1m['returns'].rolling(window=30).std()

    # 5. Efficiency Ratio & Fractal Dimension
    # ER for different windows
    for w in [15, 30, 90]:
        net_change = abs(df_1m['close'] - df_1m['close'].shift(w))
        sum_of_changes = abs(df_1m['close'].diff()).rolling(window=w).sum()
        df_1m[f'er_{w}'] = net_change / sum_of_changes.replace(0, np.nan)

    df_1m['fractal_dimension'] = fractal_dimension(df_1m, window=14)

    # Williams %R for different windows
    for w in [15, 30, 90]:
        df_1m[f'wr_{w}'] = pta.willr(df_1m['high'], df_1m['low'], df_1m['close'], length=w)

    # Rolling window change and wicks
    for w in [15, 30, 90]:
        rolling_high = df_1m['high'].rolling(window=w).max()
        rolling_low = df_1m['low'].rolling(window=w).min()
        rolling_open = df_1m['open'].shift(w - 1)

        rolling_range = (rolling_high - rolling_low).replace(0, np.nan)

        df_1m[f'change_{w}'] = (df_1m['close'] - rolling_open) / rolling_range
        df_1m[f'upper_wick_{w}'] = (rolling_high - df_1m[['open', 'close']].max(axis=1)) / rolling_range
        df_1m[f'lower_wick_{w}'] = (df_1m[['open', 'close']].min(axis=1) - rolling_low) / rolling_range


    # New Efficiency Ratio features based on 90-bar window
    window_90 = 90
    rolling_high = df_1m['high'].rolling(window=window_90)
    rolling_low = df_1m['low'].rolling(window=window_90)

    # Find the index of the rolling max and min
    # This gives us the start point for our path calculation
    idx_high = rolling_high.apply(np.argmax, raw=True).fillna(-1).astype(int)
    idx_low = rolling_low.apply(np.argmin, raw=True).fillna(-1).astype(int)

    # Get the actual high/low values
    df_1m['high_90'] = rolling_high.max()
    df_1m['low_90'] = rolling_low.min()

    # Calculate path length from the rolling high/low to the current bar
    # This is a more complex operation that can't be purely vectorized easily.
    # We will iterate for clarity and correctness.
    path_sum_down = np.full(len(df_1m), np.nan)
    path_sum_up = np.full(len(df_1m), np.nan)
    abs_diffs = abs(df_1m['close'].diff()).values

    # Convert to numpy for faster integer-based indexing
    idx_high_np = idx_high.to_numpy()
    idx_low_np = idx_low.to_numpy()

    for i in range(window_90, len(df_1m)):
        # Path for down_efficiency_ratio (from high)
        start_idx_down = i - window_90 + idx_high_np[i]
        if start_idx_down < i:
            path_sum_down[i] = np.sum(abs_diffs[start_idx_down + 1 : i + 1])

        # Path for up_efficiency_ratio (from low)
        start_idx_up = i - window_90 + idx_low_np[i]
        if start_idx_up < i:
            path_sum_up[i] = np.sum(abs_diffs[start_idx_up + 1 : i + 1])

    path_sum_down_safe = pd.Series(path_sum_down, index=df_1m.index).replace(0, np.nan)
    path_sum_up_safe = pd.Series(path_sum_up, index=df_1m.index).replace(0, np.nan)

    net_change_down = df_1m['high_90'] - df_1m['close']
    df_1m['down_efficiency_ratio'] = net_change_down / path_sum_down_safe

    net_change_up = df_1m['close'] - df_1m['low_90']
    df_1m['up_efficiency_ratio'] = net_change_up / path_sum_up_safe


    # 6. Volume Features
    # Volume-Price Correlation
    df_1m['volume_price_corr'] = df_1m['returns'].rolling(window=30).corr(df_1m['volume'].pct_change())
    # Volume Trend (Ratio of moving averages)
    df_1m['volume_trend'] = df_1m['volume'].rolling(window=10).mean() / df_1m['volume'].rolling(window=30).mean()
    # Volume Oscillator
    pvo_df = pta.pvo(df_1m['volume'], fast=12, slow=26) # Standard PVO settings
    df_1m['volume_osc'] = pvo_df.iloc[:, 0] # Use the main PVO line


    # 7. Time Features (with UTC+2 day start and session progress)

    # Define day based on UTC+2 offset
    day_start_offset = pd.Timedelta(hours=2)
    df_1m['day_utc2'] = (df_1m.index + day_start_offset).floor('D')

    # Calculate day-based features
    # VULNERABILITY FIX: Ensure day_open uses ONLY the first known bar without lookahead transformation
    def get_day_open(group):
        group = group.sort_index()
        return pd.Series(group.iloc[0], index=group.index)

    df_1m['day_open'] = df_1m.groupby('day_utc2')['open'].transform(lambda x: x.iloc[0])
    df_1m['day_high'] = df_1m.groupby('day_utc2')['high'].cummax()
    df_1m['day_low'] = df_1m.groupby('day_utc2')['low'].cummin()

    day_range = df_1m['day_high'] - df_1m['day_low']
    day_range_safe = day_range.replace(0, np.nan) # Avoid division by zero

    df_1m['change'] = (df_1m['close'] - df_1m['day_open']) / day_range_safe
    df_1m['upper_wick'] = (df_1m['day_high'] - df_1m[['day_open', 'close']].max(axis=1)) / day_range_safe
    df_1m['lower_wick'] = (df_1m[['day_open', 'close']].min(axis=1) - df_1m['day_low']) / day_range_safe

    # Calculate current bar features
    bar_range = df_1m['high'] - df_1m['low']
    bar_range_safe = bar_range.replace(0, np.nan)
    df_1m['bar_change'] = (df_1m['close'] - df_1m['open']) / bar_range_safe
    df_1m['bar_upper_wick'] = (df_1m['high'] - df_1m[['open', 'close']].max(axis=1)) / bar_range_safe
    df_1m['bar_lower_wick'] = (df_1m[['open', 'close']].min(axis=1) - df_1m['low']) / bar_range_safe

    # Calculate time elapsed since day start
    day_start_time = df_1m.groupby('day_utc2')['day_utc2'].transform('first')
    df_1m['day_progress'] = (df_1m.index - day_start_time).dt.total_seconds() / (24 * 3600)

    # Define sessions with timezones
    sessions = {
        "asia": {"timezone": ZoneInfo("Asia/Hong_Kong"), "start_hour": 8, "start_minute": 0, "end_hour": 16, "end_minute": 0},
        "london": {"timezone": ZoneInfo("Europe/London"), "start_hour": 8, "start_minute": 0, "end_hour": 16, "end_minute": 30},
        "ny": {"timezone": ZoneInfo("America/New_York"), "start_hour": 9, "start_minute": 30, "end_hour": 16, "end_minute": 0},
    }

    # Calculate session features
    session_features = pd.DataFrame(index=df_1m.index)
    for name, spec in sessions.items():
        flags_progress = df_1m.index.to_series().apply(lambda ts: _session_flag_and_progress(ts, spec))
        session_features[f'is_{name}'] = [fp[0] for fp in flags_progress]
        session_features[f'{name}_progress'] = [fp[1] for fp in flags_progress]

    df_1m = df_1m.join(session_features)

    # No lookahead — needed for live/backtest inference when models were trained with this column.
    if "atr" in df_1m.columns:
        df_1m["atr_threshold"] = df_1m["atr"] * 1.5

    if not for_live_inference:
        # 8. Define Label (Target) based on the next N bars (future_window)
        # future_window = 90 is now a parameter

        # Calculate future values first
        future_highs = df_1m['high'].shift(-future_window).rolling(window=future_window).max()
        future_lows = df_1m['low'].shift(-future_window).rolling(window=future_window).min()
        future_er = df_1m['er_30'].shift(-future_window).rolling(window=future_window).max()

        df_1m['future_max_move'] = future_highs - df_1m['close']
        df_1m['future_min_move'] = future_lows - df_1m['close']
        df_1m['future_er'] = future_er # Keep future_er for analysis

        # Define the label before dropping NaNs from feature calculation
        is_efficient_future = future_er > er_threshold
        is_up_move = df_1m['future_max_move'] > move_threshold
        is_down_move = df_1m['future_min_move'].abs() > move_threshold

        # Create mutually exclusive trend labels
        df_1m['trend_label'] = 0 # 0 = No Trend
        df_1m.loc[is_up_move & is_efficient_future, 'trend_label'] = 1 # 1 = Up Trend
        df_1m.loc[is_down_move & is_efficient_future, 'trend_label'] = -1 # -1 = Down Trend

        # Handle cases where both up and down moves qualify by prioritizing the larger move
        both_moves = is_up_move & is_down_move & is_efficient_future
        df_1m.loc[both_moves & (df_1m['future_max_move'] >= df_1m['future_min_move'].abs()), 'trend_label'] = 1
        df_1m.loc[both_moves & (df_1m['future_max_move'] < df_1m['future_min_move'].abs()), 'trend_label'] = -1

        print("Features and labels prepared.")
        # Drop all rows with any NaNs, which will include the rows at the end
        # where future values could not be calculated.
        return df_1m.dropna()
    else:
        # Live inference mode: skip future labels, only drop warm-up NaN rows at the START
        # (due to rolling windows), preserving the most recent bars.
        print("Features prepared (live inference mode — no future labels).")
        live_feature_cols = [
            'returns', 'adx', 'adx_slope', 'volatility', 'er_15', 'er_30', 'er_90',
            'fractal_dimension', 'wr_15', 'wr_30', 'wr_90',
            'change_15', 'upper_wick_15', 'lower_wick_15',
            'change_30', 'upper_wick_30', 'lower_wick_30',
            'change_90', 'upper_wick_90', 'lower_wick_90',
            'down_efficiency_ratio', 'up_efficiency_ratio',
            'volume_price_corr', 'volume_trend', 'volume_osc',
            'change', 'upper_wick', 'lower_wick',
            'bar_change', 'bar_upper_wick', 'bar_lower_wick',
            'day_progress', 'is_asia', 'asia_progress', 'is_london',
            'london_progress', 'is_ny', 'ny_progress',
        ]
        return df_1m.dropna(subset=live_feature_cols)

def visualize_features(df: pd.DataFrame):
    """Prints percentile distributions for key features."""
    print("Generating feature analysis...")

    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    print("\n--- Standard Efficiency Ratio Distribution by Trend Direction ---")
    print(df.groupby('trend_label')['er_30'].describe(percentiles=percentiles))

    print("\n--- Down Efficiency Ratio (from rolling high) Distribution by Trend ---")
    print(df.groupby('trend_label')['down_efficiency_ratio'].describe(percentiles=percentiles))

    print("\n--- Up Efficiency Ratio (from rolling low) Distribution by Trend ---")
    print(df.groupby('trend_label')['up_efficiency_ratio'].describe(percentiles=percentiles))


def main():
    """Main function to run the training pipeline."""
    df = load_price_data()

    # Reverting to a single run with the best parameters found
    best_move = 15
    best_er = 0.25
    best_window = 120
    print(f"\n--- Running with best parameters: Move > ${best_move}, ER > {best_er}, Window = {best_window} ---")

    df_featured = prepare_features(df, move_threshold=best_move, er_threshold=best_er, future_window=best_window)

    if df_featured.empty:
        print("\nError: DataFrame is empty after feature preparation.")
        return

    print("\nFeature and Labeling Complete.")
    print(f"Data shape after processing: {df_featured.shape}")
    print("Label distribution:")
    print(df_featured['trend_label'].value_counts(normalize=True))

    # --- Train the XGBoost Filter Model ---
    print("\n--- Training XGBoost Filter Model ---")

    # 1. Define Features (X) and Target (y)
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
    y = (df_featured['trend_label'] != 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    if (y_train == 1).sum() == 0:
        print("No positive samples in training set, cannot train.")
        return

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )

    print("\nTraining model...")
    model.fit(X_train, y_train, verbose=False)
    print("Model training complete.")

    # Save the trained filter model
    model_path = PROJECT_ROOT / "xgboost_filter_model" / "filter_model.joblib"
    joblib.dump(model, model_path)
    print(f"Filter model saved to {model_path}")

    # 4. Make predictions and evaluate the model
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)

    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred, target_names=['Non-Trend (0)', 'Trend (1)']))

    # Feature Importance
    print("\nFeature Importances:")
    for importance, name in sorted(zip(model.feature_importances_, features), reverse=True):
        print(f"{name}: {importance:.4f}")


if __name__ == "__main__":
    main()

