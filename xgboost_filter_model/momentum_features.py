"""Momentum features for Stage 2 (RSI, MACD, ROC). Stage 1 does not use these."""
import pandas as pd
import ta

ROC_WINDOWS = (15, 30, 60)
RSI_WINDOWS = (14, 30)


def add_roc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rate of change on close (percent change over window)."""
    df = df.copy()
    for w in ROC_WINDOWS:
        df[f"roc_{w}"] = ta.momentum.ROCIndicator(df["close"], window=w).roc()
    return df


def add_acceleration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Acceleration = bar-over-bar change in ROC (second-order momentum). Requires roc_*."""
    df = df.copy()
    for w in ROC_WINDOWS:
        col = f"roc_{w}"
        if col not in df.columns:
            df = add_roc_features(df)
            break
    for w in ROC_WINDOWS:
        df[f"accel_{w}"] = df[f"roc_{w}"].diff()
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full momentum set for Stage 2: RSI, MACD, ROC."""
    df = df.copy()
    for w in RSI_WINDOWS:
        df[f"rsi_{w}"] = ta.momentum.RSIIndicator(df["close"], window=w).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    df = add_roc_features(df)
    return df
