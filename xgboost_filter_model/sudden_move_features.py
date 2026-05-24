"""3-minute sudden rise/drop features (point move vs thresholds a, b)."""
import os
import pandas as pd
import numpy as np

FILL_NA = 9999


def _thresholds_from_env() -> tuple[float, float] | None:
    rise_a = os.environ.get("V14_SUDDEN_RISE_A", "").strip()
    drop_b = os.environ.get("V14_SUDDEN_DROP_B", "").strip()
    if not rise_a or not drop_b:
        return None
    return float(rise_a), float(drop_b)


def add_sudden_move_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    If V14_SUDDEN_RISE_A and V14_SUDDEN_DROP_B are set (points):
      sudden_move_3m      — 1 if ret_3m > a or ret_3m < -b
      time_from_sudden_move — minutes since last sudden_move_3m bar (0 on event bar)
    Requires ret_3m (from add_v14_window_features).
    """
    th = _thresholds_from_env()
    if th is None:
        return df

    a, b = th
    df = df.copy()
    if "ret_3m" not in df.columns:
        df["ret_3m"] = df["close"] - df["close"].shift(3)

    df["sudden_move_3m"] = ((df["ret_3m"] > a) | (df["ret_3m"] < -b)).astype(int)
    event_times = pd.Series(
        np.where(df["sudden_move_3m"] == 1, df.index, np.nan),
        index=df.index,
    ).ffill()
    df["time_from_sudden_move"] = (
        (df.index - event_times).dt.total_seconds() / 60.0
    )
    df["time_from_sudden_move"] = df["time_from_sudden_move"].fillna(FILL_NA)
    return df
