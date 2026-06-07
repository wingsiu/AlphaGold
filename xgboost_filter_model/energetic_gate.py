"""Energetic bar filter + S1 gate for pattern router (stack on pattern specialists)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Same exclusions as backtest_v14.py (S1 feature schema — no pattern / PA cols).
S1_EXCLUDE_COLS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "timestamp",
    "trend_label",
    "target_v10",
    "target_v14",
    "is_trend",
    "atr",
    "day_utc2",
    "future_max_move",
    "future_min_move",
    "future_er",
    "atr_threshold",
    "bar_move",
    "hour",
    "day_id",
    "day_high",
    "day_low",
    "high_90",
    "low_90",
    "closePrice_ask",
    "closePrice_bid",
    "highPrice_ask",
    "lowPrice_bid",
    "closePrice",
    "lowPrice",
    "open_price",
    "highPrice",
    "highPrice_bid",
    "lowPrice_ask",
    "openPrice_bid",
    "openPrice_ask",
    "day_open",
    "day_high_rolling",
    "day_low_rolling",
    "hmm_regime",
    "high_60m",
    "low_60m",
    "low_15m",
    "high_15m",
    "ma_60m",
    "daily_poc",
    "daily_vwap",
    "rolling_poc_4h",
    "dynamic_tp",
    "dynamic_sl",
    "fvg_bull_bottom",
    "fvg_bull_top",
    "fvg_bear_top",
    "fvg_bear_bottom",
}

S2_EXTRA_COLS = {
    "directional_change_15",
    "directional_change_30",
    "directional_change_90",
    "wick_ratio_15",
    "wick_ratio_30",
    "wick_ratio_90",
    "price_vs_ma_10",
    "price_vs_ma_30",
    "price_vs_ma_90",
    "ma_10_vs_30",
    "ma_30_vs_90",
    "rsi_14",
    "rsi_30",
    "macd",
    "macd_signal",
    "macd_diff",
    "roc_15",
    "roc_30",
    "roc_60",
}

PATTERN_ROUTER_COLS = {
    "pattern_name",
    "pattern_id",
    "matched_pattern",
    "target_pattern",
    "prob",
    "s1_prob",
    "s2_prob",
    "side_signal",
    "exec_tp",
    "exec_sl",
    "exec_horizon",
    "cycle_id",
    "model_path",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def pattern_gate_config() -> dict:
    from config.v14_config import EXECUTION_CONFIG, PATTERN_GATE_CONFIG

    cfg = dict(PATTERN_GATE_CONFIG)
    cfg["energetic_filter"] = _env_bool(
        "V14_PATTERN_ENERGETIC_GATE",
        bool(cfg.get("energetic_filter", False)),
    )
    cfg["s1_gate"] = _env_bool(
        "V14_PATTERN_S1_GATE",
        bool(cfg.get("s1_gate", False)),
    )
    thresh = cfg.get("s1_threshold")
    cfg["s1_threshold"] = (
        float(thresh) if thresh is not None else float(EXECUTION_CONFIG["s1_threshold"])
    )
    return cfg


def energetic_bar_mask(df: pd.DataFrame) -> pd.Series:
    """bar_move + volume + HMM trend-regime gate (same as v14 S1 training path)."""
    from config.v14_config import FILTER_CONFIG
    from xgboost_filter_model.hmm_regime import get_hmm_model_path

    if "bar_move" not in df.columns:
        bar_move = (df["close"] - df["open"]).abs()
    else:
        bar_move = df["bar_move"]

    model_data = joblib.load(get_hmm_model_path())
    trend_regimes = model_data["trend_regimes"]
    return (
        (bar_move > FILTER_CONFIG["min_bar_move"])
        & (df["volume"] > FILTER_CONFIG["min_volume"])
        & (df["hmm_regime"].isin(trend_regimes))
    )


@lru_cache(maxsize=1)
def get_s1_feature_names() -> tuple[str, ...]:
    """Canonical S1 columns from energetic prep path (matches filter_model_v14_wf)."""
    from xgboost_filter_model.train_filter_v14 import prepare_data_v14
    from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14

    df = prepare_data_v14(start_date="2025-06-01", end_date="2025-06-08")
    df = prepare_directional_data_v14(df)
    features = [c for c in df.columns if c not in S1_EXCLUDE_COLS]
    return tuple(f for f in features if f not in S2_EXTRA_COLS)


def s1_feature_columns(df: pd.DataFrame) -> list[str]:
    """S1 model inputs present on df (pattern matrix may have extra cols — ignore them)."""
    return [c for c in get_s1_feature_names() if c in df.columns]


def score_s1_probabilities(
    df_test: pd.DataFrame,
    s1_features: list[str] | None,
    bt_start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> None:
    """Walk-forward S1 scoring in-place on df_test['s1_prob']."""
    from config.v14_config import WF_CONFIG

    wf_dir = PROJECT_ROOT / os.environ.get(
        "V14_MODEL_OUTPUT_DIR",
        WF_CONFIG.get("model_output_dir", "runtime/bot_assets/wf_models_v14"),
    )
    retrain_days = int(WF_CONFIG.get("retrain_days", 14))
    wf_start = pd.to_datetime(WF_CONFIG["wf_start"])
    if wf_start.tzinfo is None:
        wf_start = wf_start.tz_localize("UTC")
    else:
        wf_start = wf_start.tz_convert("UTC")

    prod_s1 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v14_wf.joblib")
    if s1_features is None:
        s1_features = s1_feature_columns(df_test)
        if not s1_features:
            raise ValueError("No S1 feature columns found on pattern matrix")

    elapsed_days = max(0, (bt_start_dt - wf_start).days)
    skip_cycles = elapsed_days // retrain_days
    cycle = 1 + skip_cycles
    current_start = wf_start + pd.Timedelta(days=skip_cycles * retrain_days)

    df_test["s1_prob"] = np.nan
    while current_start < end_dt:
        current_end = min(current_start + pd.Timedelta(days=retrain_days), end_dt)
        s1_path = wf_dir / f"filter_v14_cycle_{cycle}_{current_start.date()}.joblib"
        chunk = (df_test.index >= current_start) & (df_test.index < current_end)
        if chunk.any():
            s1 = joblib.load(s1_path) if s1_path.exists() else prod_s1
            df_test.loc[chunk, "s1_prob"] = s1.predict_proba(df_test.loc[chunk, s1_features])[:, 1]
        current_start = current_end
        cycle += 1


def hybrid_config() -> dict:
    from config.v14_config import ENERGETIC_EXECUTION_CONFIG, HYBRID_CONFIG

    cfg = dict(HYBRID_CONFIG)
    cfg["enabled"] = _env_bool("V14_HYBRID", bool(cfg.get("enabled", False)))
    cfg["s1_threshold"] = float(ENERGETIC_EXECUTION_CONFIG["s1_threshold"])
    cfg["s2_threshold"] = float(ENERGETIC_EXECUTION_CONFIG["s2_threshold"])
    return cfg


@lru_cache(maxsize=1)
def get_s2_feature_names() -> tuple[str, ...]:
    from xgboost_filter_model.train_filter_v14 import prepare_data_v14
    from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14

    df = prepare_data_v14(start_date="2025-06-01", end_date="2025-06-08")
    df = prepare_directional_data_v14(df)
    return tuple(c for c in df.columns if c not in S1_EXCLUDE_COLS)


def s2_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in get_s2_feature_names() if c in df.columns]


def score_energetic_signals(
    df_test: pd.DataFrame,
    bt_start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> None:
    """Score S1/S2 on energetic bars; set energetic_s1_prob, energetic_s2_prob, energetic_side."""
    from config.v14_config import ENERGETIC_EXECUTION_CONFIG, WF_CONFIG
    from xgboost_filter_model.pattern_training import fixed_wf_cycle_from_env

    wf_dir = PROJECT_ROOT / os.environ.get(
        "V14_MODEL_OUTPUT_DIR",
        WF_CONFIG.get("model_output_dir", "runtime/bot_assets/wf_models_v14"),
    )
    retrain_days = int(WF_CONFIG.get("retrain_days", 14))
    wf_start = pd.to_datetime(WF_CONFIG["wf_start"])
    if wf_start.tzinfo is None:
        wf_start = wf_start.tz_localize("UTC")
    else:
        wf_start = wf_start.tz_convert("UTC")

    prod_s1 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v14_wf.joblib")
    prod_s2 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v14_wf.joblib")
    s1_feats = s1_feature_columns(df_test)
    s2_feats = s2_feature_columns(df_test)

    df_test["energetic_s1_prob"] = np.nan
    df_test["energetic_s2_prob"] = np.nan
    df_test["energetic_side"] = 0

    fixed_cycle = fixed_wf_cycle_from_env()
    if fixed_cycle:
        cycle, pin_start = fixed_cycle
        print(f"  Energetic models pinned: cycle_{cycle} ({pin_start.date()})")
        cycle_windows = [(cycle, bt_start_dt, end_dt, pin_start.date())]
    else:
        elapsed_days = max(0, (bt_start_dt - wf_start).days)
        skip_cycles = elapsed_days // retrain_days
        cycle = 1 + skip_cycles
        current_start = wf_start + pd.Timedelta(days=skip_cycles * retrain_days)
        cycle_windows = []
        while current_start < end_dt:
            current_end = min(current_start + pd.Timedelta(days=retrain_days), end_dt)
            cycle_windows.append((cycle, current_start, current_end, current_start.date()))
            current_start = current_end
            cycle += 1

    for cycle, win_start, win_end, model_date in cycle_windows:
        chunk = (df_test.index >= win_start) & (df_test.index < win_end)
        if chunk.any():
            s1_path = wf_dir / f"filter_v14_cycle_{cycle}_{model_date}.joblib"
            s2_path = wf_dir / f"directional_v14_cycle_{cycle}_{model_date}.joblib"
            s1 = joblib.load(s1_path) if s1_path.exists() else prod_s1
            s2 = joblib.load(s2_path) if s2_path.exists() else prod_s2
            df_test.loc[chunk, "energetic_s1_prob"] = s1.predict_proba(df_test.loc[chunk, s1_feats])[:, 1]
            s1_pass = chunk & (df_test["energetic_s1_prob"] >= ENERGETIC_EXECUTION_CONFIG["s1_threshold"])
            if s1_pass.any():
                df_test.loc[s1_pass, "energetic_s2_prob"] = s2.predict_proba(df_test.loc[s1_pass, s2_feats])[:, 1]

    energetic = energetic_bar_mask(df_test)

    # Volatility-adaptive S1/S2 thresholds
    from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold

    s1_base = float(ENERGETIC_EXECUTION_CONFIG["s1_threshold"])
    s2_base = float(ENERGETIC_EXECUTION_CONFIG["s2_threshold"])
    if os.environ.get("V14_ADAPTIVE_ENERGETIC", "0") not in ("0", "no", "false"):
        s1_adaptive = adaptive_prob_threshold(s1_base, df_test)
        s2_adaptive = adaptive_prob_threshold(s2_base, df_test)
    else:
        s1_adaptive = pd.Series(s1_base, index=df_test.index)
        s2_adaptive = pd.Series(s2_base, index=df_test.index)

    trend = energetic & (df_test["energetic_s1_prob"] >= s1_adaptive)
    long_sig = trend & (df_test["energetic_s2_prob"] >= s2_adaptive)
    short_sig = trend & (df_test["energetic_s2_prob"] <= (1.0 - s2_adaptive))
    df_test.loc[long_sig, "energetic_side"] = 1
    df_test.loc[short_sig, "energetic_side"] = -1

    print(
        f"  Energetic fallback: {int(energetic.sum())} energetic bars, "
        f"{int(trend.sum())} S1 pass, "
        f"{int((df_test['energetic_side'] != 0).sum())} entry signals"
    )


def apply_pattern_gates(df_test: pd.DataFrame, bt_start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.Series:
    """Return boolean mask of bars allowed for pattern model scoring."""
    gate = pattern_gate_config()
    mask = pd.Series(True, index=df_test.index)

    if gate["energetic_filter"]:
        from config.v14_config import FILTER_CONFIG

        energetic = energetic_bar_mask(df_test)
        print(
            f"  Energetic gate: {int(energetic.sum())} / {len(df_test)} bars "
            f"(move>{FILTER_CONFIG['min_bar_move']}, vol>{FILTER_CONFIG['min_volume']}, HMM trend)"
        )
        mask &= energetic

    if gate["s1_gate"]:
        score_s1_probabilities(df_test, None, bt_start_dt, end_dt)
        s1_pass = df_test["s1_prob"] >= gate["s1_threshold"]
        print(
            f"  S1 gate (≥{gate['s1_threshold']:.2f}): "
            f"{int(s1_pass.sum())} / {len(df_test)} bars"
        )
        mask &= s1_pass

    if gate["energetic_filter"] or gate["s1_gate"]:
        print(f"  Combined gate pass: {int(mask.sum())} / {len(df_test)} bars")
    return mask
