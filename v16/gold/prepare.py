"""v16 gold hybrid data prep + energetic S1/S2 scoring (no v15/ imports)."""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from config.hybrid_config import ENERGETIC_EXECUTION_CONFIG, WF_CONFIG
from v16._paths import PROJECT_ROOT
from v16.gold.energetic import energetic_bar_mask
from v16.gold.features import add_energetic_features
from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold
from xgboost_filter_model.energetic_gate import s1_feature_columns, s2_feature_columns
from xgboost_filter_model.pattern_training import fixed_wf_cycle_from_env
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14


def prepare_gold_hybrid_data(
    start_date: str,
    end_date: str,
    *,
    pa_groups=None,
    pattern_feature_set=None,
) -> pd.DataFrame:
    df = prepare_data_v14(
        start_date=start_date,
        end_date=end_date,
        energetic_filter=False,
        for_live_inference=True,
        pa_groups=pa_groups,
        pattern_feature_set=pattern_feature_set,
    )
    df = prepare_directional_data_v14(df)
    return add_energetic_features(df)


def score_energetic_signals(
    df_test: pd.DataFrame,
    bt_start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> None:
    """Score S1/S2 on energetic bars; sets energetic_s1_prob, energetic_s2_prob, energetic_side."""
    fixed_cycle = fixed_wf_cycle_from_env()
    wf_dir = PROJECT_ROOT / os.environ.get(
        "V14_MODEL_OUTPUT_DIR",
        WF_CONFIG.get("model_output_dir", "runtime/bot_assets/wf_models_v14"),
    )
    retrain_days = int(WF_CONFIG.get("retrain_days", 14))
    wf_start = pd.to_datetime(WF_CONFIG["wf_start"])
    wf_start = wf_start.tz_localize("UTC") if wf_start.tzinfo is None else wf_start.tz_convert("UTC")

    prod_s1 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v14_wf.joblib")
    prod_s2 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v14_wf.joblib")
    s1_feats = s1_feature_columns(df_test)
    s2_feats = s2_feature_columns(df_test)

    df_test["energetic_s1_prob"] = np.nan
    df_test["energetic_s2_prob"] = np.nan
    df_test["energetic_side"] = 0

    if fixed_cycle:
        cycle, pin_start = fixed_cycle
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
        if not chunk.any():
            continue
        s1_path = wf_dir / f"filter_v14_cycle_{cycle}_{model_date}.joblib"
        s2_path = wf_dir / f"directional_v14_cycle_{cycle}_{model_date}.joblib"
        s1 = joblib.load(s1_path) if s1_path.exists() else prod_s1
        s2 = joblib.load(s2_path) if s2_path.exists() else prod_s2
        df_test.loc[chunk, "energetic_s1_prob"] = s1.predict_proba(df_test.loc[chunk, s1_feats])[:, 1]
        s1_pass = chunk & (df_test["energetic_s1_prob"] >= ENERGETIC_EXECUTION_CONFIG["s1_threshold"])
        if s1_pass.any():
            df_test.loc[s1_pass, "energetic_s2_prob"] = s2.predict_proba(df_test.loc[s1_pass, s2_feats])[:, 1]

    energetic = energetic_bar_mask(df_test)
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
