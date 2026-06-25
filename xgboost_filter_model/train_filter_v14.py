import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xgboost_filter_model.train_filter_1min import load_price_data, prepare_features as prepare_base_features
from xgboost_filter_model.train_filter_v10 import add_liquidity_indicators
from xgboost_filter_model.hmm_regime import add_hmm_regime, get_hmm_model_path
from xgboost_filter_model.volume_profile import add_volume_profile_features
from xgboost_filter_model.time_features import add_time_features
from xgboost_filter_model.sudden_move_features import add_sudden_move_features
from xgboost_filter_model.candle_pattern_15m import add_candle_pattern_15m
from xgboost_filter_model.pattern_features import add_pattern_features

def build_target(df: pd.DataFrame, horizon: int, tp: float, sl: float) -> pd.DataFrame:
    """
    Builds the target for the v14 model.
    """
    df = df.copy()
    
    future_high = df['high'].shift(-horizon).rolling(window=horizon, min_periods=1).max()
    future_low = df['low'].shift(-horizon).rolling(window=horizon, min_periods=1).min()
    future_close = df['close'].shift(-horizon)
    
    df['future_max_move'] = future_high - df['close']
    df['future_min_move'] = df['close'] - future_low
    
    # Efficiency Ratio of the future move
    net_move = (future_close - df['close']).abs()
    path_length = df['close'].diff().abs().shift(-horizon).rolling(window=horizon, min_periods=1).sum()
    df['future_er'] = net_move / (path_length + 1e-9)
    
    return df
from config.hybrid_config import FILTER_CONFIG, TARGET_CONFIG, MODEL_CONFIG, WF_CONFIG
from xgboost_filter_model.pattern_training import (
    iter_wf_train_targets,
    wf_train_mode,
    wf_train_as_of,
)

NY_TZ = ZoneInfo("America/New_York")
HK_TZ = ZoneInfo("Asia/Hong_Kong")
LONDON_TZ = ZoneInfo("Europe/London")

def _session_info(ts, timezone, start_h, start_m, end_h, end_m):
    local_ts = ts.tz_convert(timezone)
    minute_of_day = local_ts.hour * 60 + local_ts.minute
    s = start_h * 60 + start_m
    e = end_h * 60 + end_m
    if s <= minute_of_day < e:
        return 1.0, (minute_of_day - s) / (e - s)
    return 0.0, 0.0

def add_v14_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds daily open, range, and distance features strictly without lookahead."""
    df = df.copy()
    
    # Session Progress
    def _get_sessions(ts):
        asia_f, asia_p = _session_info(ts, HK_TZ, 8, 0, 16, 0)
        lon_f, lon_p = _session_info(ts, LONDON_TZ, 8, 0, 16, 30)
        ny_f, ny_p = _session_info(ts, NY_TZ, 9, 30, 16, 0)
        return pd.Series([asia_f, asia_p, lon_f, lon_p, ny_f, ny_p])

    sessions_df = df.index.to_series().apply(_get_sessions)
    sessions_df.columns = ['is_asia', 'asia_progress', 'is_london', 'london_progress', 'is_ny', 'ny_progress']
    for c in sessions_df.columns:
        df[c] = sessions_df[c]

    # Day rolling features (UTC+2 to align with NY 17:00 cutoff)
    day_start_offset = pd.Timedelta(hours=2)
    df["day_utc2"] = (df.index + day_start_offset).floor("D")
    
    # Calculate daily open
    df["day_open"] = df.groupby("day_utc2")["open"].transform("first")
    
    # Calculate rolling high/low up to THIS bar
    df["day_high_rolling"] = df.groupby("day_utc2")["high"].cummax()
    df["day_low_rolling"] = df.groupby("day_utc2")["low"].cummin()
    
    # Distance from open (relative)
    df["dist_from_open_rel"] = (df["close"] - df["day_open"]) / (df["day_open"] + 1e-9)
    
    # Distance from rolling high/low (relative)
    df["dist_from_high_rel"] = (df["day_high_rolling"] - df["close"]) / (df["day_open"] + 1e-9)
    df["dist_from_low_rel"] = (df["close"] - df["day_low_rolling"]) / (df["day_open"] + 1e-9)
    
    # Daily range so far (relative)
    df["day_range_so_far_rel"] = (df["day_high_rolling"] - df["day_low_rolling"]) / (df["day_open"] + 1e-9)

    return df

def add_v14_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds specific 3m, 15m, and 60m window features."""
    df = df.copy()
    
    # 3-Minute Window (The Trigger)
    df['ret_3m'] = df['close'] - df['close'].shift(3)
    df['vol_3m'] = df['volume'].rolling(3).sum()
    path_3m = df['close'].diff().abs().rolling(3).sum()
    df['er_3m'] = df['ret_3m'].abs() / (path_3m + 1e-9)
    
    # 15-Minute Window (Local Structure)
    df['ret_15m'] = df['close'] - df['close'].shift(15)
    df['high_15m'] = df['high'].rolling(15).max()
    df['low_15m'] = df['low'].rolling(15).min()
    df['range_15m'] = df['high_15m'] - df['low_15m']
    
    # 60-Minute Window (Trend Context)
    df['ret_60m'] = df['close'] - df['close'].shift(60)
    df['ma_60m'] = df['close'].rolling(60).mean()
    df['dist_ma_60m'] = df['close'] - df['ma_60m']
    df['high_60m'] = df['high'].rolling(60).max()
    df['low_60m'] = df['low'].rolling(60).min()
    df['range_60m'] = df['high_60m'] - df['low_60m']
    df['breakout_up_60m'] = (df['close'] >= df['high_60m'].shift(1)).astype(int)
    df['breakout_down_60m'] = (df['close'] <= df['low_60m'].shift(1)).astype(int)
    
    return df

from xgboost_filter_model.price_action_features import add_price_action_features

def prepare_data_v14(
    start_date: str = "2020-01-01",
    end_date: str = None,
    *,
    energetic_filter: bool = True,
    for_live_inference: bool = False,
    label_horizon: int | None = None,
    label_tp: float | None = None,
    label_sl: float | None = None,
    fixed_label_tp_sl: bool = False,
    pa_groups: list[str] | tuple[str, ...] | str | None = None,
    pattern_feature_set: str | None = None,
    price_table: str | None = None,
) -> pd.DataFrame:
    """Prepares the dataset for v14 S1 Filter Model.

    energetic_filter=False skips HMM / bar_move / volume gate (pattern specialist path).
    label_horizon/tp/sl override TARGET_CONFIG for build_target (pattern retrain).
    fixed_label_tp_sl=True disables ATR-scaled dynamic_tp/sl (train label = exec params).
    pa_groups: optional price-action feature groups for pattern models.
    pattern_feature_set: "current" or "v2398" (see pattern_features.add_pattern_features).
    for_live_inference: skip future labels and preserve latest bars (hybrid backtest + live).

    Pattern specialists must label with each pattern's execution H/TP/SL via
    label_df_for_pattern() — not the global TARGET_CONFIG / EXECUTION_CONFIG defaults.
    """
    lbl_h = label_horizon if label_horizon is not None else TARGET_CONFIG["horizon"]
    lbl_tp = label_tp if label_tp is not None else TARGET_CONFIG["tp"]
    lbl_sl = label_sl if label_sl is not None else TARGET_CONFIG["sl"]
    df = load_price_data(start_date=start_date, end_date=end_date, table_name=price_table)
    
    print("Features and labels prepared.")
    df = prepare_base_features(
        df,
        move_threshold=TARGET_CONFIG["move_threshold"],
        er_threshold=TARGET_CONFIG["er_threshold"],
        future_window=TARGET_CONFIG["horizon"],
        for_live_inference=for_live_inference,
    )
    
    print("Adding liquidity zone indicators...")
    df = add_liquidity_indicators(df)
    
    print("Applying HMM Regime Filter...")
    df = add_hmm_regime(df)
    
    print("Adding v14 daily range and session features...")
    df = add_v14_daily_features(df)
    
    print("Adding v14 multi-window features (3m, 15m, 60m)...")
    df = add_v14_window_features(df)

    if os.environ.get("V14_SUDDEN_RISE_A", "").strip() and os.environ.get(
        "V14_SUDDEN_DROP_B", ""
    ).strip():
        a = os.environ["V14_SUDDEN_RISE_A"]
        b = os.environ["V14_SUDDEN_DROP_B"]
        print(f"Adding sudden 3m move features (rise>{a}, drop<-{b})...")
        df = add_sudden_move_features(df)
    
    print("Adding Volume Profile features (VWAP, POC)...")
    df = add_volume_profile_features(df)
    
    print("Adding Time features (time_from_15m, time_from_max/min)...")
    df = add_time_features(df)

    if os.environ.get("V14_CANDLE_15M", "").strip().lower() in ("1", "true", "yes", "on"):
        print("Adding 15m candle shape / pattern features...")
        df = add_candle_pattern_15m(df)
    
    if pa_groups is not None:
        groups_arg: list[str] | str
        if isinstance(pa_groups, str):
            groups_arg = pa_groups
            pa_label = pa_groups.strip().lower()
        elif pa_groups:
            groups_arg = list(pa_groups)
            pa_label = ",".join(groups_arg)
        else:
            groups_arg = []
            pa_label = ""
    else:
        groups_arg = None
        pa_label = os.environ.get("V14_PA_GROUP", "").strip().lower()
        if not pa_label and os.environ.get("V14_USE_PRICE_ACTION", "0") != "0":
            pa_label = "all"
    if pa_label:
        min_gap = os.environ.get("V14_FVG_MIN_GAP", "0")
        extra = f", FVG min_gap={min_gap}" if "fvg" in pa_label or pa_label == "all" else ""
        print(f"Adding 15m Price Action features (group={pa_label}{extra})...")
        df = add_price_action_features(df, groups=groups_arg if pa_groups is not None else None)
    else:
        print("Skipping 15m Price Action features")
    
    if not for_live_inference:
        print(f"Redefining target with Horizon={lbl_h}, TP={lbl_tp}, SL={lbl_sl}...")
        df = build_target(
            df,
            horizon=lbl_h,
            tp=lbl_tp,
            sl=lbl_sl,
        )

        if fixed_label_tp_sl:
            df["dynamic_tp"] = lbl_tp
            df["dynamic_sl"] = lbl_sl
        elif "atr_threshold" in df.columns:
            df["dynamic_tp"] = np.where(df["atr"] > df["atr_threshold"], lbl_tp * 1.5, lbl_tp)
            df["dynamic_sl"] = np.where(df["atr"] > df["atr_threshold"], lbl_sl * 1.5, lbl_sl)
        else:
            df["dynamic_tp"] = lbl_tp
            df["dynamic_sl"] = lbl_sl

        df["trend_label"] = (
            (df["future_max_move"] >= df["dynamic_tp"])
            & (df["future_min_move"].abs() <= df["dynamic_sl"])
        ).astype(int) | (
            (df["future_min_move"].abs() >= df["dynamic_tp"])
            & (df["future_max_move"] <= df["dynamic_sl"])
        ).astype(int)

        df["bar_move"] = (df["close"] - df["open"]).abs()
    else:
        df["bar_move"] = (df["close"] - df["open"]).abs()

    if not energetic_filter:
        from config.pattern_registry import pattern_feature_set as _resolve_pfs

        pfs = pattern_feature_set if pattern_feature_set is not None else _resolve_pfs()
        print(f"Adding pattern features (set={pfs})…")
        df = add_pattern_features(df, feature_set=pfs)
        if for_live_inference:
            from xgboost_filter_model.energetic_gate import s1_feature_columns

            core = [c for c in s1_feature_columns(df) if c in df.columns]
            if core:
                ok = df[core].notna().all(axis=1)
                if ok.any():
                    df = df.iloc[int(ok.argmax()) :].copy()
        else:
            df.dropna(inplace=True)
        print(f"Pattern path: {len(df)} bars (no HMM/bar_move/volume filter)")
        return df

    from xgboost_filter_model.energetic_gate import energetic_bar_mask

    mask = energetic_bar_mask(df)
    df_filtered = df.loc[mask].copy()
    df_filtered.dropna(inplace=True)
    return df_filtered

def train_walk_forward_v14():
    """Trains the v14 Filter Model using walk-forward validation."""
    print("=== Training AlphaGold v14 Filter Model ===")
    
    df = prepare_data_v14(
        start_date=WF_CONFIG["full_start"], 
        end_date=WF_CONFIG["wf_end"]
    )
    
    # Exclude non-feature columns
    exclude_cols = {
        'open', 'high', 'low', 'close', 'volume', 'timestamp',
        'trend_label', 'target_v10', 'target_v14', 'is_trend', 'atr', 'day_utc2',
        'future_max_move', 'future_min_move', 'future_er', 'atr_threshold',
        'bar_move', 'hour', 'day_id', 'day_high', 'day_low', 'high_90', 'low_90',
        'day_open', 'day_high_rolling', 'day_low_rolling',
        'openPrice_ask', 'openPrice_bid', 'closePrice_ask', 'closePrice_bid', 
        'highPrice_ask', 'highPrice_bid', 'lowPrice_ask', 'lowPrice_bid',
        'closePrice', 'lowPrice', 'open_price', 'highPrice', 'openPrice',
        'ma_60m', 'high_60m', 'low_60m', 'high_15m', 'low_15m', 'hmm_regime',
        'daily_poc', 'daily_vwap', 'rolling_poc_4h', 'dynamic_tp', 'dynamic_sl',
        'fvg_bull_bottom', 'fvg_bull_top', 'fvg_bear_top', 'fvg_bear_bottom',
    }
    features = [c for c in df.columns if c not in exclude_cols]
    print(f"Using {len(features)} features.")
    
    wf_start = pd.to_datetime(WF_CONFIG["wf_start"])
    if wf_start.tzinfo is None:
        wf_start = wf_start.tz_localize('UTC')
    else:
        wf_start = wf_start.tz_convert('UTC')
        
    retrain_days = WF_CONFIG["retrain_days"]
    out_dir = PROJECT_ROOT / os.environ.get("V14_MODEL_OUTPUT_DIR", WF_CONFIG["model_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    prod_path = PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v14_wf.joblib"

    def _cycle_path(cycle: int, start_date) -> Path:
        return out_dir / f"filter_v14_cycle_{cycle}_{start_date}.joblib"

    targets = iter_wf_train_targets(_cycle_path, as_of=wf_train_as_of())
    print(f"S1 WF train mode: {wf_train_mode()}  targets={len(targets)}")

    if wf_train_mode() == "full":
        df_train_full = df[df.index < wf_start]
        print(f"Initial training set: {len(df_train_full)} bars (up to {wf_start.date()})")
        X_train = df_train_full[features]
        y_train = df_train_full["trend_label"]
        if y_train.sum() > 0:
            scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
            print(f"Applying scale_pos_weight: {scale_pos_weight:.2f}")
            model_config = MODEL_CONFIG["s1"].copy()
            model_config["scale_pos_weight"] = scale_pos_weight
        else:
            model_config = MODEL_CONFIG["s1"]
        model = xgb.XGBClassifier(**model_config)
        model.fit(X_train, y_train)
        joblib.dump(model, prod_path)
        print(f"Saved initial PROD model to {prod_path}")
    elif not targets:
        print("S1 incremental: no new cycle — prod and cycle files unchanged.")
        return

    prod_model = joblib.load(prod_path) if prod_path.exists() else None

    for cycle, current_start in targets:
        model_path = _cycle_path(cycle, current_start.date())
        train_chunk = df[df.index < current_start]
        X_tr = train_chunk[features]
        y_tr = train_chunk["trend_label"]
        print(f"Cycle {cycle} ({current_start.date()}): Training on {len(X_tr)} bars...")
        if len(X_tr) < 10 or y_tr.sum() == 0:
            if prod_model is not None:
                joblib.dump(prod_model, model_path)
                print(f"  Using prod fallback -> {model_path.name}")
            continue
        scale_pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()
        cycle_config = MODEL_CONFIG["s1"].copy()
        cycle_config["scale_pos_weight"] = scale_pos_weight
        cycle_model = xgb.XGBClassifier(**cycle_config)
        cycle_model.fit(X_tr, y_tr)
        joblib.dump(cycle_model, model_path)

    print("Walk-forward S1 training complete.")

if __name__ == "__main__":
    train_walk_forward_v14()
