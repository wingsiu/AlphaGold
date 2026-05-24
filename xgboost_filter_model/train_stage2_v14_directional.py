import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgboost_filter_model.train_directional_model_v9 import add_momentum_features
from config.v14_config import MODEL_CONFIG, WF_CONFIG

def prepare_directional_data_v14(df: pd.DataFrame) -> pd.DataFrame:
    """Adds directional features (MA, Momentum, etc.) for Stage 2."""
    df = add_directional_features(df)
    df = add_ma_features(df)
    df = add_momentum_features(df)
    df.dropna(inplace=True)
    
    # Target for Stage 2: 1 if it hits TP before SL, 0 otherwise
    # In build_target, future_max_move is the max favorable excursion, future_min_move is the max adverse excursion
    # We want to predict if it will hit TP (30) before SL (15).
    # Since we are using build_target, we can just use the direction of the close after horizon, or we can use a simpler target.
    # Let's create `target_v14` for Stage 2: 1 if long is profitable, 0 if short is profitable.
    # A long is profitable if future_max_move >= TP and future_min_move > -SL.
    # A short is profitable if future_min_move <= -TP and future_max_move < SL.
    
    # Actually, build_target already produces target logic, but let's just use a simple directional target:
    # 1 if close will be higher after horizon, 0 if lower.
    # But let's use the TP/SL logic:
    tp = 30.0
    sl = 15.0
    
    # Simplified: 1 if future_max_move > future_min_move.abs() else 0
    df['target_v14'] = (df['future_max_move'] > df['future_min_move'].abs()).astype(int)
    
    # Ensure both classes exist in the target variable to avoid XGBoost ValueError
    if len(df['target_v14'].unique()) < 2:
        print("Warning: Only one class found in target_v14. Adding dummy rows to ensure 2 classes.")
        # Add a dummy row for each class just to satisfy XGBoost's class inference
        dummy_0 = df.iloc[-1:].copy()
        dummy_0['target_v14'] = 0
        dummy_1 = df.iloc[-1:].copy()
        dummy_1['target_v14'] = 1
        df = pd.concat([df, dummy_0, dummy_1])
    
    return df

def train_walk_forward_s2_v14():
    print("=== Training AlphaGold v14 Directional Model ===")
    
    df = prepare_data_v14(
        start_date=WF_CONFIG["full_start"], 
        end_date=WF_CONFIG["wf_end"]
    )
    df = prepare_directional_data_v14(df)
    
    # Filter to only bars that passed Stage 1 (trend_label == 1)
    df_s2 = df[df["trend_label"] == 1].copy()
    print(f"Total Stage 2 training samples (trend_label=1): {len(df_s2)}")
    
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
    features = [c for c in df_s2.columns if c not in exclude_cols]
    print(f"Using {len(features)} features for Stage 2.")
    
    wf_start = pd.to_datetime(WF_CONFIG["wf_start"])
    if wf_start.tzinfo is None:
        wf_start = wf_start.tz_localize('UTC')
    else:
        wf_start = wf_start.tz_convert('UTC')
        
    retrain_days = WF_CONFIG["retrain_days"]
    out_dir = PROJECT_ROOT / os.environ.get("V14_MODEL_OUTPUT_DIR", WF_CONFIG["model_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_train_full = df_s2[df_s2.index < wf_start]
    print(f"Initial training set: {len(df_train_full)} bars (up to {wf_start.date()})")
    
    # Train initial "production" model
    X_train = df_train_full[features]
    y_train = df_train_full["target_v14"]
    
    if len(y_train.unique()) < 2:
        print("Warning: Initial training set has < 2 classes. Adding dummy rows.")
        if len(X_train) == 0:
            # Create completely dummy rows if empty
            dummy_row = pd.DataFrame([np.zeros(len(features))], columns=features)
            X_train = pd.concat([dummy_row, dummy_row])
            y_train = pd.Series([0, 1])
        else:
            X_train = pd.concat([X_train, X_train.iloc[-2:]])
            y_train = pd.concat([y_train, pd.Series([0, 1], index=X_train.index[-2:])])
        
    model = xgb.XGBClassifier(**MODEL_CONFIG["s2"])
    model.fit(X_train, y_train)
    
    prod_path = PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v14_wf.joblib"
    joblib.dump(model, prod_path)
    print(f"Saved initial PROD S2 model to {prod_path}")
    
    current_start = wf_start
    end_dt = max(df_s2.index.max(), pd.Timestamp.now(tz="UTC"))
    cycle = 1
    
    while current_start < end_dt:
        model_path = out_dir / f"directional_v14_cycle_{cycle}_{current_start.date()}.joblib"
        
        train_chunk = df_s2[df_s2.index < current_start]
        if len(train_chunk) < 10 or len(train_chunk["target_v14"].unique()) < 2:
            print(f"Cycle {cycle} ({current_start.date()}): Not enough data or classes. Skipping training.")
            # Just copy the previous model or prod model
            if cycle > 1:
                prev_model_path = out_dir / f"directional_v14_cycle_{cycle-1}_{current_start.date() - pd.Timedelta(days=retrain_days)}.joblib"
                if prev_model_path.exists():
                    import shutil
                    shutil.copy(prev_model_path, model_path)
            else:
                import shutil
                shutil.copy(prod_path, model_path)
        else:
            X_tr = train_chunk[features]
            y_tr = train_chunk["target_v14"]
            
            print(f"Cycle {cycle} ({current_start.date()}): Training on {len(X_tr)} bars...")
            cycle_model = xgb.XGBClassifier(**MODEL_CONFIG["s2"])
            cycle_model.fit(X_tr, y_tr)
            
            joblib.dump(cycle_model, model_path)
        
        current_start += pd.Timedelta(days=retrain_days)
        cycle += 1
        
    print("Walk-forward training complete.")

if __name__ == "__main__":
    train_walk_forward_s2_v14()
