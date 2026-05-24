import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

from v14._paths import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14
from config.v14_config import WF_CONFIG, EXECUTION_CONFIG, TARGET_CONFIG

def evaluate_s2():
    print("Loading data for S2 evaluation (2026-01-01 to 2026-05-21)...")
    df = prepare_data_v14(start_date="2026-01-01", end_date="2026-05-22")
    
    print("Adding Stage 2 directional features...")
    df = prepare_directional_data_v14(df)

    
    # Add target_v14 for S2 evaluation
    if "atr_threshold" in df.columns:
        df["dynamic_tp"] = np.where(df["atr"] > df["atr_threshold"], TARGET_CONFIG["tp"] * 1.5, TARGET_CONFIG["tp"])
        df["dynamic_sl"] = np.where(df["atr"] > df["atr_threshold"], TARGET_CONFIG["sl"] * 1.5, TARGET_CONFIG["sl"])
    else:
        df["dynamic_tp"] = TARGET_CONFIG["tp"]
        df["dynamic_sl"] = TARGET_CONFIG["sl"]
        
    df['target_v14'] = (df['future_max_move'] > df['future_min_move'].abs()).astype(int)
    
    # Filter to only the bars where there was an actual trend (trend_label == 1)
    # This evaluates how well S2 predicts direction WHEN there is a trend
    df_s2 = df[df["trend_label"] == 1].copy()
    
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
        'daily_poc', 'daily_vwap', 'rolling_poc_4h', 'dynamic_tp', 'dynamic_sl'
    }
    features = [c for c in df_s2.columns if c not in exclude_cols]
    
    # Filter to the test period
    df_test = df_s2[df_s2.index >= pd.to_datetime("2026-01-01").tz_localize('UTC')].copy()
    
    df_test['s2_prob'] = np.nan
    wf_dir = PROJECT_ROOT / WF_CONFIG["model_output_dir"]
    prod_s2 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v14_wf.joblib")
    
    wf_anchor = pd.to_datetime(WF_CONFIG["wf_start"])
    if wf_anchor.tzinfo is None:
        wf_anchor = wf_anchor.tz_localize('UTC')
    else:
        wf_anchor = wf_anchor.tz_convert('UTC')
        
    retrain_days = WF_CONFIG["retrain_days"]
    
    # Align cycle calculation with backtest
    run_start = pd.to_datetime("2026-01-01").tz_localize('UTC')
    end_dt = df_test.index.max()
    
    elapsed_days = max(0, (run_start - wf_anchor).days)
    skip_cycles = elapsed_days // retrain_days
    cycle = 1 + skip_cycles
    current_start = wf_anchor + pd.Timedelta(days=skip_cycles * retrain_days)
    
    while current_start < end_dt:
        current_end = current_start + pd.Timedelta(days=retrain_days)
        chunk = (df_test.index >= current_start) & (df_test.index < current_end)
        
        if chunk.any():
            s2_path = wf_dir / f"directional_v14_cycle_{cycle}_{current_start.date()}.joblib"
            model = joblib.load(s2_path) if s2_path.exists() else prod_s2
            df_test.loc[chunk, 's2_prob'] = model.predict_proba(df_test.loc[chunk, features])[:, 1]
            
        current_start = current_end
        cycle += 1

    df_test = df_test.dropna(subset=['s2_prob'])
    
    print("\n" + "="*60)
    print("=== Stage 2 (Directional) Evaluation: 2026-01-01 to 2026-05-21 ===")
    print("="*60)
    print(f"Total True Trend bars in period: {len(df_test)}")
    print(f"Actual Longs (target_v14=1): {df_test['target_v14'].sum()} ({df_test['target_v14'].mean()*100:.1f}%)")
    print("-" * 60)
    
    for thresh in [0.5, 0.52, 0.55, 0.58, 0.6]:
        # Prediction: 1 if prob >= thresh, 0 if prob <= (1 - thresh), else NaN
        preds = pd.Series(np.nan, index=df_test.index)
        preds[df_test['s2_prob'] >= thresh] = 1
        preds[df_test['s2_prob'] <= (1.0 - thresh)] = 0
        
        valid_preds = preds.dropna()
        if len(valid_preds) > 0:
            acc = accuracy_score(df_test.loc[valid_preds.index, 'target_v14'], valid_preds)
            coverage = len(valid_preds) / len(df_test) * 100
            
            # Confusion Matrix
            true_labels = df_test.loc[valid_preds.index, 'target_v14']
            tp = ((valid_preds == 1) & (true_labels == 1)).sum()
            tn = ((valid_preds == 0) & (true_labels == 0)).sum()
            fp = ((valid_preds == 1) & (true_labels == 0)).sum()
            fn = ((valid_preds == 0) & (true_labels == 1)).sum()
            
        else:
            acc = 0.0
            coverage = 0.0
            tp = tn = fp = fn = 0
            
        print(f"Threshold {thresh:.2f} (Long >= {thresh:.2f}, Short <= {1-thresh:.2f}):")
        print(f"  Bars traded: {len(valid_preds)} ({coverage:.1f}% coverage)")
        print(f"  Directional Accuracy: {acc*100:.1f}%")
        print(f"  Confusion Matrix: TP (True Long): {tp}, TN (True Short): {tn}, FP (False Long): {fp}, FN (False Short): {fn}")
        if (tp + fp) > 0:
            print(f"  Long Precision: {tp / (tp + fp) * 100:.1f}%")
        if (tn + fn) > 0:
            print(f"  Short Precision: {tn / (tn + fn) * 100:.1f}%")
        print("-" * 60)

if __name__ == "__main__":
    evaluate_s2()
