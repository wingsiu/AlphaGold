import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from v14._paths import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from config.v14_config import WF_CONFIG

def evaluate_s1():
    print("Loading data for S1 evaluation (2026-01-01 to 2026-05-21)...")
    df = prepare_data_v14(start_date="2026-01-01", end_date="2026-05-22")
    
    exclude_cols = {
        'open', 'high', 'low', 'close', 'volume', 'timestamp',
        'trend_label', 'target_v10', 'is_trend', 'atr', 'day_utc2',
        'future_max_move', 'future_min_move', 'future_er', 'atr_threshold',
        'bar_move', 'hour', 'day_id', 'day_high', 'day_low', 'high_90', 'low_90',
        'day_open', 'day_high_rolling', 'day_low_rolling',
        'openPrice_ask', 'openPrice_bid', 'closePrice_ask', 'closePrice_bid', 
        'highPrice_ask', 'highPrice_bid', 'lowPrice_ask', 'lowPrice_bid',
        'closePrice', 'lowPrice', 'open_price', 'highPrice', 'openPrice',
        'ma_60m', 'high_60m', 'low_60m', 'high_15m', 'low_15m', 'hmm_regime',
        'daily_poc', 'daily_vwap', 'rolling_poc_4h', 'dynamic_tp', 'dynamic_sl'
    }
    features = [c for c in df.columns if c not in exclude_cols]
    
    # Filter to the test period
    df_test = df[df.index >= pd.to_datetime("2026-01-01").tz_localize('UTC')].copy()
    
    df_test['s1_prob'] = np.nan
    wf_dir = PROJECT_ROOT / WF_CONFIG["model_output_dir"]
    prod_s1 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v14_wf.joblib")
    
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
            s1_path = wf_dir / f"filter_v14_cycle_{cycle}_{current_start.date()}.joblib"
            model = joblib.load(s1_path) if s1_path.exists() else prod_s1
            df_test.loc[chunk, 's1_prob'] = model.predict_proba(df_test.loc[chunk, features])[:, 1]
            
        current_start = current_end
        cycle += 1

    df_test = df_test.dropna(subset=['s1_prob'])
    
    print("\n" + "="*60)
    print("=== Stage 1 (Filter) Evaluation: 2026-01-01 to 2026-05-21 ===")
    print("="*60)
    print(f"Total energetic bars in period: {len(df_test)}")
    print(f"Actual trend bars (Label=1): {df_test['trend_label'].sum()} ({df_test['trend_label'].mean()*100:.1f}% of energetic bars)")
    print("-" * 60)
    
    for thresh in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        preds = (df_test['s1_prob'] >= thresh).astype(int)
        passed = preds.sum()
        
        true_labels = df_test['trend_label']
        tp = ((preds == 1) & (true_labels == 1)).sum()
        tn = ((preds == 0) & (true_labels == 0)).sum()
        fp = ((preds == 1) & (true_labels == 0)).sum()
        fn = ((preds == 0) & (true_labels == 1)).sum()
        
        if passed > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
            
        print(f"Threshold {thresh:.2f}:")
        print(f"  Bars passed S1: {passed}")
        print(f"  Precision (True Trends / Passed Bars): {precision*100:.1f}%")
        print(f"  Confusion Matrix: TP (True Trend Passed): {tp}, TN (Chop Filtered): {tn}, FP (Chop Passed): {fp}, FN (True Trend Filtered): {fn}")
        print("-" * 60)

if __name__ == "__main__":
    evaluate_s1()
