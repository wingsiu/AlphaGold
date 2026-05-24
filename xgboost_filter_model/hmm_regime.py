import pandas as pd
import numpy as np
import joblib
from pathlib import Path

try:
    from hmmlearn import hmm
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hmmlearn"])
    from hmmlearn import hmm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_hmm_model_path():
    return PROJECT_ROOT / "runtime" / "bot_assets" / "hmm_model.joblib"

def train_hmm_model(df_1min: pd.DataFrame):
    print("Training HMM model on 1-hour bars...")
    df_hourly = df_1min.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_hourly['log_return'] = np.log(df_hourly['close'] / df_hourly['close'].shift(1)) * 1000
    df_hourly['range'] = (df_hourly['high'] - df_hourly['low']) / df_hourly['open'] * 1000
    df_hourly.dropna(inplace=True)
    
    X = df_hourly[['log_return', 'range']].values
    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(X)
    
    hidden_states = model.predict(X)
    df_hourly['regime'] = hidden_states
    
    stats = df_hourly.groupby('regime')['range'].mean()
    ranging_regime = int(stats.idxmin())
    trend_regimes = [i for i in range(3) if i != ranging_regime]
    
    model_data = {
        'model': model,
        'ranging_regime': ranging_regime,
        'trend_regimes': trend_regimes
    }
    
    out_path = get_hmm_model_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, out_path)
    print(f"HMM Model saved. Ranging regime: {ranging_regime}, Trend regimes: {trend_regimes}")
    return model_data

def add_hmm_regime(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Adds the 'hmm_regime' column based on the PREVIOUS 1-hour bar."""
    model_path = get_hmm_model_path()
    if not model_path.exists():
        train_hmm_model(df_1min)
        
    model_data = joblib.load(model_path)
    model = model_data['model']
    
    df_hourly = df_1min.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_hourly['log_return'] = np.log(df_hourly['close'] / df_hourly['close'].shift(1)) * 1000
    df_hourly['range'] = (df_hourly['high'] - df_hourly['low']) / df_hourly['open'] * 1000
    
    valid_hourly = df_hourly.dropna().copy()
    if len(valid_hourly) > 0:
        X = valid_hourly[['log_return', 'range']].values
        valid_hourly['regime'] = model.predict(X)
    else:
        valid_hourly['regime'] = model_data['ranging_regime']
        
    # Shift by 1 so that at 10:00, we have the regime of the 09:00-09:59 bar
    valid_hourly['prev_regime'] = valid_hourly['regime'].shift(1)
    
    regime_df = valid_hourly[['prev_regime']].copy()
    
    df_1min = df_1min.copy()
    df_1min = df_1min.sort_index()
    regime_df = regime_df.sort_index()
    
    df_1min['hmm_regime'] = pd.merge_asof(
        df_1min[[]], 
        regime_df, 
        left_index=True, 
        right_index=True, 
        direction='backward'
    )['prev_regime']
    
    # Fill NaNs at the beginning with the ranging regime
    df_1min['hmm_regime'] = df_1min['hmm_regime'].fillna(model_data['ranging_regime'])
    
    return df_1min
