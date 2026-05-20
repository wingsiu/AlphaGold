"""
TP/SL sweep on a recent backtest window using the live WF cycle models.

Re-uses the existing backtest scoring pipeline (so S1/S2 probs stay identical
across grid points) and only varies (TP, SL, horizon) in the exit/entry loop.

Usage:
    python tp_sl_sweep.py [days_back]            # default 30
    python tp_sl_sweep.py YYYY-MM-DD YYYY-MM-DD
"""
import sys, os, json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from config.v13_config import EXECUTION_CONFIG, WF_CONFIG
from xgboost_filter_model.train_filter_v13_wf_image import prepare_data_v13
from xgboost_filter_model.train_filter_1min import load_price_data
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgboost_filter_model.train_directional_model_v9 import add_momentum_features

# ── Args ──────────────────────────────────────────────────────────────────────
if len(sys.argv) == 1:
    days_back = 30
    bt_end = pd.Timestamp.utcnow().normalize().date()
    bt_start = (pd.Timestamp(bt_end) - pd.Timedelta(days=days_back)).date()
elif len(sys.argv) == 2:
    days_back = int(sys.argv[1])
    bt_end = pd.Timestamp.utcnow().normalize().date()
    bt_start = (pd.Timestamp(bt_end) - pd.Timedelta(days=days_back)).date()
else:
    bt_start = pd.to_datetime(sys.argv[1]).date()
    bt_end   = pd.to_datetime(sys.argv[2]).date()

print(f"Sweep window: {bt_start} → {bt_end}")

# ── Load / prepare ────────────────────────────────────────────────────────────
wf_start = pd.to_datetime(WF_CONFIG["wf_start"])
warmup_days = 120
load_start = (pd.to_datetime(bt_start) - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
load_end   = (pd.to_datetime(bt_end)   + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

df = prepare_data_v13(start_date=load_start, end_date=load_end)
df = add_directional_features(df)
df = add_ma_features(df)
df = add_momentum_features(df)
df.dropna(inplace=True)

exclude = {
    'open','high','low','close','volume','timestamp',
    'day_high_rolling','day_low_rolling','day_open',
    'Dchange_utc2_rel','Dupper_wick_utc2_rel','Dlower_wick_utc2_rel',
    'trend_label','target_v10','is_trend','atr','day_utc2',
    'future_max_move','future_min_move','future_er','atr_threshold',
    'bar_move','hour','day_id','day_high','day_low','high_90','low_90',
    'closePrice_ask','closePrice_bid','highPrice_ask','lowPrice_bid',
    'closePrice','lowPrice','open_price','highPrice_bid','lowPrice_ask',
    'openPrice_bid','openPrice_ask',
}
s2_extra = {
    'directional_change_15','directional_change_30','directional_change_90',
    'wick_ratio_15','wick_ratio_30','wick_ratio_90',
    'price_vs_ma_10','price_vs_ma_30','price_vs_ma_90',
    'ma_10_vs_30','ma_30_vs_90',
    'rsi_14','rsi_30','macd','macd_signal','macd_diff',
    'roc_15','roc_30','roc_60',
}
features    = [c for c in df.columns if c not in exclude]
s1_features = [f for f in features if f not in s2_extra]

df_test = df[df.index >= pd.to_datetime(bt_start).tz_localize('UTC')].copy()
print(f"Energetic bars in window: {len(df_test)}")
if df_test.empty:
    print("Empty window."); sys.exit(0)

# ── WF scoring (cycle-aligned) ────────────────────────────────────────────────
prod_s1 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "filter_model_v13_wf_image.joblib")
prod_s2 = joblib.load(PROJECT_ROOT / "xgboost_filter_model" / "directional_model_v13_wf.joblib")

df_test['s1_prob'] = np.nan
df_test['s2_prob'] = np.nan

wf_dir = PROJECT_ROOT / WF_CONFIG.get("model_output_dir", "runtime/bot_assets/wf_models_v13")
retrain_days = WF_CONFIG.get("retrain_days", 14)
wf_anchor = wf_start.tz_localize('UTC') if wf_start.tzinfo is None else wf_start.tz_convert('UTC')
run_start = pd.to_datetime(bt_start).tz_localize('UTC')
end_dt    = pd.to_datetime(bt_end).tz_localize('UTC') + pd.Timedelta(days=1)
elapsed_days = max(0, (run_start - wf_anchor).days)
skip_cycles  = elapsed_days // retrain_days
cycle = 1 + skip_cycles
current_start = wf_anchor + pd.Timedelta(days=skip_cycles * retrain_days)

s1_thresh = EXECUTION_CONFIG["s1_threshold"]
s2_base   = EXECUTION_CONFIG["s2_threshold"]
print("Scoring WF cycles…")
while current_start < end_dt:
    current_end = min(current_start + pd.Timedelta(days=retrain_days), end_dt)
    s1_path = wf_dir / f"filter_v13_cycle_{cycle}_{current_start.date()}.joblib"
    chunk = (df_test.index >= current_start) & (df_test.index < current_end)
    if chunk.any():
        s1 = joblib.load(s1_path) if s1_path.exists() else prod_s1
        df_test.loc[chunk, 's1_prob'] = s1.predict_proba(df_test.loc[chunk, s1_features])[:, 1]
        s2_path = wf_dir / f"directional_v13_cycle_{cycle}_{current_start.date()}.joblib"
        s2 = joblib.load(s2_path) if s2_path.exists() else prod_s2
        m = chunk & (df_test['s1_prob'] >= s1_thresh)
        if m.any():
            df_test.loc[m, 's2_prob'] = s2.predict_proba(df_test.loc[m, features])[:, 1]
    current_start = current_end
    cycle += 1

trend_mask = df_test['s1_prob'] >= s1_thresh
df_test['side_signal'] = 0
df_test.loc[trend_mask & (df_test['s2_prob'] >= s2_base),         'side_signal'] =  1
df_test.loc[trend_mask & (df_test['s2_prob'] <= (1.0 - s2_base)), 'side_signal'] = -1

# ── Full 1-min frame for exits ─────────────────────────────────────────────────
spread = EXECUTION_CONFIG["spread_default"]
raw_df = load_price_data(start_date=str(bt_start), end_date=load_end)
raw_df = raw_df[raw_df.index >= pd.to_datetime(bt_start).tz_localize('UTC')].copy()
if 'openPrice_ask' in raw_df.columns:
    raw_df['open_ask']  = raw_df['openPrice_ask']
    raw_df['open_bid']  = raw_df['openPrice_bid']
    raw_df['close_ask'] = raw_df['closePrice_ask']
    raw_df['close_bid'] = raw_df['closePrice_bid']
    raw_df['high_ask']  = raw_df['highPrice_ask']
    raw_df['low_bid']   = raw_df['lowPrice_bid']
else:
    raw_df['open_ask']  = raw_df['open']  + spread
    raw_df['open_bid']  = raw_df['open']  - spread
    raw_df['close_ask'] = raw_df['close'] + spread
    raw_df['close_bid'] = raw_df['close'] - spread
    raw_df['high_ask']  = raw_df['high']  + spread
    raw_df['low_bid']   = raw_df['low']   - spread

sim_df = raw_df[['open','high','low','close',
                 'open_ask','open_bid','close_ask','close_bid',
                 'high_ask','low_bid']].copy()
for col in ['side_signal','s1_prob','s2_prob']:
    sim_df[col] = df_test[col] if col in df_test.columns else np.nan
sim_df['side_signal'] = sim_df['side_signal'].fillna(0).astype(int)

# Pre-extract numpy arrays for speed
ts_arr   = sim_df.index.values
sig_arr  = sim_df['side_signal'].values
s2_arr   = sim_df['s2_prob'].values
low_b    = sim_df['low_bid'].values
high_a   = sim_df['high_ask'].values
close_b  = sim_df['close_bid'].values
close_a  = sim_df['close_ask'].values
close_m  = sim_df['close'].values
open_a   = sim_df['open_ask'].values
open_b   = sim_df['open_bid'].values
n        = len(sim_df)

# ── Sweep loop ────────────────────────────────────────────────────────────────
def simulate(tp, sl, horizon_min):
    """Returns dict of stats for a (tp, sl, horizon) combo."""
    horizon_ns = np.timedelta64(horizon_min, 'm')
    s2_max = EXECUTION_CONFIG["s2_max_threshold"]
    s2_inc = EXECUTION_CONFIG["s2_loss_increment"]

    trades = []
    active = None
    cl = 0
    for i in range(n - 1):
        now_ts = ts_arr[i]
        sig    = int(sig_arr[i])

        # exit
        if active is not None:
            s = active['side']
            exit_px = None; reason = None
            if s == 1:
                if low_b[i] <= active['stop']:
                    exit_px = active['stop']; reason = 'stop_loss'
                elif high_a[i] >= active['target']:
                    exit_px = active['target']; reason = 'target_hit'
                elif now_ts >= active['timeout']:
                    exit_px = close_b[i]; reason = 'timeout'
            else:
                if high_a[i] >= active['stop']:
                    exit_px = active['stop']; reason = 'stop_loss'
                elif low_b[i] <= active['target']:
                    exit_px = active['target']; reason = 'target_hit'
                elif now_ts >= active['timeout']:
                    exit_px = close_a[i]; reason = 'timeout'
            if exit_px is not None:
                pnl = (exit_px - active['entry_price']) * s
                trades.append({'side': s, 'entry': active['entry_price'], 'exit': exit_px,
                               'pnl': pnl, 'reason': reason})
                cl = 0 if pnl > 0 else cl + 1
                active = None

        # reverse/roll
        if active is not None and sig != 0:
            s = active['side']
            if sig == -s:
                px = close_b[i] if s == 1 else close_a[i]
                pnl = (px - active['entry_price']) * s
                trades.append({'side': s, 'entry': active['entry_price'], 'exit': px,
                               'pnl': pnl, 'reason': 'reverse_signal'})
                cl = 0 if pnl > 0 else cl + 1
                active = None
            elif sig == s:
                active['timeout'] = now_ts + horizon_ns
                new_t = close_m[i] + (tp if s == 1 else -tp)
                if (s == 1 and new_t > active['target']) or (s == -1 and new_t < active['target']):
                    active['target'] = new_t

        # entry
        if active is None and sig != 0:
            dyn_s2 = min(s2_max, s2_base + cl * s2_inc)
            s2_p = s2_arr[i]
            if np.isnan(s2_p):
                continue
            passes = (sig == 1 and s2_p >= dyn_s2) or (sig == -1 and s2_p <= (1.0 - dyn_s2))
            if passes:
                ep = open_a[i+1] if sig == 1 else open_b[i+1]
                active = {
                    'side': sig,
                    'entry_price': float(ep),
                    'stop':    ep - sl if sig == 1 else ep + sl,
                    'target':  ep + tp if sig == 1 else ep - tp,
                    'timeout': ts_arr[i+1] + horizon_ns,
                }

    if not trades:
        return {'tp': tp, 'sl': sl, 'horizon': horizon_min,
                'n': 0, 'wr': 0.0, 'avg_pnl': 0.0, 'net_pnl': 0.0,
                'profit_factor': 0.0, 'expectancy': 0.0}
    pnls = np.array([t['pnl'] for t in trades])
    wins = pnls > 0
    gross_p = pnls[wins].sum()
    gross_l = -pnls[~wins].sum()
    pf = (gross_p / gross_l) if gross_l > 0 else float('inf')
    return {
        'tp': tp, 'sl': sl, 'horizon': horizon_min,
        'n': len(pnls),
        'wr': float(wins.mean()),
        'avg_pnl': float(pnls.mean()),
        'net_pnl': float(pnls.sum()),
        'gross_profit': float(gross_p),
        'gross_loss': float(gross_l),
        'profit_factor': float(pf) if pf != float('inf') else 999.0,
        'expectancy': float(pnls.mean()),
    }

# Grid
tp_grid = [10, 15, 20, 25, 30, 35, 40, 50]
sl_grid = [8, 10, 12, 15, 20, 25, 30]
horizon_grid = [30, 45, 60, 90]

print(f"\nRunning sweep: {len(tp_grid)*len(sl_grid)*len(horizon_grid)} combos…")
results = []
for tp, sl, h in product(tp_grid, sl_grid, horizon_grid):
    r = simulate(tp, sl, h)
    results.append(r)
    print(f"  TP={tp:>3} SL={sl:>3} H={h:>3} | "
          f"n={r['n']:>4} WR={r['wr']*100:>5.1f}% "
          f"net={r['net_pnl']:>+8.1f} avg={r['avg_pnl']:>+5.2f} PF={r['profit_factor']:>5.2f}")

# Save
out = PROJECT_ROOT / "runtime" / f"tp_sl_sweep_{bt_start}_{bt_end}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(out, index=False)
print(f"\nSaved: {out}")

rdf = pd.DataFrame(results)
rdf = rdf[rdf['n'] >= 20]  # ignore combos with too few trades
print("\n=== Top 10 by NET PnL ===")
print(rdf.sort_values('net_pnl', ascending=False).head(10).to_string(index=False))
print("\n=== Top 10 by PROFIT FACTOR (n>=20) ===")
print(rdf.sort_values('profit_factor', ascending=False).head(10).to_string(index=False))
print("\n=== Top 10 by EXPECTANCY (n>=20) ===")
print(rdf.sort_values('expectancy', ascending=False).head(10).to_string(index=False))
