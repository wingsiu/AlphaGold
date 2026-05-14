#!/usr/bin/env python3
"""Diagnose why image_s1_prob differs between training and live for the same bar."""
import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
PROJECT_ROOT = Path("/Users/alpha/Desktop/python/AlphaGold")
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd, numpy as np, joblib
from trading_bot_v13 import prepare_v13_features, _session_info, HK_TZ, LONDON_TZ, NY_TZ
from xgboost_filter_model.train_filter_1min import load_price_data, prepare_features as prepare_base_features
from xgboost_filter_model.train_filter_v10 import add_liquidity_indicators
from data.data_loader import DataLoader

bundle = joblib.load(PROJECT_ROOT / "training" / "image_trend_model.joblib")
s1_img = bundle["stage1"]
win    = bundle["config"].get("window", 150)
s1_xgb = joblib.load(PROJECT_ROOT / "runtime" / "bot_assets" / "filter_model_v13_wf_image.joblib")

v13_s1_cols = [
    'returns','adx','adx_slope','volatility','er_15','er_30','er_90','fractal_dimension',
    'wr_15','wr_30','wr_90','change_15','upper_wick_15','lower_wick_15','change_30',
    'upper_wick_30','lower_wick_30','change_90','upper_wick_90','lower_wick_90',
    'down_efficiency_ratio','up_efficiency_ratio','volume_price_corr','volume_trend',
    'volume_osc','change','upper_wick','lower_wick','bar_change','bar_upper_wick',
    'bar_lower_wick','day_progress','is_asia','asia_progress','is_london',
    'london_progress','is_ny','ny_progress','is_eq_high','is_eq_low','near_high_zone',
    'near_low_zone','recovery_long','recovery_short','image_s1_prob'
]

TARGET_TS = pd.Timestamp("2026-05-14 13:42:00", tz="UTC")

def extract_img(df, idx):
    w = df.iloc[idx - win + 1 : idx + 1]
    c0 = float(w["close"].iloc[0]) or 1.0
    vol = w["volume"].to_numpy(dtype=float)
    vol_mean, vol_std = np.mean(vol), np.std(vol)
    vol_z = np.zeros_like(vol) if vol_std < 1e-9 else (vol - vol_mean) / vol_std
    v0 = float(vol[0])
    vol_rel = np.zeros_like(vol) if abs(v0) < 1e-9 else vol / v0 - 1.0
    vd = np.diff(vol, prepend=vol[0]); vd_std = np.std(vd)
    vol_diff_norm = np.zeros_like(vd) if vd_std < 1e-9 else vd / vd_std
    img = np.stack([
        w["open"].to_numpy()/c0 - 1.0, w["high"].to_numpy()/c0 - 1.0,
        w["low"].to_numpy()/c0 - 1.0,  w["close"].to_numpy()/c0 - 1.0,
        (w["close"].to_numpy() - w["open"].to_numpy()) / c0,
        (w["high"].to_numpy()  - w["low"].to_numpy())  / c0,
        vol_z, vol_rel, vol_diff_norm
    ], axis=0).flatten()
    return img, w

# ───────────────────────── TRAINING APPROACH ─────────────────────────
print("="*60)
print("APPROACH 1: Training (full dataset, no filtering)")
print("="*60)
df_full = load_price_data(start_date="2020-01-01", end_date="2026-05-15")
print(f"Loaded {len(df_full)} raw bars")
df_full = prepare_base_features(df_full, move_threshold=10, er_threshold=0.3, future_window=45)
df_full = add_liquidity_indicators(df_full)
print(f"After prepare_base_features + dropna: {len(df_full)} bars")

if TARGET_TS in df_full.index:
    i = df_full.index.get_loc(TARGET_TS)
    print(f"Target ts found at position {i}")
else:
    i = df_full.index.searchsorted(TARGET_TS) - 1
    print(f"Target ts not in index; using position {i} = {df_full.index[i]}")

img_train, w_train = extract_img(df_full, i)

# Build extra features (recompute day features on full df as training does)
from zoneinfo import ZoneInfo
dso = pd.Timedelta(hours=2)
df_full["_dut2"] = (df_full.index + dso).floor("D")
df_full["_dopen"] = df_full.groupby("_dut2")["open"].transform("first")
df_full["_dhigh"] = df_full.groupby("_dut2")["high"].cummax()
df_full["_dlow"]  = df_full.groupby("_dut2")["low"].cummin()
row_train = df_full.iloc[i]
c_t = row_train["close"]; do_t = row_train["_dopen"]
dh_t = row_train["_dhigh"]; dl_t = row_train["_dlow"]
Dchange_t = (c_t - do_t) / do_t
Dupper_t  = (dh_t - max(do_t, c_t)) / do_t
Dlower_t  = (min(do_t, c_t) - dl_t) / do_t
ts_t = df_full.index[i]
asia_f, asia_p = _session_info(ts_t, HK_TZ, 8, 0, 16, 0)
lon_f, lon_p   = _session_info(ts_t, LONDON_TZ, 8, 0, 16, 30)
ny_f, ny_p     = _session_info(ts_t, NY_TZ, 9, 30, 16, 0)
extra_t = [Dchange_t, Dupper_t, Dlower_t, asia_f, asia_p, lon_f, lon_p, ny_f, ny_p]
img_s1_train = s1_img.predict_proba(np.concatenate([img_train, extra_t]).reshape(1,-1))[0][1]

# Now also set image_s1_prob and run xgb s1
df_full.loc[ts_t, "image_s1_prob"] = img_s1_train
xgb_s1_train = s1_xgb.predict_proba(df_full.loc[[ts_t], v13_s1_cols])[0][1]

print(f"\n  Window: {w_train.index[0]}  →  {w_train.index[-1]}  ({len(w_train)} bars)")
print(f"  Dchange={Dchange_t:.6f}  Dupper={Dupper_t:.6f}  Dlower={Dlower_t:.6f}")
print(f"  img_s1_prob  (img model) = {img_s1_train:.6f}")
print(f"  S1           (xgb model) = {xgb_s1_train:.6f}")

# ───────────────────────── LIVE APPROACH ─────────────────────────
print()
print("="*60)
print("APPROACH 2: Live (2500-bar rolling window)")
print("="*60)
df_live_raw = DataLoader().load_data("gold_prices")
df_live_raw.index = pd.to_datetime(df_live_raw['timestamp'], unit='ms', utc=True)
df_live_raw = df_live_raw.rename(columns={
    'openPrice':'open','highPrice':'high','lowPrice':'low',
    'closePrice':'close','lastTradedVolume':'volume'}).sort_index()
df_live_raw = df_live_raw[df_live_raw.index <= TARGET_TS].tail(2500)
print(f"Live raw cache size: {len(df_live_raw)}")

df_live = prepare_v13_features(df_live_raw)
print(f"After prepare_v13_features + dropna: {len(df_live)} bars")

if TARGET_TS in df_live.index:
    i_l = df_live.index.get_loc(TARGET_TS)
else:
    i_l = len(df_live) - 1
    print(f"WARNING: TARGET_TS not in live df! Using last bar: {df_live.index[-1]}")

ts_l = df_live.index[i_l]
img_live, w_live = extract_img(df_live, i_l)
row_live = df_live.iloc[i_l]
Dchange_l = float(row_live["Dchange_utc2_rel"])
Dupper_l  = float(row_live["Dupper_wick_utc2_rel"])
Dlower_l  = float(row_live["Dlower_wick_utc2_rel"])
extra_l = [Dchange_l, Dupper_l, Dlower_l, asia_f, asia_p, lon_f, lon_p, ny_f, ny_p]
img_s1_live = s1_img.predict_proba(np.concatenate([img_live, extra_l]).reshape(1,-1))[0][1]

df_live.loc[ts_l, "image_s1_prob"] = img_s1_live
xgb_s1_live = s1_xgb.predict_proba(df_live.loc[[ts_l], v13_s1_cols])[0][1]

print(f"\n  Window: {w_live.index[0]}  →  {w_live.index[-1]}  ({len(w_live)} bars)")
print(f"  Dchange={Dchange_l:.6f}  Dupper={Dupper_l:.6f}  Dlower={Dlower_l:.6f}")
print(f"  img_s1_prob  (img model) = {img_s1_live:.6f}")
print(f"  S1           (xgb model) = {xgb_s1_live:.6f}")

# ───────────────────────── DIFF ANALYSIS ─────────────────────────
print()
print("="*60)
print("DIFF ANALYSIS")
print("="*60)
print(f"  img_s1_prob train={img_s1_train:.6f}  live={img_s1_live:.6f}  DIFF={img_s1_train-img_s1_live:.6f}")
print(f"  S1          train={xgb_s1_train:.6f}  live={xgb_s1_live:.6f}  DIFF={xgb_s1_train-xgb_s1_live:.6f}")
print(f"  Dchange     train={Dchange_t:.6f}  live={Dchange_l:.6f}  DIFF={Dchange_t-Dchange_l:.6f}")
print(f"  Dupper      train={Dupper_t:.6f}  live={Dupper_l:.6f}  DIFF={Dupper_t-Dupper_l:.6f}")
print(f"  Dlower      train={Dlower_t:.6f}  live={Dlower_l:.6f}  DIFF={Dlower_t-Dlower_l:.6f}")

if w_train.index[0] == w_live.index[0]:
    print(f"\n  Windows have SAME start. Checking OHLCV diffs in window...")
    for col in ['open','high','low','close','volume']:
        d = np.abs(w_train[col].to_numpy() - w_live[col].to_numpy()).max()
        print(f"    max diff {col}: {d:.6f}")
else:
    print(f"\n  DIFFERENT WINDOW STARTS!")
    print(f"  Training: {w_train.index[0]}")
    print(f"  Live:     {w_live.index[0]}")

# Check all v13_s1_cols except image_s1_prob
print("\n  Feature diffs (train vs live, top 10 largest):")
diffs = {}
for col in v13_s1_cols[:-1]:  # exclude image_s1_prob
    try:
        vt = float(df_full.loc[ts_t, col]) if col in df_full.columns else np.nan
        vl = float(df_live.loc[ts_l, col]) if col in df_live.columns else np.nan
        diffs[col] = abs(vt - vl) if not (np.isnan(vt) or np.isnan(vl)) else np.nan
    except:
        diffs[col] = np.nan
top_diffs = sorted([(k,v) for k,v in diffs.items() if not np.isnan(v)], key=lambda x:-x[1])[:10]
for col, d in top_diffs:
    vt = float(df_full.loc[ts_t, col])
    vl = float(df_live.loc[ts_l, col])
    print(f"    {col:35s}: train={vt:.6f}  live={vl:.6f}  diff={d:.6f}")

