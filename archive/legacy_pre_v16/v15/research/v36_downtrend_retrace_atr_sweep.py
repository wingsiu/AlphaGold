#!/usr/bin/env python3
"""
V36 — CORRECTED Short Retrace sweep with rolling daily ATR
===========================================================
SHORT RETRACE concept:
  We're in a DOWNTREND (price has dropped from recent 240-bar high).
  Then there's a BOUNCE UP (price rises from recent 240-bar low).
  We SHORT this bounce, expecting the downtrend to resume.
  WR_90 > -70 filters out deeply oversold conditions.

Definition:
  drop_from_high_240 >= daily_atr5 * DROP_X  (previous drop was significant)
  rise_from_low_240  >= daily_atr5 * RISE_X  (bounce is visible)
  wr_90 > -70  (not deeply oversold — still room to drop)

TP / SL (for SHORT):
  TP: price drops further with trend → entry_price - daily_atr5 * TP_X
  SL: price bounces higher against trend → entry_price + daily_atr5 * SL_X
  SL < TP for shorts (bounce is smaller than trend continuation)

Label (CORRECTED):
  fmin >= tp_abs  → price dropped at least tp from close (TP hit)
  fmax < sl_abs   → price did NOT rise at least sl from close (not stopped out)
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd, pandas_ta as pta
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from config.hybrid_config import WF_CONFIG
from xgboost_filter_model.train_filter_v14 import prepare_data_v14, build_target
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14
from xgboost_filter_model.train_filter_1min import load_price_data
from v15.features import add_v15_energetic_features

BT_START = "2025-06-01"
BT_END = WF_CONFIG["wf_end"]

print("=" * 80)
print("  V36 — SHORT RETRACE with rolling DAILY ATR (CORRECTED LABEL)")
print(f"  {BT_START} → {BT_END}")
print("=" * 80)

# Load
raw = load_price_data("2020-01-01", BT_END)
if raw.index.tz is None: raw.index = raw.index.tz_localize("UTC")

print("\n[1] Building features + rolling daily ATR...")
df = prepare_data_v14("2020-01-01", BT_END, energetic_filter=False, for_live_inference=True, pattern_feature_set="v2398")
df = prepare_directional_data_v14(df)
df = add_v15_energetic_features(df)

# Daily ATR5
dly_h = df["high"].resample("D").max(); dly_l = df["low"].resample("D").min(); dly_c = df["close"].resample("D").last()
dly_atr = pta.atr(dly_h, dly_l, dly_c, length=5)
df["day"] = df.index.floor("D"); df["daily_atr5"] = df["day"].map(dly_atr)

# Future moves for multiple horizons
print("Computing future moves...")
for h in [60, 120, 240, 360, 480]:
    fm = build_target(df[["open","high","low","close"]], h, 1.0, 1.0)
    df[f"fmax_{h}"] = fm["future_max_move"]
    df[f"fmin_{h}"] = fm["future_min_move"]

df_test = df[df.index >= pd.Timestamp(BT_START).tz_localize("UTC")].copy()
print(f"  Test bars: {len(df_test)}")

# ── Sweep ─────────────────────────────────────────────────────────────────
print("\n[2] Sweeping definition + TP/SL (corrected label only)...")

# User suggested: SL 0.10-0.20, TP 0.20-0.40 (TP > SL)
drop_mults = [0.30, 0.50, 0.80]
rise_mults = [0.10, 0.15, 0.20]
tp_mults   = [0.20, 0.25, 0.30, 0.35, 0.40]
sl_mults   = [0.10, 0.12, 0.15, 0.20]
horizons   = [120, 240, 360, 480]

results = []
total = len(horizons)*len(drop_mults)*len(rise_mults)*len(tp_mults)*len(sl_mults)
cnt = 0

for h in horizons:
    fm, fmn = f"fmax_{h}", f"fmin_{h}"
    for dx in drop_mults:
        for rx in rise_mults:
            mask = (df_test["drop_from_high_240"] >= df_test["daily_atr5"]*dx) & \
                   (df_test["rise_from_low_240"] >= df_test["daily_atr5"]*rx) & \
                   (df_test["wr_90"] > -70)
            if "near_low_zone" in df_test.columns:
                mask &= df_test["near_low_zone"] != 1.0
            n_bars = int(mask.sum())
            if n_bars < 200: continue
            
            for tpx in tp_mults:
                for slx in sl_mults:
                    cnt += 1
                    if cnt % 50 == 0: print(f"  {cnt}/{total}...")
                    
                    tp_a = df_test["daily_atr5"]*tpx
                    sl_a = df_test["daily_atr5"]*slx
                    # CORRECTED SHORT label:
                    trg = ((df_test[fmn] >= tp_a) & (df_test[fm] < sl_a)).astype(int)
                    n_pos = int(trg.loc[mask].sum())
                    if n_pos < 20: continue
                    hr = float(trg.loc[mask].mean()*100)
                    score = int(n_bars * hr / 100)
                    rr = tpx/slx if slx > 0 else 0
                    
                    results.append({
                        "h":h, "dx":dx, "rx":rx, "tp_x":tpx, "sl_x":slx,
                        "rr":rr, "bars":n_bars, "pos":n_pos, "hr":hr,
                        "tp$":float(tp_a[mask].mean()), "sl$":float(sl_a[mask].mean()),
                        "score":score,
                    })

rd = pd.DataFrame(results).sort_values("score", ascending=False)

print(f"\n{'='*130}")
print(f"  TOP 40 BY SCORE")
print(f"{'='*130}")
print(f"{'#':>4s} {'h':>4s} {'dx':>5s} {'rx':>5s} {'tp_x':>6s} {'sl_x':>6s} {'RR':>5s} {'bars':>6s} {'pos':>5s} {'HR%':>7s} {'tp$':>7s} {'sl$':>7s} {'score':>7s}")
print("-"*130)
for i, r in rd.head(40).iterrows():
    print(f"{i:4d} {int(r['h']):4d} {r['dx']:5.2f} {r['rx']:5.2f} {r['tp_x']:6.3f} {r['sl_x']:6.3f} {r['rr']:5.2f} {int(r['bars']):6d} {int(r['pos']):5d} {r['hr']:7.2f} {r['tp$']:7.1f} {r['sl$']:7.1f} {int(r['score']):7d}")

csv_path = Path(BASE / "runtime" / "v36_atr_sweep_results.csv")
rd.to_csv(csv_path, index=False)
print(f"\n  Saved to {csv_path}")
print("\nDone.")
