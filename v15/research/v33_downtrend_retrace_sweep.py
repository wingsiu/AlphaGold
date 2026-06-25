#!/usr/bin/env python3
"""V33 Downtrend Retrace (SHORT) — Daily ATR definition + TP/SL sweep."""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import numpy as np, pandas as pd, pandas_ta as pta
from config.hybrid_config import WF_CONFIG
from xgboost_filter_model.pattern_training import feature_columns, fit_pattern_model
from xgboost_filter_model.train_filter_v14 import prepare_data_v14, build_target
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14
from xgboost_filter_model.train_filter_1min import load_price_data
from backtest.core import simulate_v13_core
from v15.features import add_v15_energetic_features
from config.pattern_registry import PATTERN_MODEL_DIR, PATTERN_REGISTRY
from xgboost_filter_model.pattern_training import wf_anchor_ts, iter_wf_cycles, cycle_model_path, prod_model_path, pattern_variant_tag, execution_tp_sl, execution_target_mode
from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold
from xgboost_filter_model.pattern_router import assign_patterns
import joblib

BT_START, BT_END = "2025-06-01", WF_CONFIG["wf_end"]

print("="*70)
print(f"  V33 — DOWNTEND RETRACE V15 SWEEP | {BT_START} → {BT_END}")
print("="*70)

# Load
raw = load_price_data("2020-01-01", BT_END)
if raw.index.tz is None: raw.index = raw.index.tz_localize("UTC")
raw_sim = raw[raw.index >= pd.Timestamp(BT_START).tz_localize("UTC")].copy()

df = prepare_data_v14("2020-01-01", BT_END, energetic_filter=False, for_live_inference=True, pattern_feature_set="v2398")
df = prepare_directional_data_v14(df); df = add_v15_energetic_features(df)
df["daily_atr"] = df.index.floor("D").map(pta.atr(df["high"].resample("D").max(), df["low"].resample("D").min(), df["close"].resample("D").last(), 14))
fm120 = build_target(df[["open","high","low","close"]], 120, 1.0, 1.0)
df["fmax_120"], df["fmin_120"] = fm120["future_max_move"], fm120["future_min_move"]
df_test = df[df.index >= raw_sim.index[0]].copy()

# ── V14 Baseline ─────────────────────────────────────────────────────────
print("\n--- V14 Baseline ---")
v14_spec = PATTERN_REGISTRY["downtrend_retrace"]
v14_ex = v14_spec["execution"]
v14_tp, v14_sl = execution_tp_sl(v14_ex)
v14_mask = (df_test["drop_from_high_240"] >= 25) & (df_test["rise_from_low_240"] >= 5)
if "near_low_zone" in df_test.columns:
    v14_mask &= df_test["near_low_zone"] != 1.0
print(f"  V14 bars: {int(v14_mask.sum())}  TP={v14_tp} SL={v14_sl} H={v14_ex['horizon']}")

# V14 Sim (all bars, no model filter — for baseline reference only)
sim14 = raw_sim[["open","high","low","close"]].copy()
sim14["side_signal"] = 0; sim14["s1_prob"] = 0.5; sim14["s2_prob"] = 0.5
ci14 = df_test.index[v14_mask].intersection(sim14.index)
sim14.loc[ci14, "side_signal"] = -1  # SHORT
sim14.loc[ci14, "s2_prob"] = 1.0
t14 = simulate_v13_core(sim14, v14_tp, v14_sl, v14_ex["horizon"])
if t14:
    td14 = pd.DataFrame(t14)
    pnl14, wr14 = td14["pnl"].sum(), (td14["pnl"]>0).mean()*100
    dd14 = float((td14["pnl"].cumsum()-td14["pnl"].cumsum().cummax()).min())
    print(f"  V14 Sim: {len(td14)}t  PnL={pnl14:+.1f}  WR={wr14:.1f}%  MaxDD={dd14:+.1f}")
else:
    pnl14, wr14, dd14, td14 = 0, 0, 0, pd.DataFrame()
    print("  V14 Sim: no trades")

wf_anchor = wf_anchor_ts()
bt_start_dt = pd.Timestamp(BT_START).tz_localize("UTC")
end_dt = pd.Timestamp(BT_END).tz_localize("UTC")

# ── V15 Sweep ────────────────────────────────────────────────────────────
print("\n--- V15 Daily ATR Sweep ---")

horizons = [30, 60, 120, 240]
rise_multipliers = [0.05, 0.10, 0.15, 0.20]
drop_multipliers = [0.15, 0.20, 0.25, 0.30, 0.40]
tp_multipliers = [0.10, 0.15, 0.20, 0.25]
sl_multipliers = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

results = []
for h in horizons:
    print(f"\n  Horizon={h}min...")
    fmax_col, fmin_col = f"fmax_{h}", f"fmin_{h}"
    if fmax_col not in df_test.columns:
         fm_tmp = build_target(df[["open","high","low","close"]], h, 1.0, 1.0)
         df_test[fmax_col] = fm_tmp["future_max_move"]
         df_test[fmin_col] = fm_tmp["future_min_move"]

    for rx in drop_multipliers:
        for ry in rise_multipliers:
            ddef = df_test["drop_from_high_240"] >= df_test["daily_atr"] * rx
            rdef = df_test["rise_from_low_240"] >= df_test["daily_atr"] * ry
            mask = ddef & rdef
            if "near_low_zone" in df_test.columns:
                mask &= df_test["near_low_zone"] != 1.0
            n_bars = int(mask.sum())
            if n_bars < 100: continue

            for tpx in tp_multipliers:
                for slx in sl_multipliers:
                    tp_abs = df_test["daily_atr"] * tpx
                    sl_abs = df_test["daily_atr"] * slx
                    # For SHORT: TP hit when price drops by at least tp_abs
                    #            SL hit when price rises by at least sl_abs
                    #            Target label = TP hit AND NOT stopped out
                    trg = ((df_test[fmin_col] <= -tp_abs) & (df_test[fmax_col] < sl_abs)).astype(int)
                    pat_trg = trg.loc[mask]
                    if len(pat_trg) < 50: continue
                    hr = float(pat_trg.mean() * 100)
                    score = int(n_bars * hr / 100) if hr > 0 else 0

                    results.append({
                        "h": h, "drop_x": rx, "rise_x": ry,
                        "tp_x": tpx, "sl_x": slx,
                        "bars": n_bars, "hr": hr,
                        "tp_mean": float(tp_abs[mask].mean()),
                        "sl_mean": float(sl_abs[mask].mean()),
                        "score": score,
                    })

rd = pd.DataFrame(results).sort_values("score", ascending=False)
print(f"\n{'='*90}")
print(f"  TOP 30 CONFIGS BY SCORE (bars × HR)")
print(f"{'='*90}")
print(f"{'#':>4s} {'h':>4s} {'drop_x':>7s} {'rise_x':>7s} {'tp_x':>7s} {'sl_x':>7s} {'bars':>7s} {'HR%':>7s} {'TP$':>7s} {'SL$':>7s} {'score':>7s}")
print("-" * 90)
for i, r in rd.head(30).iterrows():
    print(f"{i:4d} {int(r['h']):4d} {r['drop_x']:7.2f} {r['rise_x']:7.2f} {r['tp_x']:7.3f} {r['sl_x']:7.3f} {int(r['bars']):7d} {r['hr']:7.2f} {r['tp_mean']:7.1f} {r['sl_mean']:7.1f} {int(r['score']):7d}")

# Best config
if len(rd):
    best = rd.iloc[0]
    print(f"\n{'='*70}")
    print(f"  BEST: H={int(best['h'])} drop≥{best['drop_x']:.2f}×ATR rise≥{best['rise_x']:.2f}×ATR")
    print(f"        TP={best['tp_x']:.3f}×ATR  SL={best['sl_x']:.3f}×ATR")
    print(f"        {int(best['bars'])} bars  HR={best['hr']:.1f}%  TP${best['tp_mean']:.1f}  SL${best['sl_mean']:.1f}")
    print(f"{'='*70}")

    # Train model + simulate
    print(f"\n--- Training model for best config ---")
    mask = (df_test["drop_from_high_240"] >= df_test["daily_atr"] * best["drop_x"]) & \
           (df_test["rise_from_low_240"] >= df_test["daily_atr"] * best["rise_x"])
    if "near_low_zone" in df_test.columns: mask &= df_test["near_low_zone"] != 1.0

    tp_abs = df_test["daily_atr"] * best["tp_x"]
    sl_abs = df_test["daily_atr"] * best["sl_x"]
    h = int(best["h"])
    fmax_c, fmin_c = f"fmax_{h}", f"fmin_{h}"
    df_test["v15_trg"] = ((df_test[fmin_c] <= -tp_abs) & (df_test[fmax_c] < sl_abs)).astype(int)
    df_pat = df_test.loc[mask].copy()
    feats = [c for c in feature_columns(df_pat) if c not in ("v15_trg","daily_atr",fmax_c,fmin_c) and df_pat[c].dtype in ("float64","float32","int64","int32","bool")]
    model = fit_pattern_model(df_pat[feats], df_pat["v15_trg"], min_samples=50)
    if model is None:
        print("  Model training FAILED")
        sys.exit(1)

    mf = list(model.feature_names_in_)
    df_test["v15_prob"] = np.nan
    for ts in df_test.index[mask]:
        row = df_test.loc[ts]; v = row[mf].values.astype(float)
        if np.isnan(v).any(): continue
        df_test.loc[ts,"v15_prob"] = float(model.predict_proba(pd.DataFrame([v], columns=mf))[:,1][0])
    df_test["v15_signal"] = mask & (df_test["v15_prob"] >= 0.45)
    print(f"  Signals: {int(df_test['v15_signal'].sum())}")

    sim15 = raw_sim[["open","high","low","close"]].copy()
    sim15["side_signal"] = 0; sim15["s1_prob"] = 0.5; sim15["s2_prob"] = 0.5
    ci15 = df_test.index[df_test["v15_signal"]].intersection(sim15.index)
    sim15.loc[ci15, "side_signal"] = -1
    sim15.loc[ci15, "s2_prob"] = df_test.loc[ci15, "v15_prob"]
    avg_tp = float(tp_abs[mask].mean()); avg_sl = float(sl_abs[mask].mean())
    t15 = simulate_v13_core(sim15, avg_tp, avg_sl, h)
    if t15:
        td15 = pd.DataFrame(t15)
        pnl15 = td15["pnl"].sum(); wr15 = (td15["pnl"]>0).mean()*100
        dd15 = float((td15["pnl"].cumsum()-td15["pnl"].cumsum().cummax()).min())
        print(f"  V15 Sim: {len(td15)}t  PnL={pnl15:+.1f}  WR={wr15:.1f}%  MaxDD={dd15:+.1f}")
    else:
        pnl15, wr15, dd15, td15 = 0, 0, 0, pd.DataFrame()
        print("  V15 Sim: no trades")

    print(f"\n{'='*55}")
    print(f"  V14 vs V15 — DOWNTEND RETRACE")
    print(f"{'='*55}")
    print(f"{'':<20} {'V14 (HMM)':>15} {'V15 (ATR)':>15}")
    v14_h_label = f'{int(v14_ex["horizon"])}min'
    v15_h_label = f'{int(best["h"])}min'
    print(f"{'Horizon':<20} {v14_h_label:>15} {v15_h_label:>15}")
    print(f"{'TP/SL':<20} {f'${v14_tp}/${v14_sl}':>15} {f'${avg_tp:.1f}/${avg_sl:.1f}':>15}")
    print(f"{'Bars':<20} {int(v14_mask.sum()):>15d} {int(mask.sum()):>15d}")
    print(f"{'Trades':<20} {len(td14) if len(td14) else 0:>15d} {len(td15) if len(td15) else 0:>15d}")
    print(f"{'PnL':<20} {pnl14:>+15.1f} {pnl15:>+15.1f}")
    print(f"{'WR':<20} {wr14:>14.1f}% {wr15:>14.1f}%")
    print(f"{'Max DD':<20} {dd14:>15.1f} {dd15:>15.1f}")

print("\nDone.")
