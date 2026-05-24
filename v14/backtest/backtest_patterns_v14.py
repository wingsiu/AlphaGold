#!/usr/bin/env python3
"""
Backtest pattern-specialist router — single-stage model per pattern.

Scores uptrend + downtrend on one timeline (priority routing). Each pattern
uses its own variant model and execution params from config/v14_patterns.py.

Usage:
  python3 backtest_patterns_v14.py 2025-06-01 2026-05-23
  V14_TIME_FILTER_JSON=runtime/v14_weak_time_slots.json python3 backtest_patterns_v14.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

from v14._paths import PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from v14.backtest.backtest_core import simulate_hybrid_two_pass, simulate_v13_core
from config.v14_config import EXECUTION_CONFIG, ENERGETIC_EXECUTION_CONFIG, TIME_FILTER_CONFIG, WF_CONFIG
from config.v14_patterns import PATTERN_MODEL_DIR, PATTERN_REGISTRY, PRODUCTION_PATTERNS, collect_pa_groups, backtest_feature_set
from xgboost_filter_model.energetic_gate import (
    apply_pattern_gates,
    hybrid_config,
    pattern_gate_config,
    score_energetic_signals,
)
from xgboost_filter_model.pattern_router import assign_patterns
from xgboost_filter_model.pattern_training import (
    cycle_model_path,
    feature_columns,
    iter_wf_cycles,
    pattern_variant_tag,
    prod_model_path,
    wf_anchor_ts,
)
from xgboost_filter_model.time_slot_filter import load_weak_filter, resolve_v14_time_filter_path
from xgboost_filter_model.train_filter_1min import load_price_data
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14

args = sys.argv[1:]
today_str = date.today().strftime("%Y-%m-%d")
bt_start = WF_CONFIG["wf_start"]
bt_end = today_str

date_args = [a for a in args if a not in PATTERN_REGISTRY]
pattern_filter = [a for a in args if a in PATTERN_REGISTRY]
if not pattern_filter:
    pattern_filter = list(PRODUCTION_PATTERNS)
active_patterns = {k: PATTERN_REGISTRY[k] for k in pattern_filter}

if len(date_args) == 1:
    if date_args[0].isdigit():
        bt_start = (date.today() - timedelta(days=int(date_args[0]))).strftime("%Y-%m-%d")
    else:
        bt_start = date_args[0]
elif len(date_args) >= 2:
    bt_start, bt_end = date_args[0], date_args[1]

print(f"\n{'='*60}")
print("  AlphaGold v14 Pattern Backtest (combined router)")
print(f"  Period : {bt_start} → {bt_end}")
print(f"  Patterns: {', '.join(pattern_filter)}")
for name, spec in active_patterns.items():
    ex = spec["execution"]
    print(
        f"  {name}: H={ex['horizon']} TP={ex['tp']} SL={ex['sl']}  "
        f"({spec['direction_bias']}, prob≥{spec['thresholds']['prob']})"
    )
print(f"  Feature set: {backtest_feature_set()} (widest matrix for mixed models)")
_gate = pattern_gate_config()
_hybrid = hybrid_config()
print(
    f"  Gates: energetic={_gate['energetic_filter']}  "
    f"s1={_gate['s1_gate']} (≥{_gate['s1_threshold']:.2f})"
)
if _hybrid["enabled"]:
    print(
        "  Hybrid: pattern-first → energetic fallback  |  "
        f"pattern refresh={_hybrid['pattern_same_dir_refresh']}  "
        f"energetic refresh={_hybrid['energetic_same_dir_refresh']}  "
        f"energetic reverse={_hybrid['energetic_close_on_reverse']}"
    )
print(f"  Refresh: {EXECUTION_CONFIG.get('same_dir_refresh', 'entry')}  "
      f"upgrade_stop={EXECUTION_CONFIG.get('upgrade_stop', False)}")
print(f"{'='*60}\n")

warmup_days = int(WF_CONFIG.get("feature_warmup_days", 120))
load_start_dt = max(
    pd.to_datetime(WF_CONFIG["full_start"]),
    pd.to_datetime(bt_start) - pd.Timedelta(days=warmup_days),
)
load_start = load_start_dt.strftime("%Y-%m-%d")
bt_end_date = bt_end.split("T")[0] if "T" in bt_end else bt_end
load_end = (pd.to_datetime(bt_end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
bt_start_date = bt_start.split("T")[0] if "T" in bt_start else bt_start

print(f"Loading pattern data {load_start} → {bt_end}…")
df = prepare_data_v14(
    start_date=load_start,
    end_date=load_end,
    energetic_filter=False,
    pa_groups=collect_pa_groups(list(active_patterns.keys())),
    pattern_feature_set=backtest_feature_set(),
)
df = prepare_directional_data_v14(df)
feats = feature_columns(df)

bt_start_dt = pd.to_datetime(bt_start)
if bt_start_dt.tzinfo is None:
    bt_start_dt = bt_start_dt.tz_localize("UTC")
else:
    bt_start_dt = bt_start_dt.tz_convert("UTC")

df_test = df[df.index >= bt_start_dt].copy()
print(f"Bars in test window: {len(df_test)}")

if df_test.empty:
    print("No bars in test window.")
    sys.exit(0)

df_test = assign_patterns(df_test)
drop = df_test["pattern_name"].notna() & ~df_test["pattern_name"].isin(pattern_filter)
df_test.loc[drop, "pattern_name"] = pd.NA
df_test.loc[drop, "pattern_id"] = -1
routed = df_test["pattern_name"].notna().sum()
print(f"Pattern-routed bars: {routed} ({routed/len(df_test)*100:.2f}%)")
for name in active_patterns:
    n = int((df_test["pattern_name"] == name).sum())
    print(f"  {name:20s}: {n}")

models: dict[str, dict] = {}
for name, spec in active_patterns.items():
    ex = spec["execution"]
    variant = pattern_variant_tag(ex["horizon"], ex["tp"], ex["sl"])
    pdir = PATTERN_MODEL_DIR / name / variant
    mp = prod_model_path(pdir)
    if not mp.exists():
        print(f"Missing model: {mp} — run train_pattern_variants.py for {name}")
        sys.exit(1)
    models[name] = {
        "prod": joblib.load(mp),
        "spec": spec,
        "dir": pdir,
        "variant": variant,
    }
    print(f"  Model {name}: {variant}")

wf_anchor = wf_anchor_ts()
end_dt = pd.to_datetime(bt_end).tz_localize("UTC") + pd.Timedelta(days=1)

if _hybrid["enabled"]:
    gate_mask = pd.Series(True, index=df_test.index)
    print("\nHybrid mode: pattern models ungated (energetic is fallback only)")
else:
    print("\nApplying pattern gates…")
    gate_mask = apply_pattern_gates(df_test, bt_start_dt, end_dt)

df_test["prob"] = np.nan
df_test["pattern_side"] = 0
df_test["s1_prob"] = np.nan
df_test["s2_prob"] = np.nan
df_test["side_signal"] = 0
df_test["matched_pattern"] = pd.NA
df_test["exec_tp"] = np.nan
df_test["exec_sl"] = np.nan
df_test["exec_horizon"] = np.nan

for cycle, current_start, current_end in iter_wf_cycles(bt_start_dt, end_dt, wf_anchor):
    chunk = (df_test.index >= current_start) & (df_test.index < current_end)
    if not chunk.any():
        continue

    for name, m in models.items():
        pat_chunk = chunk & (df_test["pattern_name"] == name) & gate_mask
        if not pat_chunk.any():
            continue

        path = cycle_model_path(m["dir"], cycle, current_start.date())
        model = joblib.load(path) if path.exists() else m["prod"]
        model_feats = list(model.feature_names_in_)
        spec = m["spec"]
        ex = spec["execution"]
        prob_thresh = spec["thresholds"]["prob"]
        bias = spec["direction_bias"]

        rows = df_test.loc[pat_chunk]
        p = model.predict_proba(rows[model_feats])[:, 1]
        df_test.loc[pat_chunk, "prob"] = p
        if not _gate["s1_gate"]:
            df_test.loc[pat_chunk, "s1_prob"] = p

        sig = pat_chunk & (df_test["prob"] >= prob_thresh)
        side = 1 if bias == "long" else -1
        df_test.loc[sig, "side_signal"] = side
        if bias == "long":
            df_test.loc[sig, "s2_prob"] = df_test.loc[sig, "prob"]
        else:
            df_test.loc[sig, "s2_prob"] = 1.0 - df_test.loc[sig, "prob"]
        df_test.loc[sig, "exec_tp"] = ex["tp"]
        df_test.loc[sig, "exec_sl"] = ex["sl"]
        df_test.loc[sig, "exec_horizon"] = ex["horizon"]
        fired = sig & df_test["matched_pattern"].isna()
        df_test.loc[fired, "matched_pattern"] = name

df_test["pattern_side"] = df_test["side_signal"].astype(int)

if _hybrid["enabled"]:
    print("\nScoring energetic fallback (S1/S2 on energetic bars)…")
    score_energetic_signals(df_test, bt_start_dt, end_dt)
else:
    df_test["energetic_side"] = 0

pat_sig_count = (df_test["pattern_side"] != 0).sum()
en_sig_count = int((df_test.get("energetic_side", 0) != 0).sum()) if _hybrid["enabled"] else 0
print(
    f"\nPattern entry signals: {pat_sig_count} "
    f"(LONG={(df_test['pattern_side']==1).sum()} SHORT={(df_test['pattern_side']==-1).sum()})"
)
if _hybrid["enabled"]:
    print(
        f"Energetic fallback signals: {en_sig_count} "
        f"(LONG={(df_test['energetic_side']==1).sum()} SHORT={(df_test['energetic_side']==-1).sum()})"
    )

if pat_sig_count == 0 and en_sig_count == 0:
    print("No signals in window.")
    sys.exit(0)

print("Loading full 1-min bars for exit simulation…")
raw_df = load_price_data(start_date=bt_start_date, end_date=load_end)
raw_df = raw_df[raw_df.index >= bt_start_dt].copy()

sim_df = raw_df[["open", "high", "low", "close"]].copy()
merge_cols = (
    "pattern_side",
    "energetic_side",
    "s1_prob",
    "s2_prob",
    "energetic_s1_prob",
    "energetic_s2_prob",
    "matched_pattern",
    "exec_tp",
    "exec_sl",
    "exec_horizon",
)
for col in merge_cols:
    sim_df[col] = df_test[col] if col in df_test.columns else np.nan
if "pattern_side" not in sim_df.columns or sim_df["pattern_side"].isna().all():
    sim_df["pattern_side"] = df_test["side_signal"] if "side_signal" in df_test.columns else 0
sim_df["pattern_side"] = sim_df["pattern_side"].fillna(0).astype(int)
sim_df["energetic_side"] = sim_df.get("energetic_side", pd.Series(0, index=sim_df.index)).fillna(0).astype(int)

weak_cells = None
_filter_path = resolve_v14_time_filter_path(PROJECT_ROOT)
if _filter_path:
    weak_cells = load_weak_filter(_filter_path)
    print(f"Time filter: blocking {len(weak_cells)} slots from {_filter_path}")

pat_exec_cfg = EXECUTION_CONFIG.copy()
pat_exec_cfg["close_on_reverse"] = (
    _hybrid["pattern_close_on_reverse"] if _hybrid["enabled"] else EXECUTION_CONFIG.get("close_on_reverse", False)
)
pat_exec_cfg["same_dir_refresh"] = (
    _hybrid["pattern_same_dir_refresh"] if _hybrid["enabled"] else EXECUTION_CONFIG.get("same_dir_refresh", "entry")
)
pat_exec_cfg["upgrade_stop"] = (
    _hybrid["pattern_upgrade_stop"] if _hybrid["enabled"] else EXECUTION_CONFIG.get("upgrade_stop", False)
)

if _hybrid["enabled"]:
    en_exec_cfg = ENERGETIC_EXECUTION_CONFIG.copy()
    en_exec_cfg["close_on_reverse"] = _hybrid["energetic_close_on_reverse"]
    en_exec_cfg["same_dir_refresh"] = _hybrid["energetic_same_dir_refresh"]
    en_exec_cfg["upgrade_stop"] = _hybrid["energetic_upgrade_stop"]
    pat_sim = sim_df.copy()
    pat_sim["side_signal"] = pat_sim["pattern_side"]
    pattern_trades = simulate_v13_core(
        pat_sim,
        EXECUTION_CONFIG["tp"],
        EXECUTION_CONFIG["sl"],
        EXECUTION_CONFIG["horizon"],
        config=pat_exec_cfg,
        weak_period_cells=weak_cells,
    )
    pat_pnl = sum(t["pnl"] for t in pattern_trades)
    print(f"\nPattern leg (standalone): {len(pattern_trades)} trades  PnL={pat_pnl:+.1f}")
    print("Running energetic fallback leg…")
    all_trades = simulate_hybrid_two_pass(
        sim_df,
        pattern_trades,
        ENERGETIC_EXECUTION_CONFIG["tp"],
        ENERGETIC_EXECUTION_CONFIG["sl"],
        ENERGETIC_EXECUTION_CONFIG["horizon"],
        config=en_exec_cfg,
        weak_period_cells=weak_cells,
    )
else:
    sim_df["side_signal"] = sim_df["pattern_side"]
    all_trades = simulate_v13_core(
        sim_df,
        EXECUTION_CONFIG["tp"],
        EXECUTION_CONFIG["sl"],
        EXECUTION_CONFIG["horizon"],
        config=pat_exec_cfg,
        weak_period_cells=weak_cells,
    )

if not all_trades:
    print("Signals found but no trades closed.")
    sys.exit(0)

tdf = pd.DataFrame(all_trades)
wins = int((tdf["pnl"] > 0).sum())
net_pnl = float(tdf["pnl"].sum())
wr = wins / len(tdf) * 100
cum = tdf["pnl"].cumsum()
max_dd = float((cum - cum.cummax()).min())

print(f"\n{'='*60}")
print(f"  COMBINED {'HYBRID ' if _hybrid['enabled'] else ''}BACKTEST  |  {bt_start} → {bt_end}")
print(f"{'='*60}")
print(f"  Trades   : {len(tdf)}")
print(f"  Win rate : {wr:.1f}%")
print(f"  Net PnL  : {net_pnl:+.2f}")
print(f"  Avg/trade: {net_pnl/len(tdf):+.2f}")
print(f"  Max DD   : {max_dd:+.2f}")

out_path = PROJECT_ROOT / "runtime" / "v14_pattern_backtest_trades.csv"
tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
signal_ts = tdf["entry_time"] - pd.Timedelta(minutes=1)
if "matched_pattern" in tdf.columns:
    tdf["pattern"] = tdf["matched_pattern"]
else:
    tdf["pattern"] = sim_df["matched_pattern"].reindex(signal_ts).values
if "source" in tdf.columns:
    tdf.loc[tdf["source"] == "energetic", "pattern"] = "energetic"
else:
    tdf["pattern"] = sim_df["matched_pattern"].reindex(signal_ts).values
tdf.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")

print("\n  By pattern:")
for name, grp in tdf.groupby("pattern", dropna=False):
    label = name if pd.notna(name) else "unknown"
    print(
        f"    {label:20s}: {len(grp):4d} trades  "
        f"PnL={grp['pnl'].sum():+.1f}  WR={(grp['pnl']>0).mean()*100:.0f}%  "
        f"avg={grp['pnl'].mean():+.2f}"
    )
if "source" in tdf.columns:
    print("\n  By source:")
    for src, grp in tdf.groupby("source", dropna=False):
        print(
            f"    {str(src):10s}: {len(grp):4d} trades  "
            f"PnL={grp['pnl'].sum():+.1f}  WR={(grp['pnl']>0).mean()*100:.0f}%"
        )

long_t = tdf[tdf["side"] == 1]
short_t = tdf[tdf["side"] == -1]
if len(long_t):
    print(f"\n  LONG total : {len(long_t):4d}  PnL={long_t['pnl'].sum():+.1f}  WR={(long_t['pnl']>0).mean()*100:.0f}%")
if len(short_t):
    print(f"  SHORT total: {len(short_t):4d}  PnL={short_t['pnl'].sum():+.1f}  WR={(short_t['pnl']>0).mean()*100:.0f}%")

print("\n  Last 10 trades (HKT):")
tdf2 = tdf.copy()
tdf2["entry_hkt"] = pd.to_datetime(tdf2["entry_time"]).dt.tz_convert("Asia/Hong_Kong")
for _, r in tdf2.tail(10).iterrows():
    d = "LONG" if r["side"] == 1 else "SHORT"
    print(
        f"  {r['entry_hkt'].strftime('%m-%d %H:%M')} {d:5s} pnl={r['pnl']:+.2f}  "
        f"pattern={r.get('pattern', '?')}"
    )
