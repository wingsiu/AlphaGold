#!/usr/bin/env python3
"""
AlphaGold v15 Backtest — Pattern-First Hybrid Router (Copy of v14)
===================================================================
Copied from v14/backtest/backtest_patterns_v14.py.
Pattern-specialist router (6 production patterns) + energetic S1/S2 fallback.
Same as v14 — no modifications to strategy logic, only paths adapted for v15.

Usage:
  V14_HYBRID=1 python3 v15/backtest/backtest_v15.py 2025-06-01 2026-06-09
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from backtest.core import simulate_hybrid_two_pass, simulate_v13_core
from backtest.trade_display import print_trades_table_hkt
from config.hybrid_config import EXECUTION_CONFIG, ENERGETIC_EXECUTION_CONFIG, TIME_FILTER_CONFIG, WF_CONFIG
from config.pattern_registry import PATTERN_MODEL_DIR, PATTERN_REGISTRY, PRODUCTION_PATTERNS, collect_pa_groups, backtest_feature_set, pattern_prob_override
from xgboost_filter_model.energetic_gate import (
    apply_pattern_gates,
    hybrid_config,
    pattern_gate_config,
)
from xgboost_filter_model.pattern_router import assign_patterns
from xgboost_filter_model.pattern_training import (
    assign_exec_tp_sl,
    cycle_model_path,
    execution_target_mode,
    execution_tp_sl,
    feature_columns,
    fixed_wf_cycle_from_env,
    iter_wf_cycles,
    pattern_variant_tag,
    prod_model_path,
    wf_anchor_ts,
)
from xgboost_filter_model.time_slot_filter import load_weak_filter, resolve_v14_time_filter_path
from xgboost_filter_model.train_filter_1min import load_price_data

from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from v15.backtest.prepare_v15 import score_energetic_signals_v15


# ── Date helpers ──────────────────────────────────────────────────────────

def _utc_ts(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault("V14_HYBRID", "1")

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

    bt_start_dt = _utc_ts(bt_start)
    bt_end_dt = _utc_ts(bt_end.split("T")[0] if "T" in bt_end else bt_end) + pd.Timedelta(days=1)

    _hybrid = hybrid_config()
    print(f"\n{'='*70}")
    print(f"  AlphaGold v15 Backtest (v14 copy)  |  {bt_start} → {bt_end}")
    print(f"  Patterns: {', '.join(pattern_filter)}")
    print(f"  Hybrid: {'ENABLED' if _hybrid['enabled'] else 'DISABLED'}")
    print(f"{'='*70}\n")

    warmup_days = int(WF_CONFIG.get("feature_warmup_days", 120))
    load_start_dt = max(
        _utc_ts(WF_CONFIG["full_start"]),
        bt_start_dt - pd.Timedelta(days=warmup_days),
    )
    load_start = load_start_dt.strftime("%Y-%m-%d")
    bt_end_date = bt_end.split("T")[0] if "T" in bt_end else bt_end
    load_end = (_utc_ts(bt_end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bt_start_date = bt_start.split("T")[0] if "T" in bt_start else bt_start

    print(f"Loading data {load_start} → {bt_end}…")
    df = prepare_data_v14(
        start_date=load_start,
        end_date=load_end,
        energetic_filter=False,
        for_live_inference=True,
        pa_groups=collect_pa_groups(list(active_patterns.keys())),
        pattern_feature_set=backtest_feature_set(),
    )
    from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14
    df = prepare_directional_data_v14(df)
    feats = feature_columns(df)

    df_test = df[df.index >= bt_start_dt].copy()
    print(f"Bars in test window: {len(df_test)}")

    if df_test.empty:
        print("No bars in test window.")
        return

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
        tp, sl = execution_tp_sl(ex)
        mode = execution_target_mode(ex)
        variant = pattern_variant_tag(ex["horizon"], tp, sl, target_mode=mode)
        pdir = PATTERN_MODEL_DIR / name / variant
        mp = prod_model_path(pdir)
        if not mp.exists():
            print(f"WARNING: Missing model {mp} — skipping {name}")
            active_patterns.pop(name, None)
            pattern_filter.remove(name)
            continue
        models[name] = {
            "prod": joblib.load(mp),
            "spec": spec,
            "dir": pdir,
            "variant": variant,
        }
        print(f"  Model {name}: {variant}")

    if not models:
        print("No pattern models — exiting.")
        return

    wf_anchor = wf_anchor_ts()
    end_dt = _utc_ts(bt_end_date) + pd.Timedelta(days=1)

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

    fixed_cycle = fixed_wf_cycle_from_env()
    if fixed_cycle:
        pin_cycle, pin_start = fixed_cycle
        cycle_iter = [(pin_cycle, bt_start_dt, end_dt)]
    else:
        cycle_iter = list(iter_wf_cycles(bt_start_dt, end_dt, wf_anchor))

    for cycle, current_start, current_end in cycle_iter:
        chunk = (df_test.index >= current_start) & (df_test.index < current_end)
        if not chunk.any():
            continue

        for name, m in models.items():
            pat_chunk = chunk & (df_test["pattern_name"] == name) & gate_mask
            if not pat_chunk.any():
                continue

            model_start = pin_start.date() if fixed_cycle else current_start.date()
            path = cycle_model_path(m["dir"], cycle, model_start)
            model = joblib.load(path) if path.exists() else m["prod"]
            model_feats = list(model.feature_names_in_)
            spec = m["spec"]
            ex = spec["execution"]
            prob_thresh = spec["thresholds"]["prob"]
            _override = pattern_prob_override()
            if _override is not None:
                prob_thresh = _override
            bias = spec["direction_bias"]

            rows = df_test.loc[pat_chunk]
            p = model.predict_proba(rows[model_feats])[:, 1]
            df_test.loc[pat_chunk, "prob"] = p
            _gate = pattern_gate_config()
            if not _gate["s1_gate"]:
                df_test.loc[pat_chunk, "s1_prob"] = p

            from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold
            adaptive_thresh = adaptive_prob_threshold(prob_thresh, df_test)
            sig = pat_chunk & (df_test["prob"] >= adaptive_thresh)
            side = 1 if bias == "long" else -1
            df_test.loc[sig, "side_signal"] = side
            if bias == "long":
                df_test.loc[sig, "s2_prob"] = df_test.loc[sig, "prob"]
            else:
                df_test.loc[sig, "s2_prob"] = 1.0 - df_test.loc[sig, "prob"]
            assign_exec_tp_sl(df_test, df_test.index[sig], ex)
            df_test.loc[sig, "exec_horizon"] = ex["horizon"]
            fired = sig & df_test["matched_pattern"].isna()
            df_test.loc[fired, "matched_pattern"] = name

    df_test["pattern_side"] = df_test["side_signal"].astype(int)

    if _hybrid["enabled"]:
        print("\nScoring v15 energetic fallback (S1/S2, deterministic gate, no HMM)…")
        score_energetic_signals_v15(df_test, bt_start_dt, end_dt)
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

    print("Loading full 1-min bars for exit simulation…")
    raw_df = load_price_data(start_date=bt_start_date, end_date=load_end)
    raw_df = raw_df[raw_df.index >= bt_start_dt].copy()

    sim_df = raw_df[["open", "high", "low", "close"]].copy()
    merge_cols = (
        "pattern_side", "energetic_side", "s1_prob", "s2_prob",
        "energetic_s1_prob", "energetic_s2_prob", "matched_pattern",
        "exec_tp", "exec_sl", "exec_horizon",
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

    if _hybrid["enabled"] and (pat_sig_count > 0 or en_sig_count > 0):
        en_exec_cfg = ENERGETIC_EXECUTION_CONFIG.copy()
        en_exec_cfg["close_on_reverse"] = _hybrid["energetic_close_on_reverse"]
        en_exec_cfg["same_dir_refresh"] = _hybrid["energetic_same_dir_refresh"]
        en_exec_cfg["upgrade_stop"] = _hybrid["energetic_upgrade_stop"]

        print("Running hybrid two-pass simulation (pattern-first, energetic fallback)…")
        sim_df["side_signal"] = sim_df["pattern_side"]
        pattern_trades = simulate_v13_core(
            sim_df,
            ENERGETIC_EXECUTION_CONFIG["tp"],
            ENERGETIC_EXECUTION_CONFIG["sl"],
            ENERGETIC_EXECUTION_CONFIG["horizon"],
            config=pat_exec_cfg,
            weak_period_cells=weak_cells,
        )
        all_trades = simulate_hybrid_two_pass(
            sim_df,
            pattern_trades,
            ENERGETIC_EXECUTION_CONFIG["tp"],
            ENERGETIC_EXECUTION_CONFIG["sl"],
            ENERGETIC_EXECUTION_CONFIG["horizon"],
            config=en_exec_cfg,
            weak_period_cells=weak_cells,
        )
    elif pat_sig_count > 0:
        sim_df["side_signal"] = sim_df["pattern_side"]
        all_trades = simulate_v13_core(
            sim_df,
            EXECUTION_CONFIG["tp"],
            EXECUTION_CONFIG["sl"],
            EXECUTION_CONFIG["horizon"],
            config=pat_exec_cfg,
            weak_period_cells=weak_cells,
        )
    else:
        all_trades = []

    if not all_trades:
        print("No trades generated.")
        return

    tdf = pd.DataFrame(all_trades)
    tdf["pnl"] = tdf["pnl"].astype(float)
    wins = int((tdf["pnl"] > 0).sum())
    net_pnl = float(tdf["pnl"].sum())
    wr = wins / len(tdf) * 100
    cum = tdf["pnl"].cumsum()
    max_dd = float((cum - cum.cummax()).min())

    print(f"\n{'='*70}")
    print(f"  v15 HYBRID BACKTEST  |  {bt_start} → {bt_end}")
    print(f"{'='*70}")
    print(f"  Total Trades: {len(tdf)}")
    print(f"  Net PnL     : {net_pnl:+.2f} pts")
    print(f"  Win Rate    : {wr:.1f}%")
    print(f"  Avg/trade   : {net_pnl/len(tdf):+.2f}")
    print(f"  Max DD      : {max_dd:+.2f}")

    out_path = PROJECT_ROOT / "runtime" / "v15_backtest_trades.csv"
    tdf.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    if "source" in tdf.columns:
        print("\n  By source:")
        for src_name, grp in tdf.groupby("source", dropna=False):
            label = src_name if pd.notna(src_name) else "unknown"
            src_pnl = float(grp["pnl"].sum())
            src_wr = (grp["pnl"] > 0).mean() * 100
            print(f"    {label:15s}: {len(grp):4d} trades  "
                  f"PnL={src_pnl:+.1f}  WR={src_wr:.0f}%  avg={src_pnl/len(grp):+.2f}")

    if "matched_pattern" in tdf.columns:
        print("\n  By matched_pattern:")
        for pat_name, grp in tdf.groupby("matched_pattern", dropna=False):
            label = pat_name if pd.notna(pat_name) else "unknown"
            pat_pnl = float(grp["pnl"].sum())
            pat_wr = (grp["pnl"] > 0).mean() * 100
            print(f"    {label:20s}: {len(grp):4d} trades  "
                  f"PnL={pat_pnl:+.1f}  WR={pat_wr:.0f}%  avg={pat_pnl/len(grp):+.2f}")

    long_t = tdf[tdf["side"] == 1] if "side" in tdf.columns else pd.DataFrame()
    short_t = tdf[tdf["side"] == -1] if "side" in tdf.columns else pd.DataFrame()
    if len(long_t):
        print(f"\n  LONG total : {len(long_t):4d}  PnL={long_t['pnl'].sum():+.1f}  "
              f"WR={(long_t['pnl']>0).mean()*100:.0f}%")
    if len(short_t):
        print(f"  SHORT total: {len(short_t):4d}  PnL={short_t['pnl'].sum():+.1f}  "
              f"WR={(short_t['pnl']>0).mean()*100:.0f}%")

    show_all = len(date_args) == 1 and date_args[0].isdigit()
    print_trades_table_hkt(tdf, show_all=show_all)


if __name__ == '__main__':
    main()
