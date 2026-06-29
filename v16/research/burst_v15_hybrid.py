#!/usr/bin/env python3
"""Burst entries scored with v15 S1/S2 + v15 hybrid execution."""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("V14_HYBRID", "1")

import numpy as np
import pandas as pd

from backtest.core import simulate_hybrid_two_pass, simulate_v13_core
from config.hybrid_config import ENERGETIC_EXECUTION_CONFIG, EXECUTION_CONFIG, TIME_FILTER_CONFIG, WF_CONFIG, hybrid_config
from config.pattern_registry import PRODUCTION_PATTERNS, PATTERN_REGISTRY, collect_pa_groups, backtest_feature_set
from v16.backtest.features import build_features
from v16.backtest.signals import burst_mask
from v16.data.load_gold import load_gold_1m
from v15.backtest.prepare_v15 import prepare_v15_data, score_energetic_signals_v15
from xgboost_filter_model.energetic_gate import apply_pattern_gates, pattern_gate_config
from xgboost_filter_model.pattern_router import assign_patterns
from xgboost_filter_model.pattern_training import (
    assign_exec_tp_sl,
    cycle_model_path,
    execution_tp_sl,
    execution_target_mode,
    fixed_wf_cycle_from_env,
    iter_wf_cycles,
    pattern_variant_tag,
    prod_model_path,
    wf_anchor_ts,
)
from xgboost_filter_model.time_slot_filter import CycleWeakFilter, load_weak_filter, resolve_v14_time_filter_path
from xgboost_filter_model.train_filter_1min import load_price_data
from config.pattern_registry import PATTERN_MODEL_DIR, pattern_prob_override
from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold
import joblib

from v16._paths import PROJECT_ROOT


def _utc_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _score_v15(df_test: pd.DataFrame, bt_start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> None:
    """Pattern + energetic scoring (same as backtest_v15)."""
    _hybrid = hybrid_config()
    active_patterns = {k: PATTERN_REGISTRY[k] for k in PRODUCTION_PATTERNS}
    models: dict = {}
    for name in list(active_patterns):
        spec = active_patterns[name]
        ex = spec["execution"]
        tp, sl = execution_tp_sl(ex)
        variant = pattern_variant_tag(ex["horizon"], tp, sl, target_mode=execution_target_mode(ex))
        mp = prod_model_path(PATTERN_MODEL_DIR / name / variant)
        if not mp.exists():
            active_patterns.pop(name, None)
            continue
        models[name] = {"prod": joblib.load(mp), "spec": spec, "dir": PATTERN_MODEL_DIR / name / variant, "variant": variant}

    gate_mask = pd.Series(True, index=df_test.index)
    df_test["prob"] = np.nan
    df_test["pattern_side"] = 0
    df_test["side_signal"] = 0
    df_test["matched_pattern"] = pd.NA
    df_test["exec_tp"] = np.nan
    df_test["exec_sl"] = np.nan
    df_test["exec_horizon"] = np.nan

    wf_anchor = wf_anchor_ts()
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
            prob_thresh = pattern_prob_override() or spec["thresholds"]["prob"]
            bias = spec["direction_bias"]
            rows = df_test.loc[pat_chunk]
            p = model.predict_proba(rows[model_feats])[:, 1]
            df_test.loc[pat_chunk, "prob"] = p
            adaptive_thresh = adaptive_prob_threshold(prob_thresh, df_test)
            sig = pat_chunk & (df_test["prob"] >= adaptive_thresh)
            side = 1 if bias == "long" else -1
            df_test.loc[sig, "side_signal"] = side
            assign_exec_tp_sl(df_test, df_test.index[sig], ex)
            df_test.loc[sig, "exec_horizon"] = ex["horizon"]
            fired = sig & df_test["matched_pattern"].isna()
            df_test.loc[fired, "matched_pattern"] = name

    df_test["pattern_side"] = df_test["side_signal"].astype(int)
    if _hybrid["enabled"]:
        score_energetic_signals_v15(df_test, bt_start_dt, end_dt)
    else:
        df_test["energetic_side"] = 0


def _apply_burst_gate(df_test: pd.DataFrame, df_raw: pd.DataFrame, mode: str) -> pd.DataFrame:
    feats = build_features(df_raw)
    burst = burst_mask(feats, df_raw.index)
    burst_on_test = burst.reindex(df_test.index, fill_value=False)
    out = df_test.copy()
    if mode == "burst_only":
        # Keep pattern/energetic signal only on burst bars
        no_burst = ~burst_on_test
        out.loc[no_burst, "pattern_side"] = 0
        out.loc[no_burst, "energetic_side"] = 0
        out.loc[no_burst, "matched_pattern"] = pd.NA
    elif mode == "burst_energetic":
        # Energetic must also be burst; patterns unchanged
        no_burst = ~burst_on_test
        out.loc[no_burst, "energetic_side"] = 0
    return out


def _simulate(df_test: pd.DataFrame, bt_start: str, bt_end: str) -> pd.DataFrame:
    _hybrid = hybrid_config()
    bt_start_dt = _utc_ts(bt_start)
    bt_end_date = bt_end.split("T")[0]
    load_end = (_utc_ts(bt_end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw_df = load_price_data(start_date=bt_start, end_date=load_end)
    raw_df = raw_df[raw_df.index >= bt_start_dt].copy()
    sim_df = raw_df[["open", "high", "low", "close"]].copy()
    for col in (
        "pattern_side", "energetic_side", "s1_prob", "s2_prob",
        "energetic_s1_prob", "energetic_s2_prob", "matched_pattern",
        "exec_tp", "exec_sl", "exec_horizon",
    ):
        sim_df[col] = df_test[col] if col in df_test.columns else np.nan
    sim_df["pattern_side"] = sim_df["pattern_side"].fillna(0).astype(int)
    sim_df["energetic_side"] = sim_df.get("energetic_side", pd.Series(0, index=sim_df.index)).fillna(0).astype(int)

    weak_cells = None
    if TIME_FILTER_CONFIG.get("enabled"):
        _filter_path = resolve_v14_time_filter_path(PROJECT_ROOT)
        if _filter_path:
            weak_cells = CycleWeakFilter(PROJECT_ROOT, fallback_path=Path(_filter_path))

    pat_exec = EXECUTION_CONFIG.copy()
    pat_exec["close_on_reverse"] = _hybrid["pattern_close_on_reverse"]
    pat_exec["same_dir_refresh"] = _hybrid["pattern_same_dir_refresh"]
    pat_exec["upgrade_stop"] = _hybrid["pattern_upgrade_stop"]

    en_exec = ENERGETIC_EXECUTION_CONFIG.copy()
    en_exec["close_on_reverse"] = _hybrid["energetic_close_on_reverse"]
    en_exec["same_dir_refresh"] = _hybrid["energetic_same_dir_refresh"]
    en_exec["upgrade_stop"] = _hybrid["energetic_upgrade_stop"]

    pat_sig = (sim_df["pattern_side"] != 0).sum()
    en_sig = (sim_df["energetic_side"] != 0).sum()
    if _hybrid["enabled"] and (pat_sig > 0 or en_sig > 0):
        pattern_trades = simulate_v13_core(
            sim_df, pat_exec["tp"], pat_exec["sl"], pat_exec["horizon"],
            config=pat_exec, weak_period_cells=weak_cells,
        )
        all_trades = simulate_hybrid_two_pass(
            sim_df, pat_exec["tp"], pat_exec["sl"], pat_exec["horizon"],
            pattern_trades, config=en_exec, pattern_config=pat_exec,
            weak_period_cells=weak_cells,
        )
    else:
        all_trades = simulate_v13_core(
            sim_df, pat_exec["tp"], pat_exec["sl"], pat_exec["horizon"],
            config=pat_exec, weak_period_cells=weak_cells,
        )
    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def _print(name: str, tdf: pd.DataFrame, baseline: float) -> dict:
    if tdf.empty:
        print(f"{name}: no trades")
        return {"model": name, "trades": 0, "wr": 0, "net": 0, "avg": 0}
    tdf = tdf.copy()
    tdf["win"] = tdf["pnl"] > 0
    s = {
        "model": name,
        "trades": len(tdf),
        "wr": round(tdf["win"].mean() * 100, 1),
        "net": round(tdf["pnl"].sum(), 1),
        "avg": round(tdf["pnl"].mean(), 2),
    }
    flag = " *** BEATS v15 ***" if s["net"] > baseline else ""
    print(f"{name}: {s['trades']} trades  WR={s['wr']}%  net={s['net']:+.1f}  avg={s['avg']:+.2f}{flag}")
    return s


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    bt_start = args[0] if args else "2025-06-01"
    bt_end = args[1] if len(args) > 1 else date.today().strftime("%Y-%m-%d")
    baseline = 4282.0

    bt_start_dt = _utc_ts(bt_start)
    end_dt = _utc_ts(bt_end) + pd.Timedelta(days=1)
    warmup = int(WF_CONFIG.get("feature_warmup_days", 120))
    load_start = (bt_start_dt - pd.Timedelta(days=warmup)).strftime("%Y-%m-%d")
    load_end = end_dt.strftime("%Y-%m-%d")

    print("=" * 70)
    print(f"  Burst × v15 hybrid  |  {bt_start} → {bt_end}")
    print("=" * 70)

    df = prepare_v15_data(load_start, load_end, pa_groups=collect_pa_groups(), pattern_feature_set=backtest_feature_set())
    df_test = df[df.index >= bt_start_dt].copy()
    df_test = assign_patterns(df_test)

    df_raw = load_gold_1m(bt_start, bt_end)

    _score_v15(df_test, bt_start_dt, end_dt)

    results = []
    # v15 reference (re-sim)
    results.append(_print("v15 full (re-sim)", _simulate(df_test, bt_start, bt_end), baseline))

    for mode, label in [
        ("burst_only", "v15 gated: burst bars only"),
        ("burst_energetic", "v15: burst∩energetic fallback"),
    ]:
        gated = _apply_burst_gate(df_test, df_raw, mode)
        results.append(_print(label, _simulate(gated, bt_start, bt_end), baseline))

    pd.DataFrame(results).to_csv(PROJECT_ROOT / "runtime" / "v16_burst_v15_hybrid.csv", index=False)


if __name__ == "__main__":
    main()
