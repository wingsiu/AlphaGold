"""v16 gold hybrid legs — pattern router + energetic fallback (standalone, no v15/)."""
from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from backtest.core import simulate_hybrid_two_pass, simulate_v13_core
from config.hybrid_config import (
    ENERGETIC_EXECUTION_CONFIG,
    EXECUTION_CONFIG,
    TIME_FILTER_CONFIG,
    WF_CONFIG,
)
from config.pattern_registry import (
    PATTERN_MODEL_DIR,
    PATTERN_REGISTRY,
    PRODUCTION_PATTERNS,
    backtest_feature_set,
    collect_pa_groups,
    pattern_prob_override,
)
from v16.gold.merge import trade_row
from v16.gold.prepare import prepare_gold_hybrid_data, score_energetic_signals
from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold
from xgboost_filter_model.energetic_gate import apply_pattern_gates, hybrid_config, pattern_gate_config
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
from xgboost_filter_model.time_slot_filter import (
    CycleWeakFilter,
    load_weak_filter,
    resolve_v14_time_filter_path,
)
from xgboost_filter_model.train_filter_1min import load_price_data
from v16._paths import PROJECT_ROOT


def _utc_ts(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _resolve_weak_filter():
    if not TIME_FILTER_CONFIG.get("enabled"):
        return None
    if os.environ.get("V14_NO_TIME_FILTER", "").strip().lower() in ("1", "true", "yes", "on"):
        return None
    if os.environ.get("V14_TIME_FILTER_JSON", "").strip():
        return load_weak_filter(os.environ["V14_TIME_FILTER_JSON"].strip())
    path = resolve_v14_time_filter_path(PROJECT_ROOT)
    if path:
        fallback = __import__("pathlib").Path(path)
        return CycleWeakFilter(PROJECT_ROOT, fallback_path=fallback)
    return None


def run_hybrid_legs(
    start: str,
    end: str,
    *,
    pattern_filter: Optional[list[str]] = None,
    verbose: bool = True,
) -> list[dict]:
    """Run pattern-first hybrid backtest; return normalized trade dicts."""
    os.environ.setdefault("V14_HYBRID", "1")
    _hybrid = hybrid_config()

    bt_start = start
    bt_end = end
    bt_start_dt = _utc_ts(bt_start)
    bt_end_dt = _utc_ts(bt_end.split("T")[0] if "T" in bt_end else bt_end) + pd.Timedelta(days=1)
    bt_start_date = bt_start.split("T")[0] if "T" in bt_start else bt_start
    bt_end_date = bt_end.split("T")[0] if "T" in bt_end else bt_end

    pattern_filter = pattern_filter or list(PRODUCTION_PATTERNS)
    active_patterns = {k: PATTERN_REGISTRY[k] for k in pattern_filter if k in PATTERN_REGISTRY}

    warmup_days = int(WF_CONFIG.get("feature_warmup_days", 120))
    load_start_dt = max(_utc_ts(WF_CONFIG["full_start"]), bt_start_dt - pd.Timedelta(days=warmup_days))
    load_start = load_start_dt.strftime("%Y-%m-%d")
    load_end = (_utc_ts(bt_end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if verbose:
        print(f"  Hybrid load {load_start} → {bt_end}…")

    df = prepare_gold_hybrid_data(
        start_date=load_start,
        end_date=load_end,
        pa_groups=collect_pa_groups(list(active_patterns.keys())),
        pattern_feature_set=backtest_feature_set(),
    )
    df_test = df[df.index >= bt_start_dt].copy()
    if df_test.empty:
        return []

    df_test = assign_patterns(df_test)
    drop = df_test["pattern_name"].notna() & ~df_test["pattern_name"].isin(pattern_filter)
    df_test.loc[drop, "pattern_name"] = pd.NA
    df_test.loc[drop, "pattern_id"] = -1

    models: dict[str, dict] = {}
    for name in list(active_patterns.keys()):
        spec = active_patterns[name]
        ex = spec["execution"]
        tp, sl = execution_tp_sl(ex)
        mode = execution_target_mode(ex)
        variant = pattern_variant_tag(ex["horizon"], tp, sl, target_mode=mode)
        pdir = PATTERN_MODEL_DIR / name / variant
        mp = prod_model_path(pdir)
        if not mp.exists():
            active_patterns.pop(name, None)
            continue
        models[name] = {"prod": joblib.load(mp), "spec": spec, "dir": pdir, "variant": variant}

    if not models:
        return []

    end_dt = _utc_ts(bt_end_date) + pd.Timedelta(days=1)
    gate_mask = pd.Series(True, index=df_test.index) if _hybrid["enabled"] else apply_pattern_gates(df_test, bt_start_dt, end_dt)

    for col in (
        "prob", "pattern_side", "s1_prob", "s2_prob", "side_signal",
        "matched_pattern", "exec_tp", "exec_sl", "exec_horizon",
    ):
        if col == "matched_pattern":
            df_test[col] = pd.NA
        elif col in ("pattern_side", "side_signal"):
            df_test[col] = 0
        else:
            df_test[col] = np.nan

    wf_anchor = wf_anchor_ts()
    fixed_cycle = fixed_wf_cycle_from_env()
    cycle_iter = [(fixed_cycle[0], bt_start_dt, end_dt)] if fixed_cycle else list(iter_wf_cycles(bt_start_dt, end_dt, wf_anchor))

    for cycle, current_start, current_end in cycle_iter:
        chunk = (df_test.index >= current_start) & (df_test.index < current_end)
        if not chunk.any():
            continue
        for name, m in models.items():
            pat_chunk = chunk & (df_test["pattern_name"] == name) & gate_mask
            if not pat_chunk.any():
                continue
            model_start = fixed_cycle[1].date() if fixed_cycle else current_start.date()
            path = cycle_model_path(m["dir"], cycle, model_start)
            model = joblib.load(path) if path.exists() else m["prod"]
            model_feats = list(model.feature_names_in_)
            spec = m["spec"]
            ex = spec["execution"]
            prob_thresh = spec["thresholds"]["prob"]
            override = pattern_prob_override()
            if override is not None:
                prob_thresh = override
            bias = spec["direction_bias"]
            rows = df_test.loc[pat_chunk]
            p = model.predict_proba(rows[model_feats])[:, 1]
            df_test.loc[pat_chunk, "prob"] = p
            if not pattern_gate_config()["s1_gate"]:
                df_test.loc[pat_chunk, "s1_prob"] = p
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
        score_energetic_signals(df_test, bt_start_dt, end_dt)
    else:
        df_test["energetic_side"] = 0

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
    sim_df["pattern_side"] = sim_df.get("pattern_side", df_test["side_signal"]).fillna(0).astype(int)
    sim_df["energetic_side"] = sim_df.get("energetic_side", pd.Series(0, index=sim_df.index)).fillna(0).astype(int)

    weak_cells = _resolve_weak_filter()
    pat_exec_cfg = EXECUTION_CONFIG.copy()
    pat_exec_cfg["close_on_reverse"] = _hybrid["pattern_close_on_reverse"] if _hybrid["enabled"] else EXECUTION_CONFIG.get("close_on_reverse", False)
    pat_exec_cfg["same_dir_refresh"] = _hybrid["pattern_same_dir_refresh"] if _hybrid["enabled"] else EXECUTION_CONFIG.get("same_dir_refresh", "entry")
    pat_exec_cfg["upgrade_stop"] = _hybrid["pattern_upgrade_stop"] if _hybrid["enabled"] else EXECUTION_CONFIG.get("upgrade_stop", False)

    pat_sig_count = (df_test["pattern_side"] != 0).sum()
    en_sig_count = int((df_test.get("energetic_side", 0) != 0).sum()) if _hybrid["enabled"] else 0

    if _hybrid["enabled"] and (pat_sig_count > 0 or en_sig_count > 0):
        en_exec_cfg = ENERGETIC_EXECUTION_CONFIG.copy()
        en_exec_cfg["close_on_reverse"] = _hybrid["energetic_close_on_reverse"]
        en_exec_cfg["same_dir_refresh"] = _hybrid["energetic_same_dir_refresh"]
        en_exec_cfg["upgrade_stop"] = _hybrid["energetic_upgrade_stop"]
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

    out = []
    for t in all_trades:
        src = str(t.get("source", "pattern"))
        pat = str(t.get("matched_pattern", src))
        typ = pat if src == "pattern" and pat not in ("nan", "None", "NoneType") else src
        out.append(
            trade_row(
                t["entry_time"],
                t["exit_time"],
                t["pnl"],
                typ,
                side=int(t["side"]),
                typ=typ,
            )
        )
    return out
