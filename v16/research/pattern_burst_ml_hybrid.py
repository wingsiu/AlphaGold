#!/usr/bin/env python3
"""
v15 patterns + energetic + burst ML union fallback.

Winner (Jun 2025→Jun 2026): UNION at p>=0.65 → +4413 vs v15 +4282.
Burst ML only fills bars where pattern and energetic did not fire.
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("V14_HYBRID", "1")

import joblib
import numpy as np
import pandas as pd

from backtest.core import simulate_hybrid_two_pass, simulate_v13_core
from config.hybrid_config import (
    ENERGETIC_EXECUTION_CONFIG,
    EXECUTION_CONFIG,
    HYBRID_CONFIG,
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
from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features, feature_columns
from v16.backtest.ml import walk_forward_dual_v15_exit
from v16.backtest.signals import build_labeled_set
from v16.config.v16_config import ML_CONFIG, UNION_FALLBACK_CONFIG
import v16.config.v16_config as v16_config
from v16.data.load_gold import load_gold_1m
from v15.backtest.prepare_v15 import prepare_v15_data, score_energetic_signals_v15
from xgboost_filter_model.adaptive_prob import adaptive_prob_threshold
from xgboost_filter_model.pattern_router import assign_patterns
from xgboost_filter_model.pattern_training import (
    assign_exec_tp_sl,
    cycle_model_path,
    execution_target_mode,
    execution_tp_sl,
    fixed_wf_cycle_from_env,
    iter_wf_cycles,
    pattern_variant_tag,
    prod_model_path,
    wf_anchor_ts,
)
from xgboost_filter_model.time_slot_filter import CycleWeakFilter, resolve_v14_time_filter_path
from xgboost_filter_model.train_filter_1min import load_price_data

V15_BASELINE_NET = 4282.0


def _utc_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _score_patterns(df_test: pd.DataFrame, bt_start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> None:
    active = {k: PATTERN_REGISTRY[k] for k in PRODUCTION_PATTERNS}
    models: dict = {}
    for name in list(active):
        spec = active[name]
        ex = spec["execution"]
        tp, sl = execution_tp_sl(ex)
        variant = pattern_variant_tag(ex["horizon"], tp, sl, target_mode=execution_target_mode(ex))
        mp = prod_model_path(PATTERN_MODEL_DIR / name / variant)
        if not mp.exists():
            active.pop(name, None)
            continue
        models[name] = {"prod": joblib.load(mp), "spec": spec, "dir": PATTERN_MODEL_DIR / name / variant}

    df_test["prob"] = np.nan
    df_test["pattern_side"] = 0
    df_test["side_signal"] = 0
    df_test["s1_prob"] = np.nan
    df_test["s2_prob"] = np.nan
    df_test["matched_pattern"] = pd.NA
    df_test["exec_tp"] = np.nan
    df_test["exec_sl"] = np.nan
    df_test["exec_horizon"] = np.nan
    df_test["energetic_side"] = 0
    df_test["energetic_s2_prob"] = np.nan

    wf_anchor = wf_anchor_ts()
    fixed_cycle = fixed_wf_cycle_from_env()
    cycle_iter = (
        [(fixed_cycle[0], bt_start_dt, end_dt)]
        if fixed_cycle
        else list(iter_wf_cycles(bt_start_dt, end_dt, wf_anchor))
    )

    for cycle, current_start, current_end in cycle_iter:
        chunk = (df_test.index >= current_start) & (df_test.index < current_end)
        if not chunk.any():
            continue
        for name, m in models.items():
            pat_chunk = chunk & (df_test["pattern_name"] == name)
            if not pat_chunk.any():
                continue
            pin_start = fixed_cycle[1] if fixed_cycle else current_start
            model_start = pin_start.date() if fixed_cycle else current_start.date()
            path = cycle_model_path(m["dir"], cycle, model_start)
            model = joblib.load(path) if path.exists() else m["prod"]
            feats = list(model.feature_names_in_)
            spec = m["spec"]
            ex = spec["execution"]
            prob_thresh = pattern_prob_override() or spec["thresholds"]["prob"]
            bias = spec["direction_bias"]
            rows = df_test.loc[pat_chunk]
            p = model.predict_proba(rows[feats])[:, 1]
            df_test.loc[pat_chunk, "prob"] = p
            sig = pat_chunk & (df_test["prob"] >= adaptive_prob_threshold(prob_thresh, df_test))
            side = 1 if bias == "long" else -1
            df_test.loc[sig, "side_signal"] = side
            df_test.loc[sig, "s1_prob"] = df_test.loc[sig, "prob"]
            if bias == "long":
                df_test.loc[sig, "s2_prob"] = df_test.loc[sig, "prob"]
            else:
                df_test.loc[sig, "s2_prob"] = 1.0 - df_test.loc[sig, "prob"]
            assign_exec_tp_sl(df_test, df_test.index[sig], ex)
            df_test.loc[sig, "exec_horizon"] = ex["horizon"]
            fired = sig & df_test["matched_pattern"].isna()
            df_test.loc[fired, "matched_pattern"] = name

    df_test["pattern_side"] = df_test["side_signal"].astype(int)


def _inject_burst_union(
    df_test: pd.DataFrame,
    ml_trades: pd.DataFrame,
    *,
    score_energetic_fn,
    bt_start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> tuple[int, int]:
    """v15 energetic first; burst ML fills gaps (no pattern bar, no energetic)."""
    df_test["energetic_side"] = 0
    df_test["energetic_s2_prob"] = np.nan
    score_energetic_fn(df_test, bt_start_dt, end_dt)
    en_n = int((df_test["energetic_side"] != 0).sum())
    burst_n = 0
    for _, tr in ml_trades.iterrows():
        ts = tr["signal_ts"]
        if ts not in df_test.index:
            continue
        if int(df_test.loc[ts, "pattern_side"]) != 0:
            continue
        if int(df_test.loc[ts, "energetic_side"]) != 0:
            continue
        side = int(tr["side"])
        df_test.loc[ts, "energetic_side"] = side
        df_test.loc[ts, "energetic_s2_prob"] = 0.99 if side == 1 else 0.01
        burst_n += 1
    return en_n, burst_n
def _inject_burst_ml(df_test: pd.DataFrame, ml_trades: pd.DataFrame, *, only_if_no_pattern: bool) -> int:
    n = 0
    for _, tr in ml_trades.iterrows():
        ts = tr["signal_ts"]
        if ts not in df_test.index:
            continue
        if only_if_no_pattern and int(df_test.loc[ts, "pattern_side"]) != 0:
            continue
        side = int(tr["side"])
        df_test.loc[ts, "energetic_side"] = side
        df_test.loc[ts, "energetic_s2_prob"] = 0.99 if side == 1 else 0.01
        n += 1
    return n


def _run_union(
    df_test: pd.DataFrame,
    ml_trades: pd.DataFrame,
    bt_start: str,
    bt_end: str,
    bt_start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> dict:
    dt = df_test.copy()
    en_n, burst_n = _inject_burst_union(
        dt, ml_trades, score_energetic_fn=score_energetic_signals_v15,
        bt_start_dt=bt_start_dt, end_dt=end_dt,
    )
    tdf = _simulate(dt, bt_start, bt_end)
    return _stats(f"UNION pat+energetic+burstML (en={en_n}, burst+{burst_n})", tdf)


def _simulate(df_test: pd.DataFrame, bt_start: str, bt_end: str) -> pd.DataFrame:
    bt_start_dt = _utc_ts(bt_start)
    bt_end_date = bt_end.split("T")[0]
    load_end = (_utc_ts(bt_end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw_df = load_price_data(start_date=bt_start, end_date=load_end)
    raw_df = raw_df[raw_df.index >= bt_start_dt].copy()
    sim_df = raw_df[["open", "high", "low", "close"]].copy()
    for col in (
        "pattern_side", "energetic_side", "matched_pattern",
        "exec_tp", "exec_sl", "exec_horizon",
        "energetic_s2_prob", "s1_prob", "s2_prob",
    ):
        sim_df[col] = df_test.reindex(sim_df.index)[col]
    sim_df["pattern_side"] = sim_df["pattern_side"].fillna(0).astype(int)
    sim_df["energetic_side"] = sim_df["energetic_side"].fillna(0).astype(int)
    sim_df["side_signal"] = sim_df["pattern_side"]

    weak_cells = None
    if TIME_FILTER_CONFIG.get("enabled"):
        fp = resolve_v14_time_filter_path(PROJECT_ROOT)
        if fp:
            weak_cells = CycleWeakFilter(PROJECT_ROOT, fallback_path=Path(fp))

    pat_exec = EXECUTION_CONFIG.copy()
    pat_exec["close_on_reverse"] = HYBRID_CONFIG["pattern_close_on_reverse"]
    pat_exec["same_dir_refresh"] = HYBRID_CONFIG["pattern_same_dir_refresh"]
    pat_exec["upgrade_stop"] = HYBRID_CONFIG["pattern_upgrade_stop"]

    en_exec = ENERGETIC_EXECUTION_CONFIG.copy()
    en_exec["close_on_reverse"] = HYBRID_CONFIG["energetic_close_on_reverse"]
    en_exec["same_dir_refresh"] = HYBRID_CONFIG["energetic_same_dir_refresh"]
    en_exec["upgrade_stop"] = HYBRID_CONFIG["energetic_upgrade_stop"]

    pattern_trades = simulate_v13_core(
        sim_df,
        en_exec["tp"],
        en_exec["sl"],
        en_exec["horizon"],
        config=pat_exec,
        weak_period_cells=weak_cells,
    )
    all_trades = simulate_hybrid_two_pass(
        sim_df,
        pattern_trades,
        en_exec["tp"],
        en_exec["sl"],
        en_exec["horizon"],
        config=en_exec,
        weak_period_cells=weak_cells,
    )
    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def _stats(name: str, tdf: pd.DataFrame) -> dict:
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
    flag = " *** BEATS v15 ***" if s["net"] > V15_BASELINE_NET else ""
    print(
        f"{name}: {s['trades']} trades  WR={s['wr']}%  "
        f"net={s['net']:+.1f}  avg={s['avg']:+.2f}{flag}"
    )
    return s


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    bt_start = args[0] if args else "2025-06-01"
    bt_end = args[1] if len(args) > 1 else date.today().strftime("%Y-%m-%d")

    bt_start_dt = _utc_ts(bt_start)
    end_dt = _utc_ts(bt_end) + pd.Timedelta(days=1)
    warmup = int(WF_CONFIG.get("feature_warmup_days", 120))
    load_start = (bt_start_dt - pd.Timedelta(days=warmup)).strftime("%Y-%m-%d")

    print("=" * 70)
    print(f"  Pattern + burst ML fallback  |  {bt_start} → {bt_end}")
    print(f"  v15 baseline net: +{V15_BASELINE_NET:.0f}")
    print("=" * 70)

    df_v15 = prepare_v15_data(
        load_start, end_dt.strftime("%Y-%m-%d"),
        pa_groups=collect_pa_groups(),
        pattern_feature_set=backtest_feature_set(),
    )
    df_test = df_v15[df_v15.index >= bt_start_dt].copy()
    df_test = assign_patterns(df_test)
    _score_patterns(df_test, bt_start_dt, end_dt)

    df_raw = load_gold_1m(bt_start, bt_end)
    feats = build_features(df_raw)
    labeled = build_labeled_set(df_raw, feats)
    feat_cols = feature_columns(feats)

    u = UNION_FALLBACK_CONFIG
    old = dict(v16_config.ML_CONFIG)
    v16_config.ML_CONFIG["prob_threshold"] = u["burst_prob_threshold"]
    v16_config.ML_CONFIG["min_edge"] = u["burst_min_edge"]
    ml_union = walk_forward_dual_v15_exit(
        df_raw,
        labeled,
        feats,
        feat_cols,
        tp=u["burst_exit_tp"],
        sl=u["burst_exit_sl"],
        horizon=int(u["burst_exit_horizon"]),
    )
    v16_config.ML_CONFIG.update(old)

    results = []
    results.append(_run_union(df_test, ml_union, bt_start, bt_end, bt_start_dt, end_dt))
    results.append(_stats("v15 pat + energetic only (baseline)", _simulate(
        _inject_union_baseline(df_test.copy(), bt_start_dt, end_dt), bt_start, bt_end
    )))

    # Optional extended sweep with --sweep
    if "--sweep" in sys.argv:
        _run_sweep(df_test, df_raw, labeled, feats, feat_cols, bt_start, bt_end, bt_start_dt, end_dt, results)

    pd.DataFrame(results).to_csv(PROJECT_ROOT / "runtime" / "v16_pat_burst_ml.csv", index=False)


def _inject_union_baseline(df_test: pd.DataFrame, bt_start_dt, end_dt) -> pd.DataFrame:
    df_test["energetic_side"] = 0
    df_test["energetic_s2_prob"] = np.nan
    score_energetic_signals_v15(df_test, bt_start_dt, end_dt)
    return df_test


def _run_sweep(df_test, df_raw, labeled, feats, feat_cols, bt_start, bt_end, bt_start_dt, end_dt, results):
    ml_variants: list[tuple[str, pd.DataFrame]] = []
    base_ml = walk_forward_dual_v15_exit(
        df_raw, labeled, feats, feat_cols, tp=30, sl=25, horizon=30
    )
    ml_variants.append(("dual p>=0.58", base_ml))

    for p, e in [(0.62, 0.05), (0.65, 0.05), (0.68, 0.08)]:
        old_p, old_e = v16_config.ML_CONFIG["prob_threshold"], v16_config.ML_CONFIG["min_edge"]
        v16_config.ML_CONFIG["prob_threshold"] = p
        v16_config.ML_CONFIG["min_edge"] = e
        ml_variants.append(
            (f"dual p>={p}", walk_forward_dual_v15_exit(
                df_raw, labeled, feats, feat_cols, tp=30, sl=25, horizon=30
            ))
        )
        v16_config.ML_CONFIG["prob_threshold"] = old_p
        v16_config.ML_CONFIG["min_edge"] = old_e

    from v16.backtest.ml import walk_forward_triclass
    ml_variants.append(
        (
            "triclass margin+trend",
            walk_forward_triclass(
                df_raw,
                labeled,
                feats,
                feat_cols,
                prob_threshold=0.50,
                require_trend_align=True,
                label_margin=2.0,
            ),
        )
    )

    for ml_name, ml in ml_variants:
        dt = df_test.copy()
        dt["energetic_side"] = 0
        dt["energetic_s2_prob"] = np.nan
        n = _inject_burst_ml(dt, ml, only_if_no_pattern=True)
        results.append(_stats(f"pat + {ml_name} replace energetic (n={n})", _simulate(dt, bt_start, bt_end)))


if __name__ == "__main__":
    main()
