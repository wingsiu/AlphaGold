#!/usr/bin/env python3
"""
Search v16 model variants vs v15 baseline (+4282 pts, Jun 2025 → Jun 2026).

Usage:
  PYTHONPATH=. python3 v16/research/model_search.py 2025-06-01 2026-06-25
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from config.hybrid_config import WF_CONFIG
from config.pattern_registry import collect_pa_groups, backtest_feature_set
from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features, feature_columns
from v16.backtest.ml import (
    walk_forward_dual,
    walk_forward_dual_v15_exit,
    walk_forward_triclass,
    walk_forward_trend_gate,
)
from v16.backtest.signals import build_labeled_set, candidate_mask
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from xgboost_filter_model.pattern_training import feature_columns as v15_feature_columns
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14

V15_BASELINE = {"trades": 722, "wr": 53.2, "net": 4282.0, "avg": 5.93}


def _summarize(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {"trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0}
    return {
        "trades": len(tdf),
        "wr": round(float(tdf["win"].mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
    }


def _load_v15_features(bt_start: str, bt_end: str) -> pd.DataFrame:
    bt_start_dt = pd.Timestamp(bt_start, tz="UTC")
    warmup = int(WF_CONFIG.get("feature_warmup_days", 120))
    load_start = (bt_start_dt - pd.Timedelta(days=warmup)).strftime("%Y-%m-%d")
    load_end = (pd.Timestamp(bt_end, tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = prepare_data_v14(
        start_date=load_start,
        end_date=load_end,
        energetic_filter=False,
        for_live_inference=True,
        pa_groups=collect_pa_groups(),
        pattern_feature_set=backtest_feature_set(),
    )
    df = prepare_directional_data_v14(df)
    return df[df.index >= bt_start_dt]


def _tight_burst_labeled(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    cfg = copy.deepcopy(v16_config.SIGNAL_CONFIG)
    cfg["min_range_pts"] = 5.0
    cfg["min_volume_ratio"] = 1.6
    cfg["min_body_pts"] = 2.0
    old = v16_config.SIGNAL_CONFIG
    v16_config.SIGNAL_CONFIG = cfg
    try:
        return build_labeled_set(df, feats)
    finally:
        v16_config.SIGNAL_CONFIG = old


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    bt_start = args[0] if args else "2025-06-01"
    bt_end = args[1] if len(args) > 1 else "2026-06-25"

    print("=" * 80)
    print(f"  v16 MODEL SEARCH  |  {bt_start} → {bt_end}")
    print(f"  v15 baseline: {V15_BASELINE['trades']} trades  WR={V15_BASELINE['wr']}%  "
          f"net={V15_BASELINE['net']:+.0f}  avg={V15_BASELINE['avg']:+.2f}")
    print("=" * 80)

    t0 = time.time()
    df = load_gold_1m(bt_start, bt_end)
    feats = build_features(df)
    feat_cols = feature_columns(feats)
    labeled = build_labeled_set(df, feats)
    n_cand = int(candidate_mask(feats, df.index).sum())
    print(f"\nBurst candidates: {n_cand}  labeled: {len(labeled)}")

    print("\nLoading v15 feature stack (warmup included)…")
    df_v15 = _load_v15_features(bt_start, bt_end)
    v15_cols = v15_feature_columns(df_v15)
    feats_v15 = df_v15.reindex(labeled.index).fillna(0.0)
    print(f"v15 features: {len(v15_cols)} cols  aligned rows: {len(feats_v15)}")

    exit_wide = {
        "first_scale_pnl": 5.0,
        "first_scale_frac": 0.5,
        "final_scale_pnl": 20.0,
        "initial_sl": 20.0,
        "runner_lock_pnl": 5.0,
        "horizon": 20,
    }

    results: list[dict] = []

    def run(name: str, fn) -> None:
        print(f"  … {name}", flush=True)
        tdf = fn()
        s = _summarize(tdf)
        s["model"] = name
        results.append(s)
        beat = "✓ BEATS v15" if s["net"] > V15_BASELINE["net"] else ""
        print(
            f"     {s['trades']:4d} trades  WR={s['wr']:5.1f}%  "
            f"net={s['net']:+8.1f}  avg={s['avg']:+.2f}  {beat}"
        )

    run("dual_scale (v16 baseline)", lambda: walk_forward_dual(df, labeled, feats, feat_cols))
    run("triclass + slot filter", lambda: walk_forward_triclass(df, labeled, feats, feat_cols))
    run(
        "triclass + trend align",
        lambda: walk_forward_triclass(df, labeled, feats, feat_cols, require_trend_align=True),
    )
    run("trend_gate dual", lambda: walk_forward_trend_gate(df, labeled, feats, feat_cols))
    run(
        "dual wide exit (+5/+20 H20)",
        lambda: walk_forward_dual(df, labeled, feats, feat_cols, exit_overrides=exit_wide),
    )
    run(
        "triclass wide exit",
        lambda: walk_forward_triclass(
            df, labeled, feats, feat_cols, exit_overrides=exit_wide, prob_threshold=0.52
        ),
    )
    run(
        "dual v15 features",
        lambda: walk_forward_dual(df, labeled, feats_v15, v15_cols),
    )
    run(
        "triclass v15 features",
        lambda: walk_forward_triclass(
            df, labeled, feats_v15, v15_cols, prob_threshold=0.52, slot_feats=feats
        ),
    )
    run(
        "dual v15 features + wide exit",
        lambda: walk_forward_dual(
            df, labeled, feats_v15, v15_cols, exit_overrides=exit_wide
        ),
    )
    run(
        "triclass margin + trend + wide",
        lambda: walk_forward_triclass(
            df,
            labeled,
            feats,
            feat_cols,
            exit_overrides=exit_wide,
            prob_threshold=0.50,
            require_trend_align=True,
            label_margin=2.0,
        ),
    )
    run(
        "dual v15 exit TP30/SL25 H30 (v16 feats)",
        lambda: walk_forward_dual_v15_exit(df, labeled, feats, feat_cols),
    )
    run(
        "dual v15 feats + v15 exit",
        lambda: walk_forward_dual_v15_exit(df, labeled, feats_v15, v15_cols, tp=30, sl=25, horizon=30),
    )

    labeled_tight = _tight_burst_labeled(df, feats)
    run(
        "tight burst + dual v15 feats",
        lambda: walk_forward_dual(df, labeled_tight, feats_v15.reindex(labeled_tight.index).fillna(0.0), v15_cols),
    )
    run(
        "tight burst + triclass v15",
        lambda: walk_forward_triclass(
            df,
            labeled_tight,
            feats_v15.reindex(labeled_tight.index).fillna(0.0),
            v15_cols,
            prob_threshold=0.50,
        ),
    )

    # Threshold sweep on best family (v15 feats dual)
    for p in (0.55, 0.60, 0.62, 0.65):
        old_p = v16_config.ML_CONFIG["prob_threshold"]
        old_e = v16_config.ML_CONFIG["min_edge"]
        v16_config.ML_CONFIG["prob_threshold"] = p
        v16_config.ML_CONFIG["min_edge"] = 0.03 if p >= 0.62 else 0.05
        run(
            f"dual v15 feats p>={p}",
            lambda: walk_forward_dual(df, labeled, feats_v15, v15_cols),
        )
        v16_config.ML_CONFIG["prob_threshold"] = old_p
        v16_config.ML_CONFIG["min_edge"] = old_e

    out_df = pd.DataFrame(results).sort_values("net", ascending=False)
    csv = PROJECT_ROOT / "runtime" / "v16_model_search.csv"
    out_df.to_csv(csv, index=False)

    print("\n" + "=" * 80)
    print("  LEADERBOARD (sorted by net PnL)")
    print("=" * 80)
    for _, r in out_df.head(12).iterrows():
        flag = " *** BEATS v15 ***" if r["net"] > V15_BASELINE["net"] else ""
        print(
            f"  {r['model'][:42]:42s}  {int(r['trades']):4d}  "
            f"WR={r['wr']:5.1f}%  net={r['net']:+8.1f}  avg={r['avg']:+.2f}{flag}"
        )
    print(f"\nSaved -> {csv}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
