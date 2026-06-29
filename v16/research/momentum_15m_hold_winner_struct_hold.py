#!/usr/bin/env python3
"""
#1 experiment: winner entry + ET ML, struct-only hold (no TP, H=480) vs baseline R=3 H=120.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_winner_struct_hold.py 2025-06-01 2026-06-25
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

from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features
from v16.backtest.impulse_features import attach_structure_features, structure_kwargs
from v16.backtest.impulse_ml import filter_signal_table, walk_forward_model_scores
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_signal_table

VARIANTS = {
    "baseline_R3_H120": {
        "tp_enabled": True,
        "tp_multiple": 3.0,
        "horizon": 120,
        "exit_on_structure_change": True,
        "exit_on_structure_change_min_pnl": -1e9,
    },
    "struct_hold_H480": {
        "tp_enabled": False,
        "horizon": 480,
        "exit_on_structure_change": True,
        "exit_on_structure_change_min_pnl": -1e9,
    },
}


def _stats(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {"trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0, "hold_med": 0.0}
    hold = (
        pd.to_datetime(tdf["exit_time"]) - pd.to_datetime(tdf["entry_time"])
    ).dt.total_seconds() / 60.0
    reasons = tdf["exit_reason"].value_counts()
    return {
        "trades": len(tdf),
        "wr": round(float(tdf["win"].mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
        "hold_med": round(float(hold.median()), 1),
        "struct": int(reasons.get("structure_change", 0)),
        "tp": int(reasons.get("target_hit", 0)),
        "sl": int(reasons.get("stop_loss", 0)),
        "to": int(reasons.get("timeout", 0)),
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    start = args[0] if args else "2025-06-01"
    end = args[1] if len(args) > 1 else "2026-06-25"
    train_start = "2024-01-01"
    oos = pd.Timestamp(start, tz="UTC")

    print("=" * 88)
    print(f"  Winner struct-hold experiment  |  OOS {start} → {end}")
    print("  Entry: pre-close winner + ET ML p≥0.50 (14d WF, same scores both variants)")
    print("=" * 88)

    t0 = time.time()
    base = copy.deepcopy(v16_config.MOMENTUM_V16_WINNER_PRECLOSE)
    df = load_gold_1m(train_start, end)
    df_oos = df[df.index >= oos]
    signals = build_signal_table(df_oos, cfg=base)

    labeled = build_labeled_set(df, cfg=base)
    feats = build_features(df)
    skw = structure_kwargs(base)
    if skw:
        feats = attach_structure_features(df, feats, **skw)

    print("ET walk-forward (14d)…")
    scores = walk_forward_model_scores(
        df, feats, labeled, "et", prob_threshold=0.0, retrain_freq="14D", cfg=base
    )
    scores_oos = scores[pd.to_datetime(scores["signal_ts"], utc=True) >= oos]
    ml_filt = filter_signal_table(signals, scores_oos[scores_oos["p_win"] >= 0.50])

    rows: list[dict] = []
    for variant, exit_kw in VARIANTS.items():
        for lane, sig in [("mech", signals), ("ml", ml_filt)]:
            cfg = copy.deepcopy(base)
            cfg["impulse_stop"] = {**cfg.get("impulse_stop", {}), **exit_kw}
            tdf = simulate_position_impulse_stop(df_oos, sig, cfg=cfg)
            st = _stats(tdf)
            rows.append({"variant": variant, "lane": lane, **st})

    out = pd.DataFrame(rows)
    path = PROJECT_ROOT / "runtime/v16_winner_struct_hold_experiment.csv"
    out.to_csv(path, index=False)

    print(f"\n{'variant':18s} {'lane':5s} {'tr':>5} {'WR%':>6} {'net':>9} {'avg':>7} {'hold':>6} {'SC':>4} {'TP':>4} {'SL':>4} {'TO':>4}")
    print("-" * 88)
    for _, r in out.iterrows():
        print(
            f"{r['variant']:18s} {r['lane']:5s} {int(r['trades']):5d} {r['wr']:6.1f} "
            f"{r['net']:+9.1f} {r['avg']:+7.2f} {r['hold_med']:6.0f} "
            f"{int(r['struct']):4d} {int(r['tp']):4d} {int(r['sl']):4d} {int(r['to']):4d}"
        )

    base_ml = out[(out["variant"] == "baseline_R3_H120") & (out["lane"] == "ml")].iloc[0]
    hold_ml = out[(out["variant"] == "struct_hold_H480") & (out["lane"] == "ml")].iloc[0]
    delta = hold_ml["net"] - base_ml["net"]
    print(f"\nML delta (struct-hold vs baseline): {delta:+.1f} pt")
    print(f"Saved {path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
