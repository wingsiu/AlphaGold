#!/usr/bin/env python3
"""
Struct-hold sweep on winner pre-close: no TP, struct-exit always, vary horizon.

Precomputes ET ML scores once (14d WF), then sweeps sim only.
Compares vs baseline R=3 H=120 and optional portfolio with dip_short_rip.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_struct_hold_sweep.py 2025-06-01 2026-06-25
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
from v16.backtest.dip_short_rip_run import run_dip_short_rip
from v16.backtest.features import build_features
from v16.backtest.impulse_features import attach_structure_features, structure_kwargs
from v16.backtest.impulse_ml import filter_signal_table, walk_forward_model_scores
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_signal_table

HORIZONS = (90, 120, 180, 240, 360, 480, 600, 720)
ML_PROB = 0.50
STRUCT_EXIT = {
    "exit_on_structure_change": True,
    "exit_on_structure_change_min_pnl": -1e9,
}


def _stats(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {"trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0, "hold_med": 0.0, "struct": 0, "tp": 0, "sl": 0, "to": 0}
    hold = (pd.to_datetime(tdf["exit_time"]) - pd.to_datetime(tdf["entry_time"])).dt.total_seconds() / 60.0
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


def _portfolio_stats(tdfs: list[pd.DataFrame], label: str) -> dict:
    """Merge non-overlapping single-position lanes is approximate: sum PnL of independent lanes."""
    if not tdfs:
        return {"label": label, "trades": 0, "net": 0.0}
    all_t = pd.concat([t for t in tdfs if not t.empty], ignore_index=True)
    if all_t.empty:
        return {"label": label, "trades": 0, "net": 0.0}
    return {
        "label": label,
        "trades": len(all_t),
        "wr": round(float(all_t["win"].mean() * 100), 1),
        "net": round(float(all_t["pnl"].sum()), 1),
        "avg": round(float(all_t["pnl"].mean()), 2),
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    start = args[0] if args else "2025-06-01"
    end = args[1] if len(args) > 1 else "2026-06-25"
    train_start = "2024-01-01"
    oos = pd.Timestamp(start, tz="UTC")

    print("=" * 92)
    print(f"  Struct-hold horizon sweep  |  OOS {start} → {end}")
    print(f"  Entry: winner pre-close | ET ML p≥{ML_PROB} (14d) + mechanical")
    print("=" * 92)

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
    ml_filt = filter_signal_table(signals, scores_oos[scores_oos["p_win"] >= ML_PROB])

    rows: list[dict] = []

    # Baseline
    for lane, sig in [("mech", signals), ("ml", ml_filt)]:
        cfg = copy.deepcopy(base)
        cfg["impulse_stop"] = {
            **cfg.get("impulse_stop", {}),
            "tp_enabled": True,
            "tp_multiple": 3.0,
            "horizon": 120,
            **STRUCT_EXIT,
        }
        tdf = simulate_position_impulse_stop(df_oos, sig, cfg=cfg)
        rows.append({"mode": "baseline_R3_H120", "lane": lane, "horizon": 120, "tp_enabled": True, **_stats(tdf)})

    # Struct-hold sweep
    for h in HORIZONS:
        for lane, sig in [("mech", signals), ("ml", ml_filt)]:
            cfg = copy.deepcopy(base)
            cfg["impulse_stop"] = {
                **cfg.get("impulse_stop", {}),
                "tp_enabled": False,
                "horizon": int(h),
                **STRUCT_EXIT,
            }
            tdf = simulate_position_impulse_stop(df_oos, sig, cfg=cfg)
            rows.append({"mode": "struct_hold", "lane": lane, "horizon": h, "tp_enabled": False, **_stats(tdf)})

    out = pd.DataFrame(rows)
    path = PROJECT_ROOT / "runtime/v16_winner_struct_hold_sweep.csv"
    out.to_csv(path, index=False)

    print(f"\n{'mode':14s} {'lane':5s} {'H':>4} {'tr':>5} {'WR%':>6} {'net':>9} {'avg':>7} {'hold':>6} {'SC':>4} {'SL':>4} {'TO':>4}")
    print("-" * 80)
    for _, r in out.sort_values(["lane", "net"], ascending=[True, False]).iterrows():
        print(
            f"{r['mode']:14s} {r['lane']:5s} {int(r['horizon']):4d} {int(r['trades']):5d} {r['wr']:6.1f} "
            f"{r['net']:+9.1f} {r['avg']:+7.2f} {r['hold_med']:6.0f} "
            f"{int(r['struct']):4d} {int(r['sl']):4d} {int(r['to']):4d}"
        )

    ml = out[out["lane"] == "ml"]
    best_sh = ml[ml["mode"] == "struct_hold"].sort_values("net", ascending=False).iloc[0]
    base_ml = ml[ml["mode"] == "baseline_R3_H120"].iloc[0]
    print(f"\nBest struct-hold ML: H={int(best_sh['horizon'])} → net={best_sh['net']:+.1f} ({int(best_sh['trades'])} tr)")
    print(f"Baseline ML:         H=120 R=3 → net={base_ml['net']:+.1f} ({int(base_ml['trades'])} tr)")

    # Portfolio: best momentum ML struct-hold + dip short rip ML
    print("\n--- Portfolio: momentum + dip_short_rip (separate patterns) ---")
    dip_cfg = copy.deepcopy(v16_config.DIP_SHORT_RIP)
    dip_feats = build_features(df_oos)
    dip_ml = run_dip_short_rip(df_oos, dip_feats, dip_cfg, mechanical=False, ml_prob=float(dip_cfg["ml_prob"]))
    dip_mech = run_dip_short_rip(df_oos, dip_feats, dip_cfg, mechanical=True)

    best_h = int(best_sh["horizon"])
    cfg_best = copy.deepcopy(base)
    cfg_best["impulse_stop"] = {**cfg_best.get("impulse_stop", {}), "tp_enabled": False, "horizon": best_h, **STRUCT_EXIT}
    mom_best = simulate_position_impulse_stop(df_oos, ml_filt, cfg=cfg_best)

    combos = [
        ("mom baseline ML only", [simulate_position_impulse_stop(
            df_oos, ml_filt, cfg={**base, "impulse_stop": {**base.get("impulse_stop", {}), "tp_enabled": True, "tp_multiple": 3.0, "horizon": 120, **STRUCT_EXIT}}
        )]),
        (f"mom struct-hold H{best_h} ML only", [mom_best]),
        ("dip short ML only", [dip_ml]),
        (f"mom H{best_h} ML + dip ML", [mom_best, dip_ml]),
        ("mom baseline ML + dip ML", [
            simulate_position_impulse_stop(df_oos, ml_filt, cfg={**base, "impulse_stop": {**base.get("impulse_stop", {}), "tp_enabled": True, "tp_multiple": 3.0, "horizon": 120, **STRUCT_EXIT}}),
            dip_ml,
        ]),
        ("dip mech only", [dip_mech]),
    ]

    print(f"{'portfolio':32s} {'tr':>6} {'WR%':>6} {'net':>10} {'avg':>7}")
    print("-" * 65)
    port_rows = []
    for label, tdfs in combos:
        st = _portfolio_stats(tdfs, label)
        port_rows.append(st)
        wr = st.get("wr", 0)
        avg = st.get("avg", 0)
        print(f"{label:32s} {st['trades']:6d} {wr:6.1f} {st['net']:+10.1f} {avg:+7.2f}")

    port_df = pd.DataFrame(port_rows)
    port_path = PROJECT_ROOT / "runtime/v16_winner_struct_hold_portfolio.csv"
    port_df.to_csv(port_path, index=False)

    print(f"\nSaved {path}")
    print(f"Saved {port_path}")
    print(f"Done {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
