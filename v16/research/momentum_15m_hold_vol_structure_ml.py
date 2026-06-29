#!/usr/bin/env python3
"""
Vol≥200 + |body|≥3pt + structure gate + ML (structure features).

Entry: open (default) or pre-close breakout via --preclose.

Usage:
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_vol_structure_ml.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_vol_structure_ml.py 2025-06-01 2026-06-25 --preclose
  PYTHONPATH=. python3 v16/research/momentum_15m_hold_vol_structure_ml.py 2025-06-01 2026-06-25 --min-volume 150 250
"""
from __future__ import annotations

import argparse
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
from v16.backtest.impulse_entry import build_resolved_entry_table
from v16.backtest.impulse_features import (
    attach_structure_features,
    impulse_ml_feature_columns,
    structure_kwargs,
)
from v16.backtest.impulse_ml import MODEL_NAMES, evaluate_threshold_sweep, walk_forward_model_scores
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_signal_table
from v16.structure.filter import apply_structure_gate

THRESHOLDS = (0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70)
DEFAULT_MODELS = ("logreg", "gbc", "lgb", "hgb", "mlp", "rf", "xgb", "ens", "et")
ALL_MODELS = tuple(m for m in MODEL_NAMES if m != "lstm")


def _cfg(min_volume: float, *, preclose: bool) -> dict:
    base = v16_config.MOMENTUM_VOL3_PRECLOSE if preclose else v16_config.MOMENTUM_VOL3_OPEN
    c = copy.deepcopy(base)
    c["min_impulse_volume"] = float(min_volume)
    return c


def _mech_summary(df: pd.DataFrame, signals: pd.DataFrame, cfg: dict) -> dict:
    is_cfg = cfg.get("impulse_stop", {})
    gated = apply_structure_gate(df, signals, cfg=cfg)
    fills = build_resolved_entry_table(df, gated, cfg=cfg)
    tdf = simulate_position_impulse_stop(
        df,
        signals,
        tp_multiple=float(is_cfg.get("tp_multiple", 3.0)),
        horizon=int(is_cfg.get("horizon", 120)),
        cfg=cfg,
    )
    if tdf.empty:
        return {"signals": len(signals), "gated": len(gated), "fills": len(fills), "trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0}
    return {
        "signals": len(signals),
        "gated": len(gated),
        "fills": len(fills),
        "trades": len(tdf),
        "wr": round(float(tdf["win"].mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oos_start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--min-move", type=float, default=3.0)
    parser.add_argument("--min-volume", nargs="+", type=float, default=[200.0])
    parser.add_argument("--preclose", action="store_true", help="use breakout pre-close≤10 entry")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    models = list(ALL_MODELS if args.all_models else (args.models or DEFAULT_MODELS))
    oos_start = pd.Timestamp(args.oos_start, tz="UTC")

    entry_label = "pre-close≤10 breakout" if args.preclose else "open"
    print("=" * 100)
    print(f"  vol + change filter + structure ML  |  train {args.train_start}  OOS {args.oos_start} → {args.end}")
    print(f"  |body|>={args.min_move:.0f}  vol>={args.min_volume}  entry={entry_label}  with-trend  structure feats")
    print(f"  models ({len(models)}): {', '.join(models)}")
    print("=" * 100)

    t0 = time.time()
    df = load_gold_1m(args.train_start, args.end)
    feats = build_features(df)

    print("\n--- Reference winners (no vol filter) ---")
    for name, ref_cfg in [
        ("V16_WINNER open+LGB", v16_config.MOMENTUM_V16_WINNER),
        ("V16_WINNER_PRECLOSE", v16_config.MOMENTUM_V16_WINNER_PRECLOSE),
    ]:
        ref_oos = df[df.index >= oos_start]
        sig = build_signal_table(ref_oos, cfg=ref_cfg)
        m = _mech_summary(ref_oos, sig, ref_cfg)
        print(
            f"  {name}: sig={m['signals']} gate={m['gated']} fills={m['fills']} "
            f"tr={m['trades']} WR={m['wr']}% net={m['net']:+.1f}"
        )

    best_rows: list[dict] = []
    all_sweeps: list[pd.DataFrame] = []

    for min_vol in args.min_volume:
        cfg = _cfg(min_vol, preclose=args.preclose)
        cfg["min_move_pts"] = float(args.min_move)

        skw = structure_kwargs(cfg)
        feats_s = attach_structure_features(df, feats, **skw) if skw else feats

        labeled = build_labeled_set(df, cfg=cfg)
        print(
            f"\n=== min_volume={min_vol:.0f} | labeled={len(labeled)} "
            f"WR={labeled['win'].mean()*100:.1f}% vol_med={labeled['impulse_volume'].median():.0f} ==="
        )
        if labeled.empty:
            continue

        feat_n = len(impulse_ml_feature_columns(feats_s, labeled, include_structure=True))
        print(f"Features (incl structure): {feat_n}")

        df_oos = df[df.index >= oos_start]
        sig_oos = build_signal_table(df_oos, cfg=cfg)
        mech = _mech_summary(df_oos, sig_oos, cfg)
        print(
            f"Mechanical OOS: sig={mech['signals']} gate={mech['gated']} fills={mech['fills']} "
            f"tr={mech['trades']} WR={mech['wr']}% net={mech['net']:+.1f} avg={mech['avg']:+.2f}"
        )

        for model in models:
            print(f"\n--- vol>={min_vol:.0f} {model.upper()} ---")
            t1 = time.time()
            scores = walk_forward_model_scores(
                df,
                feats_s,
                labeled,
                model,
                seq_len=args.seq_len,
                prob_threshold=0.0,
                cfg=cfg,
            )
            if scores.empty:
                print("  no scores")
                continue
            scores_oos = scores[pd.to_datetime(scores["signal_ts"], utc=True) >= oos_start]
            sweep = evaluate_threshold_sweep(
                df_oos,
                pd.Series(0, index=df_oos.index),
                scores_oos,
                cfg,
                THRESHOLDS,
                exit_mode="impulse_stop",
                signal_table=sig_oos,
            )
            elapsed_m = time.time() - t1
            sweep["model"] = model
            sweep["min_volume"] = min_vol
            all_sweeps.append(sweep)

            print(f"{'prob':>6} {'sig':>5} {'tr':>5} {'WR%':>6} {'net':>9} {'avg':>7}")
            for _, r in sweep.iterrows():
                print(
                    f"{r['prob']:6.2f} {int(r['signals']):5d} {int(r['trades']):5d} "
                    f"{r['wr']:6.1f} {r['net']:+9.1f} {r['avg']:+.2f}"
                )
            best = sweep.sort_values("net", ascending=False).iloc[0]
            best_rows.append(
                {
                    "min_volume": min_vol,
                    "model": model,
                    "prob": best["prob"],
                    "signals": int(best["signals"]),
                    "trades": int(best["trades"]),
                    "wr": best["wr"],
                    "net": best["net"],
                    "avg": best["avg"],
                    "mech_net": mech["net"],
                    "mech_trades": mech["trades"],
                    "elapsed_s": round(elapsed_m, 1),
                }
            )
            print(
                f"  best p>={best['prob']:.2f}: {int(best['trades'])} tr "
                f"net={best['net']:+.1f} ({elapsed_m:.1f}s)"
            )

    elapsed = time.time() - t0
    if best_rows:
        best_df = pd.DataFrame(best_rows).sort_values("net", ascending=False)
        sweep_df = pd.concat(all_sweeps, ignore_index=True)
        tag = f"mv{int(args.min_move)}_vol{int(args.min_volume[0])}{'_preclose' if args.preclose else '_open'}"
        best_path = PROJECT_ROOT / "runtime" / f"v16_vol_structure_ml_{tag}_best.csv"
        sweep_path = PROJECT_ROOT / "runtime" / f"v16_vol_structure_ml_{tag}_sweep.csv"
        best_df.to_csv(best_path, index=False)
        sweep_df.to_csv(sweep_path, index=False)

        print("\n" + "=" * 100)
        print("Best per model:")
        for model in models:
            sub = best_df[best_df["model"] == model]
            if sub.empty:
                continue
            r = sub.iloc[0]
            flag = "✓" if r["net"] > r["mech_net"] else " "
            print(
                f"  {flag} {model:6s} p>={r['prob']:.2f}: {int(r['trades'])} tr "
                f"WR={r['wr']:.1f}% net={r['net']:+.1f}  (mech {r['mech_net']:+.1f})"
            )

        top = best_df.iloc[0]
        print(
            f"\nTop: vol>={top['min_volume']:.0f} {top['model']} p>={top['prob']:.2f} "
            f"→ net={top['net']:+.1f} ({int(top['trades'])} tr)  "
            f"vs V16_WINNER ~+1150 / PRECLOSE ~+1124"
        )
        print(f"Saved {best_path}")
        print(f"Saved {sweep_path}")
    print(f"Total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
