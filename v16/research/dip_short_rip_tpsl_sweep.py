#!/usr/bin/env python3
"""
Sweep TP / SL / horizon for v16 dip_short_rip — mechanical vs ML.

ML modes:
  scaleout  — one WF model (scaleout labels), sweep exit params (fast)
  execution — WF retrained per combo (execution labels match exit; slow)

Usage:
  PYTHONPATH=. python3 v16/research/dip_short_rip_tpsl_sweep.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/dip_short_rip_tpsl_sweep.py 2025-06-01 2026-06-25 --quick
  PYTHONPATH=. python3 v16/research/dip_short_rip_tpsl_sweep.py 2025-06-01 2026-06-25 --ml-label execution
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features
from v16.backtest.position_sim import simulate_single_position
from v16.backtest.ml import walk_forward_short_scores
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.dip_short_rip import build_labeled_set, feature_columns, router_mask

GRIDS = {
    "quick": {
        "tp": [20.0, 30.0, 40.0],
        "sl": [20.0, 25.0, 30.0],
        "horizon": [20, 30, 45],
    },
    "full": {
        "tp": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0],
        "sl": [15.0, 20.0, 25.0, 30.0, 35.0],
        "horizon": [15, 20, 30, 45, 60],
    },
}


def _stats_from_trades(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {"trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0, "tp_hit": 0, "sl_hit": 0, "timeout": 0}
    reasons = tdf["exit_reason"].value_counts()
    return {
        "trades": len(tdf),
        "wr": round(float(tdf["win"].mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
        "tp_hit": int(reasons.get("target_hit", 0)),
        "sl_hit": int(reasons.get("stop_loss", 0)),
        "timeout": int(reasons.get("timeout", 0)),
    }


def _run_position_sim(
    df: pd.DataFrame,
    signals: pd.Series,
    *,
    tp: float,
    sl: float,
    horizon: int,
    cfg: dict,
) -> dict:
    tdf = simulate_single_position(
        df,
        signals,
        side=-1,
        tp=tp,
        sl=sl,
        horizon=horizon,
        same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
        upgrade_stop=bool(cfg.get("upgrade_stop", False)),
    )
    tdf["win"] = tdf["pnl"] > 0
    return _stats_from_trades(tdf)


def _run_mechanical(
    df: pd.DataFrame,
    router: pd.Series,
    grid: dict,
    cfg: dict,
) -> pd.DataFrame:
    rows = []
    combos = list(itertools.product(grid["tp"], grid["sl"], grid["horizon"]))
    for tp, sl, h in combos:
        st = _run_position_sim(df, router, tp=tp, sl=sl, horizon=h, cfg=cfg)
        rows.append({"mode": "mechanical", "ml_label": "-", "tp": tp, "sl": sl, "horizon": h, **st})
    return pd.DataFrame(rows)


def _run_ml_scaleout(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    router: pd.Series,
    grid: dict,
    *,
    ml_p: float,
    base_cfg: dict,
) -> pd.DataFrame:
    cfg = copy.deepcopy(base_cfg)
    cfg["ml_label_source"] = "scaleout"
    labeled = build_labeled_set(df, feats, cfg=cfg)
    feat_cols = feature_columns(feats)
    ml_rows = walk_forward_short_scores(labeled, feats, feat_cols, prob_threshold=ml_p)
    ml_ok = pd.Series(False, index=df.index)
    if not ml_rows.empty:
        ts = pd.to_datetime(ml_rows["signal_ts"], utc=True)
        ml_ok.loc[df.index.intersection(ts)] = True
    signals = router & ml_ok

    rows = []
    for tp, sl, h in itertools.product(grid["tp"], grid["sl"], grid["horizon"]):
        st = _run_position_sim(df, signals, tp=tp, sl=sl, horizon=h, cfg=cfg)
        rows.append({"mode": "ml", "ml_label": "scaleout", "tp": tp, "sl": sl, "horizon": h, **st})
    return pd.DataFrame(rows)


def _run_ml_execution(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    router: pd.Series,
    grid: dict,
    *,
    ml_p: float,
    base_cfg: dict,
) -> pd.DataFrame:
    feat_cols = feature_columns(feats)
    rows = []
    combos = list(itertools.product(grid["tp"], grid["sl"], grid["horizon"]))
    n = len(combos)
    t0 = time.time()
    for k, (tp, sl, h) in enumerate(combos, 1):
        cfg = copy.deepcopy(base_cfg)
        cfg["ml_label_source"] = "execution"
        cfg["execution"] = {"tp": tp, "sl": sl, "horizon": h}
        labeled = build_labeled_set(df, feats, cfg=cfg)
        ml_rows = walk_forward_short_scores(labeled, feats, feat_cols, prob_threshold=ml_p)
        ml_ok = pd.Series(False, index=df.index)
        if not ml_rows.empty:
            ts = pd.to_datetime(ml_rows["signal_ts"], utc=True)
            ml_ok.loc[df.index.intersection(ts)] = True
        signals = router & ml_ok
        st = _run_position_sim(df, signals, tp=tp, sl=sl, horizon=h, cfg=cfg)
        rows.append({"mode": "ml", "ml_label": "execution", "tp": tp, "sl": sl, "horizon": h, **st})
        if k % 10 == 0 or k == n:
            elapsed = time.time() - t0
            print(f"  ML execution: {k}/{n} ({elapsed:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def _print_top(df: pd.DataFrame, title: str, n: int = 12) -> None:
    if df.empty:
        return
    print(f"\n{title}")
    print("-" * 90)
    sub = df.sort_values("net", ascending=False).head(n)
    for _, r in sub.iterrows():
        print(
            f"  {r['mode']:11s} {str(r['ml_label']):9s} "
            f"TP{r['tp']:4.0f} SL{r['sl']:4.0f} H{r['horizon']:2d} | "
            f"{int(r['trades']):4d} tr  WR={r['wr']:5.1f}%  net={r['net']:+8.1f}  avg={r['avg']:+.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="dip_short_rip TP/SL/horizon sweep")
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--quick", action="store_true", help="smaller grid")
    parser.add_argument(
        "--ml-label",
        choices=("scaleout", "execution", "both"),
        default="both",
        help="ML label source (execution is slow)",
    )
    parser.add_argument("--ml-prob", type=float, default=None, help="ML threshold (default from config)")
    args = parser.parse_args()

    grid = GRIDS["quick" if args.quick else "full"]
    cfg = v16_config.DIP_SHORT_RIP
    ml_p = float(args.ml_prob if args.ml_prob is not None else cfg["ml_prob"])

    print("=" * 90)
    print(f"  dip_short_rip TP/SL/H sweep  |  {args.start} → {args.end}")
    print(f"  grid={'quick' if args.quick else 'full'}  combos={len(grid['tp'])*len(grid['sl'])*len(grid['horizon'])}  ML p>={ml_p}")
    print("=" * 90)

    df = load_gold_1m(args.start, args.end)
    feats = build_features(df)
    router = router_mask(feats, df.index, cfg=cfg)
    print(f"\nRouter signals: {int(router.sum())} (single-position sim)")

    t0 = time.time()
    mech = _run_mechanical(df, router, grid, cfg)
    print(f"Mechanical sweep done ({time.time()-t0:.1f}s)")

    parts = [mech]
    if args.ml_label in ("scaleout", "both"):
        t1 = time.time()
        parts.append(_run_ml_scaleout(df, feats, router, grid, ml_p=ml_p, base_cfg=cfg))
        print(f"ML scaleout sweep done ({time.time()-t1:.1f}s)")
    if args.ml_label in ("execution", "both"):
        t2 = time.time()
        parts.append(_run_ml_execution(df, feats, router, grid, ml_p=ml_p, base_cfg=cfg))
        print(f"ML execution sweep done ({time.time()-t2:.1f}s)")

    out = pd.concat(parts, ignore_index=True)
    suffix = ""
    if args.ml_label == "execution":
        suffix = "_exec"
    elif args.ml_label == "scaleout" and args.quick:
        suffix = "_quick"
    out_path = PROJECT_ROOT / "runtime" / f"v16_dip_short_rip_tpsl_sweep{suffix}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    _print_top(mech, "Top mechanical")
    if args.ml_label in ("scaleout", "both"):
        _print_top(out[(out["mode"] == "ml") & (out["ml_label"] == "scaleout")], "Top ML (scaleout labels)")
    if args.ml_label in ("execution", "both") and "execution" in out["ml_label"].values:
        _print_top(out[out["ml_label"] == "execution"], "Top ML (execution labels)")

    # Baseline row highlight
    base = out[(out["tp"] == 30) & (out["sl"] == 25) & (out["horizon"] == 30)]
    if not base.empty:
        print("\nBaseline TP30/SL25/H30:")
        for _, r in base.iterrows():
            print(
                f"  {r['mode']:11s} {str(r['ml_label']):9s} | "
                f"{int(r['trades']):4d} tr  net={r['net']:+8.1f}  avg={r['avg']:+.2f}"
            )


if __name__ == "__main__":
    main()
