#!/usr/bin/env python3
"""
Sweep initial stop (before first +5 scale) for v16 burst + ML scale-out.

Usage:
  PYTHONPATH=. python3 v16/research/sweep_stop.py 2025-06-01 2026-06-25
  PYTHONPATH=. python3 v16/research/sweep_stop.py 2025-06-01 2026-06-25 8 10 12 15 20 25
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features, feature_columns
from v16.backtest.ml import walk_forward_dual
from v16.backtest.scaleout_sim import batch_simulate
from v16.backtest.signals import build_labeled_set, candidate_mask, _exit_kwargs
from v16.config.v16_config import BACKTEST_CONFIG, EXIT_CONFIG, ML_CONFIG, SIGNAL_CONFIG
from v16.data.load_gold import load_gold_1m


def _summarize(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {"trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0, "scaled_pct": 0.0}
    return {
        "trades": len(tdf),
        "wr": float(tdf["win"].mean() * 100),
        "net": float(tdf["pnl"].sum()),
        "avg": float(tdf["pnl"].mean()),
        "scaled_pct": float(tdf["scaled_half"].mean() * 100),
        "stop_loss_n": int((tdf["exit_reason"] == "stop_loss").sum()),
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    bt_start = args[0] if args else BACKTEST_CONFIG["default_start"]
    bt_end = args[1] if len(args) > 1 else pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    sl_values = [float(x) for x in args[2:]] if len(args) > 2 else [8, 10, 12, 15, 18, 20, 25, 30]

    print("=" * 78)
    print(f"  v16 STOP SWEEP (burst + scale-out)  |  {bt_start} → {bt_end}")
    print(
        f"  Signal: burst London/NY | ML p>={ML_CONFIG['prob_threshold']} "
        f"edge>={ML_CONFIG['min_edge']}"
    )
    print(
        f"  Ladder fixed: +{EXIT_CONFIG['first_scale_pnl']:.0f} half | "
        f"+{EXIT_CONFIG['final_scale_pnl']:.0f} all | "
        f"H={EXIT_CONFIG['horizon_minutes']}m"
    )
    print("=" * 78)

    df = load_gold_1m(bt_start, bt_end)
    feats = build_features(df)
    n_cand = int(candidate_mask(feats, df.index).sum())
    print(f"\nBurst candidate bars: {n_cand}")

    rows = []
    for sl in sl_values:
        overrides = {"initial_sl": sl}
        labeled = build_labeled_set(df, feats, exit_overrides=overrides)
        kw = {**_exit_kwargs(), **overrides}

        raw = batch_simulate(
            df,
            [
                {
                    "signal_ts": ts,
                    "entry_idx": int(r["entry_idx"]),
                    "entry_price": float(df.iloc[int(r["entry_idx"])]["open_ask"]
                    if int(r["best_side"]) == 1
                    else df.iloc[int(r["entry_idx"])]["open_bid"]),
                    "side": int(r["best_side"]),
                }
                for ts, r in labeled.iterrows()
            ],
            **kw,
        )
        raw["win"] = raw["pnl"] > 0
        oracle = _summarize(raw)

        ml = walk_forward_dual(
            df,
            labeled,
            feats,
            feature_columns(feats),
            exit_overrides=overrides,
        )
        ml_s = _summarize(ml)
        rows.append({"initial_sl": sl, **{f"ml_{k}": v for k, v in ml_s.items()}, **{f"ora_{k}": v for k, v in oracle.items()}})
        print(
            f"  SL={sl:4.0f}  ML: {ml_s['trades']:4d} trades  "
            f"WR={ml_s['wr']:5.1f}%  net={ml_s['net']:+8.0f}  "
            f"avg={ml_s['avg']:+.2f}  stops={ml_s['stop_loss_n']}"
        )

    out = pd.DataFrame(rows)
    csv = PROJECT_ROOT / "runtime" / "v16_stop_sweep.csv"
    out.to_csv(csv, index=False)

    best = out.sort_values("ml_net", ascending=False).head(3)
    print(f"\nTop 3 by ML net PnL:\n{best.to_string(index=False)}")
    print(f"\nSaved -> {csv}")


if __name__ == "__main__":
    main()
