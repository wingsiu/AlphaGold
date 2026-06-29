#!/usr/bin/env python3
"""
Sweep 15m dip-long filters + ML.

Compares:
  - 1 prior 15m down vs 2 consecutive down bars
  - prev / prev2 body & range thresholds
  - mechanical long vs walk-forward long ML

Usage:
  PYTHONPATH=. python3 v16/research/dip_15m_sweep.py 2025-06-01 2026-06-25
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features, dip_ml_feature_columns
from v16.backtest.ml import walk_forward_long
from v16.backtest.scaleout_sim import simulate_scaleout_trade
from v16.backtest.signals import build_labeled_set, dip_long_15m_mask, dip_short_15m_mask, _exit_kwargs
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m


def _mechanical_long(df: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
    kw = _exit_kwargs()
    rows = []
    for ts, row in labeled.iterrows():
        i = int(row["entry_idx"])
        r = simulate_scaleout_trade(df, i, 1, float(df.iloc[i]["open_ask"]), **kw)
        rows.append({"pnl": r.pnl, "win": r.pnl > 0})
    return pd.DataFrame(rows)


def _summarize(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {"trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0}
    return {
        "trades": len(tdf),
        "wr": round(float(tdf["win"].mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
    }


def _run_variant(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    cfg_patch: dict,
    *,
    use_ml: bool,
    ml_p: float = 0.58,
) -> dict:
    cfg = copy.deepcopy(v16_config.SIGNAL_CONFIG)
    cfg.update(cfg_patch)
    cfg["mode"] = "dip_long_15m"
    saved = copy.deepcopy(v16_config.SIGNAL_CONFIG)
    v16_config.SIGNAL_CONFIG.clear()
    v16_config.SIGNAL_CONFIG.update(cfg)
    try:
        labeled = build_labeled_set(df, feats)
        n_sig = int(dip_long_15m_mask(feats, df.index).sum())
        if use_ml:
            tdf = walk_forward_long(
                df,
                labeled,
                feats,
                dip_ml_feature_columns(feats),
                prob_threshold=ml_p,
            )
        else:
            tdf = _mechanical_long(df, labeled)
        s = _summarize(tdf)
        s["signals"] = n_sig
        s["labeled"] = len(labeled)
        return s
    finally:
        v16_config.SIGNAL_CONFIG.clear()
        v16_config.SIGNAL_CONFIG.update(saved)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    bt_start = args[0] if args else "2025-06-01"
    bt_end = args[1] if len(args) > 1 else pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

    print("=" * 78)
    print(f"  15m DIP SWEEP (long)  |  {bt_start} → {bt_end}")
    print("  No ML in v1 — now: mechanical + WF long ML on prev 1/2 bar filters")
    print("=" * 78)

    df = load_gold_1m(bt_start, bt_end)
    feats = build_features(df)
    rows: list[dict] = []

    variants = [
        ("base: 1x prev down", {"dip_require_two_prev_down": False}),
        ("2x prev down (two 15m down)", {"dip_require_two_prev_down": True}),
        ("1x down + prev body>=5", {"dip_min_prev_body_pts": 5.0}),
        ("1x down + prev range>=12", {"dip_min_prev_range_pts": 12.0}),
        ("2x down + prev body>=5", {"dip_require_two_prev_down": True, "dip_min_prev_body_pts": 5.0}),
        ("2x down + prev2 body>=5", {"dip_require_two_prev_down": True, "dip_min_prev2_body_pts": 5.0}),
        ("2x down + prev range>=12", {"dip_require_two_prev_down": True, "dip_min_prev_range_pts": 12.0}),
        ("2x down + prev2 range>=12", {"dip_require_two_prev_down": True, "dip_min_prev2_range_pts": 12.0}),
        ("1x down + prev body>=10", {"dip_min_prev_body_pts": 10.0}),
        ("2x down + both range>=12", {
            "dip_require_two_prev_down": True,
            "dip_min_prev_range_pts": 12.0,
            "dip_min_prev2_range_pts": 12.0,
        }),
    ]

    for name, patch in variants:
        mech = _run_variant(df, feats, patch, use_ml=False)
        ml = _run_variant(df, feats, patch, use_ml=True, ml_p=0.58)
        rows.append({"variant": name, "type": "mech", **mech})
        rows.append({"variant": name, "type": "ML p>=0.58", **ml})
        print(
            f"  {name[:36]:36s}  sig={mech['signals']:4d}  "
            f"mech {mech['trades']:4d} net={mech['net']:+7.1f}  |  "
            f"ML {ml['trades']:4d} net={ml['net']:+7.1f}"
        )

    # Two 15m UP → short (symmetric rule)
    print("\n--- SHORT: two 15m UP + slot rip (mechanical) ---")
    short_cfg = copy.deepcopy(v16_config.SIGNAL_CONFIG)
    short_cfg["mode"] = "dip_short_15m"
    short_cfg["dip_require_two_prev_up"] = True
    saved = copy.deepcopy(v16_config.SIGNAL_CONFIG)
    v16_config.SIGNAL_CONFIG.clear()
    v16_config.SIGNAL_CONFIG.update(short_cfg)
    try:
        labeled_s = build_labeled_set(df, feats)
        n_short = int(dip_short_15m_mask(feats, df.index).sum())
        kw = _exit_kwargs()
        short_rows = []
        for ts, row in labeled_s.iterrows():
            i = int(row["entry_idx"])
            r = simulate_scaleout_trade(df, i, -1, float(df.iloc[i]["open_bid"]), **kw)
            short_rows.append({"pnl": r.pnl, "win": r.pnl > 0})
        ss = _summarize(pd.DataFrame(short_rows))
        print(f"  2x prev UP short: signals={n_short} trades={ss['trades']} WR={ss['wr']}% net={ss['net']:+.1f}")
        rows.append({"variant": "2x prev UP short", "type": "mech", "signals": n_short, **ss})
    finally:
        v16_config.SIGNAL_CONFIG.clear()
        v16_config.SIGNAL_CONFIG.update(saved)

    out = pd.DataFrame(rows).sort_values("net", ascending=False)
    csv = PROJECT_ROOT / "runtime" / "v16_dip_15m_sweep.csv"
    out.to_csv(csv, index=False)
    print(f"\nTop 5 by net PnL:\n{out.head(5).to_string(index=False)}")
    print(f"\nSaved -> {csv}")


if __name__ == "__main__":
    main()
