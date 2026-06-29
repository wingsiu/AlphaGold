#!/usr/bin/env python3
"""
Throw experiments at 15m dip / burst until something sticks (Jun 2025→Jun 2026).

Usage:
  PYTHONPATH=. python3 v16/research/profit_hunt.py 2025-06-01 2026-06-25
"""
from __future__ import annotations

import copy
import contextlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features, dip_ml_feature_columns
from v16.backtest.fixed_tpsl_sim import simulate_fixed_tpsl
from v16.backtest.ml import walk_forward_dual, walk_forward_long, walk_forward_short
from v16.backtest.scaleout_sim import simulate_scaleout_trade
from v16.backtest.signals import build_labeled_set, _exit_kwargs
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m


@contextlib.contextmanager
def signal_cfg(patch: dict):
    saved = copy.deepcopy(v16_config.SIGNAL_CONFIG)
    v16_config.SIGNAL_CONFIG.update(patch)
    try:
        yield
    finally:
        v16_config.SIGNAL_CONFIG.clear()
        v16_config.SIGNAL_CONFIG.update(saved)


def _mech_side(df: pd.DataFrame, labeled: pd.DataFrame, side: int) -> pd.DataFrame:
    kw = _exit_kwargs()
    rows = []
    for ts, row in labeled.iterrows():
        i = int(row["entry_idx"])
        if side == 1:
            r = simulate_scaleout_trade(df, i, 1, float(df.iloc[i]["open_ask"]), **kw)
        else:
            r = simulate_scaleout_trade(df, i, -1, float(df.iloc[i]["open_bid"]), **kw)
        rows.append({"signal_ts": ts, "side": side, "pnl": r.pnl, "win": r.pnl > 0})
    return pd.DataFrame(rows)


def _v15_side(df: pd.DataFrame, labeled: pd.DataFrame, side: int, tp: float, sl: float, h: int) -> pd.DataFrame:
    rows = []
    for ts, row in labeled.iterrows():
        i = int(row["entry_idx"])
        ep = float(df.iloc[i]["open_ask"] if side == 1 else df.iloc[i]["open_bid"])
        r = simulate_fixed_tpsl(df, i, side, ep, tp=tp, sl=sl, horizon=h)
        rows.append({"signal_ts": ts, "side": side, "pnl": r.pnl, "win": r.pnl > 0})
    return pd.DataFrame(rows)


def _stats(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {"trades": 0, "wr": 0.0, "net": 0.0, "avg": 0.0}
    return {
        "trades": len(tdf),
        "wr": round(float(tdf["win"].mean() * 100), 1),
        "net": round(float(tdf["pnl"].sum()), 1),
        "avg": round(float(tdf["pnl"].mean()), 2),
    }


def _combine(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if "signal_ts" in out.columns:
        out = out.sort_values("signal_ts").drop_duplicates("signal_ts", keep="first")
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    bt_start = args[0] if args else "2025-06-01"
    bt_end = args[1] if len(args) > 1 else "2026-06-25"

    print("=" * 78)
    print(f"  v16 PROFIT HUNT  |  {bt_start} → {bt_end}")
    print("=" * 78)

    df = load_gold_1m(bt_start, bt_end)
    feats = build_features(df)
    ml_cols = dip_ml_feature_columns(feats)
    rows: list[dict] = []

    def trial(name: str, tdf: pd.DataFrame) -> None:
        s = _stats(tdf)
        s["strategy"] = name
        rows.append(s)
        flag = " ***" if s["net"] > 0 else ""
        print(f"  {name[:52]:52s}  {s['trades']:4d}  WR={s['wr']:5.1f}%  net={s['net']:+8.1f}{flag}")

    # --- 2x UP short family (best mechanical so far) ---
    short_bases = [
        ("short 2xUP base", {"mode": "dip_short_15m", "dip_require_two_prev_up": True}),
        ("short 2xUP prev body>=8", {
            "mode": "dip_short_15m", "dip_require_two_prev_up": True, "dip_min_prev_body_pts": 8.0,
        }),
        ("short 2xUP prev body>=10", {
            "mode": "dip_short_15m", "dip_require_two_prev_up": True, "dip_min_prev_body_pts": 10.0,
        }),
        ("short 2xUP prev range>=12", {
            "mode": "dip_short_15m", "dip_require_two_prev_up": True, "dip_min_prev_range_pts": 12.0,
        }),
        ("short 2xUP minute<5", {
            "mode": "dip_short_15m", "dip_require_two_prev_up": True, "dip_max_minute_in_slot": 5,
        }),
        ("short 1xUP rip", {"mode": "dip_short_15m", "dip_require_two_prev_up": False}),
    ]

    for name, patch in short_bases:
        with signal_cfg(patch):
            labeled = build_labeled_set(df, feats)
            trial(f"{name} mech", _mech_side(df, labeled, -1))
            for p in (0.55, 0.58, 0.62):
                tdf = walk_forward_short(df, labeled, feats, ml_cols, prob_threshold=p)
                trial(f"{name} ML p>={p}", tdf)

    # --- Long dip family ---
    long_bases = [
        ("long 1xDN body>=10", {"mode": "dip_long_15m", "dip_min_prev_body_pts": 10.0}),
        ("long 1xDN range>=12", {"mode": "dip_long_15m", "dip_min_prev_range_pts": 12.0}),
        ("long 2xDN body>=5", {
            "mode": "dip_long_15m", "dip_require_two_prev_down": True, "dip_min_prev_body_pts": 5.0,
        }),
    ]
    for name, patch in long_bases:
        with signal_cfg(patch):
            labeled = build_labeled_set(df, feats)
            trial(f"{name} mech", _mech_side(df, labeled, 1))
            tdf = walk_forward_long(df, labeled, feats, ml_cols, prob_threshold=0.58)
            trial(f"{name} ML p>=0.58", tdf)

    # --- Combo: best short + best long ML (non-deduped union of signals) ---
    with signal_cfg({"mode": "dip_short_15m", "dip_require_two_prev_up": True, "dip_min_prev_body_pts": 8.0}):
        lab_s = build_labeled_set(df, feats)
        short_ml = walk_forward_short(df, lab_s, feats, ml_cols, prob_threshold=0.58)
    with signal_cfg({"mode": "dip_long_15m", "dip_min_prev_body_pts": 10.0}):
        lab_l = build_labeled_set(df, feats)
        long_ml = walk_forward_long(df, lab_l, feats, ml_cols, prob_threshold=0.58)
    trial("COMBO short2xUP ML + long body10 ML", _combine(short_ml, long_ml))

    with signal_cfg({"mode": "dip_short_15m", "dip_require_two_prev_up": True}):
        lab_s = build_labeled_set(df, feats)
        short_m = _mech_side(df, lab_s, -1)
    with signal_cfg({"mode": "dip_long_15m", "dip_min_prev_body_pts": 10.0}):
        lab_l = build_labeled_set(df, feats)
        long_m = walk_forward_long(df, lab_l, feats, ml_cols, prob_threshold=0.58)
    trial("COMBO short2xUP mech + long body10 ML", _combine(short_m, long_m))

    # --- v15 exits on 2xUP short ---
    with signal_cfg({"mode": "dip_short_15m", "dip_require_two_prev_up": True, "dip_min_prev_body_pts": 8.0}):
        labeled = build_labeled_set(df, feats)
        trial("short 2xUP body8 v15 TP20/SL15", _v15_side(df, labeled, -1, 20, 15, 15))
        trial("short 2xUP body8 v15 TP30/SL25", _v15_side(df, labeled, -1, 30, 25, 30))

    # --- Wider scale-out on short ---
    wide = {
        "first_scale_pnl": 5.0,
        "first_scale_frac": 0.5,
        "final_scale_pnl": 15.0,
        "initial_sl": 25.0,
        "runner_lock_pnl": 5.0,
        "horizon": 15,
    }
    with signal_cfg({"mode": "dip_short_15m", "dip_require_two_prev_up": True}):
        labeled = build_labeled_set(df, feats)
        kw = {**_exit_kwargs(), **wide}
        rows_s = []
        for ts, row in labeled.iterrows():
            i = int(row["entry_idx"])
            r = simulate_scaleout_trade(df, i, -1, float(df.iloc[i]["open_bid"]), **kw)
            rows_s.append({"pnl": r.pnl, "win": r.pnl > 0})
        trial("short 2xUP wide exit +5/+15", pd.DataFrame(rows_s))

    # --- Burst dual ML (known weak alone) ---
    with signal_cfg({"mode": "burst"}):
        labeled = build_labeled_set(df, feats)
        burst = walk_forward_dual(df, labeled, feats, list(feats.columns), exit_overrides=wide)
        trial("burst dual ML wide exit", burst)

    out = pd.DataFrame(rows).sort_values("net", ascending=False)
    profitable = out[out["net"] > 0]
    csv = PROJECT_ROOT / "runtime" / "v16_profit_hunt.csv"
    out.to_csv(csv, index=False)

    print("\n" + "=" * 78)
    print(f"  PROFITABLE ({len(profitable)} strategies)")
    print("=" * 78)
    for _, r in profitable.head(15).iterrows():
        print(
            f"  {r['strategy'][:50]:50s}  {int(r['trades']):4d}  "
            f"WR={r['wr']:5.1f}%  net={r['net']:+8.1f}  avg={r['avg']:+.2f}"
        )
    print(f"\nSaved -> {csv}")


if __name__ == "__main__":
    main()
