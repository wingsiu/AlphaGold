#!/usr/bin/env python3
"""Gold v16 alone + v15+v16 combined (single-slot merge by entry time).

  PYTHONPATH=. python3 v16/research/gold_v15_v16_combined_backtest.py [start] [end]

Default OOS: 2025-06-01 → 2026-06-25 (v16 winners window)
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from config.pattern_registry import PATTERN_REGISTRY, PRODUCTION_PATTERNS
from v15.backtest import backtest_oil as bt
from v16._paths import PROJECT_ROOT
from v16.backtest.dip_short_rip_run import run_dip_short_rip
from v16.backtest.features import build_features
from v16.backtest.impulse_features import attach_structure_features, structure_kwargs
from v16.backtest.impulse_ml import filter_signal_table, walk_forward_model_scores
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_signal_table

# Merge priorities (lower = wins tie at same entry minute)
GOLD_LEG_PRIORITY: dict[str, int] = {
    "v16_dip_short": 0,
    "dip_short_rip": 0,
}
for name, spec in PATTERN_REGISTRY.items():
    if name in PRODUCTION_PATTERNS:
        GOLD_LEG_PRIORITY[name] = int(spec.get("priority", 9))
GOLD_LEG_PRIORITY.setdefault("energetic", 25)
GOLD_LEG_PRIORITY["v16_momentum"] = 14
GOLD_LEG_PRIORITY["pattern"] = 10

MOM_H = 720
MOM_ML_PROB = 0.50
TRAIN_START = "2024-01-01"


def _utc_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _trade_row(
    entry,
    exit,
    pnl: float,
    leg: str,
    *,
    side: int = 1,
    typ: str | None = None,
) -> dict:
    return {
        "entry": _utc_ts(entry),
        "exit": _utc_ts(exit),
        "pnl": float(pnl),
        "type": typ or leg,
        "_leg": leg,
        "side": side,
    }


def _df_to_trades(tdf: pd.DataFrame, leg: str, typ: str | None = None) -> list[dict]:
    if tdf.empty:
        return []
    out = []
    for _, r in tdf.iterrows():
        side = int(r["side"]) if "side" in r and pd.notna(r["side"]) else (1 if leg != "v16_dip_short" else -1)
        out.append(
            _trade_row(r["entry_time"], r["exit_time"], r["pnl"], leg, side=side, typ=typ or leg)
        )
    return out


def merge_gold_trades(trades: list[dict]) -> list[dict]:
    """Single slot — sort by entry time, tie-break by leg priority."""
    if not trades:
        return []

    def sort_key(tr):
        entry = _utc_ts(tr["entry"])
        typ = str(tr.get("type", tr.get("_leg", "")))
        return (entry, GOLD_LEG_PRIORITY.get(typ, GOLD_LEG_PRIORITY.get(tr.get("_leg", ""), 9)))

    taken: list[dict] = []
    busy_until = None
    for tr in sorted(trades, key=sort_key):
        entry = _utc_ts(tr["entry"])
        if busy_until is not None and entry < busy_until:
            continue
        taken.append(tr)
        busy_until = _utc_ts(tr["exit"])
    return taken


def _stats(trades: list[dict], label: str) -> dict:
    if not trades:
        print(f"  {label:28s}   0 trades  PnL=+0.0")
        return {"label": label, "trades": 0, "pnl": 0.0}
    pnls = [t["pnl"] for t in trades]
    wr = 100 * sum(1 for p in pnls if p > 0) / len(pnls)
    pnl = sum(pnls)
    print(f"  {label:28s} {len(trades):4d} trades  PnL={pnl:+8.1f}  WR={wr:.1f}%  avg={pnl/len(pnls):+.2f}")
    return {"label": label, "trades": len(trades), "pnl": pnl, "wr": wr}


def _by_leg(trades: list[dict]) -> None:
    legs: dict[str, list] = {}
    for t in trades:
        legs.setdefault(str(t.get("_leg", t.get("type", "?"))), []).append(t["pnl"])
    print("\n  In portfolio:")
    for leg, pnls in sorted(legs.items()):
        print(f"    {leg:22s} {len(pnls):4d}t  PnL={sum(pnls):+.1f}")


def run_v16_momentum(df: pd.DataFrame, oos: pd.Timestamp) -> pd.DataFrame:
    cfg = copy.deepcopy(v16_config.MOMENTUM_V16_WINNER_PRECLOSE)
    df_oos = df[df.index >= oos]
    signals = build_signal_table(df_oos, cfg=cfg)
    labeled = build_labeled_set(df, cfg=cfg)
    feats = build_features(df)
    skw = structure_kwargs(cfg)
    if skw:
        feats = attach_structure_features(df, feats, **skw)
    scores = walk_forward_model_scores(df, feats, labeled, "et", prob_threshold=0.0, retrain_freq="14D", cfg=cfg)
    scores_oos = scores[pd.to_datetime(scores["signal_ts"], utc=True) >= oos]
    ml_filt = filter_signal_table(signals, scores_oos[scores_oos["p_win"] >= MOM_ML_PROB])
    cfg["impulse_stop"] = {
        **cfg.get("impulse_stop", {}),
        "tp_enabled": False,
        "horizon": MOM_H,
        "exit_on_structure_change": True,
        "exit_on_structure_change_min_pnl": -1e9,
    }
    return simulate_position_impulse_stop(df_oos, ml_filt, cfg=cfg)


def run_v16_dip(df: pd.DataFrame, oos: pd.Timestamp) -> pd.DataFrame:
    cfg = copy.deepcopy(v16_config.DIP_SHORT_RIP)
    df_oos = df[df.index >= oos]
    feats = build_features(df_oos)
    return run_dip_short_rip(df_oos, feats, cfg, mechanical=False, ml_prob=float(cfg["ml_prob"]))


def run_v15_hybrid(start: str, end: str, *, use_csv: bool = True) -> list[dict]:
    """Run v15 hybrid backtest; return trade list."""
    csv_path = PROJECT_ROOT / "runtime" / "v15_backtest_trades.csv"
    oos = _utc_ts(start)
    end_ts = _utc_ts(end) + pd.Timedelta(hours=23, minutes=59)

    if use_csv and csv_path.exists():
        tdf = pd.read_csv(csv_path)
        tdf["entry_time"] = pd.to_datetime(tdf["entry_time"], utc=True)
        tdf["exit_time"] = pd.to_datetime(tdf["exit_time"], utc=True)
        in_window = tdf[(tdf["entry_time"] >= oos) & (tdf["entry_time"] <= end_ts)]
        if len(in_window) >= 50:
            print(f"  (using {csv_path}, {len(in_window)} OOS trades)")
            tdf = in_window
        else:
            use_csv = False

    if not use_csv or not csv_path.exists():
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "V14_HYBRID": "1"}
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "v15" / "backtest" / "backtest_v15.py"), start, end],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
        )
        tdf = pd.read_csv(csv_path)
        tdf["entry_time"] = pd.to_datetime(tdf["entry_time"], utc=True)
        tdf["exit_time"] = pd.to_datetime(tdf["exit_time"], utc=True)
        tdf = tdf[(tdf["entry_time"] >= oos) & (tdf["entry_time"] <= end_ts)]
    out = []
    for _, r in tdf.iterrows():
        src = str(r.get("source", "pattern"))
        pat = str(r.get("matched_pattern", src))
        typ = pat if src == "pattern" and pat not in ("nan", "None") else src
        leg = f"v15_{typ}"
        out.append(
            _trade_row(
                r["entry_time"],
                r["exit_time"],
                r["pnl"],
                leg,
                side=int(r["side"]),
                typ=typ,
            )
        )
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    start = args[0] if args else v16_config.BACKTEST_CONFIG.get("default_start", "2025-06-01")
    end = args[1] if len(args) > 1 else "2026-06-25"
    oos = _utc_ts(start)

    print("=" * 72)
    print("  GOLD v16 ALONE + v15+v16 COMBINED")
    print(f"  OOS: {start} → {end}")
    print("=" * 72)

    print("\nLoading gold 1m…")
    df = load_gold_1m(TRAIN_START, end)

    print("Running v16 momentum (pre-close struct-hold ET ML)…")
    mom = run_v16_momentum(df, oos)
    mom_trades = _df_to_trades(mom, "v16_momentum", "v16_momentum")

    print("Running v16 dip short rip (ML)…")
    dip = run_v16_dip(df, oos)
    dip_trades = _df_to_trades(dip, "v16_dip_short", "v16_dip_short")

    v16_raw = mom_trades + dip_trades
    v16_merged = merge_gold_trades(v16_raw)

    print("\n--- v16 ALONE (pre-merge) ---")
    _stats(mom_trades, "momentum preclose")
    _stats(dip_trades, "dip short rip")
    print(f"  {'raw total':28s} {len(v16_raw):4d} trades  PnL={sum(t['pnl'] for t in v16_raw):+.1f}")

    print("\n--- v16 ALONE (single-slot merged) ---")
    v16_stat = _stats(v16_merged, "v16 combined")
    _by_leg(v16_merged)

    print("\nRunning v15 hybrid backtest…")
    v15_trades = run_v15_hybrid(start, end)

    print("\n--- v15 ALONE ---")
    v15_stat = _stats(v15_trades, "v15 hybrid")
    _by_leg(v15_trades)

    combined_raw = v15_trades + v16_raw
    combined = merge_gold_trades(combined_raw)

    print("\n--- v15 + v16 COMBINED (single slot) ---")
    comb_stat = _stats(combined, "v15+v16 combined")
    _by_leg(combined)
    print(f"\n  Raw legs total: {len(combined_raw)} → merged: {len(combined)} ({len(combined_raw)-len(combined)} dropped)")

    print("\n--- Comparison ---")
    print(f"  v15 alone     : {v15_stat['trades']:4d}t  PnL={v15_stat['pnl']:+.1f}")
    print(f"  v16 alone     : {v16_stat['trades']:4d}t  PnL={v16_stat['pnl']:+.1f}")
    print(f"  v15+v16       : {comb_stat['trades']:4d}t  PnL={comb_stat['pnl']:+.1f}")
    delta = comb_stat["pnl"] - v15_stat["pnl"]
    print(f"  vs v15 alone  : {delta:+.1f} pt ({100*delta/max(abs(v15_stat['pnl']),1):+.1f}%)")

    out = PROJECT_ROOT / "runtime" / "gold_v15_v16_combined_trades.csv"
    pd.DataFrame(combined).to_csv(out, index=False)
    print(f"\n  CSV: {out}")
    print("DONE.")


if __name__ == "__main__":
    main()
