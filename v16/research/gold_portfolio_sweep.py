#!/usr/bin/env python3
"""Sweep gold portfolio combos — proper single-slot re-merge (v16 production legs).

  PYTHONPATH=. python3 v16/research/gold_portfolio_sweep.py [start] [end]
  PYTHONPATH=. python3 v16/research/gold_portfolio_sweep.py --use-cache   # skip v16 ML if cached

Caches v16 leg trades to runtime/gold_v16_{mom,dip}_trades.csv after first run.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from config.pattern_registry import PATTERN_REGISTRY, PRODUCTION_PATTERNS
from v16._paths import PROJECT_ROOT
from v16.backtest.dip_short_rip_run import run_dip_short_rip
from v16.backtest.features import build_features
from v16.backtest.impulse_features import attach_structure_features, structure_kwargs
from v16.backtest.impulse_ml import filter_signal_table, walk_forward_model_scores
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.momentum_15m_hold import build_labeled_set, build_signal_table
from v16.config.gold_config import GOLD_TRAIN_START, MOMENTUM, DIP_SHORT
from v16.gold.hybrid_legs import run_hybrid_legs
from v16.gold.merge import df_to_trades, merge_gold_trades
from v16.gold.v16_legs import run_v16_legs


def _utc_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

MOM_CACHE = PROJECT_ROOT / "runtime" / "gold_v16_mom_trades.csv"
DIP_CACHE = PROJECT_ROOT / "runtime" / "gold_v16_dip_trades.csv"
SWEEP_CSV = PROJECT_ROOT / "runtime" / "gold_portfolio_sweep.csv"


def _trades_from_csv(path: Path) -> list[dict]:
    tdf = pd.read_csv(path)
    tdf["entry"] = pd.to_datetime(tdf["entry"], utc=True)
    tdf["exit"] = pd.to_datetime(tdf["exit"], utc=True)
    return tdf.to_dict("records")


def _save_trades(trades: list[dict], path: Path) -> None:
    pd.DataFrame(trades).to_csv(path, index=False)


def _load_or_run_v16_legs(df, oos: pd.Timestamp, use_cache: bool) -> tuple[list[dict], list[dict]]:
    if use_cache and MOM_CACHE.exists() and DIP_CACHE.exists():
        print("  (using cached v16 leg trades)")
        return _trades_from_csv(MOM_CACHE), _trades_from_csv(DIP_CACHE)

    print("Running v16 momentum ML…")
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
    ml_filt = filter_signal_table(signals, scores_oos[scores_oos["p_win"] >= MOMENTUM["ml_prob"]])
    cfg["impulse_stop"] = {
        **cfg.get("impulse_stop", {}),
        "tp_enabled": False,
        "horizon": MOMENTUM["horizon"],
        "exit_on_structure_change": True,
        "exit_on_structure_change_min_pnl": -1e9,
    }
    mom_tdf = simulate_position_impulse_stop(df_oos, ml_filt, cfg=cfg)
    mom_trades = df_to_trades(mom_tdf, "v16_momentum", "v16_momentum")

    print("Running v16 dip short ML…")
    dip_cfg = copy.deepcopy(v16_config.DIP_SHORT_RIP)
    dip_feats = build_features(df_oos)
    dip_tdf = run_dip_short_rip(df_oos, dip_feats, dip_cfg, mechanical=False, ml_prob=float(DIP_SHORT["ml_prob"]))
    dip_trades = df_to_trades(dip_tdf, "v16_dip_short", "v16_dip_short")

    _save_trades(mom_trades, MOM_CACHE)
    _save_trades(dip_trades, DIP_CACHE)
    return mom_trades, dip_trades


def _portfolio_row(label: str, trades: list[dict], raw_n: int) -> dict:
    pnls = [t["pnl"] for t in trades]
    return {
        "portfolio": label,
        "trades": len(trades),
        "raw_legs": raw_n,
        "dropped": raw_n - len(trades),
        "pnl": round(sum(pnls), 1),
        "wr": round(100 * sum(1 for p in pnls if p > 0) / max(1, len(pnls)), 1),
        "avg": round(sum(pnls) / max(1, len(pnls)), 2),
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    use_cache = "--use-cache" in flags or "--fast" in flags
    start = args[0] if args else "2025-06-01"
    end = args[1] if len(args) > 1 else "2026-06-25"
    oos = _utc_ts(start)

    print("=" * 72)
    print("  GOLD PORTFOLIO SWEEP (single-slot re-merge)")
    print(f"  OOS: {start} → {end}")
    print("=" * 72)

    df = load_gold_1m(GOLD_TRAIN_START, end)
    mom_trades, dip_trades = _load_or_run_v16_legs(df, oos, use_cache)

    print("\nRunning hybrid patterns + energetic…")
    hybrid_trades = run_hybrid_legs(start, end, verbose=False)

    combos = [
        ("hybrid alone", hybrid_trades),
        ("v16 momentum alone", mom_trades),
        ("v16 dip alone", dip_trades),
        ("v16 mom+dip", mom_trades + dip_trades),
        ("hybrid + momentum", hybrid_trades + mom_trades),
        ("hybrid + dip", hybrid_trades + dip_trades),
        ("hybrid + mom + dip", hybrid_trades + mom_trades + dip_trades),
    ]

    rows = []
    best_trades = None
    best_label = ""
    best_pnl = -1e9

    print(f"\n{'portfolio':22s} {'raw':>5} {'merged':>6} {'drop':>5} {'PnL':>10} {'WR%':>6} {'avg':>7}")
    print("-" * 72)
    for label, raw in combos:
        merged = merge_gold_trades(raw)
        row = _portfolio_row(label, merged, len(raw))
        rows.append(row)
        print(
            f"{label:22s} {row['raw_legs']:5d} {row['trades']:6d} {row['dropped']:5d} "
            f"{row['pnl']:+10.1f} {row['wr']:6.1f} {row['avg']:+7.2f}"
        )
        if row["pnl"] > best_pnl:
            best_pnl = row["pnl"]
            best_label = label
            best_trades = merged

    sweep_df = pd.DataFrame(rows).sort_values("pnl", ascending=False)
    sweep_df.to_csv(SWEEP_CSV, index=False)

    hybrid_pnl = next(r["pnl"] for r in rows if r["portfolio"] == "hybrid alone")
    best = sweep_df.iloc[0]

    print("\n" + "=" * 72)
    print(f"  BEST: {best['portfolio']}  →  {best['trades']:.0f}t  PnL={best['pnl']:+.1f}  "
          f"(vs hybrid alone {hybrid_pnl:+.1f}, delta {best['pnl']-hybrid_pnl:+.1f})")
    print("=" * 72)

    if best_trades:
        out = PROJECT_ROOT / "runtime" / "gold_best_portfolio_trades.csv"
        pd.DataFrame(best_trades).to_csv(out, index=False)
        print(f"\n  Best portfolio trades: {out}")
        print(f"  Sweep table: {SWEEP_CSV}")

        print("\n  Best portfolio leg breakdown:")
        legs: dict[str, list] = {}
        for t in best_trades:
            legs.setdefault(str(t.get("_leg", "?")), []).append(t["pnl"])
        for leg, pnls in sorted(legs.items(), key=lambda x: -sum(x[1])):
            print(f"    {leg:22s} {len(pnls):4d}t  PnL={sum(pnls):+.1f}  avg={sum(pnls)/len(pnls):+.2f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
