#!/usr/bin/env python3
"""
v16 scalp system — fresh start (no v15 hybrid / patterns / S1-S2).

Idea:
  1. Burst bars in London+NY (range + volume expansion)
  2. ML picks long vs short (dual walk-forward classifiers)
  3. Platform exits: +5 close half, runner lock +5, +10 close all

Usage:
  PYTHONPATH=. python3 v16/research/scalp_scaleout_backtest.py
  PYTHONPATH=. python3 v16/research/scalp_scaleout_backtest.py 2025-06-01 2026-06-25
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.features import build_features, feature_columns, session_mask
from v16.backtest.ml import walk_forward_dual
from v16.backtest.scaleout_sim import simulate_scaleout_trade
from v16.backtest.signals import build_labeled_set, candidate_mask, fade_side
from v16.config.v16_config import BACKTEST_CONFIG, EXIT_CONFIG, ML_CONFIG, SIGNAL_CONFIG
from v16.data.load_gold import load_gold_1m


def _exit_kwargs() -> dict:
    return {
        "first_scale_pnl": EXIT_CONFIG["first_scale_pnl"],
        "first_scale_frac": EXIT_CONFIG["first_scale_frac"],
        "final_scale_pnl": EXIT_CONFIG["final_scale_pnl"],
        "initial_sl": EXIT_CONFIG["initial_sl"],
        "runner_lock_pnl": EXIT_CONFIG["runner_lock_pnl"],
        "horizon": EXIT_CONFIG["horizon_minutes"],
    }


def _oracle_best_side(labeled: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Upper bound: always pick better of long/short at each candidate."""
    rows = []
    for ts, row in labeled.iterrows():
        side = int(row["best_side"])
        entry_idx = int(row["entry_idx"])
        nxt = df.iloc[entry_idx]
        ep = float(nxt["open_ask"] if side == 1 else nxt["open_bid"])
        res = simulate_scaleout_trade(df, entry_idx, side, ep, **_exit_kwargs())
        rows.append(
            {
                "signal_ts": ts,
                "side": side,
                "pnl": res.pnl,
                "exit_reason": res.exit_reason,
                "scaled_half": res.scaled_half,
                "win": res.pnl > 0,
            }
        )
    return pd.DataFrame(rows)


def _naive_momentum(labeled: pd.DataFrame, df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ts, row in labeled.iterrows():
        side = 1 if feats.loc[ts, "ret_3"] >= 0 else -1
        entry_idx = int(row["entry_idx"])
        nxt = df.iloc[entry_idx]
        ep = float(nxt["open_ask"] if side == 1 else nxt["open_bid"])
        res = simulate_scaleout_trade(df, entry_idx, side, ep, **_exit_kwargs())
        rows.append(
            {
                "signal_ts": ts,
                "side": side,
                "pnl": res.pnl,
                "exit_reason": res.exit_reason,
                "scaled_half": res.scaled_half,
                "win": res.pnl > 0,
            }
        )
    return pd.DataFrame(rows)


def print_stats(name: str, tdf: pd.DataFrame) -> None:
    if tdf.empty:
        print(f"\n{name}: no trades")
        return
    print(f"\n{name}")
    print(f"  Trades     : {len(tdf)}")
    print(f"  Win rate   : {tdf['win'].mean()*100:.1f}%")
    print(f"  Scaled 50% : {tdf['scaled_half'].mean()*100:.1f}%")
    print(f"  Net PnL    : {tdf['pnl'].sum():+.1f}")
    print(f"  Avg/trade  : {tdf['pnl'].mean():+.2f}")
    for reason, grp in tdf.groupby("exit_reason"):
        print(
            f"    {reason:12s}: {len(grp):4d}  "
            f"WR={grp['win'].mean()*100:.0f}%  PnL={grp['pnl'].sum():+.1f}"
        )


def _fade_rule(labeled: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Mechanical fade: short after large up 15m, long after large down."""
    rows = []
    for ts, row in labeled.iterrows():
        if pd.isna(row.get("fade_side")):
            continue
        side = int(row["fade_side"])
        entry_idx = int(row["entry_idx"])
        nxt = df.iloc[entry_idx]
        ep = float(nxt["open_ask"] if side == 1 else nxt["open_bid"])
        res = simulate_scaleout_trade(df, entry_idx, side, ep, **_exit_kwargs())
        rows.append(
            {
                "signal_ts": ts,
                "side": side,
                "pnl": res.pnl,
                "exit_reason": res.exit_reason,
                "scaled_half": res.scaled_half,
                "win": res.pnl > 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    bt_start = args[0] if args else BACKTEST_CONFIG["default_start"]
    bt_end = args[1] if len(args) > 1 else pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

    print("=" * 70)
    print(f"  v16 SCALP (standalone)  |  {bt_start} → {bt_end}")
    print(
        f"  Exit: +{EXIT_CONFIG['first_scale_pnl']:.0f} half | "
        f"+{EXIT_CONFIG['final_scale_pnl']:.0f} all | "
        f"SL={EXIT_CONFIG['initial_sl']:.0f} | H={EXIT_CONFIG['horizon_minutes']}m"
    )
    print(
        f"  Signal mode: {SIGNAL_CONFIG.get('mode', 'burst')} | "
        f"sessions={','.join(SIGNAL_CONFIG['sessions'])}"
    )
    if SIGNAL_CONFIG.get("mode") in ("fade_15m", "both"):
        print(
            f"  15m fade: open mins {SIGNAL_CONFIG['fade_open_minutes']} | "
            f"prev body>={SIGNAL_CONFIG['fade_min_prev_body_pts']}"
        )
    print("=" * 70)

    df = load_gold_1m(bt_start, bt_end)
    feats = build_features(df)
    labeled = build_labeled_set(df, feats)
    n_cand = int(candidate_mask(feats, df.index).sum())
    print(f"\nCandidate bars: {len(labeled)} (mask true on {n_cand} bars)")

    if labeled.empty:
        print("No candidates — widen SIGNAL_CONFIG or date range.")
        return

    print_stats("Oracle (best of long/short)", _oracle_best_side(labeled, df))
    if SIGNAL_CONFIG.get("mode") in ("fade_15m", "both"):
        print_stats("15m fade rule (counter prior bar)", _fade_rule(labeled, df))
    print_stats("Naive momentum (ret_3)", _naive_momentum(labeled, df, feats))

    ml_trades = walk_forward_dual(df, labeled, feats, feature_columns(feats))
    print_stats(
        f"ML dual-side (p>={ML_CONFIG['prob_threshold']}, edge>={ML_CONFIG['min_edge']})",
        ml_trades,
    )

    out = PROJECT_ROOT / BACKTEST_CONFIG["trades_csv"]
    if not ml_trades.empty:
        ml_trades.to_csv(out, index=False)
        print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
