#!/usr/bin/env python3
"""
A/B: same-direction refresh with vs without stop trailing.

Loads pattern backtest data once, then simulates:
  A) target-only upgrade (exec max TP/H, no stop trail)
  B) full upgrade     (exec max TP/SL/H + stop trail) — current default
  C) legacy refresh   (global TP/H trail only, no exec upgrade, no stop trail)

Usage:
  .venv/bin/python3 ab_pattern_upgrade.py
  .venv/bin/python3 ab_pattern_upgrade.py 2025-06-01 2026-05-23
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

from v14._paths import PROJECT_ROOT

import joblib
import numpy as np
import pandas as pd

from v14.backtest.backtest_core import simulate_v13_core
from config.v14_config import EXECUTION_CONFIG, TIME_FILTER_CONFIG, WF_CONFIG
from config.v14_patterns import PATTERN_MODEL_DIR, PATTERN_REGISTRY, collect_pa_groups
from xgboost_filter_model.pattern_router import assign_patterns
from xgboost_filter_model.pattern_training import (
    cycle_model_path,
    iter_wf_cycles,
    pattern_variant_tag,
    prod_model_path,
    wf_anchor_ts,
)
from xgboost_filter_model.time_slot_filter import is_blocked_entry, load_weak_filter
from xgboost_filter_model.train_filter_1min import load_price_data
from xgboost_filter_model.train_filter_v14 import prepare_data_v14
from xgboost_filter_model.train_stage2_v14_directional import prepare_directional_data_v14


def _parse_dates(argv: list[str]) -> tuple[str, str]:
    today_str = date.today().strftime("%Y-%m-%d")
    bt_start = WF_CONFIG["wf_start"]
    bt_end = today_str
    date_args = [a for a in argv if a not in PATTERN_REGISTRY]
    if len(date_args) == 1:
        if date_args[0].isdigit():
            bt_start = (date.today() - timedelta(days=int(date_args[0]))).strftime("%Y-%m-%d")
        else:
            bt_start = date_args[0]
    elif len(date_args) >= 2:
        bt_start, bt_end = date_args[0], date_args[1]
    return bt_start, bt_end


def build_sim_df(bt_start: str, bt_end: str) -> tuple[pd.DataFrame, list | None]:
    warmup_days = int(WF_CONFIG.get("feature_warmup_days", 120))
    load_start_dt = max(
        pd.to_datetime(WF_CONFIG["full_start"]),
        pd.to_datetime(bt_start) - pd.Timedelta(days=warmup_days),
    )
    load_start = load_start_dt.strftime("%Y-%m-%d")
    bt_end_date = bt_end.split("T")[0] if "T" in bt_end else bt_end
    load_end = (pd.to_datetime(bt_end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bt_start_date = bt_start.split("T")[0] if "T" in bt_start else bt_start

    print(f"Loading pattern data {load_start} → {bt_end}…")
    df = prepare_data_v14(start_date=load_start, end_date=load_end, energetic_filter=False,
                          pa_groups=collect_pa_groups(list(PATTERN_REGISTRY.keys())))
    df = prepare_directional_data_v14(df)

    bt_start_dt = pd.to_datetime(bt_start)
    if bt_start_dt.tzinfo is None:
        bt_start_dt = bt_start_dt.tz_localize("UTC")
    else:
        bt_start_dt = bt_start_dt.tz_convert("UTC")

    df_test = df[df.index >= bt_start_dt].copy()
    df_test = assign_patterns(df_test)

    models: dict[str, dict] = {}
    for name, spec in PATTERN_REGISTRY.items():
        ex = spec["execution"]
        variant = pattern_variant_tag(ex["horizon"], ex["tp"], ex["sl"])
        pdir = PATTERN_MODEL_DIR / name / variant
        mp = prod_model_path(pdir)
        if not mp.exists():
            raise FileNotFoundError(f"Missing model: {mp}")
        models[name] = {"prod": joblib.load(mp), "spec": spec, "dir": pdir}

    wf_anchor = wf_anchor_ts()
    end_dt = pd.to_datetime(bt_end).tz_localize("UTC") + pd.Timedelta(days=1)

    for col in ("prob", "s1_prob", "s2_prob", "matched_pattern", "exec_tp", "exec_sl", "exec_horizon"):
        df_test[col] = np.nan
    df_test["side_signal"] = 0

    for cycle, current_start, current_end in iter_wf_cycles(bt_start_dt, end_dt, wf_anchor):
        chunk = (df_test.index >= current_start) & (df_test.index < current_end)
        if not chunk.any():
            continue
        for name, m in models.items():
            pat_chunk = chunk & (df_test["pattern_name"] == name)
            if not pat_chunk.any():
                continue
            path = cycle_model_path(m["dir"], cycle, current_start.date())
            model = joblib.load(path) if path.exists() else m["prod"]
            model_feats = list(model.feature_names_in_)
            spec = m["spec"]
            ex = spec["execution"]
            prob_thresh = spec["thresholds"]["prob"]
            bias = spec["direction_bias"]
            rows = df_test.loc[pat_chunk]
            p = model.predict_proba(rows[model_feats])[:, 1]
            df_test.loc[pat_chunk, "prob"] = p
            df_test.loc[pat_chunk, "s1_prob"] = p
            sig = pat_chunk & (df_test["prob"] >= prob_thresh)
            side = 1 if bias == "long" else -1
            df_test.loc[sig, "side_signal"] = side
            if bias == "long":
                df_test.loc[sig, "s2_prob"] = df_test.loc[sig, "prob"]
            else:
                df_test.loc[sig, "s2_prob"] = 1.0 - df_test.loc[sig, "prob"]
            df_test.loc[sig, "exec_tp"] = ex["tp"]
            df_test.loc[sig, "exec_sl"] = ex["sl"]
            df_test.loc[sig, "exec_horizon"] = ex["horizon"]
            fired = sig & df_test["matched_pattern"].isna()
            df_test.loc[fired, "matched_pattern"] = name

    print(
        f"Entry signals: {(df_test['side_signal'] != 0).sum()} "
        f"(LONG={(df_test['side_signal']==1).sum()} SHORT={(df_test['side_signal']==-1).sum()})"
    )

    print("Loading full 1-min bars for exit simulation…")
    raw_df = load_price_data(start_date=bt_start_date, end_date=load_end)
    raw_df = raw_df[raw_df.index >= bt_start_dt].copy()

    sim_df = raw_df[["open", "high", "low", "close"]].copy()
    for col in ("side_signal", "s1_prob", "s2_prob", "matched_pattern", "exec_tp", "exec_sl", "exec_horizon"):
        sim_df[col] = df_test[col] if col in df_test.columns else np.nan
    sim_df["side_signal"] = sim_df["side_signal"].fillna(0).astype(int)

    weak_cells = None
    _filter_path = os.environ.get("V14_TIME_FILTER_JSON", "").strip()
    if not _filter_path and TIME_FILTER_CONFIG.get("enabled"):
        _filter_path = str(PROJECT_ROOT / TIME_FILTER_CONFIG.get("weak_slots_json", ""))
    if _filter_path:
        weak_cells = load_weak_filter(_filter_path)

    return sim_df, weak_cells


def simulate_legacy_refresh(
    df_test: pd.DataFrame,
    tp: float,
    sl: float,
    horizon_minutes: int,
    config: dict | None = None,
    weak_period_cells=None,
) -> list[dict]:
    """Pre-upgrade refresh: global tp/h on trail, global tp/sl/h at entry."""
    if config is None:
        config = EXECUTION_CONFIG.copy()
    _spread = config.get("spread_default", 0.25)
    df_test = df_test.copy()
    if "open_ask" not in df_test.columns:
        df_test["open_ask"] = df_test["open"] + _spread
        df_test["open_bid"] = df_test["open"] - _spread
        df_test["close_ask"] = df_test["close"] + _spread
        df_test["close_bid"] = df_test["close"] - _spread
        df_test["high_ask"] = df_test["high"] + _spread
        df_test["low_bid"] = df_test["low"] - _spread

    s2_base = config.get("s2_threshold", 0.55)
    s2_increment = config.get("s2_loss_increment", 0.01)
    s2_max = config.get("s2_max_threshold", 0.70)
    close_on_reverse = config.get("close_on_reverse", True)

    all_trades, active_pos = [], None
    consecutive_losses = 0

    for i in range(len(df_test) - 1):
        row = df_test.iloc[i]
        next_row = df_test.iloc[i + 1]
        now_ts = row.name
        sig = int(row["side_signal"])
        reverse_flip_sig = None

        if active_pos:
            s = active_pos["side"]
            exit_info = None
            if s == 1:
                if row["low_bid"] <= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif row["high_ask"] >= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif now_ts >= active_pos["timeout"]:
                    exit_info = (row["close_bid"], "timeout")
            else:
                if row["high_ask"] >= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif row["low_bid"] <= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif now_ts >= active_pos["timeout"]:
                    exit_info = (row["close_ask"], "timeout")
            if exit_info:
                px, reason = exit_info
                pnl = (px - active_pos["entry_price"]) * s
                all_trades.append({**active_pos, "exit_time": now_ts, "exit_price": px, "exit_reason": reason, "pnl": pnl})
                consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
                active_pos = None

        if active_pos:
            s = active_pos["side"]
            if close_on_reverse and sig != 0 and sig == -s:
                px = row["close_bid"] if s == 1 else row["close_ask"]
                pnl = (px - active_pos["entry_price"]) * s
                all_trades.append({**active_pos, "exit_time": now_ts, "exit_price": px, "exit_reason": "reverse_signal", "pnl": pnl})
                consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
                active_pos = None
                reverse_flip_sig = sig
            elif sig == s:
                active_pos["timeout"] = now_ts + pd.Timedelta(minutes=horizon_minutes)
                active_pos["target_updates"] = active_pos.get("target_updates", 0) + 1
                new_t = row["close"] + (tp if s == 1 else -tp)
                if (s == 1 and new_t > active_pos["target"]) or (s == -1 and new_t < active_pos["target"]):
                    active_pos["target"] = new_t

        if active_pos is None and sig != 0:
            if is_blocked_entry(now_ts, weak_period_cells):
                continue
            dynamic_s2 = min(s2_max, s2_base + consecutive_losses * s2_increment)
            s2_p = row["s2_prob"]
            if reverse_flip_sig is not None and sig == reverse_flip_sig:
                passes = True
            else:
                passes = (sig == 1 and s2_p >= dynamic_s2) or (sig == -1 and s2_p <= (1.0 - dynamic_s2))
            if passes:
                ep = next_row["open_ask"] if sig == 1 else next_row["open_bid"]
                active_pos = {
                    "side": sig,
                    "entry_time": next_row.name,
                    "entry_price": ep,
                    "stop": ep - sl if sig == 1 else ep + sl,
                    "target": ep + tp if sig == 1 else ep - tp,
                    "timeout": next_row.name + pd.Timedelta(minutes=horizon_minutes),
                    "target_updates": 0,
                    "stop_updates": 0,
                    "s1_prob": row["s1_prob"],
                    "s2_prob": s2_p,
                }
                if "matched_pattern" in df_test.columns and pd.notna(row.get("matched_pattern")):
                    active_pos["matched_pattern"] = row["matched_pattern"]

    return all_trades


def summarize(label: str, trades: list[dict]) -> dict:
    tdf = pd.DataFrame(trades)
    wins = int((tdf["pnl"] > 0).sum())
    net = float(tdf["pnl"].sum())
    wr = wins / len(tdf) * 100 if len(tdf) else 0.0
    max_dd = float((tdf["pnl"].cumsum() - tdf["pnl"].cumsum().cummax()).min())
    tdf["hold_min"] = (
        pd.to_datetime(tdf["exit_time"]) - pd.to_datetime(tdf["entry_time"])
    ).dt.total_seconds() / 60.0

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Trades   : {len(tdf)}")
    print(f"  Win rate : {wr:.1f}%")
    print(f"  Net PnL  : {net:+.1f}")
    print(f"  Avg/trade: {net/len(tdf):+.2f}")
    print(f"  Max DD   : {max_dd:+.1f}")
    print(f"  Avg hold : {tdf['hold_min'].mean():.1f} min  (median {tdf['hold_min'].median():.1f})")
    print("  Exit breakdown:")
    for reason, grp in tdf.groupby("exit_reason"):
        print(f"    {reason:12s}: {len(grp):4d}  avg={grp['pnl'].mean():+.2f}  hold={grp['hold_min'].mean():.1f}m")
    if "stop_updates" in tdf.columns:
        print(f"  stop_updates>0: {(tdf['stop_updates']>0).sum()} trades")

    return {
        "label": label,
        "trades": len(tdf),
        "wr": wr,
        "pnl": net,
        "max_dd": max_dd,
        "avg_hold": float(tdf["hold_min"].mean()),
    }


def main() -> None:
    bt_start, bt_end = _parse_dates(sys.argv[1:])
    sim_df, weak_cells = build_sim_df(bt_start, bt_end)

    tp = EXECUTION_CONFIG["tp"]
    sl = EXECUTION_CONFIG["sl"]
    horizon = EXECUTION_CONFIG["horizon"]
    base_cfg = EXECUTION_CONFIG.copy()
    base_cfg["close_on_reverse"] = False

    print(f"\n{'='*60}")
    print(f"  A/B: same-direction upgrade variants  |  {bt_start} → {bt_end}")
    print(f"{'='*60}")

    legacy = simulate_legacy_refresh(sim_df, tp, sl, horizon, config=base_cfg, weak_period_cells=weak_cells)
    target_only_cfg = {**base_cfg, "same_dir_refresh": "exec", "upgrade_stop": False}
    target_only = simulate_v13_core(sim_df, tp, sl, horizon, config=target_only_cfg, weak_period_cells=weak_cells)
    entry_cfg = {**base_cfg, "same_dir_refresh": "entry", "upgrade_stop": False}
    entry_refresh = simulate_v13_core(sim_df, tp, sl, horizon, config=entry_cfg, weak_period_cells=weak_cells)
    full_cfg = {**base_cfg, "same_dir_refresh": "exec", "upgrade_stop": True}
    full = simulate_v13_core(sim_df, tp, sl, horizon, config=full_cfg, weak_period_cells=weak_cells)

    rows = [
        summarize("C) Legacy — global TP/H refresh, no exec upgrade, no stop trail", legacy),
        summarize("2398) Entry refresh — entry TP/H trail (default)", entry_refresh),
        summarize("A) Exec upgrade — max exec TP/H, NO stop trail", target_only),
        summarize("B) Full — exec max TP/SL/H + stop trail", full),
    ]

    print(f"\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"  {'Variant':<8} {'Trades':>7} {'WR%':>6} {'PnL':>9} {'MaxDD':>8} {'Hold':>6}")
    for r in rows:
        short = r["label"].split(")")[0] + ")"
        print(
            f"  {short:<8} {r['trades']:>7} {r['wr']:>5.1f}% {r['pnl']:>+9.1f} "
            f"{r['max_dd']:>+8.1f} {r['avg_hold']:>5.1f}m"
        )

    delta_trades = rows[3]["trades"] - rows[2]["trades"]
    delta_pnl = rows[3]["pnl"] - rows[2]["pnl"]
    print(f"\n  Stop trail effect (B − A): {delta_trades:+d} trades, {delta_pnl:+.1f} PnL")
    print(f"  Default 2398 config: {rows[1]['trades']} trades, {rows[1]['pnl']:+.1f} PnL")


if __name__ == "__main__":
    main()
