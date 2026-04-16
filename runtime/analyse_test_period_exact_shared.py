#!/usr/bin/env python3
"""Compare saved best-base model families on exact shared test-period entries.

This is stricter than the common-window comparison: it keeps only trades whose
`entry_time` exists in all three saved trade CSVs.

Outputs:
- runtime/test_period_exact_shared_comparison.json
- runtime/test_period_exact_shared_comparison.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
TRAINING_DIR = PROJECT_ROOT / "training"
HK_TZ = ZoneInfo("Asia/Hong_Kong")
NY_TZ = ZoneInfo("America/New_York")
TRADING_DAY_CUTOFF_HOUR_NY = 17


@dataclass(frozen=True)
class ModelArtifact:
    label: str
    report_path: Path
    trades_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("ts", "entry_time", "exit_time", "last_target_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _to_hkt_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.tz_convert(HK_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def _trade_coverage(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "signal_start_utc": None,
            "signal_start_hkt": None,
            "signal_end_utc": None,
            "signal_end_hkt": None,
            "entry_start_utc": None,
            "entry_start_hkt": None,
            "entry_end_utc": None,
            "entry_end_hkt": None,
            "exit_end_utc": None,
            "exit_end_hkt": None,
        }
    signal_start = df["ts"].min().isoformat() if "ts" in df.columns else None
    signal_end = df["ts"].max().isoformat() if "ts" in df.columns else None
    entry_start = df["entry_time"].min().isoformat() if "entry_time" in df.columns else None
    entry_end = df["entry_time"].max().isoformat() if "entry_time" in df.columns else None
    exit_end = df["exit_time"].max().isoformat() if "exit_time" in df.columns else None
    return {
        "rows": int(len(df)),
        "signal_start_utc": signal_start,
        "signal_start_hkt": _to_hkt_str(signal_start),
        "signal_end_utc": signal_end,
        "signal_end_hkt": _to_hkt_str(signal_end),
        "entry_start_utc": entry_start,
        "entry_start_hkt": _to_hkt_str(entry_start),
        "entry_end_utc": entry_end,
        "entry_end_hkt": _to_hkt_str(entry_end),
        "exit_end_utc": exit_end,
        "exit_end_hkt": _to_hkt_str(exit_end),
    }


def _streak_stats(pnl_vals: np.ndarray) -> dict[str, int]:
    max_win = 0
    max_loss = 0
    cur_win = 0
    cur_loss = 0
    for v in pnl_vals:
        if v > 0:
            cur_win += 1
            cur_loss = 0
            max_win = max(max_win, cur_win)
        elif v < 0:
            cur_loss += 1
            cur_win = 0
            max_loss = max(max_loss, cur_loss)
        else:
            cur_win = 0
            cur_loss = 0
    current_streak = cur_win if cur_win > 0 else cur_loss
    return {
        "max_win_streak": int(max_win),
        "max_loss_streak": int(max_loss),
        "current_win_streak": int(cur_win),
        "current_loss_streak": int(cur_loss),
        "current_streak": int(current_streak),
    }


def _bucket_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "avg_trade": None,
            "median_trade": None,
            "profit_factor": None,
            "win_rate_pct": None,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "best_trade": None,
            "worst_trade": None,
            "avg_win": None,
            "avg_loss": None,
            "n_days": 0,
            "avg_day": None,
            "positive_days_pct": None,
            "trade_max_drawdown": 0.0,
            "daily_max_drawdown": 0.0,
            "exit_reason_counts": {},
            "side_counts": {},
            "entry_signal_prob_mean": None,
            "entry_signal_prob_median": None,
        }

    x = df.copy()
    x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True)
    x["exit_time"] = pd.to_datetime(x["exit_time"], utc=True)
    x["trading_day"] = (x["entry_time"].dt.tz_convert(NY_TZ) - pd.Timedelta(hours=TRADING_DAY_CUTOFF_HOUR_NY)).dt.floor("D")

    pnl = pd.to_numeric(x["pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    pf = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None
    daily = x.groupby("trading_day")["pnl"].sum()
    equity_trade = pnl.cumsum()
    trade_dd = float((equity_trade - equity_trade.cummax()).min())
    equity_day = daily.cumsum()
    daily_dd = float((equity_day - equity_day.cummax()).min()) if len(daily) else 0.0
    streak = _streak_stats(pnl.to_numpy(dtype=np.float64))
    entry_prob = pd.to_numeric(x.get("entry_signal_prob"), errors="coerce") if "entry_signal_prob" in x.columns else pd.Series(dtype=float)

    return {
        "trades": int(len(x)),
        "total_pnl": float(pnl.sum()),
        "avg_trade": float(pnl.mean()),
        "median_trade": float(pnl.median()),
        "profit_factor": float(pf) if pf is not None else None,
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "n_days": int(len(daily)),
        "avg_day": float(daily.mean()) if len(daily) else None,
        "positive_days_pct": float((daily > 0).mean() * 100.0) if len(daily) else None,
        "trade_max_drawdown": trade_dd,
        "daily_max_drawdown": daily_dd,
        "exit_reason_counts": {str(k): int(v) for k, v in x["exit_reason"].value_counts().to_dict().items()} if "exit_reason" in x.columns else {},
        "side_counts": {str(k): int(v) for k, v in x["side"].value_counts().to_dict().items()} if "side" in x.columns else {},
        "entry_signal_prob_mean": float(entry_prob.mean()) if len(entry_prob) else None,
        "entry_signal_prob_median": float(entry_prob.median()) if len(entry_prob) else None,
        **streak,
    }


def _directional_summary(df: pd.DataFrame) -> dict[str, Any]:
    all_stats = _bucket_stats(df)
    long_df = df[df["side"] == "up"].copy() if "side" in df.columns else df.iloc[0:0].copy()
    short_df = df[df["side"] == "down"].copy() if "side" in df.columns else df.iloc[0:0].copy()
    return {
        "all": all_stats,
        "long_up": _bucket_stats(long_df),
        "short_down": _bucket_stats(short_df),
        "coverage": _trade_coverage(df),
    }


def _filter_exact_shared(df: pd.DataFrame, shared_entries: set[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if df["entry_time"].astype(str).duplicated().any():
        dupes = df.loc[df["entry_time"].astype(str).duplicated(), "entry_time"].astype(str).unique().tolist()
        raise ValueError(f"Duplicate entry_time rows found, cannot do exact-shared comparison safely: {dupes[:5]}")
    mask = df["entry_time"].astype(str).isin(shared_entries)
    return df.loc[mask].copy().sort_values("entry_time")


def main() -> int:
    artifacts = [
        ModelArtifact(
            label="promoted_single_split",
            report_path=TRAINING_DIR / "backtest_report_best_base_corrected.json",
            trades_path=TRAINING_DIR / "backtest_trades_best_base_corrected.csv",
        ),
        ModelArtifact(
            label="walk_forward_fixed",
            report_path=TRAINING_DIR / "backtest_report_best_base_wf_10cycles.json",
            trades_path=TRAINING_DIR / "backtest_trades_best_base_wf_10cycles.csv",
        ),
        ModelArtifact(
            label="walk_forward_prob_sweep",
            report_path=TRAINING_DIR / "backtest_report_best_base_wf_10cycles_prob_sweep.json",
            trades_path=TRAINING_DIR / "backtest_trades_best_base_wf_10cycles_prob_sweep.csv",
        ),
    ]

    raw_reports = {a.label: _read_json(a.report_path) for a in artifacts}
    raw_trades = {a.label: _read_trades(a.trades_path) for a in artifacts}
    coverage = {label: _trade_coverage(df) for label, df in raw_trades.items()}

    entry_sets = {label: set(df["entry_time"].astype(str)) for label, df in raw_trades.items()}
    shared_entries = set.intersection(*entry_sets.values()) if entry_sets else set()

    shared_trades = {label: _filter_exact_shared(df, shared_entries) for label, df in raw_trades.items()}
    shared_stats = {label: _directional_summary(df) for label, df in shared_trades.items()}

    shared_start = min(pd.Timestamp(s) for s in shared_entries) if shared_entries else None
    shared_end = max(pd.Timestamp(s) for s in shared_entries) if shared_entries else None

    ranking = sorted(
        [
            {
                "label": label,
                "total_pnl": stats["all"]["total_pnl"],
                "profit_factor": stats["all"]["profit_factor"],
                "avg_day": stats["all"]["avg_day"],
                "trades": stats["all"]["trades"],
            }
            for label, stats in shared_stats.items()
        ],
        key=lambda x: (x["total_pnl"], -999.0 if x["profit_factor"] is None else x["profit_factor"]),
        reverse=True,
    )

    top_label = ranking[0]["label"] if ranking else None
    result = {
        "comparison_basis": {
            "type": "exact_shared_entry_timestamps",
            "shared_entry_count": int(len(shared_entries)),
            "shared_window_start_utc": None if shared_start is None else shared_start.isoformat(),
            "shared_window_start_hkt": _to_hkt_str(shared_start),
            "shared_window_end_utc": None if shared_end is None else shared_end.isoformat(),
            "shared_window_end_hkt": _to_hkt_str(shared_end),
        },
        "models": {},
        "ranking_by_exact_shared_total_pnl": ranking,
        "recommendation": {
            "best_exact_shared_total_pnl": top_label,
            "practical_suggestion": (
                "promoted_single_split_still_leads_on_exact_shared_subset"
                if top_label == "promoted_single_split"
                else "exact_shared_subset_favors_non_promoted_model_review_manually"
            ),
        },
    }

    for artifact in artifacts:
        full_pnl = raw_reports[artifact.label]["directional_pnl"]
        result["models"][artifact.label] = {
            "report_path": str(artifact.report_path.relative_to(PROJECT_ROOT)),
            "trades_path": str(artifact.trades_path.relative_to(PROJECT_ROOT)),
            "full_trade_coverage": coverage[artifact.label],
            "full_saved_directional_pnl": {
                "trades": int(full_pnl.get("trades", 0)),
                "total_pnl": float(full_pnl.get("total_pnl", 0.0)),
                "avg_day": full_pnl.get("avg_day"),
                "positive_days_pct": full_pnl.get("positive_days_pct"),
                "profit_factor": (full_pnl.get("all") or {}).get("profit_factor"),
            },
            "exact_shared_subset": shared_stats[artifact.label],
        }

    json_out = RUNTIME_DIR / "test_period_exact_shared_comparison.json"
    md_out = RUNTIME_DIR / "test_period_exact_shared_comparison.md"
    json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def fmt_model(label: str) -> str:
        block = result["models"][label]
        stats = block["exact_shared_subset"]
        overlap_all = stats["all"]
        overlap_long = stats["long_up"]
        overlap_short = stats["short_down"]
        return (
            f"- **{label}**\n"
            f"  - exact shared trades: {overlap_all['trades']}\n"
            f"  - total pnl: {overlap_all['total_pnl']:.2f}\n"
            f"  - profit factor: {overlap_all['profit_factor']}\n"
            f"  - avg day: {overlap_all['avg_day']}\n"
            f"  - positive days %: {overlap_all['positive_days_pct']}\n"
            f"  - long/up: trades={overlap_long['trades']} pnl={overlap_long['total_pnl']:.2f}\n"
            f"  - short/down: trades={overlap_short['trades']} pnl={overlap_short['total_pnl']:.2f}\n"
            f"  - exits: {overlap_all['exit_reason_counts']}"
        )

    md = f"""# Saved test-period exact shared comparison

## Comparison basis
- Type: exact shared `entry_time` timestamps across all three models
- Shared entry count: `{result['comparison_basis']['shared_entry_count']}`
- Shared window start (HKT): `{result['comparison_basis']['shared_window_start_hkt']}`
- Shared window end (HKT): `{result['comparison_basis']['shared_window_end_hkt']}`

## Ranking by exact-shared total pnl
{chr(10).join([f"{idx+1}. `{row['label']}` pnl={row['total_pnl']:.2f} pf={row['profit_factor']} trades={row['trades']}" for idx, row in enumerate(ranking)])}

## Per-model exact-shared results
{chr(10).join(fmt_model(a.label) for a in artifacts)}

## Practical suggestion
- Best exact-shared pnl: `{result['recommendation']['best_exact_shared_total_pnl']}`
- Suggested stance: `{result['recommendation']['practical_suggestion']}`

## Interpretation
- This is stricter than the common-window report because it compares only identical saved `entry_time` timestamps.
- If the promoted single-split still leads here, that strengthens the case that its test-period edge is not just coming from extra windows outside the shared subset.
- If fixed-gate remains closer on stability but not pnl, keep using it as the main robustness benchmark rather than replacing the promoted artifact outright.
"""
    md_out.write_text(md, encoding="utf-8")

    print(f"Saved: {json_out}")
    print(f"Saved: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

