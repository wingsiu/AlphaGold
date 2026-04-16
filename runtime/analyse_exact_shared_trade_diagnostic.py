#!/usr/bin/env python3
"""Explain why the promoted model beats the fixed WF model on shared trades.

Compares the promoted single-split trades against the fixed-gate walk-forward
trades on the exact shared entry timestamps already used in the strict test
comparison.

Outputs:
- runtime/test_period_exact_shared_fixed_gate_diagnostic.json
- runtime/test_period_exact_shared_fixed_gate_diagnostic.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
TRAINING_DIR = PROJECT_ROOT / "training"
HK_TZ = ZoneInfo("Asia/Hong_Kong")

PROMOTED_TRADES = TRAINING_DIR / "backtest_trades_best_base_corrected.csv"
FIXED_WF_TRADES = TRAINING_DIR / "backtest_trades_best_base_wf_10cycles.csv"
PROB_WF_TRADES = TRAINING_DIR / "backtest_trades_best_base_wf_10cycles_prob_sweep.csv"


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


def _read_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("ts", "entry_time", "exit_time", "last_target_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _prepare(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    x = df.copy()
    if x["entry_time"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate entry_time rows found for {prefix}")
    x["entry_time_key"] = x["entry_time"].astype(str)
    keep = {
        "ts": f"{prefix}_signal_time",
        "entry_time": "entry_time",
        "exit_time": f"{prefix}_exit_time",
        "side": f"{prefix}_side",
        "pnl": f"{prefix}_pnl",
        "entry_signal_prob": f"{prefix}_entry_signal_prob",
        "exit_reason": f"{prefix}_exit_reason",
    }
    cols = [c for c in keep if c in x.columns] + ["entry_time_key"]
    x = x[cols].rename(columns=keep)
    return x


def _shared_entries() -> set[str]:
    entry_sets = []
    for path in (PROMOTED_TRADES, FIXED_WF_TRADES, PROB_WF_TRADES):
        df = _read_trades(path)
        entry_sets.append(set(df["entry_time"].astype(str)))
    return set.intersection(*entry_sets)


def _transition_summary(df: pd.DataFrame, left: str, right: str, value_col: str) -> list[dict[str, Any]]:
    grouped = (
        df.groupby([left, right], dropna=False)[value_col]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .sort_values(["sum", "count"], ascending=[False, False])
    )
    out: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        out.append(
            {
                left: None if pd.isna(row[left]) else str(row[left]),
                right: None if pd.isna(row[right]) else str(row[right]),
                "count": int(row["count"]),
                "sum": float(row["sum"]),
                "mean": float(row["mean"]),
            }
        )
    return out


def _top_rows(df: pd.DataFrame, n: int, ascending: bool) -> list[dict[str, Any]]:
    cols = [
        "entry_time",
        "promoted_signal_time",
        "fixed_signal_time",
        "promoted_side",
        "fixed_side",
        "promoted_entry_signal_prob",
        "fixed_entry_signal_prob",
        "promoted_exit_reason",
        "fixed_exit_reason",
        "promoted_pnl",
        "fixed_pnl",
        "pnl_delta_promoted_minus_fixed",
    ]
    x = df.sort_values("pnl_delta_promoted_minus_fixed", ascending=ascending).head(n).copy()
    for col in ("entry_time", "promoted_signal_time", "fixed_signal_time"):
        if col in x.columns:
            x[col] = x[col].astype(str)
    return x[cols].to_dict(orient="records")


def main() -> int:
    shared_entries = _shared_entries()

    promoted = _prepare(_read_trades(PROMOTED_TRADES), "promoted")
    fixed = _prepare(_read_trades(FIXED_WF_TRADES), "fixed")

    promoted = promoted[promoted["entry_time_key"].isin(shared_entries)].copy()
    fixed = fixed[fixed["entry_time_key"].isin(shared_entries)].copy()

    merged = promoted.merge(fixed, on=["entry_time_key", "entry_time"], how="inner")
    merged["pnl_delta_promoted_minus_fixed"] = merged["promoted_pnl"] - merged["fixed_pnl"]
    merged["prob_delta_promoted_minus_fixed"] = (
        pd.to_numeric(merged.get("promoted_entry_signal_prob"), errors="coerce")
        - pd.to_numeric(merged.get("fixed_entry_signal_prob"), errors="coerce")
    )
    merged["same_side"] = merged["promoted_side"] == merged["fixed_side"]
    merged["same_exit_reason"] = merged["promoted_exit_reason"] == merged["fixed_exit_reason"]
    merged["side_pair"] = merged["promoted_side"].astype(str) + "->" + merged["fixed_side"].astype(str)
    merged["exit_pair"] = merged["promoted_exit_reason"].astype(str) + "->" + merged["fixed_exit_reason"].astype(str)

    summary = {
        "shared_entry_count": int(len(merged)),
        "shared_window_start_utc": merged["entry_time"].min().isoformat() if not merged.empty else None,
        "shared_window_start_hkt": _to_hkt_str(merged["entry_time"].min()) if not merged.empty else None,
        "shared_window_end_utc": merged["entry_time"].max().isoformat() if not merged.empty else None,
        "shared_window_end_hkt": _to_hkt_str(merged["entry_time"].max()) if not merged.empty else None,
        "totals": {
            "promoted_total_pnl": float(merged["promoted_pnl"].sum()) if not merged.empty else 0.0,
            "fixed_total_pnl": float(merged["fixed_pnl"].sum()) if not merged.empty else 0.0,
            "promoted_minus_fixed": float(merged["pnl_delta_promoted_minus_fixed"].sum()) if not merged.empty else 0.0,
            "avg_trade_delta": float(merged["pnl_delta_promoted_minus_fixed"].mean()) if not merged.empty else None,
        },
        "agreement": {
            "same_side_count": int(merged["same_side"].sum()) if not merged.empty else 0,
            "same_side_pct": float(merged["same_side"].mean() * 100.0) if not merged.empty else None,
            "same_exit_reason_count": int(merged["same_exit_reason"].sum()) if not merged.empty else 0,
            "same_exit_reason_pct": float(merged["same_exit_reason"].mean() * 100.0) if not merged.empty else None,
        },
        "probability": {
            "promoted_entry_prob_mean": float(pd.to_numeric(merged["promoted_entry_signal_prob"], errors="coerce").mean()) if not merged.empty else None,
            "fixed_entry_prob_mean": float(pd.to_numeric(merged["fixed_entry_signal_prob"], errors="coerce").mean()) if not merged.empty else None,
            "avg_delta_promoted_minus_fixed": float(pd.to_numeric(merged["prob_delta_promoted_minus_fixed"], errors="coerce").mean()) if not merged.empty else None,
        },
        "by_side_pair": _transition_summary(merged, "promoted_side", "fixed_side", "pnl_delta_promoted_minus_fixed"),
        "by_exit_reason_pair": _transition_summary(merged, "promoted_exit_reason", "fixed_exit_reason", "pnl_delta_promoted_minus_fixed"),
        "top_positive_delta_trades": _top_rows(merged, 10, ascending=False),
        "top_negative_delta_trades": _top_rows(merged, 10, ascending=True),
    }

    json_out = RUNTIME_DIR / "test_period_exact_shared_fixed_gate_diagnostic.json"
    md_out = RUNTIME_DIR / "test_period_exact_shared_fixed_gate_diagnostic.md"
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    top_side_pairs = summary["by_side_pair"][:6]
    top_exit_pairs = summary["by_exit_reason_pair"][:8]

    md = f"""# Exact shared trade diagnostic: promoted vs fixed WF

## Shared basis
- Shared trades: `{summary['shared_entry_count']}`
- Start (HKT): `{summary['shared_window_start_hkt']}`
- End (HKT): `{summary['shared_window_end_hkt']}`

## Net edge
- Promoted total pnl: `{summary['totals']['promoted_total_pnl']:.2f}`
- Fixed WF total pnl: `{summary['totals']['fixed_total_pnl']:.2f}`
- Promoted minus fixed: `{summary['totals']['promoted_minus_fixed']:.2f}`
- Avg trade delta: `{summary['totals']['avg_trade_delta']:.2f}`

## Agreement
- Same side count: `{summary['agreement']['same_side_count']}`
- Same side %: `{summary['agreement']['same_side_pct']}`
- Same exit reason count: `{summary['agreement']['same_exit_reason_count']}`
- Same exit reason %: `{summary['agreement']['same_exit_reason_pct']}`

## Entry probability
- Promoted mean entry prob: `{summary['probability']['promoted_entry_prob_mean']}`
- Fixed WF mean entry prob: `{summary['probability']['fixed_entry_prob_mean']}`
- Mean prob delta (promoted - fixed): `{summary['probability']['avg_delta_promoted_minus_fixed']}`

## Top side-pair contributors
{chr(10).join([f"- `{row['promoted_side']} -> {row['fixed_side']}` count={row['count']} delta_sum={row['sum']:.2f} delta_mean={row['mean']:.2f}" for row in top_side_pairs])}

## Top exit-reason contributors
{chr(10).join([f"- `{row['promoted_exit_reason']} -> {row['fixed_exit_reason']}` count={row['count']} delta_sum={row['sum']:.2f} delta_mean={row['mean']:.2f}" for row in top_exit_pairs])}

## Interpretation
- Positive `delta_sum` means the promoted model outperformed fixed WF in that bucket.
- The strongest explanatory buckets are usually where side choice differs or the exit path differs materially.
- Use the JSON report for the top positive/negative exact-shared trades if you want to inspect the specific timestamps driving the edge.
"""
    md_out.write_text(md, encoding="utf-8")

    print(f"Saved: {json_out}")
    print(f"Saved: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

