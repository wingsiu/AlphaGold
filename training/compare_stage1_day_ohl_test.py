#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BASELINE = Path(
    "/Users/alpha/Desktop/python/AlphaGold/training/_tmp_feature_single_split_aligned_state_stop_sweep/"
    "w150_h25_thr0.008_d15_lt0.006_st0.008_ls12_ss18_r0_f2.5_p10.48_p20.5_sf.json"
)
DEFAULT_CANDIDATE = Path(
    "/Users/alpha/Desktop/python/AlphaGold/training/backtest_report_best_base_wr90_filter.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _get(report: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _f(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _i(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare promoted best-base baseline vs stage1 UTC+2 Dopen/DHigh/DLow test report")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)

    baseline_metrics = {
        "total_pnl": _f(_get(baseline, "directional_pnl", "all", "total_pnl", default=_get(baseline, "directional_pnl", "total_pnl"))),
        "avg_day": _f(_get(baseline, "directional_pnl", "all", "avg_day", default=_get(baseline, "directional_pnl", "avg_day"))),
        "profit_factor": _f(_get(baseline, "directional_pnl", "all", "profit_factor")),
        "positive_days_pct": _f(_get(baseline, "directional_pnl", "all", "positive_days_pct", default=_get(baseline, "directional_pnl", "positive_days_pct"))),
        "trades": _i(_get(baseline, "directional_pnl", "all", "trades", default=_get(baseline, "directional_pnl", "trades"))),
        "stage1_bal_acc": _f(_get(baseline, "stage1", "balanced_accuracy")),
        "stage2_bal_acc": _f(_get(baseline, "stage2", "balanced_accuracy")),
        "trade_max_dd": _f(_get(baseline, "directional_pnl", "all", "trade_max_drawdown", default=_get(baseline, "directional_pnl", "max_drawdown"))),
        "daily_max_dd": _f(_get(baseline, "directional_pnl", "all", "daily_max_drawdown")),
    }
    candidate_metrics = {
        "total_pnl": _f(_get(candidate, "directional_pnl", "all", "total_pnl", default=_get(candidate, "directional_pnl", "total_pnl"))),
        "avg_day": _f(_get(candidate, "directional_pnl", "all", "avg_day", default=_get(candidate, "directional_pnl", "avg_day"))),
        "profit_factor": _f(_get(candidate, "directional_pnl", "all", "profit_factor")),
        "positive_days_pct": _f(_get(candidate, "directional_pnl", "all", "positive_days_pct", default=_get(candidate, "directional_pnl", "positive_days_pct"))),
        "trades": _i(_get(candidate, "directional_pnl", "all", "trades", default=_get(candidate, "directional_pnl", "trades"))),
        "stage1_bal_acc": _f(_get(candidate, "stage1", "balanced_accuracy")),
        "stage2_bal_acc": _f(_get(candidate, "stage2", "balanced_accuracy")),
        "trade_max_dd": _f(_get(candidate, "directional_pnl", "all", "trade_max_drawdown", default=_get(candidate, "directional_pnl", "max_drawdown"))),
        "daily_max_dd": _f(_get(candidate, "directional_pnl", "all", "daily_max_drawdown")),
    }

    candidate_cfg = _get(candidate, "config", default={})
    feature_flag = bool(candidate_cfg.get("use_stage1_day_ohl_utc2", False))
    feature_names = list(candidate.get("stage1_extra_feature_names", [])) if isinstance(candidate.get("stage1_extra_feature_names"), list) else []

    print(f"baseline={baseline_path}")
    print(f"candidate={candidate_path}")
    print(f"candidate_use_stage1_day_ohl_utc2={feature_flag}")
    print(f"candidate_stage1_extra_feature_names={feature_names}")
    print()
    print(f"{'metric':<20} {'baseline':>12} {'candidate':>12} {'delta':>12}")
    for key in [
        "total_pnl",
        "avg_day",
        "profit_factor",
        "positive_days_pct",
        "trades",
        "stage1_bal_acc",
        "stage2_bal_acc",
        "trade_max_dd",
        "daily_max_dd",
    ]:
        b = baseline_metrics[key]
        c = candidate_metrics[key]
        d = c - b
        print(f"{key:<20} {b:>12.6f} {c:>12.6f} {d:>12.6f}")

    pnl_up = candidate_metrics["total_pnl"] > baseline_metrics["total_pnl"]
    pf_up = candidate_metrics["profit_factor"] >= baseline_metrics["profit_factor"]
    s1_up = candidate_metrics["stage1_bal_acc"] >= baseline_metrics["stage1_bal_acc"]
    s2_not_bad = candidate_metrics["stage2_bal_acc"] >= baseline_metrics["stage2_bal_acc"] - 0.005
    dd_not_worse = candidate_metrics["trade_max_dd"] >= baseline_metrics["trade_max_dd"]

    print()
    if not feature_flag:
        print("VERDICT: candidate report does not show the new stage1 UTC+2 day OHLC feature flag. Re-run the experiment first.")
        return 2

    if pnl_up and pf_up and s1_up and s2_not_bad and dd_not_worse:
        print("VERDICT: keep the new stage1 UTC+2 Dopen/DHigh/DLow features. They improved the matching baseline.")
        return 0

    print("VERDICT: do not promote yet. The new stage1 UTC+2 Dopen/DHigh/DLow features did not clearly beat the matching baseline.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

