#!/usr/bin/env python3
"""Parameter sweep runner for training/image_trend_ml.py.

Runs multiple configs, collects key metrics from each JSON report,
and writes a ranked CSV + text summary.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "training" / "image_trend_ml.py"


@dataclass(frozen=True)
class SweepConfig:
    window: int
    horizon: int
    trend_threshold: float
    long_target_threshold: float
    short_target_threshold: float
    long_adverse_limit: float
    short_adverse_limit: float
    min_15m_drop: float
    min_15m_rise: float
    max_flat_ratio: float
    stage1_min_prob: float
    stage2_min_prob: float


def _parse_float_list(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one float value")
    return vals


def _parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer value")
    return vals


def _parse_window_prob_map(raw: str | None) -> dict[int, list[float]]:
    """Parse mapping like: '120:0.55,0.60;150:0.65,0.70'."""
    if raw is None:
        return {}
    out: dict[int, list[float]] = {}
    chunks = [c.strip() for c in raw.split(";") if c.strip()]
    for ch in chunks:
        if ":" not in ch:
            raise ValueError(f"Invalid window-prob mapping entry: {ch!r}")
        w_raw, vals_raw = ch.split(":", 1)
        w = int(w_raw.strip())
        vals = _parse_float_list(vals_raw.strip())
        out[w] = vals
    return out


def _auto_reuse_allowed(args: argparse.Namespace) -> bool:
    return (
        args.eval_mode == "single_split"
        and not args.use_state_features
        and not args.use_optimize_prob
        and not args.reuse_model
    )


def _training_key(args: argparse.Namespace, cfg: SweepConfig) -> str:
    payload = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "timeframe": args.timeframe,
        "eval_mode": args.eval_mode,
        "test_size": float(args.test_size),
        "test_start_date": args.test_start_date,
        "max_samples": None if args.max_samples is None else int(args.max_samples),
        "disable_time_filter": bool(args.disable_time_filter),
        "window": int(cfg.window),
        "window_15m": int(args.window_15m),
        "min_window_range": float(args.min_window_range),
        "min_15m_drop": float(cfg.min_15m_drop),
        "min_15m_rise": float(cfg.min_15m_rise),
        "last_bar_wr90_high": None if args.last_bar_wr90_high is None else float(args.last_bar_wr90_high),
        "last_bar_wr90_low": None if args.last_bar_wr90_low is None else float(args.last_bar_wr90_low),
        "horizon": int(cfg.horizon),
        "trend_threshold": float(cfg.trend_threshold),
        "adverse_limit": float(args.adverse_limit),
        "long_target_threshold": float(cfg.long_target_threshold),
        "short_target_threshold": float(cfg.short_target_threshold),
        "long_adverse_limit": float(cfg.long_adverse_limit),
        "short_adverse_limit": float(cfg.short_adverse_limit),
        "classifier": args.classifier,
        "max_flat_ratio": float(cfg.max_flat_ratio),
        "random_state": int(args.random_state),
        "two_branch": bool(args.two_branch),
        "two_branch_stage": args.two_branch_stage,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_sweep_configs(args: argparse.Namespace) -> list[SweepConfig]:
    windows = _parse_int_list(args.window_values)
    horizons = _parse_int_list(args.horizon_values)
    flat_ratios = _parse_float_list(args.max_flat_ratio_values) if args.max_flat_ratio_values is not None else [float(args.max_flat_ratio)]
    if args.target_values is not None:
        thresholds = _parse_float_list(args.target_values)
    else:
        thresholds = _parse_float_list(args.trend_threshold_values)
    long_thresholds = _parse_float_list(args.long_target_values) if args.long_target_values is not None else thresholds
    short_thresholds = _parse_float_list(args.short_target_values) if args.short_target_values is not None else thresholds
    long_stops = _parse_float_list(args.long_stop_values) if args.long_stop_values is not None else [float(args.adverse_limit)]
    short_stops = _parse_float_list(args.short_stop_values) if args.short_stop_values is not None else [float(args.adverse_limit)]
    min_drops = _parse_float_list(args.min_15m_drop_values)
    min_rises = _parse_float_list(args.min_15m_rise_values)
    s1_probs = _parse_float_list(args.stage1_prob_values)
    s2_probs = _parse_float_list(args.stage2_prob_values)
    s1_by_w = _parse_window_prob_map(args.stage1_prob_by_window)
    s2_by_w = _parse_window_prob_map(args.stage2_prob_by_window)

    # When per-run probability optimization is enabled, stage prob inputs are
    # only initial seeds; sweeping many seed pairs creates duplicate runs.
    configs: list[SweepConfig] = []
    for w in windows:
        s1_vals = s1_by_w.get(w, s1_probs)
        s2_vals = s2_by_w.get(w, s2_probs)
        if args.use_optimize_prob:
            s1_vals = [s1_vals[0]]
            s2_vals = [s2_vals[0]]
        combos = itertools.product(
            horizons,
            thresholds,
            long_thresholds,
            short_thresholds,
            long_stops,
            short_stops,
            min_drops,
            min_rises,
            flat_ratios,
            s1_vals,
            s2_vals,
        )
        for h, t, lt, st, ls, ss, d, r, f, s1, s2 in combos:
            configs.append(
                SweepConfig(
                    window=int(w),
                    horizon=int(h),
                    trend_threshold=float(t),
                    long_target_threshold=float(lt),
                    short_target_threshold=float(st),
                    long_adverse_limit=float(ls),
                    short_adverse_limit=float(ss),
                    min_15m_drop=float(d),
                    min_15m_rise=float(r),
                    max_flat_ratio=float(f),
                    stage1_min_prob=float(s1),
                    stage2_min_prob=float(s2),
                )
            )
    return configs


def _run_cmd(cmd: list[str]) -> int:
    completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
    return int(completed.returncode)


def _safe_get(d: dict, path: Iterable[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def score_row(total_pnl: float, positive_days_pct: float | None, stage2_bacc: float) -> float:
    # Composite score for ranking: pnl first, then day consistency, then directional quality.
    pdp = 0.0 if positive_days_pct is None or math.isnan(positive_days_pct) else positive_days_pct
    return total_pnl + (pdp * 10.0) + (stage2_bacc * 1000.0)


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep image_trend_ml configs and rank outputs.")
    p.add_argument("--start-date", default="2025-05-20")
    p.add_argument("--end-date", default="2026-04-10")
    p.add_argument("--timeframe", default="1min")
    p.add_argument("--window", type=int, default=150)
    p.add_argument(
        "--window-values",
        default=None,
        help="Comma-separated window list (overrides --window), e.g. 120,150",
    )
    p.add_argument("--window-15m", type=int, default=0,
                   help="15-min image window length passed to each run (0=disable, default: 0)")
    p.add_argument("--min-window-range", type=float, default=40.0)
    p.add_argument("--adverse-limit", type=float, default=15.0)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--test-start-date", default=None,
                   help="Optional UTC test split anchor passed to image_trend_ml.py. Use '' to force ratio split.")
    p.add_argument("--eval-mode", choices=["single_split", "walk_forward"], default="single_split",
                   help="Evaluation mode forwarded to image_trend_ml.py (default: single_split).")
    p.add_argument("--disable-time-filter", action="store_true",
                   help="Forward --disable-time-filter to each run.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--classifier", choices=["gradient_boosting", "logistic"], default="gradient_boosting")
    p.add_argument("--max-flat-ratio", type=float, default=4.0)
    p.add_argument(
        "--max-flat-ratio-values",
        default=None,
        help="Comma-separated max-flat-ratio grid. Defaults to the single --max-flat-ratio value.",
    )

    p.add_argument("--horizon-values", default="60")
    p.add_argument("--trend-threshold-values", default="0.01")
    p.add_argument(
        "--target-values",
        default=None,
        help="Alias for --trend-threshold-values, e.g. 0.005,0.0075,0.01",
    )
    p.add_argument("--long-target-values", default=None,
                   help="Comma-separated long target thresholds. Defaults to shared target grid.")
    p.add_argument("--short-target-values", default=None,
                   help="Comma-separated short target thresholds. Defaults to shared target grid.")
    p.add_argument("--long-stop-values", default=None,
                   help="Comma-separated long stop values. Defaults to --adverse-limit.")
    p.add_argument("--short-stop-values", default=None,
                   help="Comma-separated short stop values. Defaults to --adverse-limit.")
    p.add_argument("--min-15m-drop-values", default="10,15,20")
    p.add_argument("--min-15m-rise-values", default="0")
    p.add_argument("--last-bar-wr90-high", type=float, default=None,
                   help="Forward --last-bar-wr90-high to image_trend_ml.py.")
    p.add_argument("--last-bar-wr90-low", type=float, default=None,
                   help="Forward --last-bar-wr90-low to image_trend_ml.py.")
    p.add_argument("--stage1-prob-values", default="0.55,0.60,0.65")
    p.add_argument("--stage2-prob-values", default="0.60,0.65,0.70")
    p.add_argument(
        "--stage1-prob-by-window",
        default=None,
        help="Per-window stage1 gates, e.g. '120:0.50,0.55;150:0.65,0.70'",
    )
    p.add_argument(
        "--stage2-prob-by-window",
        default=None,
        help="Per-window stage2 gates, e.g. '120:0.55,0.60;150:0.70,0.75'",
    )

    p.add_argument("--use-optimize-prob", action="store_true", help="Enable --optimize-prob in each run")
    p.add_argument("--reuse-model", default=None,
                   help="Path to an existing model artifact. When set, each sweep run uses --model-in and skips retraining.")
    p.add_argument("--two-branch", action="store_true",
                   help="Enable --two-branch in each run (requires --window-15m > 0)")
    p.add_argument("--two-branch-stage", default="both",
                   choices=["both", "stage2", "stage1"],
                   help="Which stage(s) use two-branch model (default: both)")
    p.add_argument("--use-state-features", action="store_true",
                   help="Append causal state features to each run (--use-state-features in image_trend_ml.py).")
    p.add_argument("--pred-history-len", type=int, default=150,
                   help="Number of prior predictions to encode as state inputs (default: 150).")
    p.add_argument("--max-runs", type=int, default=None, help="Limit number of configs executed")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    p.add_argument("--tmp-dir", default="training/_tmp_image_trend_sweep")
    p.add_argument("--prep-cache-dir", default=None,
                   help="Optional shared prep cache directory forwarded to image_trend_ml.py.")
    p.add_argument("--refresh-prep-cache", action="store_true",
                   help="Forward --refresh-prep-cache to image_trend_ml.py.")
    p.add_argument("--out-csv", default="training/image_trend_sweep_results.csv")
    p.add_argument("--out-txt", default="training/image_trend_sweep_summary.txt")
    args = p.parse_args()

    if args.window_values is None:
        args.window_values = str(args.window)

    if not SCRIPT.exists():
        print(f"Missing script: {SCRIPT}", file=sys.stderr)
        return 2
    if args.reuse_model and args.use_optimize_prob:
        print("[warn] --reuse-model set: ignoring --use-optimize-prob because training is skipped.")
    if args.reuse_model and args.use_state_features:
        print("--reuse-model cannot be combined with --use-state-features", file=sys.stderr)
        return 2

    configs = build_sweep_configs(args)
    if args.use_optimize_prob:
        print("[info] --use-optimize-prob enabled: collapsing stage prob grids to one seed pair.")
    if args.target_values is not None:
        print(f"[info] using target grid from --target-values={args.target_values}")
    if args.max_runs is not None:
        configs = configs[: max(0, args.max_runs)]
    if not configs:
        print("No configs to run", file=sys.stderr)
        return 2
    auto_reuse_enabled = _auto_reuse_allowed(args)
    if auto_reuse_enabled:
        print("[info] auto model reuse enabled for gate-only single-split rows.")
    elif args.eval_mode == "single_split" and not args.reuse_model:
        if args.use_state_features:
            print("[info] auto model reuse disabled because --use-state-features changes causal feature generation.")
        elif args.use_optimize_prob:
            print("[info] auto model reuse disabled because --use-optimize-prob changes per-run training.")

    tmp_root = ROOT / args.tmp_dir
    tmp_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_all = time.time()
    total = len(configs)
    trained_model_by_key: dict[str, Path] = {}
    trained_run_id_by_key: dict[str, str] = {}

    for idx, cfg in enumerate(configs, 1):
        sf_tag = "_sf" if args.use_state_features else ""
        run_id = (
            f"w{cfg.window}_h{cfg.horizon}_thr{cfg.trend_threshold:g}_d{cfg.min_15m_drop:g}"
            f"_lt{cfg.long_target_threshold:g}_st{cfg.short_target_threshold:g}"
            f"_ls{cfg.long_adverse_limit:g}_ss{cfg.short_adverse_limit:g}"
            f"_r{cfg.min_15m_rise:g}_f{cfg.max_flat_ratio:g}"
            f"_p1{cfg.stage1_min_prob:g}_p2{cfg.stage2_min_prob:g}{sf_tag}"
        )
        model_out = tmp_root / f"{run_id}.joblib"
        report_out = tmp_root / f"{run_id}.json"

        # Avoid stale artifacts from previous sweeps causing false positives or
        # masking the real outcome of the current run.
        if model_out.exists():
            model_out.unlink()
        if report_out.exists():
            report_out.unlink()

        train_key = _training_key(args, cfg) if auto_reuse_enabled else None
        auto_reuse_model = trained_model_by_key.get(train_key) if train_key is not None else None
        auto_reuse_source = trained_run_id_by_key.get(train_key) if train_key is not None else None
        explicit_model_in = args.reuse_model

        cmd = [
            sys.executable,
            str(SCRIPT),
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--timeframe", args.timeframe,
            "--eval-mode", args.eval_mode,
            "--window", str(cfg.window),
            "--window-15m", str(args.window_15m),
            "--min-window-range", str(args.min_window_range),
            "--min-15m-drop", str(cfg.min_15m_drop),
            "--min-15m-rise", str(cfg.min_15m_rise),
            "--horizon", str(cfg.horizon),
            "--trend-threshold", str(cfg.trend_threshold),
            "--adverse-limit", str(args.adverse_limit),
            "--long-target-threshold", str(cfg.long_target_threshold),
            "--short-target-threshold", str(cfg.short_target_threshold),
            "--long-adverse-limit", str(cfg.long_adverse_limit),
            "--short-adverse-limit", str(cfg.short_adverse_limit),
            "--stage1-min-prob", str(cfg.stage1_min_prob),
            "--stage2-min-prob", str(cfg.stage2_min_prob),
            "--test-size", str(args.test_size),
            "--random-state", str(args.random_state),
            "--classifier", args.classifier,
            "--max-flat-ratio", str(cfg.max_flat_ratio),
            "--model-out", str(model_out.relative_to(ROOT)),
            "--report-out", str(report_out.relative_to(ROOT)),
        ]
        if args.disable_time_filter:
            cmd.append("--disable-time-filter")
        if args.test_start_date is not None:
            cmd.extend(["--test-start-date", args.test_start_date])
        if explicit_model_in:
            cmd.extend(["--eval-mode", "single_split", "--model-in", args.reuse_model])
        elif auto_reuse_model is not None:
            cmd.extend(["--model-in", str(auto_reuse_model)])
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])
        if args.last_bar_wr90_high is not None:
            cmd.extend(["--last-bar-wr90-high", str(args.last_bar_wr90_high)])
        if args.last_bar_wr90_low is not None:
            cmd.extend(["--last-bar-wr90-low", str(args.last_bar_wr90_low)])
        if args.use_optimize_prob and not args.reuse_model:
            cmd.append("--optimize-prob")
        if args.two_branch:
            cmd.extend(["--two-branch", "--two-branch-stage", args.two_branch_stage])
        if args.use_state_features:
            cmd.extend(["--use-state-features", "--pred-history-len", str(args.pred_history_len)])
        if args.prep_cache_dir:
            cmd.extend(["--prep-cache-dir", args.prep_cache_dir])
        if args.refresh_prep_cache:
            cmd.append("--refresh-prep-cache")

        print(f"\n[{idx}/{total}] {run_id}")
        if auto_reuse_model is not None and auto_reuse_source is not None:
            print(f"AUTO-REUSE MODEL: {auto_reuse_source} -> {run_id}")
        print("CMD:", " ".join(cmd))

        if args.dry_run:
            if train_key is not None and auto_reuse_model is None:
                trained_model_by_key[train_key] = model_out
                trained_run_id_by_key[train_key] = run_id
            rows.append(
                {
                    "run_id": run_id,
                    "window": cfg.window,
                    "horizon": cfg.horizon,
                    "trend_threshold": cfg.trend_threshold,
                    "long_target_threshold": cfg.long_target_threshold,
                    "short_target_threshold": cfg.short_target_threshold,
                    "long_adverse_limit": cfg.long_adverse_limit,
                    "short_adverse_limit": cfg.short_adverse_limit,
                    "min_15m_drop": cfg.min_15m_drop,
                    "min_15m_rise": cfg.min_15m_rise,
                    "max_flat_ratio": cfg.max_flat_ratio,
                    "stage1_min_prob": cfg.stage1_min_prob,
                    "stage2_min_prob": cfg.stage2_min_prob,
                    "model_source": auto_reuse_source if auto_reuse_source is not None else ("explicit" if explicit_model_in else "trained"),
                    "status": "DRY_RUN",
                }
            )
            continue

        t0 = time.time()
        rc = _run_cmd(cmd)
        elapsed_s = time.time() - t0

        row = {
            "run_id": run_id,
            "window": cfg.window,
            "horizon": cfg.horizon,
            "trend_threshold": cfg.trend_threshold,
            "long_target_threshold": cfg.long_target_threshold,
            "short_target_threshold": cfg.short_target_threshold,
            "long_adverse_limit": cfg.long_adverse_limit,
            "short_adverse_limit": cfg.short_adverse_limit,
            "min_15m_drop": cfg.min_15m_drop,
            "min_15m_rise": cfg.min_15m_rise,
            "max_flat_ratio": cfg.max_flat_ratio,
            "stage1_min_prob": cfg.stage1_min_prob,
            "stage2_min_prob": cfg.stage2_min_prob,
            "model_source": auto_reuse_source if auto_reuse_source is not None else ("explicit" if explicit_model_in else "trained"),
            "return_code": rc,
            "elapsed_s": round(elapsed_s, 2),
        }

        if not report_out.exists():
            row["status"] = "FAILED"
            rows.append(row)
            print(f"Run failed (rc={rc})")
            continue

        try:
            rep = json.loads(report_out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            row["status"] = "FAILED_BAD_REPORT"
            rows.append(row)
            print(f"Run failed: unreadable report (rc={rc})")
            continue

        pnl = _safe_get(rep, ["directional_pnl"], {}) or {}
        total_pnl = _safe_float(pnl.get("total_pnl", float("nan")), default=float("nan"))
        positive_days_pct_raw = pnl.get("positive_days_pct", None)
        positive_days_pct: float | None
        if isinstance(positive_days_pct_raw, (int, float, str)):
            positive_days_pct = float(positive_days_pct_raw)
        else:
            positive_days_pct = None
        stage2_bacc = _safe_float(_safe_get(rep, ["stage2", "balanced_accuracy"], 0.0))

        row.update(
            {
                "status": "OK" if rc == 0 else "OK_NONZERO_RC",
                "trades": int(pnl.get("trades", 0)),
                "n_days": int(pnl.get("n_days", 0)),
                "total_pnl": total_pnl,
                "avg_trade": _safe_float(pnl.get("avg_trade", float("nan")), default=float("nan")),
                "avg_day": pnl.get("avg_day", None),
                "positive_days_pct": positive_days_pct,
                "stage1_balanced_accuracy": _safe_float(_safe_get(rep, ["stage1", "balanced_accuracy"], 0.0)),
                "stage2_balanced_accuracy": stage2_bacc,
                "score": score_row(total_pnl, positive_days_pct, stage2_bacc),
            }
        )
        rows.append(row)

        if train_key is not None and auto_reuse_model is None and rc == 0 and model_out.exists():
            trained_model_by_key[train_key] = model_out
            trained_run_id_by_key[train_key] = run_id

    df = pd.DataFrame(rows)
    if not df.empty and "status" in df.columns:
        status_order = {"OK": 0, "OK_NONZERO_RC": 1, "DRY_RUN": 2, "FAILED_BAD_REPORT": 3, "FAILED": 4}
        df["_status_rank"] = df["status"].map(lambda s: status_order.get(str(s), 99))
        sort_cols = ["_status_rank"]
        ascending = [True]
        if "score" in df.columns:
            sort_cols.append("score")
            ascending.append(False)
        df = df.sort_values(by=sort_cols, ascending=ascending, na_position="last").drop(columns=["_status_rank"])

    out_csv = ROOT / args.out_csv
    out_txt = ROOT / args.out_txt
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)

    lines = []
    lines.append("Image Trend ML sweep summary")
    lines.append(f"runs={len(df)} elapsed={time.time() - t_all:.1f}s")
    if not df.empty and "model_source" in df.columns:
        trained_count = int((df["model_source"] == "trained").sum())
        explicit_count = int((df["model_source"] == "explicit").sum())
        reused_count = int(len(df) - trained_count - explicit_count)
        lines.append(f"model_source_counts trained={trained_count} auto_reused={reused_count} explicit={explicit_count}")
    lines.append("")

    ok_statuses = {"OK", "OK_NONZERO_RC"}
    ok = df[df["status"].isin(ok_statuses)] if "status" in df.columns else pd.DataFrame()
    if len(ok) == 0:
        lines.append("No successful runs.")
    else:
        top = ok.sort_values("score", ascending=False).head(10)
        cols = [
            "run_id",
            "total_pnl",
            "avg_day",
            "positive_days_pct",
            "trades",
            "stage2_balanced_accuracy",
            "score",
        ]
        lines.append("Top runs:")
        lines.append(top[cols].to_string(index=False))

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved CSV : {out_csv}")
    print(f"Saved TXT : {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

