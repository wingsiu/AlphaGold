#!/usr/bin/env python3
"""Replay a saved walk-forward report with current backtest logic, without retraining.

This rebuilds the dataset, reloads each saved per-cycle model, reruns inference on the
cycle's test slice, and recomputes directional PnL/trades using the current execution
semantics. It is intended for logic-only what-if checks (for example, entry timing
changes) against an existing walk-forward artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from data import DataLoader
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import DataLoader. Ensure project root is on PYTHONPATH."
    ) from exc

try:
    from training.image_trend_ml import (
        LABEL_FLAT,
        _prepare_ohlcv,
        _resolve_directional_stops,
        _resolve_directional_targets,
        build_dataset,
        directional_pnl_report,
        flatten_tensors,
        predict_two_stage_details,
    )
    from training.sweep_wf_probs_no_retrain import _build_cycle_augmented_features, _cycle_slice_bounds, _find_slice
except ModuleNotFoundError:
    from image_trend_ml import (
        LABEL_FLAT,
        _prepare_ohlcv,
        _resolve_directional_stops,
        _resolve_directional_targets,
        build_dataset,
        directional_pnl_report,
        flatten_tensors,
        predict_two_stage_details,
    )
    from sweep_wf_probs_no_retrain import _build_cycle_augmented_features, _cycle_slice_bounds, _find_slice


def _metric_block(payload: dict | None) -> dict[str, float | int | None]:
    src = dict(payload or {})
    profit_factor = src.get("profit_factor")
    avg_day = src.get("avg_day")
    positive_days_pct = src.get("positive_days_pct")
    win_rate_pct = src.get("win_rate_pct")
    return {
        "trades": int(src.get("trades", 0) or 0),
        "total_pnl": float(src.get("total_pnl", 0.0) or 0.0),
        "profit_factor": None if profit_factor is None else float(profit_factor),
        "avg_day": None if avg_day is None else float(avg_day),
        "positive_days_pct": None if positive_days_pct is None else float(positive_days_pct),
        "win_rate_pct": None if win_rate_pct is None else float(win_rate_pct),
        "trade_max_drawdown": float(src.get("trade_max_drawdown", src.get("max_drawdown", 0.0)) or 0.0),
        "daily_max_drawdown": float(src.get("daily_max_drawdown", 0.0) or 0.0),
    }


def _delta_block(old: dict[str, float | int | None], new: dict[str, float | int | None]) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {}
    for key, old_val in old.items():
        new_val = new.get(key)
        if old_val is None or new_val is None:
            out[key] = None
        elif isinstance(old_val, int) and isinstance(new_val, int):
            out[key] = int(new_val - old_val)
        else:
            out[key] = float(new_val) - float(old_val)
    return out


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _resolve_cycle_model_path(report_path: Path, cycle_model_out: str) -> Path:
    p = Path(cycle_model_out)
    if p.is_absolute():
        return p
    candidate = (report_path.parent / p.name).resolve()
    if candidate.exists():
        return candidate
    candidate = (PROJECT_ROOT / cycle_model_out).resolve()
    if candidate.exists():
        return candidate
    return p.resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a saved walk-forward report without retraining")
    ap.add_argument("--report-in", required=True, help="Existing walk-forward report JSON")
    ap.add_argument("--out-json", default=None, help="Optional JSON file to save comparison summary")
    ap.add_argument("--out-trades", default=None, help="Optional CSV file to save replayed trades")
    ap.add_argument("--max-cycles", type=int, default=None, help="Optional cycle cap for smoke tests")
    ap.add_argument("--max-state-iters", type=int, default=3)
    args = ap.parse_args()

    report_path = Path(args.report_in).resolve()
    report = json.loads(report_path.read_text())
    cfg = dict(report["config"])
    execution_semantics = dict(report.get("execution_semantics") or {})

    if report.get("evaluation_mode") != "walk_forward":
        raise ValueError("--report-in must reference a walk-forward report")

    raw = DataLoader().load_data(cfg["table"], start_date=cfg["start_date"], end_date=cfg["end_date"])
    bars = _prepare_ohlcv(raw, cfg["timeframe"])

    long_thr, short_thr = _resolve_directional_targets(
        float(cfg["trend_threshold"]),
        cfg.get("long_target_threshold"),
        cfg.get("short_target_threshold"),
    )
    long_stop, short_stop = _resolve_directional_stops(
        float(cfg["adverse_limit"]),
        cfg.get("long_adverse_limit"),
        cfg.get("short_adverse_limit"),
    )

    X_tensor, y, ts, curr_a, entry_a, fut_a, entry_ts_a, fut_ts_a, _, _ = build_dataset(
        bars,
        int(cfg["window"]),
        float(cfg["min_window_range"]),
        int(cfg["horizon"]),
        float(cfg["trend_threshold"]),
        float(cfg["adverse_limit"]),
        long_target_threshold=cfg.get("long_target_threshold"),
        short_target_threshold=cfg.get("short_target_threshold"),
        long_adverse_limit=cfg.get("long_adverse_limit"),
        short_adverse_limit=cfg.get("short_adverse_limit"),
        min_15m_drop=float(cfg.get("min_15m_drop", 0.0) or 0.0),
        min_15m_rise=float(cfg.get("min_15m_rise", 0.0) or 0.0),
        window_15m=int(cfg.get("window_15m", 0) or 0),
        apply_time_filter=not bool(cfg.get("disable_time_filter", False)),
        use_15m_wick_features=bool(cfg.get("use_15m_wick_features", False)),
        wick_feature_min_range=float(cfg.get("wick_feature_min_range", 40.0) or 40.0),
        wick_feature_min_pct=float(cfg.get("wick_feature_min_pct", 35.0) or 35.0),
        wick_feature_min_volume=float(cfg.get("wick_feature_min_volume", 3000.0) or 3000.0),
    )
    X_flat = flatten_tensors(X_tensor)

    y_pred_all = np.full(len(y), LABEL_FLAT, dtype=np.int64)
    signal_prob_all = np.zeros(len(y), dtype=np.float64)
    eval_mask = np.zeros(len(y), dtype=bool)
    cycle_rows: list[dict[str, object]] = []

    wf_cycles = list(report.get("walkforward_cycles", []))
    if args.max_cycles is not None:
        wf_cycles = wf_cycles[: max(0, args.max_cycles)]

    for cycle in wf_cycles:
        cycle_model_out = cycle.get("cycle_model_out")
        if not cycle_model_out:
            raise ValueError("Missing cycle_model_out in report. Re-run WF with cycle model saving enabled.")
        cycle_model_path = _resolve_cycle_model_path(report_path, str(cycle_model_out))
        bundle = joblib.load(cycle_model_path)
        m1, m2 = bundle["stage1"], bundle["stage2"]
        test_start_bound, test_end_bound = _cycle_slice_bounds(cycle)
        sl = _find_slice(ts, test_start_bound, test_end_bound)
        x_raw = X_flat[sl]
        cycle_cfg = cfg.copy()
        cycle_cfg.update(cycle.get("selected_config", {}))
        x_aug = _build_cycle_augmented_features(
            x_raw=x_raw,
            curr=curr_a[sl],
            entry_px=entry_a[sl],
            m1=m1,
            m2=m2,
            stage1_gate=float(cycle["selected_config"]["stage1_min_prob"]),
            stage2_gate=float(cycle["selected_config"]["stage2_min_prob"]),
            cfg=cycle_cfg,
            long_thr=long_thr,
            short_thr=short_thr,
            long_stop=long_stop,
            short_stop=short_stop,
            state_feature_names=list(bundle.get("state_feature_names", [])),
            max_state_iters=args.max_state_iters,
        )
        gate_up = float(cycle["selected_config"].get("stage2_min_prob_up", cycle["selected_config"]["stage2_min_prob"]))
        gate_down = float(cycle["selected_config"].get("stage2_min_prob_down", cycle["selected_config"]["stage2_min_prob"]))
        pm = predict_two_stage_details(
            x_aug,
            m1,
            m2,
            stage1_min_prob=float(cycle["selected_config"]["stage1_min_prob"]),
            stage2_min_prob=float(cycle["selected_config"]["stage2_min_prob"]),
            stage2_min_prob_up=gate_up,
            stage2_min_prob_down=gate_down,
            stage1_min_prob_1m=cfg.get("stage1_min_prob_1m"),
            stage1_min_prob_15m=cfg.get("stage1_min_prob_15m"),
            stage2_min_prob_1m=cfg.get("stage2_min_prob_1m"),
            stage2_min_prob_15m=cfg.get("stage2_min_prob_15m"),
        )
        y_pred_all[sl] = pm["pred"]
        signal_prob_all[sl] = pm["signal_prob"]
        eval_mask[sl] = True

        cycle_pnl, _ = directional_pnl_report(
            ts[sl],
            entry_ts_a[sl],
            fut_ts_a[sl],
            pm["pred"],
            curr_a[sl],
            entry_a[sl],
            fut_a[sl],
            signal_prob=pm["signal_prob"],
            adverse_limit=float(cfg["adverse_limit"]),
            long_adverse_limit=long_stop,
            short_adverse_limit=short_stop,
            allow_overlap=bool(cfg.get("allow_overlap_backtest", False)),
            reverse_exit_prob=float(cfg.get("reverse_exit_prob", 0.7)),
        )
        replay_all = _metric_block(cycle_pnl.get("all", cycle_pnl))
        original_all = _metric_block((cycle.get("directional_pnl") or {}).get("all", cycle.get("directional_pnl") or {}))
        cycle_rows.append(
            {
                "cycle_id": int(cycle["cycle_id"]),
                "test_start": cycle["test_start"],
                "test_end": cycle["test_end"],
                "original": original_all,
                "replay": replay_all,
                "delta": _delta_block(original_all, replay_all),
            }
        )

    idx = np.where(eval_mask)[0]
    if len(idx) == 0:
        raise ValueError("Replay produced no evaluable samples")

    replay_pnl, trades_df = directional_pnl_report(
        ts[idx],
        entry_ts_a[idx],
        fut_ts_a[idx],
        y_pred_all[idx],
        curr_a[idx],
        entry_a[idx],
        fut_a[idx],
        signal_prob=signal_prob_all[idx],
        adverse_limit=float(cfg["adverse_limit"]),
        long_adverse_limit=long_stop,
        short_adverse_limit=short_stop,
        allow_overlap=bool(cfg.get("allow_overlap_backtest", False)),
        reverse_exit_prob=float(cfg.get("reverse_exit_prob", 0.7)),
    )

    original_all = _metric_block((report.get("directional_pnl") or {}).get("all", report.get("directional_pnl") or {}))
    replay_all = _metric_block(replay_pnl.get("all", replay_pnl))
    summary = {
        "report_in": str(report_path),
        "cycles_used": len(wf_cycles),
        "execution_semantics": execution_semantics,
        "original": original_all,
        "replay": replay_all,
        "delta": _delta_block(original_all, replay_all),
        "original_exit_reason_counts": dict(
            pd.Series(
                (report.get("directional_pnl") or {}).get(
                    "exit_reason_counts",
                    ((report.get("directional_pnl") or {}).get("all") or {}).get("exit_reason_counts", {}),
                )
            ).sort_index()
        ),
        "replay_exit_reason_counts": trades_df["exit_reason"].value_counts().sort_index().to_dict(),
        "cycle_deltas": cycle_rows,
    }
    summary = _to_jsonable(summary)

    if args.out_trades:
        out_trades = Path(args.out_trades)
        out_trades.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(out_trades, index=False)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"report={report_path}")
    print(f"cycles_used={len(wf_cycles)}")
    if execution_semantics:
        print("execution_semantics:", json.dumps(execution_semantics, sort_keys=True))
    print("original:", json.dumps(original_all, sort_keys=True))
    print("replay:", json.dumps(replay_all, sort_keys=True))
    print("delta:", json.dumps(summary["delta"], sort_keys=True))
    print("replay_exit_reason_counts:", json.dumps(summary["replay_exit_reason_counts"], sort_keys=True))
    if args.out_json:
        print(f"saved summary: {args.out_json}")
    if args.out_trades:
        print(f"saved trades: {args.out_trades}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

