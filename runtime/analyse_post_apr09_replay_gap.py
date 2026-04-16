#!/usr/bin/env python3
"""Summarize the post-2026-04-09 replay gap for the promoted best-base model.

This script does not retrain anything. It compares:
- the promoted single-split replay on 2026-04-09..2026-04-15
- the latest fixed-gate walk-forward cycle replay on the same slice
- the latest probability-sweep walk-forward cycle replay on the same slice
- the saved training/backtest date coverage

It writes:
- runtime/post_apr09_replay_gap_analysis.json
- runtime/post_apr09_replay_gap_analysis.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
TRAINING_DIR = PROJECT_ROOT / "training"
HK_TZ = ZoneInfo("Asia/Hong_Kong")

REPLAY_START = cast(pd.Timestamp, pd.Timestamp("2026-04-09T00:00:00+00:00"))
REPLAY_END = cast(pd.Timestamp, pd.Timestamp("2026-04-15T23:59:59+00:00"))
PRE_WINDOW_START = cast(pd.Timestamp, pd.Timestamp("2026-04-02T00:00:00+00:00"))
PRE_WINDOW_END = cast(pd.Timestamp, pd.Timestamp("2026-04-09T00:00:00+00:00"))


@dataclass(frozen=True)
class ReplayArtifact:
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
            "signal_start": None,
            "signal_start_hkt": None,
            "signal_end": None,
            "signal_end_hkt": None,
            "entry_start": None,
            "entry_start_hkt": None,
            "entry_end": None,
            "entry_end_hkt": None,
            "exit_end": None,
            "exit_end_hkt": None,
        }
    signal_start = df["ts"].min().isoformat() if "ts" in df.columns else None
    signal_end = df["ts"].max().isoformat() if "ts" in df.columns else None
    entry_start = df["entry_time"].min().isoformat() if "entry_time" in df.columns else None
    entry_end = df["entry_time"].max().isoformat() if "entry_time" in df.columns else None
    exit_end = df["exit_time"].max().isoformat() if "exit_time" in df.columns else None
    return {
        "rows": int(len(df)),
        "signal_start": signal_start,
        "signal_start_hkt": _to_hkt_str(signal_start),
        "signal_end": signal_end,
        "signal_end_hkt": _to_hkt_str(signal_end),
        "entry_start": entry_start,
        "entry_start_hkt": _to_hkt_str(entry_start),
        "entry_end": entry_end,
        "entry_end_hkt": _to_hkt_str(entry_end),
        "exit_end": exit_end,
        "exit_end_hkt": _to_hkt_str(exit_end),
    }


def _slice_trades(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df.empty or "entry_time" not in df.columns:
        return df.iloc[0:0].copy()
    mask = (df["entry_time"] >= start) & (df["entry_time"] < end)
    return df.loc[mask].copy()


def _slice_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "avg_trade": None,
            "profit_factor": None,
            "wins": 0,
            "losses": 0,
            "side_counts": {},
            "exit_reason_counts": {},
            "entry_signal_prob_mean": None,
            "entry_signal_prob_median": None,
            "coverage": _trade_coverage(df),
        }
    pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(pnl[pnl < 0].sum())
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None
    entry_prob = pd.to_numeric(df.get("entry_signal_prob"), errors="coerce") if "entry_signal_prob" in df.columns else pd.Series(dtype=float)
    return {
        "trades": int(len(df)),
        "total_pnl": float(pnl.sum()),
        "avg_trade": float(pnl.mean()),
        "profit_factor": float(profit_factor) if profit_factor is not None else None,
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "side_counts": {str(k): int(v) for k, v in df["side"].value_counts().to_dict().items()} if "side" in df.columns else {},
        "exit_reason_counts": {str(k): int(v) for k, v in df["exit_reason"].value_counts().to_dict().items()} if "exit_reason" in df.columns else {},
        "entry_signal_prob_mean": float(entry_prob.mean()) if len(entry_prob) else None,
        "entry_signal_prob_median": float(entry_prob.median()) if len(entry_prob) else None,
        "coverage": _trade_coverage(df),
    }


def _model_summary(artifact: ReplayArtifact) -> dict[str, Any]:
    report = _read_json(artifact.report_path)
    trades = _read_trades(artifact.trades_path)
    pnl = report["directional_pnl"]
    return {
        "label": artifact.label,
        "signal_model_path": report["signal_model_path"],
        "prediction_counts": dict(report.get("prediction_counts", {})),
        "avg_signal_prob_tradable": report.get("avg_signal_prob_tradable"),
        "trades_summary": {
            "trades": int(pnl.get("trades", 0)),
            "total_pnl": float(pnl.get("total_pnl", 0.0)),
            "avg_trade": float(pnl.get("avg_trade", 0.0)) if pnl.get("avg_trade") is not None else None,
            "profit_factor": (pnl.get("all") or {}).get("profit_factor"),
            "side_split": {
                "long_trades": int((pnl.get("long") or {}).get("trades", 0)),
                "short_trades": int((pnl.get("short") or {}).get("trades", 0)),
            },
            "exit_reason_counts": dict(((pnl.get("all") or {}).get("exit_reason_counts") or {})),
        },
        "trade_file_stats": _slice_stats(trades),
    }


def main() -> int:
    corrected_report = _read_json(TRAINING_DIR / "backtest_report_best_base_corrected.json")
    wf_report = _read_json(TRAINING_DIR / "backtest_report_best_base_wf_10cycles.json")
    wf_prob_report = _read_json(TRAINING_DIR / "backtest_report_best_base_wf_10cycles_prob_sweep.json")
    bot_status = _read_json(RUNTIME_DIR / "trading_bot_status.json")

    corrected_trades = _read_trades(TRAINING_DIR / "backtest_trades_best_base_corrected.csv")
    corrected_coverage = _trade_coverage(corrected_trades)
    corrected_pre_window = _slice_trades(corrected_trades, PRE_WINDOW_START, PRE_WINDOW_END)

    promoted = ReplayArtifact(
        label="promoted_single_split",
        report_path=RUNTIME_DIR / "mock_best_base_after_2026-04-09_report_v2.json",
        trades_path=RUNTIME_DIR / "mock_best_base_after_2026-04-09_trades_v2.csv",
    )
    wf_cycle10 = ReplayArtifact(
        label="walk_forward_fixed_cycle10",
        report_path=RUNTIME_DIR / "mock_best_base_after_2026-04-09_wf_cycle10_report.json",
        trades_path=RUNTIME_DIR / "mock_best_base_after_2026-04-09_wf_cycle10_trades.csv",
    )
    wf_prob_cycle10 = ReplayArtifact(
        label="walk_forward_prob_cycle10",
        report_path=RUNTIME_DIR / "mock_best_base_after_2026-04-09_wf_prob_cycle10_report.json",
        trades_path=RUNTIME_DIR / "mock_best_base_after_2026-04-09_wf_prob_cycle10_trades.csv",
    )

    promoted_summary = _model_summary(promoted)
    wf_cycle10_summary = _model_summary(wf_cycle10)
    wf_prob_cycle10_summary = _model_summary(wf_prob_cycle10)

    corrected_exit_end_raw = corrected_coverage.get("exit_end")
    corrected_exit_end_ts = pd.Timestamp(str(corrected_exit_end_raw)) if corrected_exit_end_raw else None
    no_overlap_with_replay = bool(
        corrected_exit_end_ts is not None
        and not pd.isna(corrected_exit_end_ts)
        and corrected_exit_end_ts < REPLAY_START
    )

    result = {
        "bot_status_snapshot": {
            "mode": bot_status.get("mode"),
            "signal_model_family": bot_status.get("signal_model_family"),
            "signal_model_path": bot_status.get("signal_model_path"),
            "open_position": bot_status.get("open_position"),
            "market_data_enabled": bot_status.get("market_data_enabled"),
            "prediction_cache_last_bucket_utc": bot_status.get("prediction_cache_last_bucket_utc"),
            "prediction_cache_last_bucket_hkt": _to_hkt_str(bot_status.get("prediction_cache_last_bucket_utc")),
            "cached_rows": (bot_status.get("input_data") or {}).get("rows"),
            "candidate_samples": (bot_status.get("best_base_runtime") or {}).get("candidate_samples"),
        },
        "saved_backtest_coverage": {
            "corrected_single_split_trade_coverage": corrected_coverage,
            "saved_corrected_report_total_pnl": (corrected_report.get("directional_pnl") or {}).get("total_pnl"),
            "saved_fixed_wf_total_pnl": (wf_report.get("directional_pnl") or {}).get("total_pnl"),
            "saved_fixed_wf_profit_factor": ((wf_report.get("directional_pnl") or {}).get("all") or {}).get("profit_factor"),
            "saved_prob_wf_total_pnl": (wf_prob_report.get("directional_pnl") or {}).get("total_pnl"),
            "saved_prob_wf_profit_factor": ((wf_prob_report.get("directional_pnl") or {}).get("all") or {}).get("profit_factor"),
            "saved_backtests_overlap_post_apr09_replay_window": not no_overlap_with_replay,
        },
        "pre_replay_window_from_corrected_backtest": {
            "window_start_utc": PRE_WINDOW_START.isoformat(),
            "window_end_utc": PRE_WINDOW_END.isoformat(),
            "stats": _slice_stats(corrected_pre_window),
        },
        "replay_window": {
            "window_start_utc": REPLAY_START.isoformat(),
            "window_start_hkt": _to_hkt_str(REPLAY_START),
            "window_end_utc": REPLAY_END.isoformat(),
            "window_end_hkt": _to_hkt_str(REPLAY_END),
            "models": [promoted_summary, wf_cycle10_summary, wf_prob_cycle10_summary],
        },
        "recommendation": {
            "practical_choice_now": "keep_current_promoted_artifact_as_bot_default_but_not_as_live_validated_model",
            "robustness_preference": "fixed_gate_walk_forward_methodology_over_probability_sweep",
            "rationale": [
                "The saved corrected and walk-forward training backtests do not directly cover the weak post-2026-04-09 replay window.",
                "On the aligned post-2026-04-09 replay slice, the currently promoted artifact is negative, and the latest fixed-gate/probability-sweep walk-forward cycle alternatives are also negative.",
                "The fixed-gate walk-forward family remains the stronger robustness benchmark across saved reports, but it does not clearly beat the promoted artifact on the recent replay slice.",
                "That means the recent weakness looks more like genuine regime degradation than a simple promoted-model-only bug.",
            ],
            "next_step": "extend aligned comparison tooling or training coverage through the same recent slice before any live-execution promotion",
        },
    }

    json_out = RUNTIME_DIR / "post_apr09_replay_gap_analysis.json"
    md_out = RUNTIME_DIR / "post_apr09_replay_gap_analysis.md"
    json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    replay_window = cast(dict[str, Any], result["replay_window"])
    replay_models = cast(list[dict[str, Any]], replay_window["models"])
    model_lines: list[str] = []
    for model in replay_models:
        trade_summary = model["trades_summary"]
        preds = model["prediction_counts"]
        model_lines.append(
            f"- **{model['label']}**\n"
            f"  - model: `{model['signal_model_path']}`\n"
            f"  - signals: tradable={preds.get('tradable')} up={preds.get('up')} down={preds.get('down')} flat={preds.get('flat')}\n"
            f"  - avg tradable prob: {model['avg_signal_prob_tradable']:.4f}\n"
            f"  - trades: {trade_summary['trades']}\n"
            f"  - total pnl: {trade_summary['total_pnl']:.2f}\n"
            f"  - profit factor: {trade_summary['profit_factor']}\n"
            f"  - exits: {trade_summary['exit_reason_counts']}"
        )

    pre_replay_window = cast(dict[str, Any], result["pre_replay_window_from_corrected_backtest"])
    pre_stats = cast(dict[str, Any], pre_replay_window["stats"])
    saved_backtest_coverage = cast(dict[str, Any], result["saved_backtest_coverage"])
    corrected_trade_coverage = cast(dict[str, Any], saved_backtest_coverage["corrected_single_split_trade_coverage"])
    bot_status_snapshot = cast(dict[str, Any], result["bot_status_snapshot"])
    md = f"""# Post-Apr-09 replay gap analysis

## Current status confirmed
- Bot mode: `{bot_status_snapshot['mode']}`
- Signal family: `{bot_status_snapshot['signal_model_family']}`
- Promoted model: `{bot_status_snapshot['signal_model_path']}`
- Open position: `{bot_status_snapshot['open_position']}`
- Latest prediction-cache bucket (HKT): `{bot_status_snapshot['prediction_cache_last_bucket_hkt']}`
- Cached rows in saved status: `{bot_status_snapshot['cached_rows']}`
- Candidate samples in saved status: `{bot_status_snapshot['candidate_samples']}`

## Key finding: saved training backtests do not cover the weak replay window
- Corrected single-split trades end at (HKT): `{corrected_trade_coverage['exit_end_hkt']}`
- Post-Apr-09 replay window starts at (HKT): `{replay_window['window_start_hkt']}`
- Direct overlap exists: `{saved_backtest_coverage['saved_backtests_overlap_post_apr09_replay_window']}`

So the negative replay after `2026-04-09` is **not** contradicted by the saved corrected backtest; it is simply outside that saved trade file's coverage.

## Saved whole-period health
- Corrected single-split total pnl: `{saved_backtest_coverage['saved_corrected_report_total_pnl']}`
- Fixed-gate walk-forward total pnl: `{saved_backtest_coverage['saved_fixed_wf_total_pnl']}`
- Fixed-gate walk-forward profit factor: `{saved_backtest_coverage['saved_fixed_wf_profit_factor']}`
- Probability-sweep walk-forward total pnl: `{saved_backtest_coverage['saved_prob_wf_total_pnl']}`
- Probability-sweep walk-forward profit factor: `{saved_backtest_coverage['saved_prob_wf_profit_factor']}`

## Immediate pre-replay window from corrected backtest (`2026-04-02` .. `2026-04-09`)
- Trades: `{pre_stats['trades']}`
- Total pnl: `{pre_stats['total_pnl']:.2f}`
- Profit factor: `{pre_stats['profit_factor']}`
- Side mix: `{pre_stats['side_counts']}`
- Exit mix: `{pre_stats['exit_reason_counts']}`
- Coverage end (HKT): `{cast(dict[str, Any], pre_stats['coverage'])['exit_end_hkt']}`

## Aligned replay-window comparison (`{replay_window['window_start_hkt']}` .. `{replay_window['window_end_hkt']}`)
{chr(10).join(model_lines)}

## Conclusion
- The promoted artifact is still the strongest **saved single-split benchmark**, but it is **not fully validated for live deployment**.
- The recent weakness looks **real enough to respect**, because the latest walk-forward cycle alternatives were also negative on the same slice.
- If forced to choose a robustness framework, prefer **fixed-gate walk-forward** over the probability-sweep variant.
- But there is **not yet a clearly superior replacement artifact** from this aligned recent-slice check.

## Most sensible next step
1. Keep the current promoted artifact as the bot's default **research/signal-only** model.
2. Do **not** treat it as live-ready.
3. Extend aligned comparison coverage through the same recent dates with training-side artifacts / overlap windows before any live-execution build-out becomes primary.
"""
    md_out.write_text(md, encoding="utf-8")

    print(f"Saved: {json_out}")
    print(f"Saved: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

