#!/usr/bin/env python3
"""Batch replay saved walk-forward reports without retraining.

This wraps `replay_wf_report_no_retrain.py` so future backtest-logic changes can be
compared against the same frozen walk-forward model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve().parent / "replay_wf_report_no_retrain.py"

DEFAULT_REPORTS = [
    "training/backtest_report_wf_state_sweep_C.json",
    "training/_wfC_r0_flat250_400_u58_68.json",
    "training/_wfC_r0_flat200_300_s148_55_u58_62_d62_68.json",
    "training/_wfC_r0_narrow_flat180_250_s146_50_u54_60_d62_66.json",
    "training/_wfC_r0_flat345_s148_55_u56_62_d62_66.json",
]


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_") or "replay"


def _resolve_report(path_text: str) -> Path:
    p = Path(path_text)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _metric(summary: dict[str, Any], side: str, key: str) -> Any:
    return ((summary.get(side) or {}).get(key))


def _write_batch_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    fields = [
        "report_stem",
        "report_in",
        "summary_out",
        "trades_out",
        "trades_original",
        "cycles_used",
        "entry_execution",
        "signal_reference",
        "orig_trades",
        "orig_total_pnl",
        "orig_profit_factor",
        "orig_positive_days_pct",
        "orig_trade_max_drawdown",
        "orig_daily_max_drawdown",
        "replay_trades",
        "replay_total_pnl",
        "replay_profit_factor",
        "replay_positive_days_pct",
        "replay_trade_max_drawdown",
        "replay_daily_max_drawdown",
        "delta_trades",
        "delta_total_pnl",
        "delta_profit_factor",
        "delta_positive_days_pct",
        "delta_trade_max_drawdown",
        "delta_daily_max_drawdown",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch replay walk-forward reports without retraining")
    ap.add_argument("--logic-label", required=True, help="Label for this replay batch, e.g. next_bar_open_v1")
    ap.add_argument(
        "--reports",
        nargs="*",
        default=DEFAULT_REPORTS,
        help="Walk-forward report paths to replay. Defaults to the main comparison set.",
    )
    ap.add_argument(
        "--out-root",
        default="training/replays",
        help="Base output directory for replay batches (default: training/replays)",
    )
    args = ap.parse_args()

    out_dir = _resolve_report(args.out_root) / _slug(args.logic_label)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "logic_label": args.logic_label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "script": str(SCRIPT_PATH),
        "reports": [],
    }
    summary_rows: list[dict[str, Any]] = []

    for report_text in args.reports:
        report_path = _resolve_report(report_text)
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found: {report_path}")
        stem = report_path.stem
        summary_out = out_dir / f"{stem}.replay_summary.json"
        trades_out = out_dir / f"{stem}.replay_trades.csv"

        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--report-in",
            str(report_path),
            "--out-json",
            str(summary_out),
            "--out-trades",
            str(trades_out),
        ]
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

        source_report = _load_json(report_path)
        replay_summary = _load_json(summary_out)
        execution = dict(replay_summary.get("execution_semantics") or source_report.get("execution_semantics") or {})

        report_row = {
            "report_stem": stem,
            "report_in": str(report_path),
            "summary_out": str(summary_out),
            "trades_out": str(trades_out),
            "trades_original": (source_report.get("config") or {}).get("trades_out"),
            "cycles_used": int(replay_summary.get("cycles_used", 0)),
            "entry_execution": execution.get("entry_execution"),
            "signal_reference": execution.get("signal_reference"),
            "orig_trades": _metric(replay_summary, "original", "trades"),
            "orig_total_pnl": _metric(replay_summary, "original", "total_pnl"),
            "orig_profit_factor": _metric(replay_summary, "original", "profit_factor"),
            "orig_positive_days_pct": _metric(replay_summary, "original", "positive_days_pct"),
            "orig_trade_max_drawdown": _metric(replay_summary, "original", "trade_max_drawdown"),
            "orig_daily_max_drawdown": _metric(replay_summary, "original", "daily_max_drawdown"),
            "replay_trades": _metric(replay_summary, "replay", "trades"),
            "replay_total_pnl": _metric(replay_summary, "replay", "total_pnl"),
            "replay_profit_factor": _metric(replay_summary, "replay", "profit_factor"),
            "replay_positive_days_pct": _metric(replay_summary, "replay", "positive_days_pct"),
            "replay_trade_max_drawdown": _metric(replay_summary, "replay", "trade_max_drawdown"),
            "replay_daily_max_drawdown": _metric(replay_summary, "replay", "daily_max_drawdown"),
            "delta_trades": _metric(replay_summary, "delta", "trades"),
            "delta_total_pnl": _metric(replay_summary, "delta", "total_pnl"),
            "delta_profit_factor": _metric(replay_summary, "delta", "profit_factor"),
            "delta_positive_days_pct": _metric(replay_summary, "delta", "positive_days_pct"),
            "delta_trade_max_drawdown": _metric(replay_summary, "delta", "trade_max_drawdown"),
            "delta_daily_max_drawdown": _metric(replay_summary, "delta", "daily_max_drawdown"),
        }
        summary_rows.append(report_row)
        manifest["reports"].append(
            {
                "report_in": str(report_path),
                "summary_out": str(summary_out),
                "trades_out": str(trades_out),
                "cycles_used": int(replay_summary.get("cycles_used", 0)),
                "execution_semantics": execution,
            }
        )

    batch_summary_csv = out_dir / "batch_summary.csv"
    batch_manifest_json = out_dir / "batch_manifest.json"
    _write_batch_summary(summary_rows, batch_summary_csv)
    batch_manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"saved batch summary: {batch_summary_csv}")
    print(f"saved batch manifest: {batch_manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

