#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "training" / "image_trend_ml.py"
MODEL_IN = PROJECT_ROOT / "runtime" / "bot_assets" / "backtest_model_best_base_weak_nostate.joblib"
WEAK = PROJECT_ROOT / "runtime" / "bot_assets" / "weak-filter.json"
OUTDIR = PROJECT_ROOT / "training" / "testing" / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = OUTDIR / f"sweep_holdcap_h45_{ts}.csv"

    fixed = [
        "--start-date", "2025-05-20",
        "--end-date", "2026-04-10",
        "--test-start-date", "2025-11-25T17:02:00+00:00",
        "--timeframe", "1min",
        "--eval-mode", "single_split",
        "--disable-time-filter",
        "--window", "150",
        "--window-15m", "0",
        "--min-window-range", "30",
        "--min-15m-drop", "15",
        "--min-15m-rise", "0",
        "--horizon", "45",
        "--trend-threshold", "0.008",
        "--adverse-limit", "15",
        "--long-target-threshold", "0.008",
        "--short-target-threshold", "0.008",
        "--long-adverse-limit", "15",
        "--short-adverse-limit", "18",
        "--classifier", "gradient_boosting",
        "--max-flat-ratio", "2.5",
        "--stage1-min-prob", "0.55",
        "--stage2-min-prob", "0.58",
        "--stage2-min-prob-up", "0.65",
        "--stage2-min-prob-down", "0.62",
        "--model-in", str(MODEL_IN),
    ]

    caps = [45, 50, 60, 75]
    rows: list[dict[str, object]] = []

    for i, cap in enumerate(caps, 1):
        stem = OUTDIR / f"holdcap_h45_{ts}_{cap}"
        cmd = [
            sys.executable,
            str(SCRIPT),
            *fixed,
            "--max-hold-minutes",
            str(cap),
            "--weak-periods-json",
            str(WEAK),
            "--model-out",
            str(stem) + "_model.joblib",
            "--report-out",
            str(stem) + "_report.json",
            "--trades-out",
            str(stem) + "_trades.csv",
        ]
        print(f"[{i}/{len(caps)}] running max_hold={cap}")
        rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False).returncode
        if rc != 0:
            rows.append({"horizon": 45, "max_hold_minutes": cap, "error": f"rc={rc}"})
            continue

        rpt = json.loads((Path(str(stem) + "_report.json")).read_text())
        dp = rpt.get("directional_pnl", rpt)
        all_stats = dp.get("all", dp)
        rows.append(
            {
                "horizon": 45,
                "max_hold_minutes": cap,
                "trades": int(all_stats.get("trades", 0)),
                "total_pnl": float(all_stats.get("total_pnl", 0.0)),
                "avg_trade": float(all_stats.get("avg_trade", 0.0)),
                "win_rate_pct": float(all_stats.get("win_rate_pct", 0.0)),
                "profit_factor": float(all_stats.get("profit_factor") or 0.0),
                "avg_trades_per_day": float(all_stats.get("avg_trades_per_day", 0.0)),
                "trade_max_drawdown": float(all_stats.get("trade_max_drawdown", 0.0)),
                "daily_max_drawdown": float(all_stats.get("daily_max_drawdown", 0.0)),
                "error": "",
            }
        )

    fieldnames = [
        "horizon",
        "max_hold_minutes",
        "trades",
        "total_pnl",
        "avg_trade",
        "win_rate_pct",
        "profit_factor",
        "avg_trades_per_day",
        "trade_max_drawdown",
        "daily_max_drawdown",
        "error",
    ]
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("saved", summary)
    valid = [r for r in rows if not r.get("error")]
    if valid:
        best = max(valid, key=lambda r: float(r["total_pnl"]))
        print("best", best)


if __name__ == "__main__":
    main()

