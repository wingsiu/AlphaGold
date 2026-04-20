#!/usr/bin/env python3
"""Small launcher for running trading_bot.py in live best-base mode.

PyCharm Play-button friendly: run this file directly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TRADING_BOT_SCRIPT = PROJECT_ROOT / "trading_bot.py"
DEFAULT_WEAK_PERIODS_JSON = "runtime/bot_assets/weak-filter.json"


def _weak_cells_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    cells = payload.get("weak_cells") if isinstance(payload, dict) else None
    return len(cells) if isinstance(cells, list) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Launch trading_bot.py in live best-base mode.")
    p.add_argument("--once", action="store_true", help="Run a single cycle and exit.")
    p.add_argument("--sleep-seconds", type=int, default=5)
    p.add_argument("--prediction-poll-second", type=int, default=5)
    p.add_argument("--market-data-poll-second", type=int, default=30)
    p.add_argument("--prediction-cache-max-rows", type=int, default=1200)
    p.add_argument("--size", type=float, default=2)
    p.add_argument("--stop-loss-pct", type=float, default=15.0,
        help="Absolute point stop distance for long trades (default: 15 pts)")
    p.add_argument("--short-stop-loss-pct", type=float, default=18.0,
        help="Absolute point stop distance for short trades (default: 18 pts)")
    p.add_argument("--take-profit-pct", type=float, default=0.80,
        help="Target as %% of entry price (default: 0.80 = 0.8%%)")
    p.add_argument("--disable-dynamic-target-stop", action="store_true", help="Disable dynamic TP/SL updates.")
    p.add_argument("--max-hold-minutes", type=float, default=60.0)
    p.add_argument("--signal-model-path", default=None)
    p.add_argument("--weak-periods-json", default=DEFAULT_WEAK_PERIODS_JSON)
    return p


def main() -> int:
    args = build_parser().parse_args()

    weak_arg_path = Path(args.weak_periods_json)
    weak_abs_path = weak_arg_path if weak_arg_path.is_absolute() else (PROJECT_ROOT / weak_arg_path)
    weak_cells = _weak_cells_count(weak_abs_path)

    effective_weak_periods_json = str(args.weak_periods_json)
    if weak_cells == 0:
        default_abs = PROJECT_ROOT / DEFAULT_WEAK_PERIODS_JSON
        default_cells = _weak_cells_count(default_abs)
        if default_cells > 0:
            print(
                f"[run_live_bot] weak filter override '{args.weak_periods_json}' is missing/empty; "
                f"falling back to {DEFAULT_WEAK_PERIODS_JSON} ({default_cells} cells)."
            )
            effective_weak_periods_json = DEFAULT_WEAK_PERIODS_JSON
        else:
            print(
                f"[run_live_bot] weak filter override '{args.weak_periods_json}' is missing/empty, "
                "and fallback weak-filter is also missing/empty; continuing with requested path."
            )

    cmd = [
        sys.executable,
        "-u",
        str(TRADING_BOT_SCRIPT),
        "--signal-model-family", "best_base_state",
        "--mode", "live",
        "--sleep-seconds", str(args.sleep_seconds),
        "--prediction-poll-second", str(args.prediction_poll_second),
        "--market-data-poll-second", str(args.market_data_poll_second),
        "--prediction-cache-max-rows", str(args.prediction_cache_max_rows),
        "--weak-periods-json", effective_weak_periods_json,
        "--size", str(args.size),
        "--stop-loss-pct", str(args.stop_loss_pct),
        "--short-stop-loss-pct", str(args.short_stop_loss_pct),
        "--take-profit-pct", str(args.take_profit_pct),
        "--max-hold-minutes", str(args.max_hold_minutes),
    ]
    if args.once:
        cmd.append("--once")
    if args.disable_dynamic_target_stop:
        cmd.append("--disable-dynamic-target-stop")
    if args.signal_model_path:
        cmd.extend(["--signal-model-path", str(args.signal_model_path)])

    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())

