#!/usr/bin/env python3
"""Interactive runner for best-base backtest (no retrain, no state features).

What it does:
- Runs training/image_trend_ml.py with --model-in (no retrain) and weak filters.
- Supports 3 test-data options:
  A) today only
  B) custom date range (testing range only, date split)
  C) full predefined range
- Shows full statistics and time-distribution heatmaps.
- Optionally saves run artifacts (overwrite per option key).
- Optionally regenerates/saves weak filters from latest fullstats.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import pandas as pd
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl


MODEL_IN = PROJECT_ROOT / "runtime/bot_assets/backtest_model_best_base_weak_nostate.joblib"
SCRIPT = PROJECT_ROOT / "training/image_trend_ml.py"
DEFAULT_WEAK_FILTER_PATH = PROJECT_ROOT / "runtime/bot_assets/weak-filter.json"

# Best-base constants — Candidate E (promoted 2026-04-20)
# D + max_hold_minutes=60  → PnL=$5,217  WR=53.4%  PF=1.749  DD=-$171 (unchanged)
BASE_PARAMS = {
    "timeframe": "1min",
    "eval_mode": "single_split",
    "test_size": 0.4,
    "window": 150,
    "window_15m": 0,
    "min_window_range": 30,
    "min_15m_drop": 15,
    "min_15m_rise": 0,
    "horizon": 25,
    "trend_threshold": 0.008,
    "adverse_limit": 15,
    "long_target_threshold": 0.008,
    "short_target_threshold": 0.008,
    "long_adverse_limit": 15,
    "short_adverse_limit": 18,
    "classifier": "gradient_boosting",
    "max_flat_ratio": 2.5,
    "stage1_min_prob": 0.55,
    "stage2_min_prob": 0.58,       # fallback (overridden by up/down below)
    "stage2_min_prob_up": 0.65,    # long threshold (Candidate D)
    "stage2_min_prob_down": 0.62,  # short threshold (Candidate D)
    "max_hold_minutes": 60,        # hard timeout cap (Candidate E)
}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _hour_key(h: str) -> tuple[int, int]:
    if h == "09:30":
        return (9, 30)
    hh, mm = h.split(":")
    return (int(hh), int(mm))


PREDEFINED_START = "2025-05-20"
PREDEFINED_END = "2026-04-10"
PREDEFINED_TEST_START = "2025-11-25T17:02:00+00:00"
UTC_PLUS_2 = timezone(timedelta(hours=2))


@dataclass
class RunSpec:
    option_key: str
    label: str
    start_date: str
    end_date: str
    test_start_date: str | None
    engine_start_date: str | None = None


def _prompt_yes_no(msg: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{msg} {suffix} ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def _today_utc2_date() -> date:
    # Trading day starts at HKT 06:00, which is UTC+2 00:00.
    return datetime.now(UTC_PLUS_2).date()


def _parse_date_yyyy_mm_dd(raw: str, field_name: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}: {raw}. Expected YYYY-MM-DD.") from exc


def _default_b_dates(today: date) -> tuple[date, date]:
    # Default to the latest 7 dates including today.
    return today - timedelta(days=6), today


def _choose_option(
    latest_days_arg: int | None,
    b_start_date_arg: str | None,
    b_end_date_arg: str | None,
    a_pretrain_days_arg: int,
    b_pretrain_days_arg: int,
    option_arg: str | None,
    interactive_inputs: bool,
) -> RunSpec:
    choose_prompt = option_arg is None and interactive_inputs
    if choose_prompt:
        print("Choose test-data option:")
        print("  A) today only")
        print("  B) custom start/end dates")
        print("  C) full predefined range")
        option_arg = input("Option [A/B/C]: ").strip().upper() or "A"

    option = option_arg.strip().upper()
    today = _today_utc2_date()

    if option == "A":
        if a_pretrain_days_arg < 1:
            raise ValueError("--a-pretrain-days must be >= 1")
        engine_start = today - timedelta(days=a_pretrain_days_arg)
        return RunSpec(
            option_key="A_today",
            label="today_only",
            start_date=today.isoformat(),
            end_date=today.isoformat(),
            # Keep testing on today only, with pre-context for split stability.
            test_start_date=today.isoformat(),
            engine_start_date=engine_start.isoformat(),
        )

    if option == "B":
        default_start, default_end = _default_b_dates(today)
        start_date = default_start
        end_date = default_end

        if b_start_date_arg:
            start_date = _parse_date_yyyy_mm_dd(b_start_date_arg, "--b-start-date")
        if b_end_date_arg:
            end_date = _parse_date_yyyy_mm_dd(b_end_date_arg, "--b-end-date")

        if latest_days_arg is not None and not b_start_date_arg and not b_end_date_arg:
            if latest_days_arg <= 0:
                raise ValueError("--latest-days must be > 0")
            start_date = today - timedelta(days=latest_days_arg - 1)
            end_date = today

        if b_start_date_arg is None and b_end_date_arg is None and latest_days_arg is None and interactive_inputs:
            raw_start = input(f"Start date [default {default_start.isoformat()}]: ").strip()
            raw_end = input(f"End date   [default {default_end.isoformat()}]: ").strip()
            if raw_start:
                start_date = _parse_date_yyyy_mm_dd(raw_start, "start date")
            if raw_end:
                end_date = _parse_date_yyyy_mm_dd(raw_end, "end date")

        if start_date > end_date:
            raise ValueError("Option B requires start_date <= end_date")
        if b_pretrain_days_arg < 1:
            raise ValueError("--b-pretrain-days must be >= 1")

        engine_start = start_date - timedelta(days=b_pretrain_days_arg)

        return RunSpec(
            option_key="B_date_range",
            label=f"date_range_{start_date.isoformat()}_{end_date.isoformat()}",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            # Force date-based split so Option B ignores ratio splitting.
            test_start_date=start_date.isoformat(),
            engine_start_date=engine_start.isoformat(),
        )

    if option == "C":
        return RunSpec(
            option_key="C_full_predefined",
            label="full_predefined",
            start_date=PREDEFINED_START,
            end_date=PREDEFINED_END,
            test_start_date=PREDEFINED_TEST_START,
        )

    raise ValueError(f"Unknown option: {option}")


def _ensure_weak_filter_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"weak_cells": []}, indent=2) + "\n", encoding="utf-8")


def _resolve_weak_filter_path(path: Path) -> Path | None:
    """Return path only when weak-filter file exists and contains non-empty weak_cells."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    cells = payload.get("weak_cells") if isinstance(payload, dict) else None
    if not isinstance(cells, list) or len(cells) == 0:
        return None
    return path


def _paths_for_run(spec: RunSpec, save_results: bool) -> dict[str, Path]:
    if save_results:
        base = PROJECT_ROOT / "runtime" / "backtest_no_retrain"
        base.mkdir(parents=True, exist_ok=True)
        stem = spec.option_key
        if spec.option_key.startswith("B_"):
            # Keep separate saved artifacts for each Option B date window.
            stem = f"{spec.option_key}_{spec.start_date}_to_{spec.end_date}"
    else:
        base = PROJECT_ROOT / "runtime" / "_tmp_backtest_no_retrain"
        base.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        stem = f"{spec.option_key}_{ts}"

    out = {
        "model": base / f"{stem}_model.joblib",
        "report": base / f"{stem}_report.json",
        "trades": base / f"{stem}_trades.csv",
        "fullstats": base / f"{stem}_directional_pnl_fullstats.json",
        "report_fullstats": base / f"{stem}_report_fullstats.json",
        "signals_table": base / f"{stem}_signals_table.csv",
        "signals_table_by_time": base / f"{stem}_signals_table_by_time.csv",
        "trades_table_entry": base / f"{stem}_trades_table_by_entry_time.csv",
    }
    if spec.option_key.startswith("C_"):
        out["monthly_stats"] = base / f"{stem}_monthly_stats.csv"
    return out


def _build_cmd(spec: RunSpec, weak_filter_path: Path | None, out: dict[str, Path]) -> list[str]:
    cmd_start_date = spec.engine_start_date or spec.start_date
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--start-date",
        cmd_start_date,
        "--end-date",
        spec.end_date,
        "--timeframe",
        BASE_PARAMS["timeframe"],
        "--eval-mode",
        BASE_PARAMS["eval_mode"],
        "--disable-time-filter",
        "--window",
        str(BASE_PARAMS["window"]),
        "--window-15m",
        str(BASE_PARAMS["window_15m"]),
        "--min-window-range",
        str(BASE_PARAMS["min_window_range"]),
        "--min-15m-drop",
        str(BASE_PARAMS["min_15m_drop"]),
        "--min-15m-rise",
        str(BASE_PARAMS["min_15m_rise"]),
        "--horizon",
        str(BASE_PARAMS["horizon"]),
        "--trend-threshold",
        str(BASE_PARAMS["trend_threshold"]),
        "--adverse-limit",
        str(BASE_PARAMS["adverse_limit"]),
        "--long-target-threshold",
        str(BASE_PARAMS["long_target_threshold"]),
        "--short-target-threshold",
        str(BASE_PARAMS["short_target_threshold"]),
        "--long-adverse-limit",
        str(BASE_PARAMS["long_adverse_limit"]),
        "--short-adverse-limit",
        str(BASE_PARAMS["short_adverse_limit"]),
        "--classifier",
        str(BASE_PARAMS["classifier"]),
        "--max-flat-ratio",
        str(BASE_PARAMS["max_flat_ratio"]),
        "--stage1-min-prob",
        str(BASE_PARAMS["stage1_min_prob"]),
        "--stage2-min-prob",
        str(BASE_PARAMS["stage2_min_prob"]),
        "--stage2-min-prob-up",
        str(BASE_PARAMS["stage2_min_prob_up"]),
        "--stage2-min-prob-down",
        str(BASE_PARAMS["stage2_min_prob_down"]),
        "--max-hold-minutes",
        str(BASE_PARAMS["max_hold_minutes"]),
        "--model-in",
        str(MODEL_IN),
        "--model-out",
        str(out["model"].relative_to(PROJECT_ROOT)),
        "--report-out",
        str(out["report"].relative_to(PROJECT_ROOT)),
        "--trades-out",
        str(out["trades"].relative_to(PROJECT_ROOT)),
    ]
    if weak_filter_path is not None:
        cmd.extend(["--weak-periods-json", str(weak_filter_path)])
    if spec.test_start_date:
        cmd.extend(["--test-start-date", spec.test_start_date])
    return cmd

    if spec.test_start_date is None:
        cmd.extend(["--test-size", str(BASE_PARAMS["test_size"])])

def _print_full_stats(pnl: dict[str, Any]) -> None:
    all_stats = pnl["all"]
    print("\n=== Full Statistics ===")
    print(f"trades={pnl['trades']}")
    print(f"total_pnl={pnl['total_pnl']:.2f}")
    print(f"avg_trade={pnl['avg_trade']:.4f}")
    print(f"n_days={pnl['n_days']}")
    print(f"avg_day={pnl['avg_day']}")
    print(f"positive_days_pct={pnl['positive_days_pct']}")
    print(f"win_rate_pct={all_stats['win_rate_pct']}")
    print(f"profit_factor={all_stats['profit_factor']}")
    print(f"trade_max_drawdown={all_stats['trade_max_drawdown']}")
    print(f"daily_max_drawdown={all_stats['daily_max_drawdown']}")

    long_stats = pnl["long_up"]
    short_stats = pnl["short_down"]
    print("\n=== Long / Short ===")
    print(f"long: trades={long_stats['trades']} pnl={long_stats['total_pnl']:.2f} win_rate={long_stats['win_rate_pct']}")
    print(f"short: trades={short_stats['trades']} pnl={short_stats['total_pnl']:.2f} win_rate={short_stats['win_rate_pct']}")


def _print_heatmaps(pnl: dict[str, Any]) -> None:
    print("\n=== Time Distribution Heatmaps ===")
    session_heatmaps = pnl["all"]["time_distribution"]["session_heatmaps"]
    for session in ("hkt", "london", "ny"):
        sm = session_heatmaps.get(session)
        if not sm:
            continue
        print("\n" + sm["rendered_tables"]["trade_count"])
        print("\n" + sm["rendered_tables"]["win_rate_pct"])
        print("\n" + sm["rendered_tables"]["avg_trade"])
        print("\n" + sm["rendered_tables"]["total_pnl"])


def _print_testing_period(report: dict[str, Any], trades_path: Path) -> None:
    cfg = report.get("config", {}) if isinstance(report, dict) else {}
    split_mode = report.get("split_mode", "unknown") if isinstance(report, dict) else "unknown"
    signal_start = cfg.get("test_start_date") or cfg.get("start_date") or "unknown"
    signal_end = cfg.get("end_date") or "unknown"
    exec_start = signal_start
    exec_end = signal_end

    try:
        trades_df = pd.read_csv(trades_path)
    except Exception:
        trades_df = pd.DataFrame()

    if not trades_df.empty:
        if "ts" in trades_df.columns:
            ts = pd.to_datetime(trades_df["ts"], errors="coerce", utc=True).dropna()
            if not ts.empty:
                signal_start = ts.min().isoformat()
                signal_end = ts.max().isoformat()
        if "entry_time" in trades_df.columns:
            entry_ts = pd.to_datetime(trades_df["entry_time"], errors="coerce", utc=True).dropna()
            if not entry_ts.empty:
                exec_start = entry_ts.min().isoformat()
        if "exit_time" in trades_df.columns:
            exit_ts = pd.to_datetime(trades_df["exit_time"], errors="coerce", utc=True).dropna()
            if not exit_ts.empty:
                exec_end = exit_ts.max().isoformat()

    print("\n=== Testing Data Period ===")
    print(f"split_mode={split_mode}")
    print(f"signals: {signal_start} -> {signal_end}")
    print(f"entries/exits: {exec_start} -> {exec_end}")


def _extract_weak_cells(pnl: dict[str, Any], min_trades: int = 3, max_win_rate: float = 40.0) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    session_heatmaps = pnl["all"]["time_distribution"]["session_heatmaps"]
    for session in ("hkt", "london", "ny"):
        day_map = session_heatmaps[session]["cell_stats"]
        for day, hour_map in day_map.items():
            for hour, st in hour_map.items():
                if not st:
                    continue
                t = int(st.get("trades", 0))
                pnl_total = float(st.get("total_pnl", 0.0))
                wr = st.get("win_rate_pct", None)
                wr = float(wr) if wr is not None else None
                if t >= min_trades and pnl_total < 0.0 and (wr is not None and wr < max_win_rate):
                    out.append({"session": session, "day": day, "hour": hour})
    out.sort(key=lambda c: (c["session"], DAY_ORDER.index(c["day"]), _hour_key(c["hour"])))
    return out


def _load_existing_weak_cells(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    cells = payload.get("weak_cells") if isinstance(payload, dict) else None
    out: list[dict[str, str]] = []
    if isinstance(cells, list):
        for c in cells:
            if not isinstance(c, dict):
                continue
            s = str(c.get("session", "")).strip()
            d = str(c.get("day", "")).strip()
            h = str(c.get("hour", "")).strip()
            if s and d and h:
                out.append({"session": s, "day": d, "hour": h})
    return out


def _merge_weak_cells(existing: list[dict[str, str]], new_cells: list[dict[str, str]]) -> list[dict[str, str]]:
    merged = list(existing)
    seen = {(c["session"], c["day"], c["hour"]) for c in existing}
    for c in new_cells:
        key = (c["session"], c["day"], c["hour"])
        if key not in seen:
            merged.append(c)
            seen.add(key)
    merged.sort(key=lambda c: (c["session"], DAY_ORDER.index(c["day"]) if c["day"] in DAY_ORDER else 99, _hour_key(c["hour"])))
    return merged


def _build_signals_table(report: dict[str, Any]) -> pd.DataFrame:
    labels = report.get("full_confusion_labels", [])
    matrix = report.get("full_confusion_matrix", [])
    if not labels or not matrix:
        return pd.DataFrame(columns=pd.Index(["label", "true_count", "pred_count"]))
    m = pd.DataFrame(matrix, index=labels, columns=labels)
    return pd.DataFrame(
        {
            "label": labels,
            "true_count": [int(m.loc[l].sum()) for l in labels],
            "pred_count": [int(m[l].sum()) for l in labels],
        }
    )


def _build_signals_detail_by_time(trades_path: Path) -> pd.DataFrame:
    t = pd.read_csv(trades_path)
    if t.empty:
        return pd.DataFrame()
    if "ts" in t.columns:
        t["ts"] = pd.to_datetime(t["ts"], errors="coerce", utc=True)
    if "last_target_time" in t.columns:
        t["last_target_time"] = pd.to_datetime(t["last_target_time"], errors="coerce", utc=True)

    rows: list[dict[str, Any]] = []
    for _, row in t.iterrows():
        entry_price = row.get("entry_price", None)
        target_price = row.get("last_target_price", None)
        side = row.get("side", "")
        rows.append(
            {
                "trigger_event": "entry",
                "signal_idx": row.get("signal_idx", None),
                "ts": row.get("ts", pd.NaT),
                "side": side,
                "entry_price": entry_price,
                "target_price": target_price,
                "signal_prob": row.get("entry_signal_prob", None),
                "trade_exit_reason": row.get("exit_reason", ""),
            }
        )

        updates = int(row.get("target_updates", 0) or 0)
        if updates > 0:
            event_price = row.get("last_target_price", entry_price)
            rows.append(
                {
                    "trigger_event": "change_target_stop",
                    "signal_idx": row.get("last_target_signal_idx", None),
                    "ts": row.get("last_target_time", pd.NaT),
                    "side": side,
                    "entry_price": event_price,
                    "target_price": target_price,
                    "signal_prob": row.get("last_signal_prob", None),
                    "trade_exit_reason": row.get("exit_reason", ""),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if "ts" in out.columns:
        out = out.sort_values(["ts", "trigger_event"], kind="stable")
    return out.reset_index(drop=True)


def _print_signals_table_hkt(signals_df: pd.DataFrame, max_rows: int = 100) -> None:
    shown = signals_df.head(max_rows).copy()
    if shown.empty:
        print("(empty)")
        return

    ts_hkt = shown["ts"].dt.tz_convert("Asia/Hong_Kong") if "ts" in shown.columns else pd.Series(dtype="datetime64[ns]")
    print(
        f"{'#':<5} {'Date':<12} {'Time (HKT)':<12} {'Side':<8} {'Price':<10} {'Target':<10} {'Prob':<8} {'Event':<20} {'Signal#':<8}"
    )
    print("-" * 108)
    for i, row in shown.iterrows():
        ts_i = ts_hkt.iloc[i] if i < len(ts_hkt) else pd.NaT
        date_str = ts_i.strftime("%Y-%m-%d") if pd.notna(ts_i) else "-"
        time_str = ts_i.strftime("%H:%M") if pd.notna(ts_i) else "-"
        side = str(row.get("side", ""))
        entry_price = row.get("entry_price", None)
        target_price = row.get("target_price", None)
        prob = row.get("signal_prob", None)
        event = str(row.get("trigger_event", ""))
        signal_idx = row.get("signal_idx", None)
        entry_str = f"{float(entry_price):.2f}" if pd.notna(entry_price) else "-"
        target_str = f"{float(target_price):.2f}" if pd.notna(target_price) else "-"
        prob_str = f"{float(prob):.3f}" if pd.notna(prob) else "-"
        idx_str = str(int(signal_idx)) if pd.notna(signal_idx) else "-"
        print(
            f"{i + 1:<5} {date_str:<12} {time_str:<12} {side:<8} {entry_str:<10} {target_str:<10} "
            f"{prob_str:<8} {event:<20} {idx_str:<8}"
        )

    if len(signals_df) > max_rows:
        print(f"... showing first {max_rows} of {len(signals_df)} rows")


def _build_trades_table_by_entry_time(trades_path: Path) -> pd.DataFrame:
    t = pd.read_csv(trades_path)
    if t.empty:
        return pd.DataFrame()
    t["entry_time"] = pd.to_datetime(t["entry_time"], errors="coerce", utc=True)
    if "exit_time" in t.columns:
        t["exit_time"] = pd.to_datetime(t["exit_time"], errors="coerce", utc=True)
    cols = [
        c
        for c in ["entry_time", "exit_time", "side", "entry_price", "exit_price", "pnl", "exit_reason"]
        if c in t.columns
    ]
    return t[cols].sort_values("entry_time").reset_index(drop=True)


def _fmt_exit_hkt(value: Any) -> str:
    if pd.isna(value):
        return "-"
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return "-"
    return ts.tz_convert("Asia/Hong_Kong").strftime("%Y-%m-%d %H:%M")


def _print_trades_table_hkt(by_entry: pd.DataFrame, max_rows: int = 100) -> None:
    shown = by_entry.head(max_rows).copy()
    if shown.empty:
        print("(empty)")
        return

    entry_hkt = shown["entry_time"].dt.tz_convert("Asia/Hong_Kong") if "entry_time" in shown.columns else pd.Series(dtype="datetime64[ns]")

    print(
        f"{'#':<5} {'Date':<12} {'Time (HKT)':<12} {'Side':<8} {'Entry':<10} "
        f"{'Exit':<10} {'PnL':<10} {'Exit Time (HKT)':<18} {'Reason':<16}"
    )
    print("-" * 115)

    for i, row in shown.iterrows():
        entry_dt = entry_hkt.iloc[i] if i < len(entry_hkt) else pd.NaT
        date_str = entry_dt.strftime("%Y-%m-%d") if pd.notna(entry_dt) else "-"
        time_str = entry_dt.strftime("%H:%M") if pd.notna(entry_dt) else "-"
        side = str(row.get("side", ""))
        entry_price = row.get("entry_price", None)
        exit_price = row.get("exit_price", None)
        pnl = row.get("pnl", None)
        exit_reason = str(row.get("exit_reason", ""))
        entry_str = f"{float(entry_price):.2f}" if pd.notna(entry_price) else "-"
        exit_str = f"{float(exit_price):.2f}" if pd.notna(exit_price) else "-"
        pnl_str = f"{float(pnl):.2f}" if pd.notna(pnl) else "-"
        exit_time_str = _fmt_exit_hkt(row.get("exit_time", None))
        print(
            f"{i + 1:<5} {date_str:<12} {time_str:<12} {side:<8} {entry_str:<10} "
            f"{exit_str:<10} {pnl_str:<10} {exit_time_str:<18} {exit_reason:<16}"
        )

    if len(by_entry) > max_rows:
        print(f"... showing first {max_rows} of {len(by_entry)} rows")


def _print_option_ab_tables(report: dict[str, Any], trades_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signals = _build_signals_table(report)
    signals_detail = _build_signals_detail_by_time(trades_path)
    by_entry = _build_trades_table_by_entry_time(trades_path)
    print("\n=== Signals Table ===")
    print(signals.to_string(index=False) if not signals.empty else "(empty)")
    print("\n=== Signals Table (By Time, HKT) ===")
    _print_signals_table_hkt(signals_detail, max_rows=100)
    print("\n=== Trades Table (By Entry Time) ===")
    _print_trades_table_hkt(by_entry, max_rows=100)
    return signals, signals_detail, by_entry


def _build_monthly_stats_from_trades(trades_path: Path) -> pd.DataFrame:
    t = pd.read_csv(trades_path)
    if t.empty or "entry_time" not in t.columns or "pnl" not in t.columns:
        return pd.DataFrame(columns=pd.Index(["month", "trades", "total_pnl", "avg_trade", "win_rate_pct"]))

    t["entry_time"] = pd.to_datetime(t["entry_time"], errors="coerce", utc=True)
    t["pnl"] = pd.to_numeric(t["pnl"], errors="coerce")
    t = t.dropna(subset=["entry_time", "pnl"]).copy()
    if t.empty:
        return pd.DataFrame(columns=pd.Index(["month", "trades", "total_pnl", "avg_trade", "win_rate_pct"]))

    # Convert to naive UTC before month bucketing to avoid timezone-drop warnings.
    t["month"] = t["entry_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)
    g = t.groupby("month", sort=True)
    out = g["pnl"].agg(trades="size", total_pnl="sum", avg_trade="mean").reset_index()
    win_rate_by_month = g["pnl"].apply(lambda s: (s > 0).mean())
    out["win_rate_pct"] = (out["month"].map(win_rate_by_month) * 100.0)
    out["total_pnl"] = out["total_pnl"].round(2)
    out["avg_trade"] = out["avg_trade"].round(4)
    out["win_rate_pct"] = out["win_rate_pct"].round(2)
    return out


def _print_monthly_stats(monthly_df: pd.DataFrame) -> None:
    print("\n=== Monthly Statistics (Option C) ===")
    if monthly_df.empty:
        print("(empty)")
        return
    print(monthly_df.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run best-base backtest (no retrain, no state features) with weak filters.")
    parser.add_argument("--option", choices=["A", "B", "C"], default="A", help="A=today only, B=date range (date split), C=full predefined (default: A)")
    parser.add_argument("--latest-days", type=int, default=None, help="Option B helper: latest N dates ending today (used only if B start/end are not provided)")
    parser.add_argument("--b-start-date", default=None, help="Option B start date in YYYY-MM-DD (default: latest 7 dates including today)")
    parser.add_argument("--b-end-date", default=None, help="Option B end date in YYYY-MM-DD (default: today)")
    parser.add_argument("--a-pretrain-days", type=int, default=7, help="Option A pre-test context window in days to avoid empty dataset (default: 7)")
    parser.add_argument("--b-pretrain-days", type=int, default=14, help="Option B pre-test context window in days to prevent ratio fallback (default: 14)")
    parser.add_argument("--non-interactive", action="store_true", help="Auto-save outputs and weak filters")
    args = parser.parse_args()

    if not SCRIPT.exists():
        raise FileNotFoundError(f"Missing backtest script: {SCRIPT}")
    if not MODEL_IN.exists():
        raise FileNotFoundError(f"Missing no-retrain model: {MODEL_IN}")

    weak_filter_path = _resolve_weak_filter_path(DEFAULT_WEAK_FILTER_PATH)

    spec = _choose_option(
        args.latest_days,
        args.b_start_date,
        args.b_end_date,
        args.a_pretrain_days,
        args.b_pretrain_days,
        args.option,
        interactive_inputs=not args.non_interactive,
    )
    while True:
        print(f"\nSelected option: {spec.option_key} ({spec.label})")
        print(f"Testing date range: {spec.start_date} -> {spec.end_date}")
        if spec.engine_start_date and spec.engine_start_date != spec.start_date:
            print(f"Pre-test context start: {spec.engine_start_date}")
        if weak_filter_path is None:
            print("Weak filter: disabled (file missing/empty)")
        else:
            print(f"Weak filter: {weak_filter_path}")

        # Always run to temp first, then prompt to persist after stats are shown.
        run_paths = _paths_for_run(spec, save_results=False)

        cmd = _build_cmd(spec, weak_filter_path, run_paths)
        print("\nRunning:")
        print(" ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False).returncode
        if rc == 0:
            break

        if spec.option_key.startswith("A_"):
            print("\nToday-only backtest has no usable data (or filters are too strict).")
            if not args.non_interactive and _prompt_yes_no("Fallback to option B latest 7 days and rerun now?", default=True):
                today = _today_utc2_date()
                start, end = _default_b_dates(today)
                spec = RunSpec(
                    option_key="B_date_range",
                    label=f"date_range_{start.isoformat()}_{end.isoformat()}",
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    test_start_date=start.isoformat(),
                    engine_start_date=(start - timedelta(days=args.b_pretrain_days)).isoformat(),
                )
                continue
            print("Try option B with recent days, for example:")
            print("  python3 -u runtime/bot_assets/backtest_no_retrain.py --option B --latest-days 7")
            print("  python3 -u runtime/bot_assets/backtest_no_retrain.py --option B --b-start-date 2026-03-01 --b-end-date 2026-04-17")
        print(f"Backtest failed with return code={rc}")
        return rc

    pnl = rebuild_directional_pnl(run_paths["trades"])
    _print_full_stats(pnl)
    _print_heatmaps(pnl)

    signals_df = pd.DataFrame()
    signals_detail_df = pd.DataFrame()
    trades_entry_df = pd.DataFrame()
    monthly_df = pd.DataFrame()
    run_report = json.loads(run_paths["report"].read_text(encoding="utf-8"))
    if spec.option_key.startswith("B_"):
        split_mode = str(run_report.get("split_mode", ""))
        if not split_mode.startswith("date:"):
            print("\nOption B requires date-based split, but engine returned:", split_mode or "unknown")
            print("Try increasing --b-pretrain-days (for example: 30).")
            return 2
    _print_testing_period(run_report, run_paths["trades"])
    signals_df, signals_detail_df, trades_entry_df = _print_option_ab_tables(run_report, run_paths["trades"])
    if spec.option_key.startswith("C_"):
        monthly_df = _build_monthly_stats_from_trades(run_paths["trades"])
        _print_monthly_stats(monthly_df)

    save_results = True if args.non_interactive else _prompt_yes_no("Save all run results to option files?", default=True)
    if save_results:
        out_paths = _paths_for_run(spec, save_results=True)
        shutil.copy2(run_paths["model"], out_paths["model"])
        shutil.copy2(run_paths["report"], out_paths["report"])
        shutil.copy2(run_paths["trades"], out_paths["trades"])
        out_paths["fullstats"].write_text(json.dumps(pnl, indent=2) + "\n", encoding="utf-8")
        report = json.loads(out_paths["report"].read_text(encoding="utf-8"))
        report["directional_pnl"] = pnl
        out_paths["report_fullstats"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        signals_df.to_csv(out_paths["signals_table"], index=False)
        signals_detail_df.to_csv(out_paths["signals_table_by_time"], index=False)
        trades_entry_df.to_csv(out_paths["trades_table_entry"], index=False)
        if "monthly_stats" in out_paths:
            monthly_df.to_csv(out_paths["monthly_stats"], index=False)
        print("\nSaved result files (overwritten for this option):")
        for k, p in out_paths.items():
            print(f"- {k}: {p}")
    else:
        run_paths["fullstats"].write_text(json.dumps(pnl, indent=2) + "\n", encoding="utf-8")
        report = json.loads(run_paths["report"].read_text(encoding="utf-8"))
        report["directional_pnl"] = pnl
        run_paths["report_fullstats"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        signals_df.to_csv(run_paths["signals_table"], index=False)
        signals_detail_df.to_csv(run_paths["signals_table_by_time"], index=False)
        trades_entry_df.to_csv(run_paths["trades_table_entry"], index=False)
        if "monthly_stats" in run_paths:
            monthly_df.to_csv(run_paths["monthly_stats"], index=False)
        print("\nResult files kept in temporary folder:")
        for k, p in run_paths.items():
            print(f"- {k}: {p}")

    save_weak: bool
    if spec.option_key.startswith("A_") or spec.option_key.startswith("B_"):
        save_weak = False
        print("\nSkip saving weak filters for options A/B.")
    else:
        save_weak = True if args.non_interactive else _prompt_yes_no("Save weak filters from this run?", default=False)
    if save_weak:
        weak_cells = _extract_weak_cells(pnl)
        existing_cells = _load_existing_weak_cells(DEFAULT_WEAK_FILTER_PATH)
        merged_cells = _merge_weak_cells(existing_cells, weak_cells)
        DEFAULT_WEAK_FILTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_WEAK_FILTER_PATH.write_text(json.dumps({"weak_cells": merged_cells}, indent=2) + "\n", encoding="utf-8")
        print(f"Saved weak filters (merged into existing): {DEFAULT_WEAK_FILTER_PATH}")
        print(f"new_weak_cells={len(weak_cells)}")
        print(f"total_weak_cells={len(merged_cells)}")
        for cell in weak_cells:
            print(f"- {cell['session']}:{cell['day']}:{cell['hour']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

