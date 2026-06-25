"""Compare live journal trades vs v15 hybrid backtest for the current trading day."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any
from pathlib import Path

import pandas as pd

from mobile_api.journal import (
    SignalJournal,
    _backtest_row_to_trade,
    trading_day_label,
    trading_day_start_utc,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _summary_from_trades(trades: list[dict]) -> dict:
    return SignalJournal().trades_summary(trades)


def _summary_from_backtest_csv(csv_path: Path, day_start: datetime) -> dict:
    if not csv_path.exists():
        return {
            "trade_count": 0,
            "closed_count": 0,
            "open_count": 0,
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "trades": [],
        }
    df = pd.read_csv(csv_path)
    if df.empty:
        return {
            "trade_count": 0,
            "closed_count": 0,
            "open_count": 0,
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "trades": [],
        }
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    day_end = day_start + pd.Timedelta(days=1)
    start_ts = pd.Timestamp(day_start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(day_end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    day_df = (
        df[(df["entry_time"] >= start_ts) & (df["entry_time"] < end_ts)]
        .sort_values("entry_time", ascending=False)
        .copy()
    )
    pnls = day_df["pnl"].astype(float).tolist() if "pnl" in day_df.columns else []
    wins = sum(1 for p in pnls if p > 0)
    rows = [_backtest_row_to_trade(r, i) for i, (_, r) in enumerate(day_df.iterrows())]
    summary = SignalJournal().trades_summary(
        [
            {
                "status": "closed",
                "pnl": float(r["pnl"]) if r.get("pnl") is not None else None,
            }
            for r in rows
        ]
    )
    summary["trades"] = rows
    return summary


def _write_day_snap_from_csv(day_start: datetime) -> Path:
    """Persist only this trading day's rows into the mobile snapshot CSV."""
    import pandas as pd

    snap = _snap_path(day_start)
    src = _backtest_csv_path()
    if not src.exists():
        snap.write_text(
            "entry_time,exit_time,pnl,side,source,pattern,exit_reason\n",
            encoding="utf-8",
        )
        return snap
    df = pd.read_csv(src)
    if df.empty or "entry_time" not in df.columns:
        return snap
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    day_end = pd.Timestamp(day_start) + pd.Timedelta(days=1)
    start_ts = pd.Timestamp(day_start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(day_end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    day_df = df[(df["entry_time"] >= start_ts) & (df["entry_time"] < end_ts)]
    day_df.to_csv(snap, index=False)
    return snap


def _backtest_summary_for_day(
    day_start: datetime, *, refresh: bool = False
) -> tuple[dict[str, Any], str | None]:
    note: str | None = None

    # Check CSV + snapshot first
    bt_summary = _summary_from_backtest_csv(_backtest_csv_path(), day_start)
    if bt_summary["trade_count"] > 0:
        return bt_summary, note

    snap = _snap_path(day_start)
    if snap.exists():
        snap_summary = _summary_from_backtest_csv(snap, day_start)
        if snap_summary["trade_count"] > 0:
            return snap_summary, note

    # No data for this trading day — auto-refresh once
    note = "Backtest refreshed for this trading day."
    run_backtest_for_day(day_start, refresh=True)
    _write_day_snap_from_csv(day_start)

    bt_summary = _summary_from_backtest_csv(_backtest_csv_path(), day_start)
    if bt_summary["trade_count"] > 0:
        return bt_summary, note

    # Still no data after refresh — the day may not have started yet
    note = "No backtest trades in this window — trading day may not have started."
    return bt_summary, note


def _snap_path(day_start: datetime) -> Path:
    out_dir = PROJECT_ROOT / "runtime" / "mobile"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = trading_day_label(day_start)
    return out_dir / f"backtest_{label}.csv"


def _backtest_csv_path() -> Path:
    from mobile_api.journal import _backtest_csv_path as journal_bt_path

    return journal_bt_path()


def _backtest_python() -> str:
    """Prefer project venv — mobile API may run under launchd with system Python."""
    venv_py = PROJECT_ROOT / ".venv" / "bin" / "python3"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def run_backtest_for_day(day_start: datetime, *, refresh: bool = False) -> Path:
    """Run v15 hybrid backtest for trading day (22:00 UTC cutoff) → day snapshot CSV."""
    snap = _snap_path(day_start)
    if snap.exists() and not refresh:
        age_h = (datetime.now(timezone.utc).timestamp() - snap.stat().st_mtime) / 3600.0
        if age_h < 1.0:
            return snap

    start_str = pd.Timestamp(day_start).strftime("%Y-%m-%dT%H:%M:%S")
    end_str = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%S")
    env = os.environ.copy()
    env["V14_HYBRID"] = "1"
    env.setdefault("V14_FVG_MIN_GAP", "0")
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    log_path = PROJECT_ROOT / "runtime" / "mobile" / "backtest_refresh.log"

    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n--- refresh {start_str} → {end_str} ---\n")
        log_f.flush()
        subprocess.run(
            [
                _backtest_python(),
                str(PROJECT_ROOT / "v15" / "backtest" / "backtest_v15.py"),
                start_str,
                end_str,
            ],
            cwd=PROJECT_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return _write_day_snap_from_csv(day_start)


def run_today_backtest(*, refresh: bool = False) -> Path:
    return run_backtest_for_day(trading_day_start_utc(), refresh=refresh)


def _fmt_hkt_window(day_start: datetime) -> str:
    start = pd.Timestamp(day_start).tz_convert("UTC")
    end = start + pd.Timedelta(days=1)
    hkt = __import__("zoneinfo").ZoneInfo("Asia/Hong_Kong")
    return (
        f"{start.tz_convert(hkt).strftime('%a %H:%M')} – "
        f"{end.tz_convert(hkt).strftime('%a %H:%M')} HKT"
    )


def compare_today(*, refresh_backtest: bool = False) -> dict:
    journal = SignalJournal()
    day_start = trading_day_start_utc()
    journal.reconcile_open_trades(day_start)
    trades_view = journal.resolve_trades_view()
    day_label = trades_view["trading_day"]
    day_start = datetime.fromisoformat(trades_view["trading_day_start_utc"])
    if trades_view["source"] == "journal":
        live_trades = trades_view["trades"]
    else:
        live_trades = []
    live_summary = _summary_from_trades(live_trades)
    live_summary["trades"] = live_trades

    bt_summary, backtest_note = _backtest_summary_for_day(
        day_start, refresh=refresh_backtest
    )
    snap = _snap_path(day_start)

    both_empty = live_summary["trade_count"] == 0 and bt_summary["trade_count"] == 0
    if both_empty and backtest_note is None:
        backtest_note = "No trades yet today — live and backtest both flat."
    elif live_summary["trade_count"] > 0 and bt_summary["trade_count"] == 0:
        parity = (
            " v15 backtest had 0 entry signals for this window (live/backtest parity gap)."
        )
        if backtest_note is None or "Refresh BT" not in backtest_note:
            backtest_note = (
                backtest_note or "No backtest trades in this window."
            ) + " Tap Refresh BT to re-run." + parity
        elif "parity gap" not in (backtest_note or ""):
            backtest_note = backtest_note + parity

    result = {
        "trading_day": day_label,
        "trading_day_start_utc": day_start.isoformat(),
        "trading_day_window_hkt": _fmt_hkt_window(day_start),
        "is_fallback": trades_view["is_fallback"],
        "source": trades_view["source"],
        "both_empty": both_empty,
        "note": backtest_note,
        "backtest_snapshot": snap.name if snap.exists() else None,
        "live": live_summary,
        "backtest": bt_summary,
        "delta": {
            "trade_count": live_summary["trade_count"] - bt_summary["trade_count"],
            "net_pnl": round(live_summary["net_pnl"] - bt_summary["net_pnl"], 2),
            "win_rate": round(live_summary["win_rate"] - bt_summary["win_rate"], 1),
        },
    }
    journal.save_daily_compare(day_label, live_summary, bt_summary)
    return result
