"""Compare live journal trades vs hybrid backtest for the current trading day."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mobile_api.journal import SignalJournal, trading_day_start_utc

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _trading_day_label(now: datetime | None = None) -> str:
    start = trading_day_start_utc(now)
    ts = pd.Timestamp(start).tz_convert("America/New_York")
    return ts.strftime("%Y-%m-%d")


def _summary_from_trades(trades: list[dict]) -> dict:
    return SignalJournal().trades_summary(trades)


def _summary_from_backtest_csv(csv_path: Path, day_start: datetime) -> dict:
    if not csv_path.exists():
        return {"trade_count": 0, "closed_count": 0, "net_pnl": 0.0, "win_rate": 0.0, "trades": []}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"trade_count": 0, "closed_count": 0, "net_pnl": 0.0, "win_rate": 0.0, "trades": []}
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    day_end = day_start + pd.Timedelta(days=1)
    start_ts = pd.Timestamp(day_start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(day_end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    day_df = df[(df["entry_time"] >= start_ts) & (df["entry_time"] < end_ts)].copy()
    pnls = day_df["pnl"].astype(float).tolist() if "pnl" in day_df.columns else []
    wins = sum(1 for p in pnls if p > 0)
    rows = []
    for _, r in day_df.iterrows():
        rows.append(
            {
                "side": int(r.get("side", 0)),
                "source": r.get("source"),
                "pattern": r.get("pattern") if "pattern" in day_df.columns else r.get("pattern_name"),
                "entry_time": r["entry_time"].isoformat(),
                "exit_time": pd.Timestamp(r["exit_time"]).isoformat() if pd.notna(r.get("exit_time")) else None,
                "pnl": float(r["pnl"]) if pd.notna(r.get("pnl")) else None,
                "exit_reason": r.get("exit_reason"),
            }
        )
    return {
        "trade_count": len(day_df),
        "closed_count": len(day_df),
        "net_pnl": round(float(sum(pnls)), 2) if pnls else 0.0,
        "win_rate": round(100.0 * wins / len(pnls), 1) if pnls else 0.0,
        "trades": rows,
    }


def run_today_backtest(*, refresh: bool = False) -> Path:
    """Run hybrid backtest for current trading day → filtered CSV snapshot."""
    out_dir = PROJECT_ROOT / "runtime" / "mobile"
    out_dir.mkdir(parents=True, exist_ok=True)
    day_label = _trading_day_label()
    snap = out_dir / f"backtest_today_{day_label}.csv"
    if snap.exists() and not refresh:
        age_h = (datetime.now(timezone.utc).timestamp() - snap.stat().st_mtime) / 3600.0
        if age_h < 1.0:
            return snap

    start = trading_day_start_utc()
    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_str = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    env = os.environ.copy()
    env["V14_HYBRID"] = "1"
    env.setdefault("V14_FVG_MIN_GAP", "0")

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run_hybrid_backtest.py"), start_str, end_str],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    src = PROJECT_ROOT / "runtime" / "v14_pattern_backtest_trades.csv"
    if src.exists():
        snap.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return snap


def compare_today(*, refresh_backtest: bool = False) -> dict:
    journal = SignalJournal()
    day_label = _trading_day_label()
    day_start = trading_day_start_utc()
    live_trades = journal.trades_today()
    live_summary = _summary_from_trades(live_trades)
    live_summary["trades"] = live_trades

    snap = run_today_backtest(refresh=refresh_backtest)
    bt_summary = _summary_from_backtest_csv(snap, day_start)

    result = {
        "trading_day": day_label,
        "trading_day_start_utc": day_start.isoformat(),
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
