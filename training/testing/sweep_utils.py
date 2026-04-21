#!/usr/bin/env python3
"""
Shared helpers for all sweep scripts.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl

WEAK_MIN_TRADES   = 3
WEAK_MAX_WIN_RATE = 40.0


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt_sec(s: float) -> str:
    """Format seconds as '2m05s' or '47s'."""
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec:02d}s" if m else f"{sec}s"


def progress_bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total) if total else 0
    pct    = int(100 * done / total)   if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:3d}%"


def print_progress(i: int, total: int, sweep_start: float, durations: list[float]) -> None:
    """Print a progress line with ETA based on average run time so far."""
    now_str = datetime.now().strftime("%H:%M:%S")
    elapsed = __import__("time").time() - sweep_start
    if durations:
        avg_sec      = sum(durations) / len(durations)
        remaining    = avg_sec * (total - i + 1)
        eta_str      = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M:%S")
        bar          = progress_bar(i - 1, total)
        print(f"\n{bar} [{i}/{total}] {now_str}  elapsed={fmt_sec(elapsed)}  eta={eta_str}")
    else:
        print(f"\n[{i}/{total}] {now_str}  elapsed={fmt_sec(elapsed)}")


# ── Report parsing ─────────────────────────────────────────────────────────────

def parse_report(report_path: Path) -> dict:
    """
    Return a flat dict of core metrics from a report JSON.
    Includes long_up / short_down break-down where available.
    Returns {'error': str} on failure.
    """
    try:
        rpt      = json.loads(report_path.read_text(encoding="utf-8"))
        dp       = rpt.get("directional_pnl", rpt)
        a        = dp.get("all", dp)
        long_s   = dp.get("long_up",    {})
        short_s  = dp.get("short_down", {})
        return {
            "trades":             int(a.get("trades", 0)),
            "total_pnl":          round(float(a.get("total_pnl", 0.0)), 2),
            "avg_trade":          round(float(a.get("avg_trade", 0.0)), 4),
            "win_rate_pct":       round(float(a.get("win_rate_pct", 0.0)), 2),
            "profit_factor":      round(float(a.get("profit_factor") or 0.0), 3),
            "avg_trades_per_day": round(float(a.get("avg_trades_per_day", 0.0)), 2),
            "avg_day":            round(float(dp.get("avg_day") or 0.0), 2),
            "positive_days_pct":  round(float(dp.get("positive_days_pct") or 0.0), 2),
            "trade_max_drawdown": round(float(a.get("trade_max_drawdown", 0.0)), 2),
            "daily_max_drawdown": round(float(a.get("daily_max_drawdown", 0.0)), 2),
            "long_trades":        int(long_s.get("trades", 0)),
            "long_wr":            round(float(long_s.get("win_rate_pct", 0.0)), 1),
            "long_pnl":           round(float(long_s.get("total_pnl", 0.0)), 2),
            "short_trades":       int(short_s.get("trades", 0)),
            "short_wr":           round(float(short_s.get("win_rate_pct", 0.0)), 1),
            "short_pnl":          round(float(short_s.get("total_pnl", 0.0)), 2),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── Weak-filter generation ─────────────────────────────────────────────────────

def generate_weak_filter(trades_csv: Path, out_json: Path) -> list[dict]:
    """
    Rebuild directional PnL from trades CSV, identify weak session/day/hour cells,
    write them to *out_json*, and return the list of weak cell dicts.
    """
    pnl = rebuild_directional_pnl(trades_csv)
    try:
        session_heatmaps = pnl["all"]["time_distribution"]["session_heatmaps"]
    except KeyError:
        out_json.write_text(json.dumps({"weak_cells": []}, indent=2) + "\n", encoding="utf-8")
        return []

    weak_cells: list[dict] = []
    for session in ("hkt", "london", "ny"):
        day_map = session_heatmaps.get(session, {}).get("cell_stats", {})
        for day, hour_map in day_map.items():
            for hour, st in hour_map.items():
                if not st:
                    continue
                t         = int(st.get("trades", 0))
                pnl_total = float(st.get("total_pnl", 0.0))
                wr        = st.get("win_rate_pct")
                wr        = float(wr) if wr is not None else None
                if (t >= WEAK_MIN_TRADES
                        and pnl_total < 0.0
                        and wr is not None
                        and wr < WEAK_MAX_WIN_RATE):
                    weak_cells.append({"session": session, "day": str(day), "hour": hour})

    out_json.write_text(json.dumps({"weak_cells": weak_cells}, indent=2) + "\n", encoding="utf-8")
    return weak_cells


# ── Top-N printer ──────────────────────────────────────────────────────────────

def print_top(
    rows: list[dict],
    sort_key: str = "p2_total_pnl",
    param_keys: list[str] | None = None,
    n: int = 10,
) -> None:
    """
    Print a ranked table of the top *n* rows sorted by *sort_key* (descending).
    Falls back to 'total_pnl' if *sort_key* is not present in any row.
    """
    valid = [r for r in rows if not r.get("error")]
    if not valid:
        print("  (no valid rows to rank)")
        return

    # Determine the actual key to sort by
    actual_key = sort_key if any(sort_key in r for r in valid) else "total_pnl"
    ranked = sorted(valid, key=lambda r: float(r.get(actual_key, 0)), reverse=True)

    print(f"\n=== Top {min(n, len(ranked))} by {actual_key} ===")
    header_parts = [f"{'pnl':>9}", f"{'avg':>8}", f"{'wr%':>6}", f"{'pf':>5}",
                    f"{'trades':>7}", f"{'ntpd':>5}", f"{'dd_tr':>7}", f"{'dd_dy':>7}"]
    if param_keys:
        params_hdr = "  " + " ".join(f"{k}" for k in param_keys)
    else:
        params_hdr = ""
    print("  ".join(header_parts) + params_hdr)
    print("-" * (60 + len(params_hdr)))

    for r in ranked[:n]:
        pnl    = float(r.get(actual_key, r.get("total_pnl", 0)))
        avg    = float(r.get("p2_avg_trade", r.get("avg_trade", 0)))
        wr     = float(r.get("p2_win_rate_pct", r.get("win_rate_pct", 0)))
        pf     = float(r.get("p2_profit_factor", r.get("profit_factor", 0)))
        trades = int(r.get("p2_trades", r.get("trades", 0)))
        ntpd   = float(r.get("p2_avg_trades_per_day", r.get("avg_trades_per_day", 0)))
        dd_tr  = float(r.get("p2_trade_max_drawdown", r.get("trade_max_drawdown", 0)))
        dd_dy  = float(r.get("p2_daily_max_drawdown", r.get("daily_max_drawdown", 0)))
        param_vals = ("  " + "  ".join(f"{k}={r.get(k,'?')}" for k in param_keys)) if param_keys else ""
        print(f"{pnl:>9.0f}  {avg:>8.4f}  {wr:>6.1f}  {pf:>5.3f}  {trades:>7}  {ntpd:>5.1f}"
              f"  {dd_tr:>7.0f}  {dd_dy:>7.0f}{param_vals}")

