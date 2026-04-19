#!/usr/bin/env python3
"""Run top candidates with their own auto-generated weak filter (2-pass).

Pass 1: run without weak filter → get trades
Pass 2: identify weak cells from pass-1 trades → run with new weak filter

Candidates:
  A: s1=0.48 s2=0.58 ladv=12 ltgt=0.008 mwr=30  (best total PnL)
  B: s1=0.55 s2=0.58 ladv=15 ltgt=0.008 mwr=50  (best avg trade / quality)

Usage:
    python3 training/testing/run_candidates.py
    python3 training/testing/run_candidates.py --candidate A
    python3 training/testing/run_candidates.py --candidate B
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Import rebuild function so we can generate session_heatmaps from trades CSV
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_TREND_ML = PROJECT_ROOT / "training" / "image_trend_ml.py"
MODEL_IN       = PROJECT_ROOT / "runtime" / "bot_assets" / "backtest_model_best_base_weak_nostate.joblib"
RESULTS_DIR    = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Candidate definitions ─────────────────────────────────────────────────────
CANDIDATES: dict[str, dict] = {
    "A": {
        "label": "Best Total PnL",
        "stage1_min_prob":       0.48,
        "stage2_min_prob":       0.58,
        "long_adverse_limit":    12,
        "long_target_threshold": 0.008,
        "min_window_range":      30,
    },
    "B": {
        "label": "Best Quality (avg trade)",
        "stage1_min_prob":       0.55,
        "stage2_min_prob":       0.58,
        "long_adverse_limit":    15,
        "long_target_threshold": 0.008,
        "min_window_range":      50,
    },
    "C": {
        "label": "Custom: s1=0.55 s2=0.58 ladv=15 ltgt=0.008 mwr=30",
        "stage1_min_prob":       0.55,
        "stage2_min_prob":       0.58,
        "long_adverse_limit":    15,
        "long_target_threshold": 0.008,
        "min_window_range":      30,
    },
    "D": {
        "label": "Best s2-directional: s1=0.55 s2up=0.65 s2down=0.62 ladv=15 ltgt=0.008 mwr=30",
        "stage1_min_prob":        0.55,
        "stage2_min_prob":        0.58,    # fallback; overridden below
        "stage2_min_prob_up":     0.65,
        "stage2_min_prob_down":   0.62,
        "long_adverse_limit":     15,
        "long_target_threshold":  0.008,
        "min_window_range":       30,
    },
}

# Fixed params shared by all candidates
FIXED = {
    "short_adverse_limit":    18,
    "short_target_threshold": 0.008,
    "adverse_limit":          15,
    "trend_threshold":        0.008,
    "window":                 150,
    "horizon":                25,
    "min_15m_drop":           15,
    "max_flat_ratio":         2.5,
    "classifier":             "gradient_boosting",
}

# Weak-filter generation thresholds — same as backtest_no_retrain.py
WEAK_MIN_TRADES   = 3     # minimum trades in cell
WEAK_MAX_WIN_RATE = 40.0  # filter if win_rate < this AND total_pnl < 0

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _build_cmd(p: dict, out_base: Path, weak_filter: Path | None) -> list[str]:
    cmd = [
        sys.executable, str(IMAGE_TREND_ML),
        "--start-date", "2025-05-20",
        "--end-date", "2026-04-10",
        "--test-start-date", "2025-11-25T17:02:00+00:00",
        "--timeframe", "1min",
        "--eval-mode", "single_split",
        "--disable-time-filter",
        "--window", "150",
        "--window-15m", "0",
        "--min-window-range",       str(p["min_window_range"]),
        "--min-15m-drop",           str(FIXED["min_15m_drop"]),
        "--min-15m-rise",           "0",
        "--horizon",                str(FIXED["horizon"]),
        "--trend-threshold",        str(FIXED["trend_threshold"]),
        "--adverse-limit",          str(FIXED["adverse_limit"]),
        "--long-target-threshold",  str(p["long_target_threshold"]),
        "--short-target-threshold", str(FIXED["short_target_threshold"]),
        "--long-adverse-limit",     str(p["long_adverse_limit"]),
        "--short-adverse-limit",    str(FIXED["short_adverse_limit"]),
        "--classifier",             FIXED["classifier"],
        "--max-flat-ratio",         str(FIXED["max_flat_ratio"]),
        "--stage1-min-prob",        str(p["stage1_min_prob"]),
        "--stage2-min-prob",        str(p["stage2_min_prob"]),
        "--model-in",  str(MODEL_IN),
        "--model-out", str(out_base) + "_model.joblib",
        "--report-out", str(out_base) + "_report.json",
        "--trades-out", str(out_base) + "_trades.csv",
    ]
    # Optional per-direction stage2 thresholds (Candidate D)
    if p.get("stage2_min_prob_up") is not None:
        cmd += ["--stage2-min-prob-up", str(p["stage2_min_prob_up"])]
    if p.get("stage2_min_prob_down") is not None:
        cmd += ["--stage2-min-prob-down", str(p["stage2_min_prob_down"])]
    if weak_filter:
        cmd += ["--weak-periods-json", str(weak_filter)]
    return cmd


def _run(label: str, cmd: list[str]) -> int:
    print(f"\n  Running: {label}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False).returncode
    print(f"  Done in {int(time.time()-t0)}s  (rc={rc})")
    return rc


def _parse_report(report_path: Path) -> dict:
    try:
        r = json.loads(report_path.read_text())
        dp = r.get("directional_pnl", r)
        a  = dp.get("all", dp)
        return {
            "trades":             int(a.get("trades", 0)),
            "total_pnl":          round(float(a.get("total_pnl", 0)), 2),
            "avg_trade":          round(float(a.get("avg_trade", 0)), 4),
            "win_rate_pct":       round(float(a.get("win_rate_pct", 0)), 2),
            "profit_factor":      round(float(a.get("profit_factor") or 0), 3),
            "avg_trades_per_day": round(float(a.get("avg_trades_per_day", 0)), 2),
            "avg_day":            round(float(dp.get("avg_day") or 0), 2),
            "positive_days_pct":  round(float(dp.get("positive_days_pct") or 0), 2),
            "trade_max_drawdown": round(float(a.get("trade_max_drawdown", 0)), 2),
            "daily_max_drawdown": round(float(a.get("daily_max_drawdown", 0)), 2),
        }
    except Exception as e:
        return {"error": str(e)}


def _generate_weak_filter(trades_csv: Path, out_json: Path) -> list[dict]:
    """Extract weak cells using same rule as backtest_no_retrain.py:
       total_pnl < 0  AND  win_rate < 40%  AND  trades >= 3
       across all sessions: hkt, london, ny
       Uses rebuild_directional_pnl to get session_heatmaps from trades CSV.
    """
    pnl = rebuild_directional_pnl(trades_csv)
    try:
        session_heatmaps = pnl["all"]["time_distribution"]["session_heatmaps"]
    except KeyError:
        print("  WARNING: session_heatmaps not found — no weak filter generated")
        out_json.write_text(json.dumps({"weak_cells": []}, indent=2))
        return []

    weak_cells: list[dict] = []
    for session in ("hkt", "london", "ny"):
        day_map = session_heatmaps.get(session, {}).get("cell_stats", {})
        for day, hour_map in day_map.items():
            for hour, st in hour_map.items():
                if not st:
                    continue
                t = int(st.get("trades", 0))
                pnl_total = float(st.get("total_pnl", 0.0))
                wr = st.get("win_rate_pct", None)
                wr = float(wr) if wr is not None else None
                if t >= WEAK_MIN_TRADES and pnl_total < 0.0 and wr is not None and wr < WEAK_MAX_WIN_RATE:
                    weak_cells.append({
                        "session": session, "day": str(day), "hour": hour,
                        "_trades": t, "_wr": round(wr, 1), "_pnl": round(pnl_total, 2),
                    })

    weak_cells.sort(key=lambda c: (
        c["session"],
        DAY_ORDER.index(c["day"]) if c["day"] in DAY_ORDER else 99,
        c["hour"],
    ))
    payload = {"weak_cells": [{"session": c["session"], "day": c["day"], "hour": c["hour"]} for c in weak_cells]}
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\n  Generated weak filter: {len(weak_cells)} cells → {out_json.name}")
    for c in weak_cells:
        print(f"    {c['session']}:{c['day']}:{c['hour']}  wr={c['_wr']}%  total_pnl={c['_pnl']}  trades={c['_trades']}")
    return weak_cells


def _print_result(label: str, m: dict) -> None:
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    if "error" in m:
        print(f"  ERROR: {m['error']}")
        return
    print(f"  trades={m['trades']}  total_pnl={m['total_pnl']:.2f}  avg={m['avg_trade']:.4f}")
    print(f"  win_rate={m['win_rate_pct']:.1f}%  pf={m['profit_factor']:.3f}  tpd={m['avg_trades_per_day']:.1f}")
    print(f"  avg_day={m['avg_day']:.2f}  pos_days={m['positive_days_pct']:.1f}%  dd={m['trade_max_drawdown']:.2f}")


def run_candidate(cid: str) -> None:
    p = CANDIDATES[cid]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = RESULTS_DIR / f"cand{cid}_{ts}"

    print(f"\n{'='*65}")
    print(f"CANDIDATE {cid}: {p['label']}")
    print(f"  s1={p['stage1_min_prob']}  s2={p['stage2_min_prob']}  ladv={p['long_adverse_limit']}  ltgt={p['long_target_threshold']}  mwr={p['min_window_range']}")
    print(f"{'='*65}")

    # ── Pass 1: no weak filter ────────────────────────────────────────────────
    base1 = Path(str(prefix) + "_pass1")
    cmd1  = _build_cmd(p, base1, weak_filter=None)
    rc1   = _run("Pass 1 — no weak filter", cmd1)
    m1    = _parse_report(Path(str(base1) + "_report.json")) if rc1 == 0 else {"error": f"rc={rc1}"}
    _print_result("Pass 1 (no filter)", m1)

    if rc1 != 0:
        print("  Pass 1 failed — skipping pass 2")
        return

    # ── Generate per-candidate weak filter ───────────────────────────────────
    trades1   = Path(str(base1) + "_trades.csv")
    weak_json = Path(str(prefix) + "_weak_filter.json")
    cells = _generate_weak_filter(trades1, weak_json)

    # ── Pass 2: with generated weak filter ───────────────────────────────────
    base2 = Path(str(prefix) + "_pass2")
    cmd2  = _build_cmd(p, base2, weak_filter=weak_json if cells else None)
    rc2   = _run("Pass 2 — with generated weak filter", cmd2)
    m2    = _parse_report(Path(str(base2) + "_report.json")) if rc2 == 0 else {"error": f"rc={rc2}"}
    _print_result("Pass 2 (with filter)", m2)

    # ── Compare ───────────────────────────────────────────────────────────────
    if "error" not in m1 and "error" not in m2:
        print(f"\n  COMPARISON — Candidate {cid}")
        print(f"  {'':20s} {'No filter':>12} {'With filter':>12} {'Delta':>10}")
        print(f"  {'-'*58}")
        for k, fmt in [("total_pnl",":.2f"), ("avg_trade",":.4f"), ("win_rate_pct",":.1f"), ("profit_factor",":.3f"),
                       ("trades",":d"), ("avg_day",":.2f"), ("positive_days_pct",":.1f"), ("trade_max_drawdown",":.2f")]:
            v1 = m1[k]; v2 = m2[k]
            delta = v2 - v1 if isinstance(v1, float) else v2 - v1
            sign = "+" if delta > 0 else ""
            if k == "trades":
                print(f"  {k:20s} {int(v1):>12d} {int(v2):>12d} {sign+str(int(delta)):>10}")
            else:
                print(f"  {k:20s} {v1:>12{fmt[1:]}} {v2:>12{fmt[1:]}} {sign+format(delta, fmt[1:]):>10}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=["A", "B", "C", "D"], default=None, help="Run only this candidate (default: all)")
    args = ap.parse_args()

    to_run = [args.candidate] if args.candidate else ["A", "B", "C", "D"]
    for cid in to_run:
        run_candidate(cid)

    print(f"\n{'='*65}")
    print("All done. Check training/testing/results/ for output files.")


if __name__ == "__main__":
    main()

