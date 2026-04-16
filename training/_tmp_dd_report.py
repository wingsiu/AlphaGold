from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

RUNS = {
    "BASE_C": Path("/Users/alpha/Desktop/python/AlphaGold/training/backtest_trades_wf_state_sweep_C.csv"),
    "R0": Path("/Users/alpha/Desktop/python/AlphaGold/training/_wfC_r0_flat250_400_u58_68.csv"),
    "REFINED": Path("/Users/alpha/Desktop/python/AlphaGold/training/_wfC_r0_flat200_300_s148_55_u58_62_d62_68.csv"),
    "NARROW": Path("/Users/alpha/Desktop/python/AlphaGold/training/_wfC_r0_narrow_flat180_250_s146_50_u54_60_d62_66.csv"),
}


def parse_ts(row: dict[str, str]) -> datetime:
    for key in ("exit_time", "entry_time", "ts", "ts_event"):
        val = row.get(key)
        if val:
            return datetime.fromisoformat(val)
    raise ValueError("no timestamp column found")


def parse_pnl(row: dict[str, str]) -> float:
    for key in ("pnl_usd", "pnl", "pnl_points", "points"):
        val = row.get(key)
        if val not in (None, ""):
            return float(val)
    raise ValueError("no pnl column found")


def dd_stats(path: Path) -> dict[str, object]:
    rows: list[tuple[datetime, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append((parse_ts(row), parse_pnl(row)))
    rows.sort(key=lambda x: x[0])

    equity = 0.0
    peak = 0.0
    peak_time: datetime | None = None
    best_dd = 0.0
    trough_time: datetime | None = None
    peak_before_dd_time: datetime | None = None
    dd_peak_equity = 0.0

    for ts, pnl in rows:
        equity += pnl
        if equity > peak:
            peak = equity
            peak_time = ts
        dd = equity - peak
        if dd < best_dd:
            best_dd = dd
            trough_time = ts
            peak_before_dd_time = peak_time
            dd_peak_equity = peak

    recovery_time: datetime | None = None
    if trough_time is not None and peak_before_dd_time is not None:
        equity = 0.0
        for ts, pnl in rows:
            equity += pnl
            if ts > trough_time and equity >= dd_peak_equity - 1e-12:
                recovery_time = ts
                break

    return {
        "trades": len(rows),
        "max_drawdown": best_dd,
        "peak_before_drawdown_time": None if peak_before_dd_time is None else peak_before_dd_time.isoformat(),
        "max_drawdown_time": None if trough_time is None else trough_time.isoformat(),
        "recovery_time": None if recovery_time is None else recovery_time.isoformat(),
        "recovered": recovery_time is not None,
        "recovery_minutes": None if recovery_time is None or trough_time is None else (recovery_time - trough_time).total_seconds() / 60.0,
    }


def main() -> None:
    out_lines: list[str] = []
    for name, path in RUNS.items():
        stats = dd_stats(path)
        out_lines.append(f"[{name}]")
        for k, v in stats.items():
            out_lines.append(f"{k}={v}")
        out_lines.append("")
    out_path = Path("/Users/alpha/Desktop/python/AlphaGold/training/_drawdown_timestamps_summary.txt")
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

