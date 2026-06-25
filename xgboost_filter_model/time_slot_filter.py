"""Time-slot filters — same session heatmaps + weak-cell rules as v10 (sweep_utils)."""
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from training.rebuild_directional_pnl_from_trades import rebuild_directional_pnl

HK_TZ = ZoneInfo("Asia/Hong_Kong")
LONDON_TZ = ZoneInfo("Europe/London")
NY_TZ = ZoneInfo("America/New_York")

# Match training/image_trend_ml_regime.py SESSION_PERIOD_SPECS (v10 bot)
SESSION_PERIOD_SPECS: dict[str, dict[str, object]] = {
    "hkt": {
        "timezone": HK_TZ,
        "start_hour": 8,
        "start_minute": 0,
        "end_hour": 16,
        "end_minute": 0,
    },
    "london": {
        "timezone": LONDON_TZ,
        "start_hour": 8,
        "start_minute": 0,
        "end_hour": 16,
        "end_minute": 30,
    },
    "ny": {
        "timezone": NY_TZ,
        "start_hour": 9,
        "start_minute": 30,
        "end_hour": 16,
        "end_minute": 0,
    },
}

WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]

# v10 defaults (training/testing/sweep_utils.py)
DEFAULT_MIN_TRADES = 3
DEFAULT_MAX_WIN_RATE = 40.0


def session_hour_label(session_key: str, hour: int) -> str:
    if session_key == "ny" and int(hour) == 9:
        return "09:30"
    return f"{int(hour):02d}:00"


def find_bad_slots(
    trades_csv: Path | str,
    *,
    min_trades: int = DEFAULT_MIN_TRADES,
    min_trades_exclusive: bool = True,
    max_total_pnl: float = 0.0,
    max_win_rate: float = DEFAULT_MAX_WIN_RATE,
    require_low_win_rate: bool = False,
) -> list[dict]:
    """
    Weak-filter rule on session heatmaps (hkt / london / ny):

    Default (exclusive): trades > min_trades AND total_pnl < max_total_pnl
    v10 legacy (exclusive=False): trades >= min_trades AND pnl<0 AND wr<40%
    """
    stats = rebuild_directional_pnl(trades_csv)
    heatmaps = stats["all"]["time_distribution"]["session_heatmaps"]
    cells: list[dict] = []
    for session in ("hkt", "london", "ny"):
        day_map = heatmaps.get(session, {}).get("cell_stats", {})
        for day, hour_map in day_map.items():
            for hour, st in hour_map.items():
                if not st:
                    continue
                n = int(st.get("trades", 0))
                pnl_total = float(st.get("total_pnl", 0.0))
                wr_raw = st.get("win_rate_pct")
                wr = float(wr_raw) if wr_raw is not None else None
                if min_trades_exclusive:
                    if n <= min_trades or pnl_total >= max_total_pnl:
                        continue
                elif n < min_trades or pnl_total >= max_total_pnl:
                    continue
                if require_low_win_rate and (wr is None or wr >= max_win_rate):
                    continue
                cells.append(
                    {
                        "session": session,
                        "day": str(day),
                        "hour": str(hour),
                        "_trades": n,
                        "_total_pnl": round(pnl_total, 2),
                        "_win_rate_pct": wr if wr is not None else 0.0,
                    }
                )
    cells.sort(
        key=lambda c: (
            c["session"],
            WEEKDAY_ORDER.index(c["day"]) if c["day"] in WEEKDAY_ORDER else 99,
            c["hour"],
        )
    )
    return cells


def save_weak_filter(cells: list[dict], path: Path | str) -> None:
    payload = {
        "weak_cells": [
            {"session": c["session"], "day": c["day"], "hour": c["hour"]} for c in cells
        ]
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_weak_filter(path: Path | str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("weak_cells", payload if isinstance(payload, list) else [])
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        session = str(item.get("session", "")).strip().lower()
        day = str(item.get("day", "")).strip()
        hour = str(item.get("hour", "")).strip()
        if session not in SESSION_PERIOD_SPECS or not day or not hour:
            continue
        key = (session, day, hour)
        if key in seen:
            continue
        seen.add(key)
        out.append({"session": session, "day": day, "hour": hour})
    return out


def matches_weak_cell(ts: pd.Timestamp, cell: dict) -> bool:
    ts_utc = pd.Timestamp(ts)
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    else:
        ts_utc = ts_utc.tz_convert("UTC")

    session = cell["session"]
    spec = SESSION_PERIOD_SPECS[session]
    local_ts = ts_utc.tz_convert(spec["timezone"])
    minute_of_day = local_ts.hour * 60 + local_ts.minute
    start_min = int(spec["start_hour"]) * 60 + int(spec["start_minute"])
    end_min = int(spec["end_hour"]) * 60 + int(spec["end_minute"])
    if not (start_min <= minute_of_day < end_min):
        return False
    return (
        local_ts.day_name() == cell["day"]
        and session_hour_label(session, int(local_ts.hour)) == cell["hour"]
    )


def is_blocked_entry(ts: pd.Timestamp, weak_cells: list[dict] | None) -> bool:
    if not weak_cells:
        return False
    return any(matches_weak_cell(ts, cell) for cell in weak_cells)


def resolve_weak_time_filter_path(project_root: Path | str | None = None) -> str | None:
    """Resolve hybrid weak-slot JSON path from env / config."""
    import os

    from config.hybrid_config import TIME_FILTER_CONFIG

    no_filter = os.environ.get("AG_NO_TIME_FILTER", "") or os.environ.get(
        "V14_NO_TIME_FILTER", ""
    )
    if no_filter.strip().lower() in ("1", "true", "yes", "on"):
        return None

    env_path = (
        os.environ.get("AG_TIME_FILTER_JSON", "").strip()
        or os.environ.get("V14_TIME_FILTER_JSON", "").strip()
    )
    if env_path:
        return env_path

    if not TIME_FILTER_CONFIG.get("enabled"):
        return None

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    default = root / TIME_FILTER_CONFIG.get(
        "weak_slots_json", "runtime/hybrid_weak_time_slots.json"
    )
    if default.exists():
        return str(default)
    legacy = root / "runtime" / "v14_weak_time_slots.json"
    return str(legacy) if legacy.exists() else str(default)


def resolve_v14_time_filter_path(project_root: Path | str | None = None) -> str | None:
    """Deprecated alias — use resolve_weak_time_filter_path."""
    return resolve_weak_time_filter_path(project_root)


def print_blocked_cells(cells: list[dict]) -> None:
    for c in cells:
        print(
            f"  {c['session']:6s} {c['day']:9s} {c['hour']:5s}  "
            f"trades={c['_trades']}  pnl={c['_total_pnl']:+.1f}  wr={c['_win_rate_pct']:.0f}%"
        )
