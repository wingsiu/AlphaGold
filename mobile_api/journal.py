"""SQLite journal for live signals and trades (iOS / Watch API backend)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "runtime" / "mobile" / "alphagold.db"

NY_CUTOFF_HOUR = 17  # trading day rolls at 17:00 America/New_York


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def trading_day_start_utc(now: datetime | None = None) -> datetime:
    """NY 17:00 cutoff — same convention as the live bot."""
    import pandas as pd

    now = now or datetime.now(timezone.utc)
    ts = pd.Timestamp(now).tz_convert("America/New_York")
    day = (ts - pd.Timedelta(hours=NY_CUTOFF_HOUR)).floor("D")
    start_ny = day + pd.Timedelta(hours=NY_CUTOFF_HOUR)
    return start_ny.tz_convert("UTC").to_pydatetime()


class SignalJournal:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or DEFAULT_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bar_time TEXT NOT NULL,
                    pattern_name TEXT,
                    pattern_side INTEGER,
                    pattern_prob REAL,
                    energetic_side INTEGER,
                    energetic_prob REAL,
                    action TEXT NOT NULL DEFAULT 'score',
                    detail TEXT,
                    open_source TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_signals_bar ON signals(bar_time);
                CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    deal_id TEXT UNIQUE,
                    source TEXT,
                    pattern_name TEXT,
                    side INTEGER,
                    entry_time TEXT,
                    exit_time TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    exit_reason TEXT,
                    tp REAL,
                    sl REAL,
                    horizon INTEGER,
                    probability REAL,
                    status TEXT NOT NULL DEFAULT 'open',
                    meta_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades(entry_time);

                CREATE TABLE IF NOT EXISTS daily_compare (
                    trading_day TEXT PRIMARY KEY,
                    live_json TEXT NOT NULL,
                    backtest_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def record_score(
        self,
        bar_time: str,
        *,
        pattern_name: str | None = None,
        pattern_side: int = 0,
        pattern_prob: float | None = None,
        energetic_side: int = 0,
        energetic_prob: float | None = None,
        action: str = "score",
        detail: str | None = None,
        open_source: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO signals (
                    bar_time, pattern_name, pattern_side, pattern_prob,
                    energetic_side, energetic_prob, action, detail, open_source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bar_time,
                    pattern_name,
                    pattern_side or None,
                    pattern_prob,
                    energetic_side or None,
                    energetic_prob,
                    action,
                    detail,
                    open_source,
                    _utc_now_iso(),
                ),
            )

    def open_trade(self, row: dict[str, Any]) -> None:
        now = _utc_now_iso()
        meta = {k: v for k, v in row.items() if k not in {
            "deal_id", "source", "pattern_name", "side", "entry_time", "entry_price",
            "tp", "sl", "horizon", "probability",
        }}
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    deal_id, source, pattern_name, side, entry_time, entry_price,
                    tp, sl, horizon, probability, status, meta_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
                ON CONFLICT(deal_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    meta_json=excluded.meta_json
                """,
                (
                    row.get("deal_id"),
                    row.get("source"),
                    row.get("pattern_name"),
                    row.get("side"),
                    row.get("entry_time"),
                    row.get("entry_price"),
                    row.get("tp"),
                    row.get("sl"),
                    row.get("horizon"),
                    row.get("probability"),
                    json.dumps(meta) if meta else None,
                    now,
                    now,
                ),
            )

    def close_trade(
        self,
        deal_id: str,
        *,
        exit_time: str | None = None,
        exit_price: float | None = None,
        pnl: float | None = None,
        exit_reason: str | None = None,
    ) -> None:
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE trades SET
                    exit_time=?, exit_price=?, pnl=?, exit_reason=?,
                    status='closed', updated_at=?
                WHERE deal_id=?
                """,
                (exit_time, exit_price, pnl, exit_reason, now, deal_id),
            )

    def signals_since_minutes(self, minutes: int = 30) -> list[dict[str, Any]]:
        import pandas as pd

        cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM signals WHERE created_at >= ? ORDER BY created_at DESC
                """,
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def trades_today(self) -> list[dict[str, Any]]:
        start = trading_day_start_utc().isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trades
                WHERE entry_time >= ? OR (exit_time IS NOT NULL AND exit_time >= ?)
                ORDER BY COALESCE(entry_time, exit_time) DESC
                """,
                (start, start),
            ).fetchall()
        return [dict(r) for r in rows]

    def trades_summary(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        closed = [t for t in trades if t.get("status") == "closed"]
        pnls = [float(t["pnl"]) for t in closed if t.get("pnl") is not None]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "trade_count": len(trades),
            "closed_count": len(closed),
            "open_count": sum(1 for t in trades if t.get("status") == "open"),
            "net_pnl": round(sum(pnls), 2) if pnls else 0.0,
            "win_rate": round(100.0 * wins / len(pnls), 1) if pnls else 0.0,
        }

    def save_daily_compare(
        self, trading_day: str, live: dict[str, Any], backtest: dict[str, Any]
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO daily_compare (trading_day, live_json, backtest_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(trading_day) DO UPDATE SET
                    live_json=excluded.live_json,
                    backtest_json=excluded.backtest_json,
                    updated_at=excluded.updated_at
                """,
                (trading_day, json.dumps(live), json.dumps(backtest), _utc_now_iso()),
            )

    def load_daily_compare(self, trading_day: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM daily_compare WHERE trading_day=?", (trading_day,)
            ).fetchone()
        if not row:
            return None
        return {
            "trading_day": row["trading_day"],
            "live": json.loads(row["live_json"]),
            "backtest": json.loads(row["backtest_json"]),
            "updated_at": row["updated_at"],
        }

    def status_snapshot(self, hybrid_state: dict[str, Any] | None = None) -> dict[str, Any]:
        today_trades = self.trades_today()
        return {
            "server_time_utc": _utc_now_iso(),
            "trading_day_start_utc": trading_day_start_utc().isoformat(),
            "today": self.trades_summary(today_trades),
            "open_position": hybrid_state or {},
            "recent_signals_30m": len(self.signals_since_minutes(30)),
        }
