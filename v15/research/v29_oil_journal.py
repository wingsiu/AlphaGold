"""SQLite journal for oil live signals and trades (separate tables from gold)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = PROJECT_ROOT / "runtime" / "mobile" / "alphagold.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class OilSignalJournal:
    """Oil-specific journal using oil_* table prefix to separate from gold bot data."""

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
                CREATE TABLE IF NOT EXISTS oil_bar_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bar_time TEXT UNIQUE NOT NULL,
                    features_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_oil_bar_features_bar ON oil_bar_features(bar_time);

                CREATE TABLE IF NOT EXISTS oil_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bar_time TEXT NOT NULL,
                    pattern_name TEXT,
                    pattern_side INTEGER,
                    pattern_prob REAL,
                    action TEXT NOT NULL DEFAULT 'score',
                    detail TEXT,
                    features_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_oil_signals_bar ON oil_signals(bar_time);

                CREATE TABLE IF NOT EXISTS oil_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    deal_id TEXT UNIQUE,
                    source TEXT,
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
                CREATE INDEX IF NOT EXISTS idx_oil_trades_entry ON oil_trades(entry_time);
                """
            )

    def record_bar_feature(self, bar_time: str, features_json: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO oil_bar_features (bar_time, features_json, created_at)
                   VALUES (?, ?, ?)""",
                (bar_time, features_json, _utc_now_iso()),
            )

    def record_score(
        self,
        bar_time: str,
        *,
        pattern_name: str | None = None,
        pattern_side: int = 0,
        pattern_prob: float | None = None,
        action: str = "score",
        detail: str | None = None,
        features_json: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO oil_signals
                   (bar_time, pattern_name, pattern_side, pattern_prob,
                    action, detail, features_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (bar_time, pattern_name, pattern_side or None, pattern_prob,
                 action, detail, features_json, _utc_now_iso()),
            )

    def open_trade(self, row: dict[str, Any]) -> None:
        now = _utc_now_iso()
        meta = {k: v for k, v in row.items() if k not in {
            "deal_id", "source", "side", "entry_time",
            "entry_price", "tp", "sl", "horizon", "probability",
        }}
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO oil_trades
                   (deal_id, source, side, entry_time, entry_price,
                    tp, sl, horizon, probability, status, meta_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
                   ON CONFLICT(deal_id) DO UPDATE SET
                       updated_at=excluded.updated_at,
                       meta_json=excluded.meta_json""",
                (row.get("deal_id"), row.get("source"), row.get("side"),
                 row.get("entry_time"), row.get("entry_price"),
                 row.get("tp"), row.get("sl"), row.get("horizon"),
                 row.get("probability"),
                 json.dumps(meta) if meta else None, now, now),
            )

    def close_trade(
        self, deal_id: str, *,
        exit_time: str | None = None, exit_price: float | None = None,
        pnl: float | None = None, exit_reason: str | None = None,
    ) -> None:
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """UPDATE oil_trades SET
                   exit_time=?, exit_price=?, pnl=?, exit_reason=?,
                   status='closed', updated_at=?
                   WHERE deal_id=?""",
                (exit_time, exit_price, pnl, exit_reason, now, deal_id),
            )

    def latest_bar_features(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM oil_bar_features ORDER BY bar_time DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
