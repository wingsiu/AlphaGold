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
CLOSE_BRIDGE_PATH = PROJECT_ROOT / "runtime" / "live_oil_trade_closes.json"

from mobile_api.journal import (  # noqa: E402
    UNRELIABLE_EXIT_REASONS,
    trading_day_for_timestamp,
    trading_day_label,
    trading_day_start_utc,
)


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
        self._migrate_trades_table()
        meta = {k: v for k, v in row.items() if k not in {
            "deal_id", "source", "side", "entry_time", "entry_price",
            "backtest_entry_price", "real_entry_price",
            "tp", "sl", "horizon", "probability",
        }}
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO oil_trades
                   (deal_id, source, side, entry_time, entry_price,
                    tp, sl, horizon, probability, status, meta_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
                   ON CONFLICT(deal_id) DO UPDATE SET
                       entry_price=excluded.entry_price,
                       updated_at=excluded.updated_at,
                       meta_json=excluded.meta_json""",
                (row.get("deal_id"), row.get("source"), row.get("side"),
                 row.get("entry_time"), row.get("entry_price"),
                 row.get("tp"), row.get("sl"), row.get("horizon"),
                 row.get("probability"),
                 json.dumps(meta) if meta else None, now, now),
            )

    def _migrate_trades_table(self) -> None:
        migrations = {
            "backtest_entry_price": "ALTER TABLE oil_trades ADD COLUMN backtest_entry_price REAL",
            "real_entry_price": "ALTER TABLE oil_trades ADD COLUMN real_entry_price REAL",
        }
        with self._conn() as conn:
            existing = {row[1] for row in conn.execute("PRAGMA table_info(oil_trades)").fetchall()}
            for col, sql in migrations.items():
                if col not in existing:
                    conn.execute(sql)

    def reopen_trade(self, deal_id: str) -> None:
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """UPDATE oil_trades SET
                   exit_time=NULL, exit_price=NULL, pnl=NULL, exit_reason=NULL,
                   status='open', updated_at=?
                   WHERE deal_id=?""",
                (now, deal_id),
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

    @staticmethod
    def _effective_entry_price(row: dict[str, Any]) -> float | None:
        rep = row.get("real_entry_price")
        if rep is not None:
            return float(rep)
        ep = row.get("entry_price")
        return float(ep) if ep is not None else None

    @staticmethod
    def is_pnl_confirmed(row: dict[str, Any]) -> bool:
        if row.get("status") == "open":
            return False
        if row.get("pnl") is None:
            return False
        reason = str(row.get("exit_reason") or "")
        if reason in UNRELIABLE_EXIT_REASONS:
            return False
        if reason.startswith("Position/s closed"):
            return True
        return reason not in UNRELIABLE_EXIT_REASONS

    def _enrich_trade_row(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        if out.get("meta_json"):
            try:
                meta = json.loads(out["meta_json"])
                if meta.get("real_entry_price") is not None and out.get("real_entry_price") is None:
                    out["real_entry_price"] = meta["real_entry_price"]
                if meta.get("backtest_entry_price") is not None and out.get("backtest_entry_price") is None:
                    out["backtest_entry_price"] = meta["backtest_entry_price"]
            except Exception:
                pass
        out["display_entry_price"] = self._effective_entry_price(out)
        out["pnl_confirmed"] = self.is_pnl_confirmed(out)
        if out.get("status") == "open":
            out["exit_time"] = None
            out["exit_price"] = None
            out["pnl"] = None
            out["exit_reason"] = None
        elif not out["pnl_confirmed"]:
            out["pnl"] = None
            if str(out.get("exit_reason") or "") in UNRELIABLE_EXIT_REASONS:
                out["exit_time"] = None
                out["exit_price"] = None
        return out

    def _trades_since_day_start(self, day_start: datetime) -> list[dict[str, Any]]:
        start = day_start.isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM oil_trades
                WHERE entry_time >= ? OR (exit_time IS NOT NULL AND exit_time >= ?)
                ORDER BY COALESCE(entry_time, exit_time) DESC
                """,
                (start, start),
            ).fetchall()
        return [self._enrich_trade_row(dict(r)) for r in rows]

    def _trade_by_deal_id(self, deal_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM oil_trades WHERE deal_id = ?", (deal_id,)
            ).fetchone()
        return self._enrich_trade_row(dict(row)) if row else None

    def trades_summary(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        closed = [t for t in trades if t.get("status") == "closed"]
        open_rows = [t for t in trades if t.get("status") == "open"]
        confirmed_closed = [t for t in closed if self.is_pnl_confirmed(t)]
        pnls = [float(t["pnl"]) for t in confirmed_closed if t.get("pnl") is not None]
        wins = sum(1 for p in pnls if p > 0)
        closed_pnl = round(sum(pnls), 2) if pnls else 0.0
        pending_pnl_count = sum(
            1
            for t in closed
            if t.get("pnl") is None
            or str(t.get("exit_reason") or "") in UNRELIABLE_EXIT_REASONS
        )
        return {
            "trade_count": len(trades),
            "closed_count": len(closed),
            "open_count": len(open_rows),
            "closed_pnl": closed_pnl,
            "unrealized_pnl": 0.0,
            "net_pnl": closed_pnl,
            "win_rate": round(100.0 * wins / len(pnls), 1) if pnls else 0.0,
            "pending_pnl_count": pending_pnl_count,
        }

    def _fetch_ig_closed_trade(self, deal_id: str) -> dict[str, Any] | None:
        try:
            from ig_scripts.ig_data_api import get_closed_trade_by_deal_id
            from mobile_api.ig_client import create_ig_service

            return get_closed_trade_by_deal_id(
                create_ig_service(), deal_id, lookback_hours=168
            )
        except Exception:
            return None

    def reconcile_from_close_bridge(self, day_start: datetime | None = None) -> int:
        if not CLOSE_BRIDGE_PATH.exists():
            return 0
        day_start = day_start or trading_day_start_utc()
        try:
            rows = json.loads(CLOSE_BRIDGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return 0
        if not isinstance(rows, list):
            return 0
        day_deals = {
            str(t.get("deal_id") or "")
            for t in self._trades_since_day_start(day_start)
        }
        updated = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            deal_id = str(row.get("deal_id") or "").strip()
            if not deal_id or deal_id not in day_deals:
                continue
            trade = self._trade_by_deal_id(deal_id)
            if trade and self.is_pnl_confirmed(trade):
                continue
            pnl = row.get("pnl")
            self.close_trade(
                deal_id,
                exit_time=row.get("exit_time"),
                exit_price=row.get("exit_price"),
                pnl=round(float(pnl), 2) if pnl is not None else None,
                exit_reason=str(row.get("exit_reason") or row.get("reason") or "bot_close"),
            )
            updated += 1
        return updated

    def reconcile_with_bot_state(
        self, day_start: datetime | None, bot_state: dict[str, Any] | None
    ) -> int:
        state = bot_state or {}
        open_deal = str(state.get("open_deal_id") or "").strip()
        suspect_broker_closed = False
        closed_first = state.get("closed_first_seen_at")
        if open_deal and closed_first:
            try:
                age = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(str(closed_first))
                ).total_seconds()
            except Exception:
                age = 0.0
            if age >= 120:
                suspect_broker_closed = True
                row = self._trade_by_deal_id(open_deal)
                if row and row.get("status") == "open":
                    self.close_trade(
                        open_deal,
                        exit_time=None,
                        exit_price=None,
                        pnl=None,
                        exit_reason="pnl_pending",
                    )

        day_start = day_start or trading_day_start_utc()
        updated = 0
        for trade in self._trades_since_day_start(day_start):
            deal_id = str(trade.get("deal_id") or "")
            if not deal_id:
                continue
            if (
                open_deal
                and not suspect_broker_closed
                and deal_id == open_deal
                and trade.get("status") == "closed"
            ):
                self.reopen_trade(deal_id)
                updated += 1
            elif open_deal and trade.get("status") == "open" and deal_id != open_deal:
                self.close_trade(
                    deal_id,
                    exit_time=None,
                    exit_price=None,
                    pnl=None,
                    exit_reason="pnl_pending",
                )
                updated += 1
        return updated

    def reconcile_broker_pnl(self, day_start: datetime | None = None) -> int:
        day_start = day_start or trading_day_start_utc()
        updated = 0
        for trade in self._trades_since_day_start(day_start):
            if trade.get("status") != "closed" or not trade.get("deal_id"):
                continue
            if self.is_pnl_confirmed(trade):
                continue
            closed = self._fetch_ig_closed_trade(str(trade["deal_id"]))
            if not closed or closed.get("pnl") is None:
                continue
            self.close_trade(
                str(trade["deal_id"]),
                exit_time=closed.get("exit_time") or trade.get("exit_time"),
                exit_price=closed.get("exit_price") or trade.get("exit_price"),
                pnl=round(float(closed["pnl"]), 2),
                exit_reason=str(closed.get("reason") or trade.get("exit_reason") or "broker_close"),
            )
            updated += 1
        return updated

    def resolve_trades_view(
        self, *, bot_state: dict[str, Any] | None = None, allow_ig: bool = True
    ) -> dict[str, Any]:
        day_start = trading_day_start_utc()
        state = bot_state or {}
        self.reconcile_with_bot_state(day_start, state)
        self.reconcile_from_close_bridge(day_start)
        if allow_ig:
            self.reconcile_broker_pnl(day_start)
        trades = self._trades_since_day_start(day_start)
        return {
            "trading_day": trading_day_label(day_start),
            "trading_day_start_utc": day_start.isoformat(),
            "trades": trades,
            "summary": self.trades_summary(trades),
        }
