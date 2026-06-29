"""SQLite journal for live signals and trades (iOS / Watch API backend)."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "runtime" / "mobile" / "alphagold.db"
CLOSE_BRIDGE_PATH = PROJECT_ROOT / "runtime" / "live_trade_closes.json"

TRADING_DAY_CUTOFF_UTC_HOUR = 22  # trading day rolls at 22:00 UTC (= 06:00 HKT)
IG_RECONCILE_MIN_INTERVAL_SEC = 120.0
_IG_RECONCILE_LAST_TS = 0.0

# Journal closes with these reasons must not be shown as confirmed live PnL.
UNRELIABLE_EXIT_REASONS = frozenset({
    "estimated_ohlc",
    "broker_not_open",
    "reconciled",
    "reconciled_bt",
    "pnl_pending",
    "stale_sync",
})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def trading_day_start_utc(now: datetime | None = None) -> datetime:
    """22:00 UTC cutoff (= 06:00 HKT)."""
    import pandas as pd

    ts = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC")
    day = (ts - pd.Timedelta(hours=TRADING_DAY_CUTOFF_UTC_HOUR)).floor("D")
    start = day + pd.Timedelta(hours=TRADING_DAY_CUTOFF_UTC_HOUR)
    return start.to_pydatetime()


def trading_day_label(day_start: datetime) -> str:
    """Trading day name in HKT (session opens 06:00 HKT)."""
    import pandas as pd

    return pd.Timestamp(day_start).tz_convert("Asia/Hong_Kong").strftime("%Y-%m-%d")


def trading_day_for_timestamp(ts: datetime | str) -> datetime:
    import pandas as pd

    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    day = (stamp - pd.Timedelta(hours=TRADING_DAY_CUTOFF_UTC_HOUR)).floor("D")
    start = day + pd.Timedelta(hours=TRADING_DAY_CUTOFF_UTC_HOUR)
    return start.to_pydatetime()


def _backtest_csv_path() -> Path:
    return PROJECT_ROOT / "runtime" / "gold_v16_hybrid_backtest_trades.csv"


def _load_backtest_df():
    import pandas as pd

    path = _backtest_csv_path()
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "entry_time" not in df.columns:
        return None
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["_day_start"] = df["entry_time"].map(
        lambda ts: trading_day_for_timestamp(ts.to_pydatetime())
    )
    return df


def _backtest_row_to_trade(row, idx: int) -> dict[str, Any]:
    import pandas as pd

    pattern = row.get("pattern") if hasattr(row, "get") else None
    if pattern is None and "matched_pattern" in row.index:
        pattern = row.get("matched_pattern")
    entry_time = row["entry_time"]
    exit_time = row.get("exit_time")
    entry_ts = pd.Timestamp(entry_time) if pd.notna(entry_time) else None
    signal_ts = (entry_ts - pd.Timedelta(minutes=1)) if entry_ts is not None else None
    raw_pnl = float(row["pnl"]) if pd.notna(row.get("pnl")) else None
    # Scale backtest PnL to match live bot size (ENERGETIC_EXECUTION_CONFIG.size)
    try:
        from config.hybrid_config import ENERGETIC_EXECUTION_CONFIG
        bt_size = float(ENERGETIC_EXECUTION_CONFIG.get("size", 3.0))
    except Exception:
        bt_size = 3.0
    return {
        "id": -(idx + 1),
        "deal_id": f"bt-{idx}",
        "source": row.get("source"),
        "pattern_name": pattern,
        "side": int(row.get("side", 0) or 0),
        "signal_time": signal_ts.isoformat() if signal_ts is not None else None,
        "entry_time": entry_ts.isoformat() if entry_ts is not None else None,
        "exit_time": pd.Timestamp(exit_time).isoformat() if pd.notna(exit_time) else None,
        "entry_price": float(row["entry_price"]) if pd.notna(row.get("entry_price")) else None,
        "exit_price": float(row["exit_price"]) if pd.notna(row.get("exit_price")) else None,
        "pnl": round(raw_pnl * bt_size, 2) if raw_pnl is not None else None,
        "exit_reason": row.get("exit_reason"),
        "status": "closed",
    }


def _backtest_trades_for_day(day_start: datetime) -> list[dict[str, Any]]:
    import pandas as pd

    df = _load_backtest_df()
    if df is None:
        return []
    day_end = pd.Timestamp(day_start) + pd.Timedelta(days=1)
    start_ts = pd.Timestamp(day_start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(day_end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    day_df = df[(df["entry_time"] >= start_ts) & (df["entry_time"] < end_ts)].sort_values(
        "entry_time", ascending=False
    )
    return [_backtest_row_to_trade(row, i) for i, (_, row) in enumerate(day_df.iterrows())]


def latest_backtest_trading_day_start() -> datetime | None:
    df = _load_backtest_df()
    if df is None or df.empty:
        return None
    latest_entry = df["entry_time"].max()
    return trading_day_for_timestamp(latest_entry.to_pydatetime())


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
                    features_json TEXT,
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
                    backtest_entry_price REAL,
                    real_entry_price REAL,
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
            self._migrate_signals_table()

    def _migrate_signals_table(self) -> None:
        with self._conn() as conn:
            existing = {row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()}
            if "bar_close" not in existing:
                conn.execute("ALTER TABLE signals ADD COLUMN bar_close REAL")

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
        features_json: str | None = None,
        bar_close: float | None = None,
    ) -> None:
        self._migrate_signals_table()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO signals (
                    bar_time, pattern_name, pattern_side, pattern_prob,
                    energetic_side, energetic_prob, action, detail, open_source,
                    features_json, bar_close, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    features_json,
                    bar_close,
                    _utc_now_iso(),
                ),
            )

    def record_bar_feature(self, bar_time: str, features_json: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO bar_features (bar_time, features_json, created_at) VALUES (?, ?, ?)",
                (bar_time, features_json, _utc_now_iso()),
            )

    def _migrate_trades_table(self) -> None:
        """Add new columns if missing from old schema."""
        migrations = {
            "backtest_entry_price": "ALTER TABLE trades ADD COLUMN backtest_entry_price REAL",
            "real_entry_price": "ALTER TABLE trades ADD COLUMN real_entry_price REAL",
        }
        with self._conn() as conn:
            existing = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
            for col, sql in migrations.items():
                if col not in existing:
                    conn.execute(sql)

    def open_trade(self, row: dict[str, Any]) -> None:
        self._migrate_trades_table()
        now = _utc_now_iso()
        meta = {k: v for k, v in row.items() if k not in {
            "deal_id", "source", "pattern_name", "side", "signal_time", "entry_time",
            "entry_price", "tp", "sl", "horizon", "probability",
        }}
        signal_time = row.get("signal_time")
        entry_time = row.get("entry_time")
        if signal_time and not entry_time:
            import pandas as pd

            entry_time = (pd.Timestamp(signal_time) + pd.Timedelta(minutes=1)).isoformat()
        if signal_time:
            meta["signal_time"] = signal_time
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    deal_id, source, pattern_name, side, entry_time, entry_price,
                    backtest_entry_price, real_entry_price,
                    tp, sl, horizon, probability, status, meta_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
                ON CONFLICT(deal_id) DO UPDATE SET
                    entry_price=excluded.entry_price,
                    backtest_entry_price=excluded.backtest_entry_price,
                    real_entry_price=excluded.real_entry_price,
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
                    row.get("backtest_entry_price"),
                    row.get("real_entry_price"),
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

    def reopen_trade(self, deal_id: str) -> None:
        """Undo a false journal close when the broker still has the position open."""
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE trades SET
                    exit_time=NULL, exit_price=NULL, pnl=NULL, exit_reason=NULL,
                    status='open', updated_at=?
                WHERE deal_id=?
                """,
                (now, deal_id),
            )

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
        return str(row.get("exit_reason") or "") not in UNRELIABLE_EXIT_REASONS

    def _enrich_trade_row(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        if out.get("meta_json"):
            try:
                meta = json.loads(out["meta_json"])
                if meta.get("signal_time") and not out.get("signal_time"):
                    out["signal_time"] = meta["signal_time"]
                if meta.get("horizon_deadline") and not out.get("horizon_deadline"):
                    out["horizon_deadline"] = meta["horizon_deadline"]
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
        # Add gold price to recent signal rows
        if out.get("entry_time"):
            try:
                from pathlib import Path
                import json as _json
                bp = Path(__file__).resolve().parent.parent / "runtime" / "live_price.json"
                if bp.exists():
                    price = _json.loads(bp.read_text(encoding="utf-8"))
                    out["gold_price"] = price.get("close")
            except Exception:
                pass
        return out

    def _trades_since_day_start(self, day_start: datetime) -> list[dict[str, Any]]:
        start = day_start.isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trades
                WHERE entry_time >= ? OR (exit_time IS NOT NULL AND exit_time >= ?)
                ORDER BY COALESCE(entry_time, exit_time) DESC
                """,
                (start, start),
            ).fetchall()
        return [self._enrich_trade_row(dict(r)) for r in rows]

    def _trade_by_deal_id(self, deal_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE deal_id = ?", (deal_id,)
            ).fetchone()
        return self._enrich_trade_row(dict(row)) if row else None

    def _latest_trade_timestamp(self) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT MAX(COALESCE(entry_time, exit_time)) AS latest
                FROM trades
                """
            ).fetchone()
        return row["latest"] if row and row["latest"] else None

    def _latest_signal_timestamp(self) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(created_at) AS latest FROM signals").fetchone()
        return row["latest"] if row and row["latest"] else None

    def resolve_trades_view(
        self, *, hybrid_state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        today_start = trading_day_start_utc()
        state = hybrid_state or {}
        self.reconcile_with_bot_state(today_start, state)
        self.reconcile_from_close_bridge(today_start)

        allow_ig = self._ig_reconcile_allowed()
        if allow_ig:
            broker_open, broker_ok = self._broker_open_snapshot()
        else:
            broker_open, broker_ok = self._open_deals_from_bot_state(state), True

        self.reconcile_unconfirmed_closes(
            today_start, broker_open, broker_ok, allow_ig=allow_ig
        )
        self.reconcile_open_trades(today_start, broker_open, broker_ok, allow_ig=allow_ig)
        if allow_ig:
            self.reconcile_broker_pnl(today_start)
            self.reconcile_missing_pnl(today_start, broker_open, broker_ok)
        meta = {
            "is_fallback": False,
            "source": "journal",
            "trading_day": trading_day_label(today_start),
            "trading_day_start_utc": today_start.isoformat(),
        }

        trades = self._trades_since_day_start(today_start)
        if trades:
            trades, summary = self._align_trades_with_hybrid_state(
                trades, hybrid_state, self.trades_summary(trades)
            )
            return {**meta, "trades": trades, "summary": summary}

        latest_ts = self._latest_trade_timestamp()
        if latest_ts:
            day_start = trading_day_for_timestamp(latest_ts)
            trades = self._trades_since_day_start(day_start)
            if trades:
                trades, summary = self._align_trades_with_hybrid_state(
                    trades, hybrid_state, self.trades_summary(trades)
                )
                return {
                    **meta,
                    "is_fallback": True,
                    "source": "journal",
                    "trading_day": trading_day_label(day_start),
                    "trading_day_start_utc": day_start.isoformat(),
                    "trades": trades,
                    "summary": summary,
                }

        day_start = latest_backtest_trading_day_start()
        if day_start:
            trades = _backtest_trades_for_day(day_start)
            if trades:
                return {
                    **meta,
                    "is_fallback": True,
                    "source": "backtest",
                    "trading_day": trading_day_label(day_start),
                    "trading_day_start_utc": day_start.isoformat(),
                    "trades": trades,
                    "summary": self.trades_summary(trades),
                }

        return {**meta, "trades": [], "summary": self.trades_summary([])}

    def resolve_signals_view(
        self, minutes: int = 30, *, fallback_limit: int = 100
    ) -> dict[str, Any]:
        import pandas as pd

        meta = {
            "is_fallback": False,
            "source": "journal",
            "minutes": minutes,
            "requested_minutes": minutes,
        }
        grid = self._build_minute_grid(minutes)
        has_live = any(r.get("action") != "no_data" for r in grid)
        if has_live:
            return {**meta, "count": len(grid), "signals": grid}

        # No rows in the requested window — backfill minutes from latest journal rows.
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM signals
                ORDER BY bar_time DESC, created_at DESC
                LIMIT ?
                """,
                (fallback_limit,),
            ).fetchall()
        fallback_rows = [dict(r) for r in rows]
        if fallback_rows:
            grid = self._build_minute_grid(
                minutes, seed_rows=fallback_rows, anchor_on_seed=True
            )
            return {
                **meta,
                "is_fallback": True,
                "source": "journal",
                "minutes": minutes,
                "count": len(grid),
                "signals": grid,
            }

        return {**meta, "count": len(grid), "signals": grid}

    def _minute_key(self, ts: "pd.Timestamp") -> str:
        import pandas as pd

        t = pd.Timestamp(ts).tz_convert("UTC").floor("min")
        return t.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    def _bar_minute_key(self, bar_time: str) -> "pd.Timestamp":
        import pandas as pd

        ts = pd.Timestamp(bar_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.floor("min")

    def _signals_in_bar_range(
        self, start: "pd.Timestamp", end: "pd.Timestamp"
    ) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM signals
                WHERE bar_time >= ? AND bar_time <= ?
                ORDER BY bar_time ASC, created_at ASC
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return [dict(r) for r in rows]

    def _group_signals_by_minute(
        self, rows: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = self._minute_key(self._bar_minute_key(str(row["bar_time"])))
            grouped.setdefault(key, []).append(row)
        return grouped

    def _merge_minute_rows(
        self, minute_ts: "pd.Timestamp", rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        import zlib

        bar_time = self._minute_key(minute_ts)
        if not rows:
            sid = -(zlib.crc32(bar_time.encode()) % 1_000_000_000)
            result = {
                "id": sid,
                "bar_time": bar_time,
                "pattern_name": None,
                "pattern_side": 0,
                "pattern_prob": None,
                "energetic_side": 0,
                "energetic_prob": None,
                "action": "no_data",
                "detail": None,
                "open_source": None,
                "created_at": bar_time,
                "bar_close": None,
            }
            self._enrich_signal_row(result)
            if self._bar_close_from_features(bar_time) is not None:
                result["action"] = "no_score"
            return result

        score_row = next((r for r in rows if r.get("action") == "score"), None)
        pred_row = score_row or rows[-1]
        action_row = next(
            (
                r
                for r in reversed(rows)
                if r.get("action") not in (None, "score", "no_data")
            ),
            None,
        )
        action = action_row["action"] if action_row else "score"
        merged = dict(pred_row)
        merged["action"] = action
        merged["bar_time"] = bar_time
        if action_row and action_row is not pred_row:
            if action_row.get("detail"):
                merged["detail"] = action_row["detail"]
            if action_row.get("open_source"):
                merged["open_source"] = action_row["open_source"]
        self._enrich_signal_row(merged)
        return merged

    def _bar_close_from_features(self, bar_time: str | None) -> float | None:
        if not bar_time:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT features_json FROM bar_features WHERE bar_time = ?",
                (bar_time,),
            ).fetchone()
        if not row or not row[0]:
            return None
        try:
            feat = json.loads(row[0])
            for key in ("close", "closePrice_ask", "close_ask"):
                val = feat.get(key)
                if val is not None:
                    return float(val)
        except Exception:
            return None
        return None

    def _enrich_signal_row(self, row: dict[str, Any]) -> None:
        """Add per-bar close (gold_price) to a signal row (mutates in place)."""
        price = row.get("bar_close")
        if price is None:
            price = self._bar_close_from_features(row.get("bar_time"))
        if price is None:
            try:
                from pathlib import Path
                import json as _json
                bp = Path(__file__).resolve().parent.parent / "runtime" / "live_price.json"
                if bp.exists():
                    live = _json.loads(bp.read_text(encoding="utf-8"))
                    price = live.get("close")
            except Exception:
                pass
        if price is not None:
            row["bar_close"] = float(price)
            row["gold_price"] = float(price)

    def _build_minute_grid(
        self,
        minutes: int,
        *,
        seed_rows: list[dict[str, Any]] | None = None,
        anchor_on_seed: bool = False,
    ) -> list[dict[str, Any]]:
        import pandas as pd

        now = pd.Timestamp.now(tz="UTC").floor("min")
        end = now - pd.Timedelta(minutes=1)
        if anchor_on_seed and seed_rows:
            end = max(self._bar_minute_key(str(r["bar_time"])) for r in seed_rows)
        start = end - pd.Timedelta(minutes=max(minutes - 1, 0))

        if seed_rows:
            grouped = self._group_signals_by_minute(seed_rows)
        else:
            rows = self._signals_in_bar_range(start, end)
            grouped = self._group_signals_by_minute(rows)

        grid: list[dict[str, Any]] = []
        probe = end
        while probe >= start:
            key = self._minute_key(probe)
            grid.append(self._merge_minute_rows(probe, grouped.get(key, [])))
            probe -= pd.Timedelta(minutes=1)
        return grid

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
        return self.resolve_trades_view()["trades"]

    def trades_summary(
        self,
        trades: list[dict[str, Any]],
        *,
        unrealized_pnl: float | None = None,
    ) -> dict[str, Any]:
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
        open_unrealized = unrealized_pnl
        if open_unrealized is None:
            open_unrealized = sum(
                float(t["pnl"]) for t in open_rows if t.get("pnl") is not None
            )
        open_unrealized = round(float(open_unrealized or 0.0), 2)
        net_pnl = round(closed_pnl + open_unrealized, 2)
        return {
            "trade_count": len(trades),
            "closed_count": len(closed),
            "open_count": len(open_rows),
            "closed_pnl": closed_pnl,
            "unrealized_pnl": open_unrealized,
            "net_pnl": net_pnl,
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

    def _estimate_pnl_from_ohlc(self, trade: dict[str, Any]) -> dict[str, Any] | None:
        """Approximate closed PnL from DB 1m bars when IG history is not available yet."""
        import pandas as pd

        entry_price = self._effective_entry_price(trade)
        entry_time = trade.get("entry_time")
        exit_time = trade.get("exit_time")
        signal_time = trade.get("signal_time")
        side = int(trade.get("side") or 0)
        if entry_price is None or side == 0 or (not entry_time and not signal_time):
            return None
        try:
            from config.hybrid_config import ENERGETIC_EXECUTION_CONFIG
            from xgboost_filter_model.train_filter import load_price_data

            size = float(ENERGETIC_EXECUTION_CONFIG.get("size", 2.0))
        except Exception:
            size = 2.0

        if signal_time:
            entry_ts = pd.Timestamp(signal_time)
        elif entry_time:
            entry_ts = pd.Timestamp(entry_time) - pd.Timedelta(minutes=1)
        else:
            entry_ts = pd.Timestamp(entry_time)
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        else:
            entry_ts = entry_ts.tz_convert("UTC")

        meta: dict[str, Any] = {}
        if trade.get("meta_json"):
            try:
                meta = json.loads(trade["meta_json"])
            except Exception:
                meta = {}
        if meta.get("horizon_deadline"):
            horizon_exit = pd.Timestamp(meta["horizon_deadline"])
            if horizon_exit.tzinfo is None:
                horizon_exit = horizon_exit.tz_localize("UTC")
            else:
                horizon_exit = horizon_exit.tz_convert("UTC")
        else:
            horizon_min = int(trade.get("horizon") or 15)
            pad_min = 6 if str(trade.get("source") or "") == "pattern" else 1
            horizon_exit = entry_ts + pd.Timedelta(minutes=horizon_min + pad_min)
        exit_reason = str(trade.get("exit_reason") or "")
        use_horizon_bar = exit_reason in (
            "broker_not_open",
            "stale_sync",
            "reconciled",
            "estimated_ohlc",
        ) or not exit_time
        if use_horizon_bar:
            exit_ts = horizon_exit
        else:
            exit_ts = pd.Timestamp(exit_time)
            if exit_ts.tzinfo is None:
                exit_ts = exit_ts.tz_localize("UTC")
            else:
                exit_ts = exit_ts.tz_convert("UTC")
            if exit_ts > horizon_exit + pd.Timedelta(minutes=5):
                exit_ts = horizon_exit

        day = entry_ts.strftime("%Y-%m-%d")
        try:
            ohlc = load_price_data(day, (entry_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        except Exception:
            return None
        if ohlc.empty:
            return None

        idx = ohlc.index.get_indexer([exit_ts], method="nearest")[0]
        exit_price = float(ohlc.iloc[idx]["close"])
        ep = float(entry_price)
        pnl = (ep - exit_price) * size if side < 0 else (exit_price - ep) * size
        return {
            "pnl": round(pnl, 2),
            "exit_price": exit_price,
            "exit_time": pd.Timestamp(ohlc.index[idx]).isoformat(),
            "reason": "estimated_ohlc",
        }

    @staticmethod
    def _ig_reconcile_allowed() -> bool:
        from ig_scripts.ig_request_gate import ig_second_slot_open

        if not ig_second_slot_open("mobile_api"):
            return False
        global _IG_RECONCILE_LAST_TS
        now = time.time()
        if now - _IG_RECONCILE_LAST_TS < IG_RECONCILE_MIN_INTERVAL_SEC:
            return False
        _IG_RECONCILE_LAST_TS = now
        return True

    @staticmethod
    def _open_deals_from_bot_state(state: dict[str, Any] | None) -> set[str]:
        deal_id = str((state or {}).get("open_deal_id") or "").strip()
        return {deal_id} if deal_id else set()

    def reconcile_with_bot_state(
        self, day_start: datetime | None, hybrid_state: dict[str, Any] | None
    ) -> int:
        """Align journal open/closed rows with the live bot state file (no IG calls)."""
        state = hybrid_state or {}
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

    def reconcile_from_close_bridge(self, day_start: datetime | None = None) -> int:
        """Apply close events written by the live bot (zero IG calls)."""
        if not CLOSE_BRIDGE_PATH.exists():
            return 0
        day_start = day_start or trading_day_start_utc()
        try:
            rows = json.loads(CLOSE_BRIDGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return 0
        if not isinstance(rows, list):
            return 0
        updated = 0
        day_deals = {
            str(t.get("deal_id") or "")
            for t in self._trades_since_day_start(day_start)
        }
        for row in rows:
            if not isinstance(row, dict):
                continue
            deal_id = str(row.get("deal_id") or "").strip()
            if not deal_id or deal_id not in day_deals:
                continue
            trade = next(
                (
                    t
                    for t in self._trades_since_day_start(day_start)
                    if str(t.get("deal_id") or "") == deal_id
                ),
                None,
            )
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

    def reconcile_missing_pnl(
        self,
        day_start: datetime | None = None,
        broker_open: set[str] | None = None,
        broker_ok: bool | None = None,
    ) -> int:
        """Fill null PnL on closed trades from IG only (no OHLC estimates)."""
        day_start = day_start or trading_day_start_utc()
        if broker_open is None or broker_ok is None:
            broker_open, broker_ok = self._broker_open_snapshot()
        updated = 0
        for trade in self._trades_since_day_start(day_start):
            if trade.get("status") != "closed":
                continue
            deal_id = str(trade.get("deal_id") or "")
            if not deal_id:
                continue
            if broker_ok and deal_id in broker_open:
                continue
            if trade.get("pnl") is not None and self.is_pnl_confirmed(trade):
                continue
            closed = self._fetch_ig_closed_trade(deal_id) if deal_id else None
            if closed and closed.get("pnl") is not None:
                self.close_trade(
                    deal_id,
                    exit_time=closed.get("exit_time") or trade.get("exit_time"),
                    exit_price=closed.get("exit_price") or trade.get("exit_price"),
                    pnl=round(float(closed["pnl"]), 2),
                    exit_reason=str(
                        closed.get("reason") or trade.get("exit_reason") or "broker_close"
                    ),
                )
                updated += 1
        return updated

    def reconcile_broker_pnl(self, day_start: datetime | None = None) -> int:
        """Refresh closed journal PnL from IG (broker is source of truth for live)."""
        day_start = day_start or trading_day_start_utc()
        updated = 0
        for trade in self._trades_since_day_start(day_start):
            if trade.get("status") != "closed" or not trade.get("deal_id"):
                continue
            closed = self._fetch_ig_closed_trade(str(trade["deal_id"]))
            if not closed or closed.get("pnl") is None:
                continue
            broker_pnl = round(float(closed["pnl"]), 2)
            journal_pnl = trade.get("pnl")
            if journal_pnl is not None and abs(float(journal_pnl) - broker_pnl) < 0.01:
                continue
            self.close_trade(
                str(trade["deal_id"]),
                exit_time=closed.get("exit_time") or trade.get("exit_time"),
                exit_price=closed.get("exit_price") or trade.get("exit_price"),
                pnl=broker_pnl,
                exit_reason=str(closed.get("reason") or trade.get("exit_reason") or "broker_close"),
            )
            updated += 1
        return updated

    def _broker_open_snapshot(self) -> tuple[set[str], bool]:
        """Return (open deal ids, fetch_succeeded). Empty set + False when IG is unavailable."""
        try:
            from ig_scripts.ig_data_api import fetch_open_positions
            from mobile_api.ig_client import create_ig_service

            positions = fetch_open_positions(create_ig_service())
            return {
                str(p.get("position", {}).get("dealId") or "").strip()
                for p in positions
                if p.get("position", {}).get("dealId")
            }, True
        except Exception:
            return set(), False

    def _broker_open_deal_ids(self) -> set[str]:
        open_ids, _ok = self._broker_open_snapshot()
        return open_ids

    def reconcile_unconfirmed_closes(
        self,
        day_start: datetime | None,
        broker_open: set[str],
        broker_ok: bool,
        *,
        allow_ig: bool = True,
    ) -> int:
        """Re-open false closes and strip unconfirmed estimated PnL."""
        day_start = day_start or trading_day_start_utc()
        updated = 0
        for trade in self._trades_since_day_start(day_start):
            deal_id = str(trade.get("deal_id") or "")
            if not deal_id:
                continue
            reason = str(trade.get("exit_reason") or "")
            status = trade.get("status")

            if status == "closed" and broker_ok and deal_id in broker_open:
                if reason in UNRELIABLE_EXIT_REASONS or trade.get("pnl") is None:
                    self.reopen_trade(deal_id)
                    updated += 1
                continue

            if status != "closed" or reason not in UNRELIABLE_EXIT_REASONS:
                continue

            closed = self._fetch_ig_closed_trade(deal_id) if allow_ig else None
            if closed and closed.get("pnl") is not None:
                self.close_trade(
                    deal_id,
                    exit_time=closed.get("exit_time") or trade.get("exit_time"),
                    exit_price=closed.get("exit_price") or trade.get("exit_price"),
                    pnl=round(float(closed["pnl"]), 2),
                    exit_reason=str(closed.get("reason") or "broker_close"),
                )
                updated += 1
                continue

            if trade.get("pnl") is not None or reason == "estimated_ohlc":
                self.close_trade(
                    deal_id,
                    exit_time=None,
                    exit_price=None,
                    pnl=None,
                    exit_reason="pnl_pending",
                )
                updated += 1
        return updated

    def reconcile_open_trades(
        self,
        day_start: datetime | None = None,
        broker_open: set[str] | None = None,
        broker_ok: bool | None = None,
        *,
        allow_ig: bool = True,
    ) -> int:
        """Close journal rows still marked open when broker/backtest shows them done."""
        import pandas as pd

        day_start = day_start or trading_day_start_utc()
        if broker_open is None or broker_ok is None:
            if allow_ig:
                broker_open, broker_ok = self._broker_open_snapshot()
            else:
                broker_open, broker_ok = set(), False
        if not broker_ok or not allow_ig:
            return 0

        open_trades = [
            t
            for t in self._trades_since_day_start(day_start)
            if t.get("status") == "open" and t.get("deal_id")
        ]
        if not open_trades:
            return 0

        broker_open = self._broker_open_deal_ids()
        bt_df = _load_backtest_df()
        closed_n = 0
        for trade in open_trades:
            deal_id = str(trade["deal_id"])
            pnl: float | None = None
            exit_time: str | None = None
            exit_price: float | None = None
            exit_reason = "reconciled"

            closed = self._fetch_ig_closed_trade(deal_id)
            if closed:
                if closed.get("pnl") is not None:
                    pnl = float(closed["pnl"])
                exit_time = closed.get("exit_time") or exit_time
                exit_price = closed.get("exit_price") or exit_price
                exit_reason = str(closed.get("reason") or "broker_close")

            if pnl is None and bt_df is not None and trade.get("entry_time"):
                entry = pd.Timestamp(trade["entry_time"])
                if entry.tzinfo is None:
                    entry = entry.tz_localize("UTC")
                else:
                    entry = entry.tz_convert("UTC")
                window = (bt_df["entry_time"] >= entry - pd.Timedelta(minutes=3)) & (
                    bt_df["entry_time"] <= entry + pd.Timedelta(minutes=3)
                )
                if trade.get("side") is not None:
                    window &= bt_df["side"] == int(trade["side"])
                if trade.get("pattern_name"):
                    pat_col = (
                        "pattern"
                        if "pattern" in bt_df.columns
                        else "matched_pattern"
                    )
                    if pat_col in bt_df.columns:
                        window &= bt_df[pat_col] == trade["pattern_name"]
                matches = bt_df.loc[window]
                if len(matches) == 1:
                    row = matches.iloc[0]
                    if pd.notna(row.get("pnl")):
                        pnl = float(row["pnl"])
                        exit_time = (
                            pd.Timestamp(row["exit_time"]).isoformat()
                            if pd.notna(row.get("exit_time"))
                            else None
                        )
                        exit_price = (
                            float(row["exit_price"])
                            if pd.notna(row.get("exit_price"))
                            else None
                        )
                        exit_reason = str(row.get("exit_reason") or "reconciled_bt")

            if pnl is None and deal_id not in broker_open:
                exit_time = exit_time or _utc_now_iso()
                exit_reason = exit_reason if closed else "pnl_pending"

            if pnl is None and deal_id in broker_open:
                continue

            if pnl is None and not closed:
                self.close_trade(
                    deal_id,
                    exit_time=exit_time,
                    exit_price=exit_price,
                    pnl=None,
                    exit_reason=exit_reason,
                )
                closed_n += 1
                continue

            self.close_trade(
                deal_id,
                exit_time=exit_time,
                exit_price=exit_price,
                pnl=pnl,
                exit_reason=exit_reason,
            )
            closed_n += 1
        return closed_n

    def _align_trades_with_hybrid_state(
        self,
        trades: list[dict[str, Any]],
        hybrid_state: dict[str, Any] | None,
        summary: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        state = hybrid_state or {}
        open_deal = str(state.get("open_deal_id") or "").strip()
        if not open_deal:
            return trades, summary
        closed_first = state.get("closed_first_seen_at")
        if closed_first:
            try:
                age = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(str(closed_first))
                ).total_seconds()
                if age >= 120:
                    return trades, summary
            except Exception:
                pass
        patched: list[dict[str, Any]] = []
        changed = False
        for row in trades:
            if row.get("deal_id") == open_deal and row.get("status") == "closed":
                row = dict(row)
                row["status"] = "open"
                row["exit_time"] = None
                row["exit_price"] = None
                row["pnl"] = None
                row["exit_reason"] = None
                row["pnl_confirmed"] = False
                changed = True
            patched.append(self._enrich_trade_row(row))
        if not changed:
            return trades, summary
        return patched, self.trades_summary(patched)

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
        from mobile_api.ig_account import get_ig_account_summary
        from mobile_api.market_price import get_gold_price_summary

        today_start = trading_day_start_utc()
        state = hybrid_state or {}
        trades_view = self.resolve_trades_view(hybrid_state=state)
        signals_view = self.resolve_signals_view(30)
        return {
            "server_time_utc": _utc_now_iso(),
            "trading_day_start_utc": trades_view["trading_day_start_utc"],
            "trading_day": trades_view["trading_day"],
            "is_fallback": trades_view["is_fallback"],
            "source": trades_view["source"],
            "today": trades_view["summary"],
            "open_position": state,
            "ig_account": get_ig_account_summary(),
            "gold": get_gold_price_summary(),
            "recent_signals_30m": signals_view["count"] if not signals_view["is_fallback"] else 0,
            "recent_signals_display": signals_view["count"],
            "signals_is_fallback": signals_view["is_fallback"],
        }
