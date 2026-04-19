#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import mysql.connector

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ig_scripts.ig_data_api import API_CONFIG, DB_CONFIG, IGService, Price, fetch_prices, insert_prices


def _pick_instrument(table: str) -> Price:
    mapping = {
        "gold_prices": Price.Gold,
        "aud_prices": Price.AUD,
        "prices": Price.Oil,
    }
    if table not in mapping:
        raise ValueError(f"Unsupported table for IG mapping: {table}")
    return mapping[table]


def _utc_ts(text: str) -> datetime:
    ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _count_rows_from(table: str, start_ms: int) -> int:
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE timestamp >= %s", (start_ms,))
    count = int(cur.fetchone()[0])
    conn.close()
    return count


def _minmax_from(table: str, start_ms: int) -> dict[str, int | None]:
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table} WHERE timestamp >= %s", (start_ms,))
    row = cur.fetchone()
    conn.close()
    return {"min_ts": None if row[0] is None else int(row[0]), "max_ts": None if row[1] is None else int(row[1])}


def _delete_from(table: str, start_ms: int) -> int:
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {table} WHERE timestamp >= %s", (start_ms,))
    affected = int(cur.rowcount)
    conn.commit()
    conn.close()
    return affected


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test IG fetch and optionally refill MySQL from a cutoff timestamp.")
    p.add_argument("--table", default="gold_prices", choices=["gold_prices", "aud_prices", "prices"])
    p.add_argument("--start-utc", default="2026-04-09T00:00:00+00:00")
    p.add_argument("--end-utc", default=None, help="Default is now-1 minute UTC")
    p.add_argument("--execute", action="store_true", help="Actually delete and refill MySQL.")
    p.add_argument("--report-out", default="runtime/_refill_mysql_from_ig_report.json")
    return p


def main() -> int:
    args = build_parser().parse_args()
    instrument = _pick_instrument(args.table)

    start_utc = _utc_ts(args.start_utc).astimezone(timezone.utc)
    if args.end_utc:
        end_utc = _utc_ts(args.end_utc).astimezone(timezone.utc)
    else:
        end_utc = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)

    if end_utc < start_utc:
        raise SystemExit("end_utc must be >= start_utc")

    start_ms = int(start_utc.timestamp() * 1000)

    service = IGService(
        API_CONFIG["api_key"],
        API_CONFIG["username"],
        API_CONFIG["password"],
        API_CONFIG["base_url"],
    )

    ig_rows = fetch_prices(service, instrument, start_time=start_utc, end_time=end_utc)
    if not ig_rows:
        raise SystemExit("IG fetch returned 0 rows; aborting.")

    unique_ts = sorted({int(r["timestamp"]) for r in ig_rows if r.get("timestamp") is not None})
    ig_first = datetime.fromtimestamp(unique_ts[0] / 1000, tz=timezone.utc).isoformat()
    ig_last = datetime.fromtimestamp(unique_ts[-1] / 1000, tz=timezone.utc).isoformat()

    pre_count = _count_rows_from(args.table, start_ms)
    pre_mm = _minmax_from(args.table, start_ms)

    report: dict[str, object] = {
        "table": args.table,
        "instrument": instrument.name,
        "start_utc": start_utc.isoformat(),
        "end_utc": end_utc.isoformat(),
        "ig_fetched_rows": int(len(ig_rows)),
        "ig_unique_timestamps": int(len(unique_ts)),
        "ig_first_utc": ig_first,
        "ig_last_utc": ig_last,
        "mysql_rows_from_start_before": pre_count,
        "mysql_minmax_before": pre_mm,
        "executed": bool(args.execute),
    }

    if args.execute:
        deleted = _delete_from(args.table, start_ms)
        written = insert_prices(ig_rows, instrument)
        post_count = _count_rows_from(args.table, start_ms)
        post_mm = _minmax_from(args.table, start_ms)
        report.update(
            {
                "mysql_deleted_rows": int(deleted),
                "mysql_insert_rowcount": int(written),
                "mysql_rows_from_start_after": int(post_count),
                "mysql_minmax_after": post_mm,
            }
        )

    out = ROOT / args.report_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved_report={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

