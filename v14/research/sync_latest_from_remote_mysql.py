import argparse
import os
from datetime import datetime, timezone
from typing import Any

import pymysql
from dotenv import load_dotenv


TABLES = {
    "aud": "aud_prices",
    "gold": "gold_prices",
    "oil": "prices",
}

PRICE_COLUMNS = [
    "timestamp",
    "openPrice",
    "openPrice_ask",
    "openPrice_bid",
    "closePrice",
    "closePrice_ask",
    "closePrice_bid",
    "highPrice",
    "highPrice_ask",
    "highPrice_bid",
    "lowPrice",
    "lowPrice_ask",
    "lowPrice_bid",
    "lastTradedVolume",
]


def _utc_iso_from_ms(epoch_ms: int | None) -> str:
    if epoch_ms is None:
        return "None"
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()


def _table_max_timestamp(conn: pymysql.connections.Connection, table_name: str) -> int | None:
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT MAX(timestamp) FROM {table_name}")
        result = cursor.fetchone()
        if not result:
            return None
        return result[0]


def _sync_table(
    remote_conn: pymysql.connections.Connection,
    local_conn: pymysql.connections.Connection,
    table_name: str,
    batch_size: int,
    dry_run: bool,
) -> tuple[int, int, int | None, int | None]:
    local_max = _table_max_timestamp(local_conn, table_name)

    where_sql = ""
    params: tuple[Any, ...] = ()
    if local_max is not None:
        where_sql = "WHERE timestamp > %s"
        params = (local_max,)

    remote_count_sql = f"SELECT COUNT(*) FROM {table_name} {where_sql}"
    with remote_conn.cursor() as remote_cursor:
        remote_cursor.execute(remote_count_sql, params)
        remote_new_rows = remote_cursor.fetchone()[0]

    if remote_new_rows == 0:
        return 0, 0, local_max, local_max

    columns_sql = ", ".join(PRICE_COLUMNS)
    placeholders = ", ".join(["%s"] * len(PRICE_COLUMNS))
    insert_sql = (
        f"INSERT INTO {table_name} ({columns_sql}) VALUES ({placeholders}) "
        "ON DUPLICATE KEY UPDATE "
        "openPrice=VALUES(openPrice), "
        "openPrice_ask=VALUES(openPrice_ask), "
        "openPrice_bid=VALUES(openPrice_bid), "
        "closePrice=VALUES(closePrice), "
        "closePrice_ask=VALUES(closePrice_ask), "
        "closePrice_bid=VALUES(closePrice_bid), "
        "highPrice=VALUES(highPrice), "
        "highPrice_ask=VALUES(highPrice_ask), "
        "highPrice_bid=VALUES(highPrice_bid), "
        "lowPrice=VALUES(lowPrice), "
        "lowPrice_ask=VALUES(lowPrice_ask), "
        "lowPrice_bid=VALUES(lowPrice_bid), "
        "lastTradedVolume=VALUES(lastTradedVolume)"
    )

    select_sql = f"SELECT {columns_sql} FROM {table_name} {where_sql} ORDER BY timestamp ASC"

    fetched = 0
    inserted_or_updated = 0
    latest_remote_ts: int | None = local_max

    with remote_conn.cursor() as remote_cursor:
        remote_cursor.execute(select_sql, params)

        while True:
            rows = remote_cursor.fetchmany(batch_size)
            if not rows:
                break

            fetched += len(rows)
            latest_remote_ts = rows[-1][0]

            if dry_run:
                continue

            with local_conn.cursor() as local_cursor:
                local_cursor.executemany(insert_sql, rows)
                inserted_or_updated += local_cursor.rowcount
            local_conn.commit()

    return fetched, inserted_or_updated, local_max, latest_remote_ts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync only rows newer than local MAX(timestamp) from remote MySQL for AUD/Gold/Oil tables."
    )

    parser.add_argument("--remote-host", default="192.168.0.4", help="Remote MySQL host")
    parser.add_argument("--remote-port", type=int, default=3306, help="Remote MySQL port")
    parser.add_argument("--remote-user", default=None, help="Remote MySQL user")
    parser.add_argument("--remote-password", default=None, help="Remote MySQL password")
    parser.add_argument("--remote-database", default=None, help="Remote MySQL database")

    parser.add_argument("--local-host", default=None, help="Local MySQL host (defaults to LOCAL_DB_HOST, then DB_HOST)")
    parser.add_argument("--local-port", type=int, default=None, help="Local MySQL port (defaults to LOCAL_DB_PORT, then 3306)")
    parser.add_argument("--local-user", default=None, help="Local MySQL user")
    parser.add_argument("--local-password", default=None, help="Local MySQL password")
    parser.add_argument("--local-database", default=None, help="Local MySQL database")

    parser.add_argument("--batch-size", type=int, default=5000, help="Rows per transfer batch")
    parser.add_argument("--dry-run", action="store_true", help="Show counts only; do not write to local DB")

    args = parser.parse_args()

    load_dotenv()

    remote_user = args.remote_user or os.getenv("REMOTE_DB_USER") or os.getenv("DB_USER")
    remote_password = args.remote_password or os.getenv("REMOTE_DB_PASSWORD") or os.getenv("DB_PASSWORD")
    remote_database = args.remote_database or os.getenv("REMOTE_DB_NAME") or os.getenv("DB_NAME")

    local_host = args.local_host or os.getenv("LOCAL_DB_HOST") or os.getenv("DB_HOST") or "127.0.0.1"
    local_port = args.local_port or int(os.getenv("LOCAL_DB_PORT", "3306"))
    local_user = args.local_user or os.getenv("LOCAL_DB_USER") or os.getenv("DB_USER")
    local_password = args.local_password or os.getenv("LOCAL_DB_PASSWORD") or os.getenv("DB_PASSWORD")
    local_database = args.local_database or os.getenv("LOCAL_DB_NAME") or os.getenv("DB_NAME")

    missing = []
    if not remote_user:
        missing.append("remote user")
    if not remote_password:
        missing.append("remote password")
    if not remote_database:
        missing.append("remote database")
    if not local_user:
        missing.append("local user")
    if not local_password:
        missing.append("local password")
    if not local_database:
        missing.append("local database")

    if missing:
        raise ValueError("Missing required DB settings: " + ", ".join(missing))

    remote_conn = pymysql.connect(
        host=args.remote_host,
        port=args.remote_port,
        user=remote_user,
        password=remote_password,
        database=remote_database,
        cursorclass=pymysql.cursors.Cursor,
        read_timeout=120,
        write_timeout=120,
        autocommit=False,
    )

    local_conn = pymysql.connect(
        host=local_host,
        port=local_port,
        user=local_user,
        password=local_password,
        database=local_database,
        cursorclass=pymysql.cursors.Cursor,
        read_timeout=120,
        write_timeout=120,
        autocommit=False,
    )

    try:
        for short_name, table_name in TABLES.items():
            fetched, inserted_or_updated, before_max, after_max = _sync_table(
                remote_conn=remote_conn,
                local_conn=local_conn,
                table_name=table_name,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )

            print(
                f"{short_name.upper()} ({table_name}) -> remote_new_rows={fetched}, "
                f"written={inserted_or_updated if not args.dry_run else 0}, "
                f"local_max_before={_utc_iso_from_ms(before_max)}, "
                f"latest_remote_seen={_utc_iso_from_ms(after_max)}"
            )

        if args.dry_run:
            print("Dry run finished. No local writes were made.")
        else:
            print("Sync finished.")
    finally:
        remote_conn.close()
        local_conn.close()


if __name__ == "__main__":
    main()

