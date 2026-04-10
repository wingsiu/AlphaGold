import argparse
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import pymysql
from dotenv import load_dotenv


TABLES = {
    "aud": "aud_prices",
    "gold": "gold_prices",
    "oil": "prices",
}

# Regular daily maintenance breaks in UTC. These windows are expected no-trade periods.
MAINTENANCE_WINDOWS_UTC: dict[str, list[tuple[int, int, int, int]]] = {
    "gold_prices": [(21, 0, 22, 0)],
    "prices": [(5, 0, 6, 0), (21, 0, 22, 0)],
}


@dataclass
class GapRecord:
    table: str
    previous_ts_utc: datetime
    current_ts_utc: datetime
    gap_minutes: int
    missing_bars: int


def _to_utc_datetime(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)


def _date_range(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def _load_active_dates(conn: pymysql.connections.Connection, table_name: str) -> set[date]:
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT DISTINCT DATE(FROM_UNIXTIME(timestamp / 1000)) AS trading_day
            FROM {table_name}
            """
        )
        rows = cursor.fetchall()
    return {row[0] for row in rows if row and row[0] is not None}


def _is_excluded_gap(prev_dt: datetime, curr_dt: datetime, active_dates: set[date]) -> bool:
    if curr_dt <= prev_dt + timedelta(minutes=1):
        return True

    prev_day = prev_dt.date()
    curr_day = curr_dt.date()
    if curr_day <= prev_day:
        return False

    between_days = list(_date_range(prev_day + timedelta(days=1), curr_day - timedelta(days=1)))
    if not between_days:
        return False

    has_active_weekday = any(d.weekday() < 5 and d in active_dates for d in between_days)
    if has_active_weekday:
        return False

    has_weekend = any(d.weekday() >= 5 for d in between_days)
    has_inactive_weekday = any(d.weekday() < 5 and d not in active_dates for d in between_days)
    return has_weekend or has_inactive_weekday


def _matches_maintenance_window(table_name: str, prev_dt: datetime, curr_dt: datetime) -> bool:
    windows = MAINTENANCE_WINDOWS_UTC.get(table_name, [])
    if not windows:
        return False

    # Allow small drift around expected close/open timestamps.
    close_tolerance = timedelta(minutes=2)
    open_tolerance = timedelta(minutes=3)

    for close_h, close_m, open_h, open_m in windows:
        expected_close = prev_dt.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
        expected_open = prev_dt.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
        if expected_open <= expected_close:
            expected_open += timedelta(days=1)

        expected_last_bar = expected_close - timedelta(minutes=1)
        if abs(prev_dt - expected_last_bar) <= close_tolerance and abs(curr_dt - expected_open) <= open_tolerance:
            return True

    return False


def _find_non_holiday_gaps(
    conn: pymysql.connections.Connection,
    table_name: str,
    max_allowed_missing_bars: int,
    recurring_gap_min_occurrences: int,
) -> list[GapRecord]:
    active_dates = _load_active_dates(conn, table_name)

    candidate_issues: list[GapRecord] = []
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT timestamp FROM {table_name} ORDER BY timestamp ASC")

        prev_ts_ms = None
        for (ts_ms,) in cursor:
            if prev_ts_ms is None:
                prev_ts_ms = ts_ms
                continue

            diff_minutes = int((ts_ms - prev_ts_ms) / 60000)
            missing_bars = max(diff_minutes - 1, 0)
            if missing_bars > max_allowed_missing_bars:
                prev_dt = _to_utc_datetime(prev_ts_ms)
                curr_dt = _to_utc_datetime(ts_ms)
                if _matches_maintenance_window(table_name, prev_dt, curr_dt):
                    prev_ts_ms = ts_ms
                    continue

                if not _is_excluded_gap(prev_dt, curr_dt, active_dates):
                    candidate_issues.append(
                        GapRecord(
                            table=table_name,
                            previous_ts_utc=prev_dt,
                            current_ts_utc=curr_dt,
                            gap_minutes=diff_minutes,
                            missing_bars=missing_bars,
                        )
                    )
            prev_ts_ms = ts_ms

    # Treat frequently recurring long gaps as scheduled closures rather than missing data.
    signatures = Counter(
        (
            g.previous_ts_utc.hour,
            g.previous_ts_utc.minute,
            g.current_ts_utc.hour,
            g.current_ts_utc.minute,
            g.gap_minutes,
        )
        for g in candidate_issues
        if g.missing_bars >= 30
    )

    recurring_signatures = {
        signature
        for signature, occurrences in signatures.items()
        if occurrences >= recurring_gap_min_occurrences
    }

    filtered_issues: list[GapRecord] = []
    for gap in candidate_issues:
        signature = (
            gap.previous_ts_utc.hour,
            gap.previous_ts_utc.minute,
            gap.current_ts_utc.hour,
            gap.current_ts_utc.minute,
            gap.gap_minutes,
        )
        if signature in recurring_signatures:
            continue
        filtered_issues.append(gap)

    return filtered_issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check for missing minute bars (> threshold) in AUD, Gold and Oil data, excluding holiday/weekend spans."
    )
    parser.add_argument("--host", default="192.168.0.4", help="MySQL host (default: 192.168.0.4)")
    parser.add_argument("--port", type=int, default=3306, help="MySQL port (default: 3306)")
    parser.add_argument("--user", default=None, help="MySQL user (defaults to DB_USER in .env)")
    parser.add_argument("--password", default=None, help="MySQL password (defaults to DB_PASSWORD in .env)")
    parser.add_argument("--database", default=None, help="MySQL database (defaults to DB_NAME in .env)")
    parser.add_argument(
        "--max-missing-bars",
        type=int,
        default=2,
        help="Maximum allowed missing bars before reporting (default: 2)",
    )
    parser.add_argument(
        "--recurring-gap-min-occurrences",
        type=int,
        default=5,
        help="Exclude recurring long-gap signatures seen at least this many times (default: 5)",
    )
    args = parser.parse_args()

    load_dotenv()
    user = args.user or os.getenv("DB_USER")
    password = args.password or os.getenv("DB_PASSWORD")
    database = args.database or os.getenv("DB_NAME")

    if not user or not password or not database:
        raise ValueError("Missing credentials. Provide --user/--password/--database or set DB_USER/DB_PASSWORD/DB_NAME in .env")

    conn = pymysql.connect(
        host=args.host,
        port=args.port,
        user=user,
        password=password,
        database=database,
        cursorclass=pymysql.cursors.Cursor,
        read_timeout=120,
        write_timeout=120,
    )

    try:
        all_issues: list[GapRecord] = []
        for short_name, table_name in TABLES.items():
            issues = _find_non_holiday_gaps(
                conn,
                table_name,
                args.max_missing_bars,
                args.recurring_gap_min_occurrences,
            )
            all_issues.extend(issues)
            print(f"{short_name.upper()} ({table_name}): {len(issues)} suspicious gaps")

            for issue in issues[:10]:
                print(
                    f"  - prev={issue.previous_ts_utc.isoformat()} current={issue.current_ts_utc.isoformat()} "
                    f"gap_min={issue.gap_minutes} missing_bars={issue.missing_bars}"
                )
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")

        if not all_issues:
            print("\nNo suspicious gaps found (after excluding weekend/holiday spans).")
        else:
            print(f"\nTotal suspicious gaps across all symbols: {len(all_issues)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

