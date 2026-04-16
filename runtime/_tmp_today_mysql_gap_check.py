from __future__ import annotations

import os
from datetime import datetime, timezone

import pymysql
from dotenv import load_dotenv


load_dotenv()
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")
host = os.getenv("DB_HOST", "127.0.0.1")
port = int(os.getenv("DB_PORT", "3306"))

if not user or not password or not database:
    raise SystemExit("Missing DB credentials from environment/.env")

conn = pymysql.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    database=database,
    cursorclass=pymysql.cursors.Cursor,
    read_timeout=60,
    write_timeout=60,
)

tables = [("gold", "gold_prices"), ("aud", "aud_prices"), ("oil", "prices")]
now_utc = datetime.now(timezone.utc)
today = now_utc.date()
start_dt = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
start_ms = int(start_dt.timestamp() * 1000)
now_ms = int(now_utc.timestamp() * 1000)

out_lines = [f"utc_now={now_utc.isoformat()}", f"today_start_utc={start_dt.isoformat()}"]
with conn:
    for short, table in tables:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT timestamp FROM {table} WHERE timestamp >= %s AND timestamp <= %s ORDER BY timestamp ASC",
                (start_ms, now_ms),
            )
            rows = cur.fetchall()
        ts_list = [int(r[0]) for r in rows]
        out_lines.append(f"[{short}] rows_today={len(ts_list)}")
        if not ts_list:
            out_lines.append(f"[{short}] no rows found for today")
            continue
        first_dt = datetime.fromtimestamp(ts_list[0] / 1000, tz=timezone.utc)
        last_dt = datetime.fromtimestamp(ts_list[-1] / 1000, tz=timezone.utc)
        out_lines.append(f"[{short}] first_utc={first_dt.isoformat()} last_utc={last_dt.isoformat()}")
        gaps = []
        dupes = 0
        for prev, curr in zip(ts_list, ts_list[1:]):
            diff_min = int((curr - prev) / 60000)
            if diff_min == 0:
                dupes += 1
            if diff_min > 1:
                gaps.append((prev, curr, diff_min, diff_min - 1))
        out_lines.append(f"[{short}] duplicate_adjacent_pairs={dupes}")
        out_lines.append(f"[{short}] gaps_gt_1m={len(gaps)}")
        for prev, curr, diff_min, missing in gaps[:20]:
            prev_dt = datetime.fromtimestamp(prev / 1000, tz=timezone.utc)
            curr_dt = datetime.fromtimestamp(curr / 1000, tz=timezone.utc)
            out_lines.append(
                f"[{short}] gap prev={prev_dt.isoformat()} curr={curr_dt.isoformat()} gap_min={diff_min} missing={missing}"
            )
        if len(gaps) > 20:
            out_lines.append(f"[{short}] ... {len(gaps) - 20} more gaps")

with open("runtime/_tmp_today_mysql_gap_check.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")

