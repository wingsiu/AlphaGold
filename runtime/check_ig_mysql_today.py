#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from data import DataLoader
from ig_scripts.ig_data_api import API_CONFIG, IGService, Price, fetch_prices


def _pick_instrument(table: str) -> Price:
	mapping = {
		"gold_prices": Price.Gold,
		"aud_prices": Price.AUD,
		"prices": Price.Oil,
	}
	if table not in mapping:
		raise ValueError(f"Unsupported table for IG mapping: {table}")
	return mapping[table]


def _hkt_trading_window(now_utc: datetime) -> tuple[datetime, datetime]:
	hkt = ZoneInfo("Asia/Hong_Kong")
	now_hkt = now_utc.astimezone(hkt)
	start_hkt = datetime(now_hkt.year, now_hkt.month, now_hkt.day, 6, 0, 0, tzinfo=hkt)
	if now_hkt < start_hkt:
		start_hkt = start_hkt - timedelta(days=1)
	end_utc = now_utc.replace(second=0, microsecond=0) - timedelta(minutes=1)
	return start_hkt.astimezone(timezone.utc), end_utc


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Compare today's IG data versus MySQL data.")
	p.add_argument("--table", default="gold_prices", choices=["gold_prices", "aud_prices", "prices"])
	p.add_argument("--start-utc", default=None, help="Optional override start timestamp in UTC, e.g. 2026-04-17T00:00:00+00:00")
	p.add_argument("--end-utc", default=None, help="Optional override end timestamp in UTC, e.g. 2026-04-17T20:59:00+00:00")
	p.add_argument("--csv-out", default="runtime/_ig_mysql_today_compare.csv")
	p.add_argument("--json-out", default="runtime/_ig_mysql_today_compare_summary.json")
	return p


def main() -> int:
	args = build_parser().parse_args()
	instrument = _pick_instrument(args.table)

	now_utc = datetime.now(timezone.utc)
	if args.start_utc or args.end_utc:
		if not (args.start_utc and args.end_utc):
			raise SystemExit("Please provide both --start-utc and --end-utc together")
		start_utc = pd.Timestamp(args.start_utc).tz_convert("UTC") if pd.Timestamp(args.start_utc).tzinfo else pd.Timestamp(args.start_utc).tz_localize("UTC")
		end_utc = pd.Timestamp(args.end_utc).tz_convert("UTC") if pd.Timestamp(args.end_utc).tzinfo else pd.Timestamp(args.end_utc).tz_localize("UTC")
		start_utc = start_utc.to_pydatetime()
		end_utc = end_utc.to_pydatetime()
	else:
		start_utc, end_utc = _hkt_trading_window(now_utc)

	db_raw = DataLoader().load_data(
		args.table,
		start_date=start_utc.astimezone(ZoneInfo("Asia/Hong_Kong")).date().isoformat(),
		end_date=end_utc.astimezone(ZoneInfo("Asia/Hong_Kong")).date().isoformat(),
	)
	if db_raw.empty:
		raise SystemExit("MySQL returned no rows in the selected window")

	db = db_raw.copy()
	db["ts"] = pd.to_datetime(db["timestamp"], unit="ms", utc=True)
	db = db[(db["ts"] >= pd.Timestamp(start_utc)) & (db["ts"] <= pd.Timestamp(end_utc))].copy()
	db = db.sort_values("ts").drop_duplicates("ts", keep="last")

	service = IGService(
		API_CONFIG["api_key"],
		API_CONFIG["username"],
		API_CONFIG["password"],
		API_CONFIG["base_url"],
	)
	ig_rows = fetch_prices(service, instrument, start_time=start_utc, end_time=end_utc)
	ig = pd.DataFrame(ig_rows)
	if ig.empty:
		raise SystemExit("IG returned no rows in the selected window")

	ig["ts"] = pd.to_datetime(ig["timestamp"], unit="ms", utc=True)
	ig = ig[(ig["ts"] >= pd.Timestamp(start_utc)) & (ig["ts"] <= pd.Timestamp(end_utc))].copy()
	ig = ig.sort_values("ts").drop_duplicates("ts", keep="last")

	value_cols = ["openPrice", "highPrice", "lowPrice", "closePrice", "lastTradedVolume"]
	joined = db[["ts"] + value_cols].merge(
		ig[["ts"] + value_cols],
		on="ts",
		how="outer",
		suffixes=("_mysql", "_ig"),
		indicator=True,
	)

	both = joined[joined["_merge"] == "both"].copy()
	for c in ["openPrice", "highPrice", "lowPrice", "closePrice"]:
		both[f"{c}_abs_diff"] = (both[f"{c}_mysql"] - both[f"{c}_ig"]).abs()

	max_abs_price_diff = None
	if not both.empty:
		max_abs_price_diff = float(
			both[["openPrice_abs_diff", "highPrice_abs_diff", "lowPrice_abs_diff", "closePrice_abs_diff"]]
			.max()
			.max()
		)

	summary = {
		"table": args.table,
		"instrument": instrument.name,
		"start_utc": start_utc.isoformat(),
		"end_utc": end_utc.isoformat(),
		"rows_mysql": int(len(db)),
		"rows_ig": int(len(ig)),
		"rows_overlap": int(len(both)),
		"mysql_only": int((joined["_merge"] == "left_only").sum()),
		"ig_only": int((joined["_merge"] == "right_only").sum()),
		"max_abs_price_diff": max_abs_price_diff,
	}

	out_csv = ROOT / args.csv_out
	out_json = ROOT / args.json_out
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	joined.sort_values("ts").to_csv(out_csv, index=False)
	out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	print(json.dumps(summary, indent=2))
	print(f"saved_csv={out_csv}")
	print(f"saved_json={out_json}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

