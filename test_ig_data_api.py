import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
from typing import cast

from ig_scripts.ig_data_api import (
	IGService,
	Price,
	fetch_and_store_market_snapshot,
	fetch_and_store_prices_from_latest,
	fetch_prices,
	snapshot_to_price_row,
)


class IGDataApiTest(unittest.TestCase):
	def test_snapshot_to_price_row_maps_live_snapshot_into_existing_schema(self) -> None:
		row = snapshot_to_price_row(
			{
				"instrument": "gold",
				"epic": Price.Gold.epic,
				"fetch_time_utc": "2026-01-05T12:00:45+00:00",
				"bid": 100.0,
				"offer": 102.0,
				"mid": 101.0,
				"high": 103.0,
				"low": 99.0,
				"lastTradedVolume": 7,
			}
		)

		expected_ts = int(datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
		self.assertEqual(row["timestamp"], expected_ts)
		self.assertEqual(row["openPrice"], 101.0)
		self.assertEqual(row["closePrice"], 101.0)
		self.assertEqual(row["openPrice_ask"], 102.0)
		self.assertEqual(row["openPrice_bid"], 100.0)
		self.assertEqual(row["highPrice"], 103.0)
		self.assertEqual(row["lowPrice"], 99.0)
		self.assertEqual(row["lastTradedVolume"], 7)

	def test_snapshot_to_price_row_falls_back_when_mid_is_missing(self) -> None:
		row = snapshot_to_price_row(
			{
				"instrument": "oil",
				"epic": Price.Oil.epic,
				"fetch_time_utc": "2026-01-05T12:00:10+00:00",
				"bid": 80.0,
				"offer": 82.0,
				"mid": None,
				"high": None,
				"low": None,
			}
		)

		self.assertEqual(row["openPrice"], 81.0)
		self.assertEqual(row["closePrice"], 81.0)
		self.assertEqual(row["highPrice"], 81.0)
		self.assertEqual(row["lowPrice"], 81.0)
		self.assertEqual(row["openPrice_ask"], 82.0)
		self.assertEqual(row["openPrice_bid"], 80.0)
		self.assertEqual(row["lastTradedVolume"], 0)

	@patch("ig_scripts.ig_data_api.insert_prices")
	@patch("ig_scripts.ig_data_api.fetch_market_snapshot")
	def test_fetch_and_store_market_snapshot_uses_existing_table_upsert(self, mock_fetch_snapshot, mock_insert_prices) -> None:
		mock_fetch_snapshot.return_value = {
			"instrument": "aud",
			"epic": Price.AUD.epic,
			"fetch_time_utc": "2026-01-05T12:00:45+00:00",
			"bid": 0.65,
			"offer": 0.67,
			"mid": 0.66,
			"high": 0.68,
			"low": 0.64,
			"lastTradedVolume": 11,
		}
		mock_insert_prices.return_value = 1

		row = fetch_and_store_market_snapshot(cast(IGService, object()), Price.AUD)

		mock_fetch_snapshot.assert_called_once()
		mock_insert_prices.assert_called_once_with([row], Price.AUD)
		self.assertEqual(row["openPrice"], 0.66)
		self.assertEqual(row["lastTradedVolume"], 11)

	@patch("ig_scripts.ig_data_api.requests.get")
	@patch("ig_scripts.ig_data_api.fetch_last_date")
	def test_fetch_prices_starts_from_latest_mysql_time_and_ends_at_requested_current_time(self, mock_fetch_last_date, mock_get) -> None:
		mock_fetch_last_date.return_value = datetime(2026, 1, 5, 12, 3, tzinfo=timezone.utc)
		mock_get.return_value = SimpleNamespace(
			status_code=200,
			json=lambda: {"intervalsDataPoints": []},
			text="",
		)
		service = cast(IGService, cast(object, SimpleNamespace(headers={}, refresh_tokens_if_needed=lambda: None, authenticate=lambda: None)))

		rows = fetch_prices(
			service,
			Price.Gold,
			end_time=datetime(2026, 1, 5, 12, 10, 30, tzinfo=timezone.utc),
		)

		self.assertEqual(rows, [])
		called_url = mock_get.call_args.args[0]
		self.assertIn("start/2026/1/5/12/3/0/0/", called_url)
		self.assertIn("end/2026/1/5/12/10/30/0?format=json", called_url)

	@patch("ig_scripts.ig_data_api.insert_prices")
	@patch("ig_scripts.ig_data_api.fetch_prices")
	@patch("ig_scripts.ig_data_api.fetch_last_date")
	def test_fetch_and_store_prices_from_latest_returns_resulting_period_summary(self, mock_fetch_last_date, mock_fetch_prices, mock_insert_prices) -> None:
		mock_fetch_last_date.return_value = datetime(2026, 1, 5, 12, 3, tzinfo=timezone.utc)
		mock_fetch_prices.return_value = [
			{"timestamp": int(datetime(2026, 1, 5, 12, 4, tzinfo=timezone.utc).timestamp() * 1000)},
			{"timestamp": int(datetime(2026, 1, 5, 12, 5, tzinfo=timezone.utc).timestamp() * 1000)},
		]
		mock_insert_prices.return_value = 2

		summary = fetch_and_store_prices_from_latest(
			cast(IGService, object()),
			Price.Gold,
			end_time=datetime(2026, 1, 5, 12, 10, 30, tzinfo=timezone.utc),
		)

		self.assertEqual(summary["latest_db_before_utc"], "2026-01-05T12:03:00+00:00")
		self.assertEqual(summary["requested_start_utc"], "2026-01-05T12:03:00+00:00")
		self.assertEqual(summary["requested_end_utc"], "2026-01-05T12:10:30+00:00")
		self.assertEqual(summary["fetched_rows"], 2)
		self.assertEqual(summary["written_rows"], 2)
		self.assertEqual(summary["inserted_rows"], 2)
		self.assertEqual(summary["fetched_period_start_utc"], "2026-01-05T12:04:00+00:00")
		self.assertEqual(summary["fetched_period_end_utc"], "2026-01-05T12:05:00+00:00")


if __name__ == "__main__":
	unittest.main(verbosity=2)

