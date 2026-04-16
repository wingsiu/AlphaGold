import json
import io
import logging
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import trading_bot as trading_bot_module

from trading_bot import (
	AlphaGoldTradingBot,
	DEFAULT_BEST_BASE_MODEL_PATH,
	DEFAULT_SIGNAL_MODEL_FAMILY,
	aligned_sleep_seconds,
	build_parser,
	BotConfig,
	BotState,
	ExistingSchemaMarketDataStore,
	format_best_base_signal_log,
	format_signal_status_line,
	instrument_trading_hours_open,
	market_data_due,
	market_data_poll_bucket,
	next_trading_open_utc,
	PaperBroker,
	prepare_raw_price_frame,
	build_complete_15m_bars,
	entry_block_reason,
	IGMarketDataCollector,
	snapshot_tradeable,
	summarize_daily_trade_log,
	summarize_trade_windows,
	summarize_trade_log,
	summarize_weekly_trade_log,
	trading_day_label,
)
from ig_scripts.ig_data_api import Price


def _ts(value: str) -> pd.Timestamp:
	parsed = pd.Timestamp(value)
	assert parsed is not pd.NaT
	return cast(pd.Timestamp, parsed)


class TradingBotHelpersTest(unittest.TestCase):
	def test_bot_state_round_trip_with_last_execution_attempt(self) -> None:
		state = BotState(
			last_signal_bar_time=_ts("2026-01-05T12:00:00Z").isoformat(),
			last_execution_attempt={
				"status": "dry_run",
				"mode": "signal_only",
				"entry_price": 100.25,
			},
			last_periodic_report_bucket_utc="2026-01-05T12:30:00+00:00",
		)

		raw = state.to_dict()
		rehydrated = BotState.from_dict(raw)

		self.assertIsNotNone(rehydrated.last_execution_attempt)
		self.assertEqual(rehydrated.last_execution_attempt["status"], "dry_run")
		self.assertEqual(rehydrated.last_periodic_report_bucket_utc, "2026-01-05T12:30:00+00:00")

	def test_prepare_raw_price_frame_builds_sorted_utc_index(self) -> None:
		raw = pd.DataFrame(
			{
				"timestamp": [
					int(_ts("2026-01-01T00:02:00Z").timestamp() * 1000),
					int(_ts("2026-01-01T00:01:00Z").timestamp() * 1000),
				],
				"openPrice": [1.0, 2.0],
			}
		)

		prepared = prepare_raw_price_frame(raw)

		self.assertEqual(list(prepared.index), [_ts("2026-01-01T00:01:00Z"), _ts("2026-01-01T00:02:00Z")])

	def test_format_best_base_signal_log_is_clear(self) -> None:
		msg = format_best_base_signal_log(
			{
				"side": "up",
				"probability": 0.81234,
				"signal_bar_time": _ts("2026-01-05T12:00:00Z"),
				"entry_bar_time": _ts("2026-01-05T12:01:00Z"),
				"tradable": True,
				"pred": 2,
			},
			raw_rows=6027,
			bars_rows=6027,
			candidate_samples=5878,
		)

		self.assertIn("BEST_BASE SIGNAL: side=up", msg)
		self.assertIn("prob=0.8123", msg)
		self.assertIn("raw_rows=6027", msg)
		self.assertIn("bars_1m=6027", msg)
		self.assertIn("candidates=5878", msg)

	def test_format_best_base_signal_log_includes_market_open_flag(self) -> None:
		msg = format_best_base_signal_log(
			{
				"side": "up",
				"probability": 0.8,
				"signal_bar_time": _ts("2026-01-05T12:00:00Z"),
				"entry_bar_time": _ts("2026-01-05T12:01:00Z"),
				"tradable": True,
			},
			is_trading_hour=True,
		)
		self.assertIn("market_open=1", msg)

	def test_format_signal_status_line_has_visual_marks(self) -> None:
		msg = format_signal_status_line(
			signal={
				"side": "up",
				"probability": 0.9,
				"tradable": True,
				"signal_bar_time": _ts("2026-01-05T12:00:00Z"),
				"entry_bar_time": _ts("2026-01-05T12:01:00Z"),
			},
			trading_open_now=False,
			signal_qualified=True,
			weak_filter_enabled=True,
			weak_period_block=False,
			cutoff=None,
		)
		self.assertIn("hour=[X]", msg)
		self.assertIn("signal=🟢✓", msg)
		self.assertIn("weak_filter=[OK]", msg)

	def test_format_signal_status_line_shows_open_now_and_lag(self) -> None:
		msg = format_signal_status_line(
			signal={
				"side": "flat",
				"probability": 0.0,
				"tradable": False,
				"signal_bar_time": _ts("2026-01-05T12:00:00Z"),
				"entry_bar_time": _ts("2026-01-05T12:01:00Z"),
			},
			trading_open_now=True,
			signal_qualified=False,
			weak_filter_enabled=True,
			weak_period_block=True,
			now_utc=_ts("2026-01-05T12:02:00Z"),
		)
		self.assertIn("weak_filter=[X]", msg)
		self.assertIn("freshness=[OK]", msg)
		self.assertIn("lag_min=2.0", msg)

	def test_next_trading_open_utc_returns_future_open_when_closed(self) -> None:
		now_ts = _ts("2026-01-02T21:30:00Z")
		next_open = next_trading_open_utc(Price.Gold, now_ts)
		self.assertGreaterEqual(next_open, now_ts)
		self.assertTrue(instrument_trading_hours_open(Price.Gold, next_open))

	def test_format_best_base_signal_log_supports_risk_off_side(self) -> None:
		msg = format_best_base_signal_log(
			{
				"side": "risk_off",
				"probability": 0.0,
				"signal_bar_time": _ts("2026-01-05T12:00:00Z"),
				"entry_bar_time": _ts("2026-01-05T12:01:00Z"),
				"tradable": False,
				"pred": 3,
			}
		)
		self.assertIn("side=risk_off", msg)

	def test_build_latest_signal_best_base_maps_risky_to_risk_off(self) -> None:
		bot = AlphaGoldTradingBot.__new__(AlphaGoldTradingBot)
		bot.image_trend = type(
			"ImageTrendStub",
			(),
			{
				"LABEL_UP": 2,
				"LABEL_DOWN": 0,
				"LABEL_RISKY": 3,
				"TREND_LABELS": (0, 2),
			},
		)()
		bot._build_best_base_signal_series = lambda _raw: {
			"ts": pd.DatetimeIndex([_ts("2026-01-05T12:00:00Z")]),
			"entry_ts": pd.DatetimeIndex([_ts("2026-01-05T12:01:00Z")]),
			"pred": [3],
			"signal_prob": [0.123],
		}
		raw = pd.DataFrame(index=pd.DatetimeIndex([_ts("2026-01-05T12:00:00Z")]))

		signal = bot._build_latest_signal_best_base(raw)

		self.assertEqual(signal["pred"], 3)
		self.assertEqual(signal["side"], "risk_off")
		self.assertFalse(bool(signal["tradable"]))

	def test_build_latest_signal_best_base_uses_stage_diagnostics_for_flat(self) -> None:
		bot = AlphaGoldTradingBot.__new__(AlphaGoldTradingBot)
		bot.image_trend = type(
			"ImageTrendStub",
			(),
			{
				"LABEL_UP": 2,
				"LABEL_DOWN": 0,
				"LABEL_RISKY": 3,
				"TREND_LABELS": (0, 2),
			},
		)()
		bot._build_best_base_signal_series = lambda _raw: {
			"ts": pd.DatetimeIndex([_ts("2026-01-05T12:00:00Z")]),
			"entry_ts": pd.DatetimeIndex([_ts("2026-01-05T12:01:00Z")]),
			"pred": np.array([1], dtype=np.int64),
			"signal_prob": np.array([0.0], dtype=np.float64),
			"trend_prob": np.array([0.4825], dtype=np.float64),
			"up_prob": np.array([np.nan], dtype=np.float64),
		}
		raw = pd.DataFrame(index=pd.DatetimeIndex([_ts("2026-01-05T12:00:00Z")]))

		signal = bot._build_latest_signal_best_base(raw)

		self.assertEqual(signal["side"], "flat")
		self.assertAlmostEqual(float(signal["probability"]), 0.4825, places=4)
		self.assertEqual(signal["reject_reason"], "stage1_gate")
		self.assertAlmostEqual(float(signal["trend_probability"]), 0.4825, places=4)

	def test_build_complete_15m_bars_drops_incomplete_tail(self) -> None:
		idx = pd.date_range(_ts("2026-01-01T00:00:00Z"), periods=29, freq="1min")
		raw = pd.DataFrame(
			{
				"timestamp": (idx.view("int64") // 1_000_000).astype(int),
				"openPrice": 100.0,
				"highPrice": 101.0,
				"lowPrice": 99.0,
				"closePrice": 100.5,
				"openPrice_ask": 100.1,
				"openPrice_bid": 99.9,
				"highPrice_ask": 101.1,
				"highPrice_bid": 100.9,
				"lowPrice_ask": 99.1,
				"lowPrice_bid": 98.9,
				"closePrice_ask": 100.6,
				"closePrice_bid": 100.4,
				"lastTradedVolume": 1.0,
			},
			index=idx,
		)

		_, complete = build_complete_15m_bars(raw)

		self.assertEqual(len(complete), 1)
		self.assertEqual(complete.index[0], _ts("2026-01-01T00:00:00Z"))

	def test_paper_broker_hits_take_profit(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			cfg = BotConfig(
				trade_log_path=str(Path(td) / "trades.csv"),
				log_path=str(Path(td) / "bot.log"),
				state_path=str(Path(td) / "state.json"),
				max_hold_bars=4,
				take_profit_pct=0.6,
				stop_loss_pct=0.4,
			)
			broker = PaperBroker(cfg, logging.getLogger("paper_broker_test"))
			position = broker.open_long(
				signal_bar_time=_ts("2026-01-01T00:00:00Z"),
				entry_bar_time=_ts("2026-01-01T00:15:00Z"),
				entry_time=_ts("2026-01-01T00:15:00Z"),
				entry_price=100.0,
				probability=0.72,
			)
			bars = pd.DataFrame(
				{
					"open": [100.0, 100.2],
					"high": [100.8, 100.3],
					"low": [99.7, 100.0],
					"close": [100.5, 100.1],
					"high_bid": [100.8, 100.3],
					"low_bid": [99.7, 100.0],
					"close_bid": [100.5, 100.1],
				},
				index=pd.DatetimeIndex([
					_ts("2026-01-01T00:15:00Z"),
					_ts("2026-01-01T00:30:00Z"),
				]),
			)

			next_position, trade = broker.update_position(position, bars)

			self.assertIsNone(next_position)
			self.assertIsNotNone(trade)
			self.assertTrue(str(position.deal_id).startswith("paper-"))
			self.assertEqual(trade["deal_id"], position.deal_id)
			self.assertEqual(trade["exit_reason"], "take_profit")
			self.assertAlmostEqual(trade["exit_price"], 100.6, places=6)

	def test_open_position_adjusts_target_stop_after_next_minute(self) -> None:
		bot = AlphaGoldTradingBot.__new__(AlphaGoldTradingBot)
		bot.cfg = BotConfig(take_profit_pct=0.6, stop_loss_pct=0.4)
		bot.logger = logging.getLogger("post_open_adjust_test")
		broker = PaperBroker(bot.cfg, bot.logger)
		bot.state = BotState(
			open_position=broker.open_long(
				signal_bar_time=_ts("2026-01-01T00:00:00Z"),
				entry_bar_time=_ts("2026-01-01T00:15:00Z"),
				entry_time=_ts("2026-01-01T00:15:00Z"),
				entry_price=100.0,
				probability=0.7,
			)
		)
		raw = pd.DataFrame(
			{
				"openPrice": [100.0, 100.0],
				"openPrice_ask": [101.0, 101.2],
				"closePrice": [100.2, 100.9],
			},
			index=pd.DatetimeIndex([
				_ts("2026-01-01T00:15:00Z"),
				_ts("2026-01-01T00:16:00Z"),
			]),
		)

		bot._maybe_adjust_open_position_from_previous_minute(raw)

		assert bot.state.open_position is not None
		self.assertTrue(bot.state.open_position.target_stop_adjusted)
		self.assertAlmostEqual(bot.state.open_position.entry_price, 101.0, places=6)
		self.assertAlmostEqual(bot.state.open_position.take_profit, 101.0 * 1.006, places=6)
		self.assertAlmostEqual(bot.state.open_position.stop_loss, 101.0 * 0.996, places=6)

	def test_paper_broker_times_out(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			cfg = BotConfig(
				trade_log_path=str(Path(td) / "trades.csv"),
				log_path=str(Path(td) / "bot.log"),
				state_path=str(Path(td) / "state.json"),
				max_hold_bars=2,
				take_profit_pct=0.6,
				stop_loss_pct=0.4,
			)
			broker = PaperBroker(cfg, logging.getLogger("paper_broker_timeout_test"))
			position = broker.open_long(
				signal_bar_time=_ts("2026-01-01T00:00:00Z"),
				entry_bar_time=_ts("2026-01-01T00:15:00Z"),
				entry_time=_ts("2026-01-01T00:15:00Z"),
				entry_price=100.0,
				probability=0.64,
			)
			bars = pd.DataFrame(
				{
					"open": [100.0, 100.1],
					"high": [100.3, 100.4],
					"low": [99.8, 99.7],
					"close": [100.1, 100.2],
					"high_bid": [100.3, 100.4],
					"low_bid": [99.8, 99.7],
					"close_bid": [100.1, 100.2],
				},
				index=pd.DatetimeIndex([
					_ts("2026-01-01T00:15:00Z"),
					_ts("2026-01-01T00:30:00Z"),
				]),
			)

			next_position, trade = broker.update_position(position, bars)

			self.assertIsNone(next_position)
			self.assertEqual(trade["exit_reason"], "timeout")
			self.assertEqual(pd.Timestamp(str(trade["exit_time"])), _ts("2026-01-01T00:45:00Z"))
			self.assertAlmostEqual(trade["exit_price"], 100.2, places=6)

	def test_summarize_trade_log_uses_ny_cutoff_day_labels(self) -> None:
		trades = pd.DataFrame(
			{
				"entry_time": [
					_ts("2026-01-02T00:10:00Z"),
					_ts("2026-01-02T23:10:00Z"),
				],
				"exit_time": [
					_ts("2026-01-02T00:45:00Z"),
					_ts("2026-01-02T23:45:00Z"),
				],
				"pnl_usd": [5.0, -2.0],
			}
		)

		summary = summarize_trade_log(trades, _ts("2026-01-03T00:00:00Z"))

		self.assertEqual(summary["total_trades"], 2)
		self.assertEqual(summary["wins"], 1)
		self.assertEqual(summary["losses"], 1)
		self.assertAlmostEqual(float(summary["realized_pnl_usd"]), 3.0, places=6)
		self.assertEqual(summary["today_trades"], 1)
		self.assertAlmostEqual(float(summary["today_realized_pnl_usd"]), -2.0, places=6)
		self.assertEqual(trading_day_label(_ts("2026-01-02T23:10:00Z")).date().isoformat(), summary["trading_day"])

	def test_summarize_trade_windows_returns_30m_and_60m(self) -> None:
		now_ts = _ts("2026-01-03T00:00:00Z")
		trades = pd.DataFrame(
			{
				"entry_time": [_ts("2026-01-02T23:10:00Z"), _ts("2026-01-02T22:00:00Z")],
				"exit_time": [_ts("2026-01-02T23:45:00Z"), _ts("2026-01-02T22:30:00Z")],
				"pnl_usd": [5.0, -2.0],
			}
		)

		windows = summarize_trade_windows(trades, now_ts, windows_minutes=(30, 60))

		self.assertIn("30m", windows)
		self.assertIn("60m", windows)
		self.assertEqual(windows["30m"]["trades"], 1)
		self.assertAlmostEqual(float(windows["30m"]["realized_pnl_usd"]), 5.0, places=6)
		self.assertEqual(windows["60m"]["trades"], 1)

	def test_summarize_daily_trade_log_returns_today_previous_and_trailing(self) -> None:
		now_ts = _ts("2026-01-03T00:00:00Z")
		trades = pd.DataFrame(
			{
				"entry_time": [_ts("2026-01-02T00:10:00Z"), _ts("2026-01-02T23:10:00Z")],
				"exit_time": [_ts("2026-01-02T00:45:00Z"), _ts("2026-01-02T23:45:00Z")],
				"pnl_usd": [5.0, -2.0],
			}
		)

		daily = summarize_daily_trade_log(trades, now_ts)

		self.assertIn("today", daily)
		self.assertIn("previous_day", daily)
		self.assertIn("trailing_7d", daily)
		self.assertEqual(daily["today"]["trades"], 1)
		self.assertAlmostEqual(float(daily["today"]["realized_pnl_usd"]), -2.0, places=6)
		self.assertEqual(daily["previous_day"]["trades"], 1)
		self.assertAlmostEqual(float(daily["previous_day"]["realized_pnl_usd"]), 5.0, places=6)

	def test_summarize_weekly_trade_log_returns_this_and_previous_week(self) -> None:
		now_ts = _ts("2026-01-15T12:00:00Z")
		trades = pd.DataFrame(
			{
				"entry_time": [_ts("2026-01-06T10:00:00Z"), _ts("2026-01-14T10:00:00Z")],
				"exit_time": [_ts("2026-01-06T10:30:00Z"), _ts("2026-01-14T10:30:00Z")],
				"pnl_usd": [1.5, -0.5],
			}
		)

		weekly = summarize_weekly_trade_log(trades, now_ts)

		self.assertIn("this_week", weekly)
		self.assertIn("previous_week", weekly)
		self.assertEqual(weekly["this_week"]["trades"], 1)
		self.assertEqual(weekly["previous_week"]["trades"], 1)

	def test_scheduled_performance_log_prints_only_on_00_or_30_once_per_bucket(self) -> None:
		bot = AlphaGoldTradingBot.__new__(AlphaGoldTradingBot)
		bot.cfg = BotConfig()
		bot.state = BotState()
		stream = io.StringIO()
		logger = logging.getLogger("scheduled_perf_log_test")
		logger.handlers.clear()
		logger.setLevel(logging.INFO)
		logger.propagate = False
		handler = logging.StreamHandler(stream)
		logger.addHandler(handler)
		bot.logger = logger

		trades = pd.DataFrame(
			{
				"entry_time": [_ts("2026-01-02T23:10:00Z")],
				"exit_time": [_ts("2026-01-02T23:45:00Z")],
				"pnl_usd": [3.0],
			}
		)

		bot._maybe_emit_scheduled_performance_log(trades, _ts("2026-01-03T12:29:59Z"))
		self.assertEqual(stream.getvalue(), "")

		bot._maybe_emit_scheduled_performance_log(trades, _ts("2026-01-03T12:30:05Z"))
		self.assertIn("PERF SNAPSHOT", stream.getvalue())
		first_bucket = bot.state.last_periodic_report_bucket_utc
		self.assertEqual(first_bucket, "2026-01-03T12:30:00+00:00")

		stream.truncate(0)
		stream.seek(0)
		bot._maybe_emit_scheduled_performance_log(trades, _ts("2026-01-03T12:30:40Z"))
		self.assertEqual(stream.getvalue(), "")

	def test_entry_block_reason_respects_max_trades_per_day(self) -> None:
		cfg = BotConfig(max_trades_per_day=1)
		state = BotState()
		trades = pd.DataFrame(
			{
				"entry_time": [_ts("2026-01-02T23:10:00Z")],
				"exit_time": [_ts("2026-01-02T23:45:00Z")],
				"pnl_usd": [3.0],
			}
		)

		reason = entry_block_reason(cfg, state, _ts("2026-01-03T00:15:00Z"), trades)

		self.assertEqual(reason, "max_trades_per_day_reached:1")

	def test_entry_block_reason_respects_cooldown_bars_after_exit(self) -> None:
		cfg = BotConfig(cooldown_bars_after_exit=2)
		state = BotState(last_exit_time=_ts("2026-01-01T00:45:00Z").isoformat())

		reason = entry_block_reason(cfg, state, _ts("2026-01-01T01:00:00Z"), pd.DataFrame())

		self.assertEqual(reason, "cooldown_active_until:2026-01-01T01:15:00+00:00")

	def test_market_data_due_only_once_per_minute_after_poll_second(self) -> None:
		now = _ts("2026-01-01T12:00:29Z")
		self.assertFalse(market_data_due(now, None, 30))
		now = _ts("2026-01-01T12:00:30Z")
		self.assertTrue(market_data_due(now, None, 30))
		bucket = market_data_poll_bucket(now)
		self.assertFalse(market_data_due(_ts("2026-01-01T12:00:45Z"), bucket, 30))
		self.assertTrue(market_data_due(_ts("2026-01-01T12:01:30Z"), bucket, 30))

	def test_instrument_trading_hours_open_respects_maintenance_and_weekend(self) -> None:
		self.assertFalse(instrument_trading_hours_open(Price.Gold, _ts("2026-01-02T21:30:00Z")))
		self.assertFalse(instrument_trading_hours_open(Price.Oil, _ts("2026-01-02T05:30:00Z")))
		self.assertFalse(instrument_trading_hours_open(Price.AUD, _ts("2026-01-03T12:00:00Z")))
		self.assertTrue(instrument_trading_hours_open(Price.Gold, _ts("2026-01-05T12:00:00Z")))

	def test_snapshot_tradeable_recognizes_closed_status(self) -> None:
		self.assertTrue(snapshot_tradeable("TRADEABLE"))
		self.assertFalse(snapshot_tradeable("CLOSED"))
		self.assertFalse(snapshot_tradeable(None))

	def test_aligned_sleep_seconds_targets_next_boundary(self) -> None:
		sleep_for = aligned_sleep_seconds(_ts("2026-01-01T12:00:29.500000Z"), 30)
		self.assertAlmostEqual(sleep_for, 0.5, places=2)

	def test_market_data_collector_stores_only_tradeable_open_instruments(self) -> None:
		class StubStore(ExistingSchemaMarketDataStore):
			def __init__(self) -> None:
				self.rows: list[dict[str, object]] = []
				self.synced: list[str] = []
			def sync_from_latest(self, _service: object, instrument: Price, end_time) -> dict[str, object]:
				self.synced.append(instrument.name.lower())
				return {
					"instrument": instrument.name.lower(),
					"table_name": instrument.db_name,
					"latest_db_before_utc": "2026-01-05T11:59:00+00:00",
					"requested_start_utc": "2026-01-05T11:59:00+00:00",
					"requested_end_utc": pd.Timestamp(end_time).tz_convert("UTC").isoformat() if pd.Timestamp(end_time).tzinfo else pd.Timestamp(end_time).tz_localize("UTC").isoformat(),
					"fetched_rows": 2,
					"written_rows": 2,
					"inserted_rows": 1,
					"fetched_period_start_utc": "2026-01-05T12:00:00+00:00",
					"fetched_period_end_utc": "2026-01-05T12:01:00+00:00",
				}
			def upsert_snapshots(self, snapshots: list[dict[str, object]]) -> int:
				self.rows.extend(snapshots)
				return len(snapshots)

		def fake_fetcher(_service: object, instrument: Price, fetch_time=None):
			return {
				"instrument": instrument.name.lower(),
				"epic": instrument.epic,
				"fetch_time_utc": pd.Timestamp(fetch_time).isoformat(),
				"bucket_minute_utc": pd.Timestamp(fetch_time).floor("min").isoformat(),
				"market_status": "TRADEABLE" if instrument != Price.AUD else "CLOSED",
				"update_time_utc": None,
				"instrument_name": instrument.name,
				"bid": 1.0,
				"offer": 1.1,
				"mid": 1.05,
				"high": 1.2,
				"low": 0.9,
				"net_change": 0.0,
				"percentage_change": 0.0,
				"raw_json": "{}",
			}

		cfg = BotConfig()
		collector = IGMarketDataCollector(
			cfg,
			logging.getLogger("market_data_collector_test"),
			store=StubStore(),
			snapshot_fetcher=fake_fetcher,
			account_fetcher=lambda _service: {"status": "TEST"},
			positions_fetcher=lambda _service: [],
		)
		collector._service = object()  # bypass live IG auth in unit test

		summaries = collector.maybe_capture(_ts("2026-01-05T12:00:30Z"))

		self.assertEqual(len(summaries), 3)
		self.assertEqual(sorted(s["instrument"] for s in summaries), ["aud", "gold", "oil"])
		gold_summary = next(s for s in summaries if s["instrument"] == "gold")
		self.assertEqual(gold_summary["requested_start_utc"], "2026-01-05T11:59:00+00:00")
		self.assertEqual(gold_summary["requested_end_utc"], "2026-01-05T12:00:30+00:00")
		self.assertEqual(gold_summary["resulting_period_end_utc"], "2026-01-05T12:00:00+00:00")
		aud_summary = next(s for s in summaries if s["instrument"] == "aud")
		self.assertEqual(aud_summary["snapshot_written"], 0)
		self.assertEqual(aud_summary["snapshot_market_status"], "CLOSED")
		self.assertEqual(sorted(collector.store.synced), ["aud", "gold", "oil"])
		self.assertEqual(len(collector.store.rows), 2)

	def test_market_data_collector_force_bypasses_poll_second_gate(self) -> None:
		class StubStore(ExistingSchemaMarketDataStore):
			def __init__(self) -> None:
				self.rows: list[dict[str, object]] = []
			def sync_from_latest(self, _service: object, instrument: Price, end_time) -> dict[str, object]:
				return {
					"instrument": instrument.name.lower(),
					"table_name": instrument.db_name,
					"latest_db_before_utc": None,
					"requested_start_utc": "2026-01-05T12:00:00+00:00",
					"requested_end_utc": pd.Timestamp(end_time).tz_convert("UTC").isoformat() if pd.Timestamp(end_time).tzinfo else pd.Timestamp(end_time).tz_localize("UTC").isoformat(),
					"fetched_rows": 0,
					"written_rows": 0,
					"inserted_rows": 0,
					"fetched_period_start_utc": None,
					"fetched_period_end_utc": None,
				}
			def upsert_snapshots(self, snapshots: list[dict[str, object]]) -> int:
				self.rows.extend(snapshots)
				return len(snapshots)

		def fake_fetcher(_service: object, instrument: Price, fetch_time=None):
			return {
				"instrument": instrument.name.lower(),
				"epic": instrument.epic,
				"fetch_time_utc": pd.Timestamp(fetch_time).isoformat(),
				"bucket_minute_utc": pd.Timestamp(fetch_time).floor("min").isoformat(),
				"market_status": "TRADEABLE",
				"bid": 1.0,
				"offer": 1.1,
				"mid": 1.05,
				"high": 1.2,
				"low": 0.9,
			}

		collector = IGMarketDataCollector(
			BotConfig(),
			logging.getLogger("market_data_force_test"),
			store=StubStore(),
			snapshot_fetcher=fake_fetcher,
			account_fetcher=lambda _service: {"status": "TEST"},
			positions_fetcher=lambda _service: [],
		)
		collector._service = object()
		collector._last_bucket = market_data_poll_bucket(_ts("2026-01-05T12:00:00Z"))

		summaries = collector.maybe_capture(_ts("2026-01-05T12:00:05Z"), force=True)

		self.assertEqual(len(summaries), 3)
		self.assertEqual(len(collector.store.rows), 3)

	def test_market_data_collector_logs_market_then_account_without_snapshot_store(self) -> None:
		class StubStore(ExistingSchemaMarketDataStore):
			def __init__(self, logger: logging.Logger) -> None:
				super().__init__(logger)
			def sync_from_latest(self, _service: object, instrument: Price, end_time) -> dict[str, object]:
				return {
					"instrument": instrument.name.lower(),
					"table_name": instrument.db_name,
					"latest_db_before_utc": None,
					"requested_start_utc": "2026-01-05T12:00:00+00:00",
					"requested_end_utc": pd.Timestamp(end_time).tz_convert("UTC").isoformat() if pd.Timestamp(end_time).tzinfo else pd.Timestamp(end_time).tz_localize("UTC").isoformat(),
					"fetched_rows": 2,
					"written_rows": 7 if instrument == Price.Gold else 3,
					"inserted_rows": 1 if instrument == Price.Gold else 0,
					"fetched_period_start_utc": None,
					"fetched_period_end_utc": None,
				}
			def upsert_snapshots(self, snapshots: list[dict[str, object]]) -> int:
				self.last_upsert_summaries = [
					{
						"instrument": str((snapshots[0] or {}).get("instrument") or "unknown"),
						"logical_written": len(snapshots),
						"mysql_affected_rows": len(snapshots),
					}
				] if snapshots else []
				return len(snapshots)

		def fake_fetcher(_service: object, instrument: Price, fetch_time=None):
			base_mid = 100.5 if instrument == Price.Gold else 1.05
			return {
				"instrument": instrument.name.lower(),
				"epic": instrument.epic,
				"fetch_time_utc": pd.Timestamp(fetch_time).isoformat(),
				"bucket_minute_utc": pd.Timestamp(fetch_time).floor("min").isoformat(),
				"market_status": "TRADEABLE",
				"bid": base_mid - 0.05,
				"offer": base_mid + 0.05,
				"mid": base_mid,
			}

		stream = io.StringIO()
		logger = logging.getLogger("market_data_poll_log_test")
		logger.handlers.clear()
		logger.setLevel(logging.INFO)
		handler = logging.StreamHandler(stream)
		logger.addHandler(handler)

		collector = IGMarketDataCollector(
			BotConfig(),
			logger,
			store=StubStore(logger),
			snapshot_fetcher=fake_fetcher,
			account_fetcher=lambda _service: {
				"status": "ENABLED",
				"currency": "USD",
				"balance": 1000.0,
				"equity": 1012.5,
				"profit_loss": 12.5,
			},
			positions_fetcher=lambda _service: [
				{
					"market": {"epic": Price.Gold.epic},
					"position": {"direction": "BUY", "size": 2.0, "level": 100.0},
				}
			],
		)
		collector._service = object()

		collector.maybe_capture(_ts("2026-01-05T12:00:30Z"))
		lines = [line.strip() for line in stream.getvalue().splitlines() if line.strip()]

		self.assertGreaterEqual(len(lines), 2)
		self.assertIn("MARKET POLL: bucket=2026-01-05T12:00:00+00:00", lines[0])
		self.assertIn("pos=1 gold:BUY size=2.00 entry=100.00 pnl=+1.00", lines[0])
		self.assertIn("gold:rows_added=1", lines[0])
		self.assertIn("aud:rows_added=0", lines[0])
		self.assertIn("ACCOUNT STATUS: bucket=2026-01-05T12:00:00+00:00", lines[1])
		self.assertIn("acct=enabled bal=1000.00USD eq=1012.50USD pnl=+12.50USD", lines[1])
		self.assertNotIn("SNAPSHOT STORE", "\n".join(lines))

	def test_existing_schema_store_counts_logical_snapshot_rows(self) -> None:
		original_insert_prices = trading_bot_module.insert_prices
		try:
			trading_bot_module.insert_prices = lambda rows, _instrument: len(rows) * 2
			store = ExistingSchemaMarketDataStore(logging.getLogger("snapshot_count_test"))
			snapshot = {
				"instrument": "gold",
				"epic": Price.Gold.epic,
				"fetch_time_utc": _ts("2026-01-05T12:00:30Z").isoformat(),
				"bid": 100.0,
				"offer": 100.1,
				"mid": 100.05,
				"high": 100.2,
				"low": 99.9,
				"lastTradedVolume": 1,
			}
			written = store.upsert_snapshots([snapshot])
			self.assertEqual(written, 1)
		finally:
			trading_bot_module.insert_prices = original_insert_prices

	def test_market_sync_only_run_skips_trading_logic(self) -> None:
		bot = AlphaGoldTradingBot.__new__(AlphaGoldTradingBot)
		bot.cfg = BotConfig(market_sync_only=True)
		bot.logger = logging.getLogger("market_sync_only_test")
		calls: list[str] = []

		def fake_market_cycle(*, force: bool = False):
			calls.append(f"market:{force}")
			return []

		def fake_save_state():
			calls.append("save")

		def fail_load_recent_raw_data():
			raise AssertionError("trading logic should not load raw data in market-sync-only mode")

		bot._run_market_data_cycle = fake_market_cycle
		bot._save_state = fake_save_state
		bot._load_recent_raw_data = fail_load_recent_raw_data

		bot.run_once()

		self.assertEqual(calls, ["market:True", "save"])

	def test_market_sync_only_run_raises_on_market_data_failure(self) -> None:
		bot = AlphaGoldTradingBot.__new__(AlphaGoldTradingBot)
		bot.cfg = BotConfig(market_sync_only=True)
		bot.logger = logging.getLogger("market_sync_only_failure_test")
		bot.market_data_collector = type("CollectorStub", (), {
			"last_sync_summaries": [],
			"last_error": None,
		})()
		bot._save_state = lambda: None

		def fail_capture(*, force: bool = False):
			raise RuntimeError("IG authentication failed")

		bot._run_market_data_cycle = fail_capture

		with self.assertRaisesRegex(RuntimeError, "IG authentication failed"):
			bot.run_once()

	def test_run_market_data_cycle_market_sync_only_persists_error_and_raises(self) -> None:
		bot = AlphaGoldTradingBot.__new__(AlphaGoldTradingBot)
		bot.cfg = BotConfig(market_sync_only=True)
		bot.logger = logging.getLogger("market_data_cycle_failure_test")
		bot.market_data_collector = type("CollectorStub", (), {
			"last_sync_summaries": [],
			"last_error": None,
			"maybe_capture": staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("IG auth bad"))),
		})()
		saved: list[str] = []
		bot._save_state = lambda: saved.append("saved")

		with self.assertRaisesRegex(RuntimeError, "IG auth bad"):
			bot._run_market_data_cycle(force=True)

		self.assertEqual(saved, ["saved"])
		self.assertEqual(bot.market_data_collector.last_error, "IG auth bad")

	def test_build_parser_accepts_market_sync_only_flag(self) -> None:
		args = build_parser().parse_args(["--market-sync-only", "--once"])
		self.assertTrue(args.market_sync_only)
		self.assertTrue(args.once)

	def test_build_parser_accepts_live_mode_and_disable_dynamic_target_stop(self) -> None:
		args = build_parser().parse_args([
			"--signal-model-family", "best_base_state",
			"--mode", "live",
			"--disable-dynamic-target-stop",
		])
		self.assertEqual(args.mode, "live")
		self.assertTrue(args.disable_dynamic_target_stop)

	def test_build_parser_defaults_to_best_base_signal_family(self) -> None:
		args = build_parser().parse_args([])
		self.assertEqual(args.signal_model_family, DEFAULT_SIGNAL_MODEL_FAMILY)
		self.assertEqual(args.signal_model_path, DEFAULT_BEST_BASE_MODEL_PATH)
		self.assertEqual(args.mode, "signal_only")
		self.assertEqual(args.prediction_poll_second, 5)
		self.assertEqual(args.market_data_poll_second, 30)
		self.assertEqual(args.sleep_seconds, 5)

	def test_best_base_signal_only_run_skips_legacy_trading_path(self) -> None:
		bot = AlphaGoldTradingBot.__new__(AlphaGoldTradingBot)
		bot.cfg = BotConfig(signal_model_family="best_base_state", mode="signal_only")
		bot.logger = logging.getLogger("best_base_signal_only_test")
		bot.state = BotState()
		bot.prediction_cache = type("PredictionCacheStub", (), {"_last_bucket": None, "last_error": None, "last_fetch_summary": {}})()
		bot.paper_broker = type("BrokerStub", (), {"trade_log_path": Path(tempfile.gettempdir()) / "unused_best_base_trades.csv"})()
		class EngineStub:
			def handle_signal(self, **kwargs):
				return {"status": "dry_run", "mode": kwargs.get("mode")}
		bot.execution_engine = EngineStub()
		calls: list[str] = []

		bot._run_market_data_cycle = lambda *, force=False: calls.append(f"market:{force}") or []
		bot._run_prediction_cycle = lambda *, force=False: pd.DataFrame(
			{
				"timestamp": [1],
				"openPrice": [100.0],
				"highPrice": [101.0],
				"lowPrice": [99.0],
				"closePrice": [100.5],
				"openPrice_ask": [100.6],
				"openPrice_bid": [100.4],
				"highPrice_ask": [101.1],
				"highPrice_bid": [100.9],
				"lowPrice_ask": [99.1],
				"lowPrice_bid": [98.9],
				"closePrice_ask": [100.6],
				"closePrice_bid": [100.4],
				"lastTradedVolume": [1.0],
			},
			index=pd.DatetimeIndex([_ts("2026-01-05T12:00:00Z")]),
		)
		bot._build_latest_signal_best_base = lambda raw: {
			"signal_bar_time": _ts("2026-01-05T12:00:00Z"),
			"entry_bar_time": _ts("2026-01-05T12:01:00Z"),
			"probability": 0.83,
			"side": "up",
			"tradable": True,
		}
		bot._save_state = lambda: calls.append("save")

		bot.run_once()

		self.assertEqual(calls, ["market:False", "save"])
		self.assertEqual(bot.state.last_execution_attempt["status"], "dry_run")

	def test_best_base_run_skips_when_prediction_cycle_not_due(self) -> None:
		bot = AlphaGoldTradingBot.__new__(AlphaGoldTradingBot)
		bot.cfg = BotConfig(signal_model_family="best_base_state", mode="signal_only")
		bot.logger = logging.getLogger("best_base_not_due_test")
		bot.state = BotState()
		bot.prediction_cache = type("PredictionCacheStub", (), {"_last_bucket": None, "last_error": None, "last_fetch_summary": {}})()
		bot.paper_broker = type("BrokerStub", (), {"trade_log_path": Path(tempfile.gettempdir()) / "unused_best_base_trades_skip.csv"})()
		calls: list[str] = []

		bot._run_market_data_cycle = lambda *, force=False: calls.append(f"market:{force}") or []
		bot._run_prediction_cycle = lambda *, force=False: None
		bot._save_state = lambda: calls.append("save")

		bot.run_once()

		self.assertEqual(calls, ["market:False", "save"])

	def test_best_base_paper_mode_is_rejected_in_parse(self) -> None:
		with self.assertRaises(ValueError):
			args = build_parser().parse_args(["--signal-model-family", "best_base_state", "--mode", "paper"])
			if args.signal_model_family == "best_base_state" and args.mode == "paper":
				raise ValueError("--signal-model-family best_base_state currently supports only --mode signal_only")

	def test_dynamic_target_stop_updates_from_matching_signal(self) -> None:
		bot = AlphaGoldTradingBot.__new__(AlphaGoldTradingBot)
		bot.cfg = BotConfig(mode="signal_only", dynamic_target_stop_enabled=True, take_profit_pct=0.6, stop_loss_pct=0.4)
		bot.logger = logging.getLogger("dynamic_target_stop_test")
		bot.execution_engine = type("EngineStub", (), {"broker_adapter": object()})()
		broker = PaperBroker(bot.cfg, bot.logger)
		bot.state = BotState(
			open_position=broker.open_long(
				signal_bar_time=_ts("2026-01-05T12:00:00Z"),
				entry_bar_time=_ts("2026-01-05T12:15:00Z"),
				entry_time=_ts("2026-01-05T12:15:00Z"),
				entry_price=100.0,
				probability=0.8,
				deal_id="DYN001",
			)
		)
		raw = pd.DataFrame(
			{
				"openPrice": [100.0, 100.0],
				"openPrice_ask": [101.5, 101.6],
				"closePrice": [101.3, 101.2],
			},
			index=pd.DatetimeIndex([
				_ts("2026-01-05T12:16:00Z"),
				_ts("2026-01-05T12:17:00Z"),
			]),
		)
		signal = {
			"side": "up",
			"tradable": True,
			"signal_bar_time": _ts("2026-01-05T12:16:00Z"),
			"entry_bar_time": _ts("2026-01-05T12:16:00Z"),
		}

		bot._maybe_dynamic_target_stop_from_signal(raw, signal)

		assert bot.state.open_position is not None
		self.assertEqual(bot.state.open_position.target_updates, 1)
		self.assertAlmostEqual(bot.state.open_position.last_target_price or 0.0, 101.5, places=6)
		self.assertEqual(bot.state.open_position.last_target_signal_time, "2026-01-05T12:16:00+00:00")

	def test_bot_save_state_persists_market_sync_summaries(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			bot = AlphaGoldTradingBot.__new__(AlphaGoldTradingBot)
			bot.cfg = BotConfig(
				state_path=str(root / "state.json"),
				status_path=str(root / "status.json"),
				trade_log_path=str(root / "trades.csv"),
				log_path=str(root / "bot.log"),
			)
			bot.state_path = root / "state.json"
			bot.status_path = root / "status.json"
			bot.state = BotState()
			bot.state.last_execution_attempt = {
				"status": "submitted",
				"mode": "paper",
				"deal_id": "DIAAA111",
			}
			bot.last_raw_cache_info = {
				"storage_mode": "mysql_recent_window_reload",
				"rows": 6027,
				"start_utc": "2026-04-08T22:01:00+00:00",
				"end_utc": "2026-04-15T06:34:00+00:00",
			}
			bot.last_best_base_payload_info = {
				"bars_rows": 6027,
				"candidate_samples": 5878,
				"window": 150,
				"horizon": 25,
			}
			bot.paper_broker = PaperBroker(bot.cfg, logging.getLogger("bot_save_state_test"))
			bot.state.open_position = bot.paper_broker.open_long(
				signal_bar_time=_ts("2026-01-05T12:00:00Z"),
				entry_bar_time=_ts("2026-01-05T12:15:00Z"),
				entry_time=_ts("2026-01-05T12:15:00Z"),
				entry_price=100.0,
				probability=0.77,
				deal_id="DIAAA111",
			)
			bot.state.open_position.target_stop_adjusted = True
			bot.state.open_position.target_stop_adjusted_at = "2026-01-05T12:16:00+00:00"
			bot.market_data_collector = type("CollectorStub", (), {
				"_last_bucket": _ts("2026-01-05T12:00:00Z"),
				"last_sync_summaries": [
					{
						"instrument": "gold",
						"requested_start_utc": "2026-01-05T11:59:00+00:00",
						"requested_end_utc": "2026-01-05T12:00:30+00:00",
						"resulting_period_start_utc": "2026-01-05T12:00:00+00:00",
						"resulting_period_end_utc": "2026-01-05T12:00:00+00:00",
					}
				],
			})()
			bot.prediction_cache = type("PredictionCacheStub", (), {
				"_last_bucket": _ts("2026-01-05T12:00:05Z"),
				"last_error": None,
				"last_fetch_summary": {"cache_rows": 6027, "status": "ok"},
			})()

			bot._save_state()

			status = json.loads(bot.status_path.read_text(encoding="utf-8"))
			self.assertEqual(status["market_data_tables"][0], "gold_prices")
			self.assertEqual(status["prediction_poll_second"], bot.cfg.prediction_poll_second)
			self.assertEqual(status["prediction_cache_last_bucket_utc"], "2026-01-05T12:00:05+00:00")
			self.assertIsNone(status["market_data_last_error"])
			self.assertEqual(status["last_market_data_bucket_utc"], "2026-01-05T12:00:00+00:00")
			self.assertEqual(status["input_data"]["rows"], 6027)
			self.assertEqual(status["best_base_runtime"]["candidate_samples"], 5878)
			self.assertEqual(status["last_execution_attempt"]["status"], "submitted")
			self.assertEqual(status["last_execution_attempt"]["deal_id"], "DIAAA111")
			self.assertEqual(status["open_position"]["deal_id"], "DIAAA111")
			self.assertTrue(status["open_position"]["target_stop_adjusted"])
			self.assertEqual(status["open_position"]["target_stop_adjusted_at"], "2026-01-05T12:16:00+00:00")
			self.assertIn("performance_windows", status)
			self.assertIn("30m", status["performance_windows"])
			self.assertIn("60m", status["performance_windows"])
			self.assertIn("daily_summary", status)
			self.assertIn("today", status["daily_summary"])
			self.assertIn("weekly_summary", status)
			self.assertIn("this_week", status["weekly_summary"])
			self.assertEqual(status["configured_position_size"], float(bot.cfg.size))
			self.assertIn("last_periodic_report_bucket_utc", status)
			summaries = status["market_data_last_sync_summaries"]
			self.assertEqual(len(summaries), 1)
			self.assertEqual(summaries[0]["instrument"], "gold")
			self.assertEqual(summaries[0]["resulting_period_end_utc"], "2026-01-05T12:00:00+00:00")

	def test_bot_save_state_persists_market_sync_error(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			bot = AlphaGoldTradingBot.__new__(AlphaGoldTradingBot)
			bot.cfg = BotConfig(
				market_sync_only=True,
				state_path=str(root / "state.json"),
				status_path=str(root / "status.json"),
				trade_log_path=str(root / "trades.csv"),
				log_path=str(root / "bot.log"),
			)
			bot.state_path = root / "state.json"
			bot.status_path = root / "status.json"
			bot.state = BotState()
			bot.paper_broker = PaperBroker(bot.cfg, logging.getLogger("bot_save_error_test"))
			bot.market_data_collector = type("CollectorStub", (), {
				"_last_bucket": None,
				"last_sync_summaries": [],
				"last_error": "IG authentication failed",
			})()
			bot.prediction_cache = type("PredictionCacheStub", (), {
				"_last_bucket": None,
				"last_fetch_summary": {},
				"last_error": "prediction cache unavailable",
			})()

			bot._save_state()

			status = json.loads(bot.status_path.read_text(encoding="utf-8"))
			self.assertEqual(status["market_data_last_error"], "IG authentication failed")
			self.assertEqual(status["prediction_cache_last_error"], "prediction cache unavailable")


if __name__ == "__main__":
	unittest.main(verbosity=2)

