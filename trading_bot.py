#!/usr/bin/env python3
"""Paper-trading bot scaffold for AlphaGold.

This first version is intentionally conservative:
- reads recent candles from MySQL using the workspace's DataLoader
- aggregates to 15-minute bars with the existing ML pipeline
- loads the current 15-minute next-bar model artifacts
- generates one long-only signal on the latest completed 15-minute bar
- enters on the next bar open in paper mode
- manages exits with TP / SL / timeout rules aligned to the backtest

It defaults to paper/signal-only use because this repo does not yet expose the
full live IG order-management layer referenced by old_bot.py.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "training") not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT / "training"))

from data import DataLoader
from brokers.base import DryRunBrokerAdapter
from brokers.ig_live import IGLiveBrokerAdapter
from execution.engine import ExecutionEngine
from ig_scripts.ig_data_api import (
	API_CONFIG,
	fetch_prices,
	fetch_and_store_prices_from_latest,
	fetch_open_positions,
	fetch_primary_account_summary,
	IGService,
	Price,
	fetch_market_snapshot,
	insert_prices,
	snapshot_to_price_row,
)


UTC = timezone.utc
NY_TZ = ZoneInfo("America/New_York")
TRADING_DAY_CUTOFF_HOUR_NY = 17
BAR_INTERVAL = pd.Timedelta(minutes=15)
MIN_FEATURE_BARS = 120
DEFAULT_SIGNAL_MODEL_FAMILY = "best_base_state"
DEFAULT_MODEL_DIR = "training/ml_models_15m_nextbar_060_corr"
DEFAULT_BEST_BASE_MODEL_PATH = "runtime/bot_assets/backtest_model_best_base_weak_nostate.joblib"
DEFAULT_WEAK_PERIODS_JSON = "runtime/bot_assets/weak-filter.json"
DEFAULT_MAX_ADVERSE_LOW_PCT = 0.40
DEFAULT_FORWARD_BARS = 4
DEFAULT_THRESHOLD_PCT = 0.60
PREDICTION_POLL_SECOND = 5
MARKET_DATA_POLL_SECOND = 30
TRADEABLE_MARKET_STATUSES = {"TRADEABLE", "EDIT", "ONLINE", "ON_AUCTION", "DEAL_NO_EDIT"}
MARKET_DATA_INSTRUMENTS = (Price.Gold, Price.AUD, Price.Oil)
MARKET_DATA_TABLES = tuple(instrument.db_name for instrument in MARKET_DATA_INSTRUMENTS)


def _load_15m_nextbar_module():
	last_exc: Optional[Exception] = None
	for module_name in ("training.ml_alpha_model_15m_nextbar", "ml_alpha_model_15m_nextbar"):
		try:
			return importlib.import_module(module_name)
		except ModuleNotFoundError as exc:
			last_exc = exc
	raise ModuleNotFoundError(
		"Could not import ml_alpha_model_15m_nextbar. Expected either 'training.ml_alpha_model_15m_nextbar' "
		"or 'ml_alpha_model_15m_nextbar' to be importable."
	) from last_exc


def _load_image_trend_module():
	last_exc: Optional[Exception] = None
	for module_name in ("training.image_trend_ml", "image_trend_ml"):
		try:
			return importlib.import_module(module_name)
		except ModuleNotFoundError as exc:
			last_exc = exc
	raise ModuleNotFoundError(
		"Could not import image_trend_ml. Expected either 'training.image_trend_ml' or 'image_trend_ml' to be importable."
	) from last_exc


@dataclass
class BotConfig:
	table: str = "gold_prices"
	signal_model_family: str = DEFAULT_SIGNAL_MODEL_FAMILY
	signal_model_path: str = DEFAULT_BEST_BASE_MODEL_PATH
	model_type: str = "gradient_boosting"
	model_dir: str = DEFAULT_MODEL_DIR
	market_sync_only: bool = False
	recent_days: int = 10
	probability_cutoff: float = 0.50
	take_profit_pct: float = DEFAULT_THRESHOLD_PCT
	stop_loss_pct: float = DEFAULT_MAX_ADVERSE_LOW_PCT
	max_hold_bars: int = DEFAULT_FORWARD_BARS
	size: float = 1.0
	mode: str = "signal_only"  # paper | signal_only | live
	once: bool = False
	sleep_seconds: int = 30
	state_path: str = "runtime/trading_bot_state.json"
	status_path: str = "runtime/trading_bot_status.json"
	trade_log_path: str = "runtime/trading_bot_trades.csv"
	log_path: str = "runtime/trading_bot.log"
	lock_path: str = "/tmp/alphagold_trading_bot.lock"
	max_trades_per_day: int = 0
	cooldown_bars_after_exit: int = 0
	market_data_enabled: bool = True
	prediction_poll_second: int = PREDICTION_POLL_SECOND
	market_data_poll_second: int = MARKET_DATA_POLL_SECOND
	prediction_cache_max_rows: int = 1200
	weak_periods_json: Optional[str] = DEFAULT_WEAK_PERIODS_JSON
	dynamic_target_stop_enabled: bool = True
	max_hold_minutes: Optional[float] = None  # hard timeout for live positions (60 = Candidate E)


@dataclass
class PaperPosition:
	direction: str
	deal_id: str
	signal_bar_time: str
	entry_bar_time: str
	entry_time: str
	entry_price: float
	stop_loss: float
	take_profit: float
	probability: float
	size: float
	entry_price_initial: Optional[float] = None
	target_stop_adjusted: bool = False
	target_stop_adjusted_at: Optional[str] = None
	target_updates: int = 0
	last_target_signal_time: Optional[str] = None
	last_target_price: Optional[float] = None
	bars_checked: int = 0


@dataclass
class BotState:
	last_signal_bar_time: Optional[str] = None
	open_position: Optional[PaperPosition] = None
	last_exit_time: Optional[str] = None
	last_exit_reason: Optional[str] = None
	last_execution_attempt: Optional[dict[str, object]] = None
	last_periodic_report_bucket_utc: Optional[str] = None

	def to_dict(self) -> dict[str, object]:
		return {
			"last_signal_bar_time": self.last_signal_bar_time,
			"open_position": asdict(self.open_position) if self.open_position else None,
			"last_exit_time": self.last_exit_time,
			"last_exit_reason": self.last_exit_reason,
			"last_execution_attempt": self.last_execution_attempt,
			"last_periodic_report_bucket_utc": self.last_periodic_report_bucket_utc,
		}

	@classmethod
	def from_dict(cls, raw: dict[str, object]) -> "BotState":
		pos_raw = raw.get("open_position")
		if isinstance(pos_raw, dict) and not pos_raw.get("deal_id"):
			pos_raw = dict(pos_raw)
			pos_raw["deal_id"] = "paper-legacy"
		return cls(
			last_signal_bar_time=raw.get("last_signal_bar_time"),
			open_position=PaperPosition(**pos_raw) if isinstance(pos_raw, dict) else None,
			last_exit_time=raw.get("last_exit_time"),
			last_exit_reason=raw.get("last_exit_reason"),
			last_execution_attempt=raw.get("last_execution_attempt") if isinstance(raw.get("last_execution_attempt"), dict) else None,
			last_periodic_report_bucket_utc=str(raw.get("last_periodic_report_bucket_utc")) if raw.get("last_periodic_report_bucket_utc") else None,
		)


def trading_day_label(ts: pd.Timestamp) -> pd.Timestamp:
	ts_utc = pd.Timestamp(ts)
	if ts_utc.tzinfo is None:
		ts_utc = ts_utc.tz_localize(UTC)
	else:
		ts_utc = ts_utc.tz_convert(UTC)
	return pd.Timestamp((ts_utc.tz_convert(NY_TZ) - pd.Timedelta(hours=TRADING_DAY_CUTOFF_HOUR_NY)).floor("D"))


def load_trade_log(trade_log_path: Path) -> pd.DataFrame:
	if not trade_log_path.exists():
		return pd.DataFrame()
	trades = pd.read_csv(trade_log_path)
	if trades.empty:
		return trades
	for col in ("signal_bar_time", "entry_bar_time", "entry_time", "exit_time"):
		if col in trades.columns:
			trades[col] = pd.to_datetime(trades[col], utc=True)
	if "pnl_usd" in trades.columns:
		trades["pnl_usd"] = pd.to_numeric(trades["pnl_usd"], errors="coerce").fillna(0.0)
	return trades


def summarize_trade_log(trades: pd.DataFrame, now_ts: pd.Timestamp) -> dict[str, object]:
	now_label = trading_day_label(now_ts).date().isoformat()
	if trades.empty:
		return {
			"trading_day": now_label,
			"total_trades": 0,
			"wins": 0,
			"losses": 0,
			"breakeven": 0,
			"realized_pnl_usd": 0.0,
			"today_trades": 0,
			"today_realized_pnl_usd": 0.0,
			"last_trade": None,
		}

	trades = trades.copy()
	entry_label = trades["entry_time"].apply(trading_day_label) if "entry_time" in trades.columns else pd.Series(dtype="datetime64[ns, UTC]")
	exit_label = trades["exit_time"].apply(trading_day_label) if "exit_time" in trades.columns else pd.Series(dtype="datetime64[ns, UTC]")
	pnl = pd.to_numeric(trades.get("pnl_usd", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
	today_entries = int((entry_label == trading_day_label(now_ts)).sum()) if len(entry_label) else 0
	today_realized = float(pnl[exit_label == trading_day_label(now_ts)].sum()) if len(exit_label) else 0.0
	last_trade = trades.iloc[-1].to_dict()
	for key, value in list(last_trade.items()):
		if isinstance(value, pd.Timestamp):
			last_trade[key] = value.isoformat()
	return {
		"trading_day": now_label,
		"total_trades": int(len(trades)),
		"wins": int((pnl > 0).sum()),
		"losses": int((pnl < 0).sum()),
		"breakeven": int((pnl == 0).sum()),
		"realized_pnl_usd": float(pnl.sum()),
		"today_trades": today_entries,
		"today_realized_pnl_usd": today_realized,
		"last_trade": last_trade,
	}


def summarize_trade_windows(
	trades: pd.DataFrame,
	now_ts: pd.Timestamp,
	*,
	windows_minutes: tuple[int, ...] = (30, 60),
) -> dict[str, dict[str, object]]:
	now_utc = pd.Timestamp(now_ts)
	if now_utc.tzinfo is None:
		now_utc = now_utc.tz_localize(UTC)
	else:
		now_utc = now_utc.tz_convert(UTC)

	def _empty_window(window_min: int) -> dict[str, object]:
		start_utc = now_utc - pd.Timedelta(minutes=int(window_min))
		return {
			"window_minutes": int(window_min),
			"window_start_utc": start_utc.isoformat(),
			"window_end_utc": now_utc.isoformat(),
			"trades": 0,
			"wins": 0,
			"losses": 0,
			"breakeven": 0,
			"realized_pnl_usd": 0.0,
			"avg_trade_pnl_usd": 0.0,
			"win_rate_pct": 0.0,
			"last_exit_utc": None,
		}

	if trades.empty or "exit_time" not in trades.columns:
		return {f"{int(window)}m": _empty_window(int(window)) for window in windows_minutes}

	t = trades.copy()
	t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
	t = t.dropna(subset=["exit_time"])
	if t.empty:
		return {f"{int(window)}m": _empty_window(int(window)) for window in windows_minutes}

	pnl_all = pd.to_numeric(t.get("pnl_usd", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
	windows: dict[str, dict[str, object]] = {}
	for window in windows_minutes:
		window_min = int(window)
		start_utc = now_utc - pd.Timedelta(minutes=window_min)
		mask = (t["exit_time"] >= start_utc) & (t["exit_time"] <= now_utc)
		tw = t.loc[mask]
		if tw.empty:
			windows[f"{window_min}m"] = _empty_window(window_min)
			continue
		pnl = pnl_all.loc[tw.index]
		trades_n = int(len(tw))
		wins = int((pnl > 0).sum())
		losses = int((pnl < 0).sum())
		breakeven = int((pnl == 0).sum())
		windows[f"{window_min}m"] = {
			"window_minutes": window_min,
			"window_start_utc": start_utc.isoformat(),
			"window_end_utc": now_utc.isoformat(),
			"trades": trades_n,
			"wins": wins,
			"losses": losses,
			"breakeven": breakeven,
			"realized_pnl_usd": float(pnl.sum()),
			"avg_trade_pnl_usd": float(pnl.mean()) if trades_n else 0.0,
			"win_rate_pct": float((wins / trades_n) * 100.0) if trades_n else 0.0,
			"last_exit_utc": pd.Timestamp(tw["exit_time"].max()).isoformat(),
		}
	return windows


def _aggregate_closed_trade_slice(
	trades: pd.DataFrame,
	*,
	start_label: pd.Timestamp,
	end_label: pd.Timestamp,
) -> dict[str, object]:
	if trades.empty or "exit_time" not in trades.columns:
		return {
			"trades": 0,
			"wins": 0,
			"losses": 0,
			"breakeven": 0,
			"realized_pnl_usd": 0.0,
			"avg_trade_pnl_usd": 0.0,
			"win_rate_pct": 0.0,
		}
	t = trades.copy()
	t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
	t = t.dropna(subset=["exit_time"])
	if t.empty:
		return {
			"trades": 0,
			"wins": 0,
			"losses": 0,
			"breakeven": 0,
			"realized_pnl_usd": 0.0,
			"avg_trade_pnl_usd": 0.0,
			"win_rate_pct": 0.0,
		}
	exit_labels = t["exit_time"].apply(trading_day_label)
	mask = (exit_labels >= start_label) & (exit_labels <= end_label)
	tw = t.loc[mask]
	if tw.empty:
		return {
			"trades": 0,
			"wins": 0,
			"losses": 0,
			"breakeven": 0,
			"realized_pnl_usd": 0.0,
			"avg_trade_pnl_usd": 0.0,
			"win_rate_pct": 0.0,
		}
	pnl = pd.to_numeric(tw.get("pnl_usd", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
	trades_n = int(len(tw))
	wins = int((pnl > 0).sum())
	losses = int((pnl < 0).sum())
	breakeven = int((pnl == 0).sum())
	return {
		"trades": trades_n,
		"wins": wins,
		"losses": losses,
		"breakeven": breakeven,
		"realized_pnl_usd": float(pnl.sum()),
		"avg_trade_pnl_usd": float(pnl.mean()) if trades_n else 0.0,
		"win_rate_pct": float((wins / trades_n) * 100.0) if trades_n else 0.0,
	}


def summarize_daily_trade_log(trades: pd.DataFrame, now_ts: pd.Timestamp) -> dict[str, object]:
	now_label = trading_day_label(now_ts)
	prev_label = now_label - pd.Timedelta(days=1)
	trailing_start = now_label - pd.Timedelta(days=6)
	return {
		"trading_day": now_label.date().isoformat(),
		"today": {
			"trading_day": now_label.date().isoformat(),
			**_aggregate_closed_trade_slice(trades, start_label=now_label, end_label=now_label),
		},
		"previous_day": {
			"trading_day": prev_label.date().isoformat(),
			**_aggregate_closed_trade_slice(trades, start_label=prev_label, end_label=prev_label),
		},
		"trailing_7d": {
			"start_trading_day": trailing_start.date().isoformat(),
			"end_trading_day": now_label.date().isoformat(),
			**_aggregate_closed_trade_slice(trades, start_label=trailing_start, end_label=now_label),
		},
	}


def summarize_weekly_trade_log(trades: pd.DataFrame, now_ts: pd.Timestamp) -> dict[str, object]:
	now_label = trading_day_label(now_ts)
	this_week_start = now_label - pd.Timedelta(days=int(now_label.weekday()))
	this_week_end = this_week_start + pd.Timedelta(days=6)
	prev_week_start = this_week_start - pd.Timedelta(days=7)
	prev_week_end = this_week_start - pd.Timedelta(days=1)
	return {
		"week_start_trading_day": this_week_start.date().isoformat(),
		"week_end_trading_day": this_week_end.date().isoformat(),
		"this_week": {
			"start_trading_day": this_week_start.date().isoformat(),
			"end_trading_day": this_week_end.date().isoformat(),
			**_aggregate_closed_trade_slice(trades, start_label=this_week_start, end_label=this_week_end),
		},
		"previous_week": {
			"start_trading_day": prev_week_start.date().isoformat(),
			"end_trading_day": prev_week_end.date().isoformat(),
			**_aggregate_closed_trade_slice(trades, start_label=prev_week_start, end_label=prev_week_end),
		},
	}


def entry_block_reason(
	cfg: BotConfig,
	state: BotState,
	signal_bar_time: pd.Timestamp,
	trades: pd.DataFrame,
) -> Optional[str]:
	if cfg.max_trades_per_day > 0 and not trades.empty and "entry_time" in trades.columns:
		today_entries = int((trades["entry_time"].apply(trading_day_label) == trading_day_label(signal_bar_time)).sum())
		if today_entries >= cfg.max_trades_per_day:
			return f"max_trades_per_day_reached:{today_entries}"

	if cfg.cooldown_bars_after_exit > 0 and state.last_exit_time:
		last_exit = pd.Timestamp(state.last_exit_time)
		if last_exit.tzinfo is None:
			last_exit = last_exit.tz_localize(UTC)
		else:
			last_exit = last_exit.tz_convert(UTC)
		next_allowed_signal = last_exit + (BAR_INTERVAL * cfg.cooldown_bars_after_exit)
		if signal_bar_time < next_allowed_signal:
			return f"cooldown_active_until:{next_allowed_signal.isoformat()}"

	return None


def market_data_poll_bucket(ts: pd.Timestamp) -> pd.Timestamp:
	ts_utc = pd.Timestamp(ts)
	if ts_utc.tzinfo is None:
		ts_utc = ts_utc.tz_localize(UTC)
	else:
		ts_utc = ts_utc.tz_convert(UTC)
	return ts_utc.floor("min")


def market_data_due(now_ts: pd.Timestamp, last_bucket: Optional[pd.Timestamp], poll_second: int) -> bool:
	if poll_second < 0 or poll_second > 59:
		raise ValueError("poll_second must be within [0, 59]")
	now_utc = pd.Timestamp(now_ts)
	if now_utc.tzinfo is None:
		now_utc = now_utc.tz_localize(UTC)
	else:
		now_utc = now_utc.tz_convert(UTC)
	bucket = market_data_poll_bucket(now_utc)
	if now_utc.second < poll_second:
		return False
	if last_bucket is None:
		return True
	last_utc = pd.Timestamp(last_bucket)
	if last_utc.tzinfo is None:
		last_utc = last_utc.tz_localize(UTC)
	else:
		last_utc = last_utc.tz_convert(UTC)
	return bucket > last_utc


def instrument_trading_hours_open(instrument: Price, now_ts: pd.Timestamp) -> bool:
	now_utc = pd.Timestamp(now_ts)
	if now_utc.tzinfo is None:
		now_utc = now_utc.tz_localize(UTC)
	else:
		now_utc = now_utc.tz_convert(UTC)
	wd = now_utc.weekday()
	minute_of_day = now_utc.hour * 60 + now_utc.minute

	if instrument == Price.AUD:
		if wd == 5:
			return False
		if wd == 6 and minute_of_day < 22 * 60:
			return False
		if wd == 4 and minute_of_day >= 22 * 60:
			return False
		return True

	if wd == 5:
		return False
	if wd == 6 and minute_of_day < 22 * 60:
		return False
	if wd == 4 and minute_of_day >= 21 * 60:
		return False
	if instrument == Price.Gold and 21 * 60 <= minute_of_day < 22 * 60:
		return False
	if instrument == Price.Oil and ((5 * 60 <= minute_of_day < 6 * 60) or (21 * 60 <= minute_of_day < 22 * 60)):
		return False
	return True


def next_trading_open_utc(instrument: Price, now_ts: pd.Timestamp, *, max_scan_hours: int = 168) -> pd.Timestamp:
	now_utc = pd.Timestamp(now_ts)
	if now_utc.tzinfo is None:
		now_utc = now_utc.tz_localize(UTC)
	else:
		now_utc = now_utc.tz_convert(UTC)
	if instrument_trading_hours_open(instrument, now_utc):
		return now_utc
	probe = now_utc.ceil("min")
	for _ in range(max_scan_hours * 60):
		if instrument_trading_hours_open(instrument, probe):
			return probe
		probe += pd.Timedelta(minutes=1)
	return probe


def snapshot_tradeable(status: Optional[str]) -> bool:
	if not status:
		return False
	return str(status).upper() in TRADEABLE_MARKET_STATUSES


def aligned_sleep_seconds(now_ts: pd.Timestamp, interval_seconds: int) -> float:
	if interval_seconds <= 0:
		return 0.0
	now_utc = pd.Timestamp(now_ts)
	if now_utc.tzinfo is None:
		now_utc = now_utc.tz_localize(UTC)
	else:
		now_utc = now_utc.tz_convert(UTC)
	epoch = now_utc.timestamp()
	remainder = epoch % interval_seconds
	sleep_for = interval_seconds - remainder
	if sleep_for < 0.05:
		sleep_for += interval_seconds
	return float(sleep_for)


def prepare_raw_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
	if raw.empty:
		raise ValueError("No raw data returned from DataLoader")
	raw = raw.copy()
	raw.index = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
	raw = raw.sort_index()
	return raw


def format_best_base_signal_log(
	signal: dict[str, object],
	*,
	raw_rows: Optional[int] = None,
	bars_rows: Optional[int] = None,
	candidate_samples: Optional[int] = None,
	is_trading_hour: Optional[bool] = None,
	latest_close: Optional[float] = None,
	range150_ok: Optional[bool] = None,
	drop15m_ok: Optional[bool] = None,
) -> str:
	parts = [
		f"BEST_BASE SIGNAL: side={signal.get('side', 'flat')}",
		f"prob={float(signal.get('probability', 0.0)):.4f}",
		f"tradable={int(bool(signal.get('tradable', False)))}",
	]
	if signal.get("trend_probability") is not None:
		parts.append(f"trend_prob={float(signal['trend_probability']):.4f}")
	if signal.get("direction_probability") is not None:
		parts.append(f"dir_prob={float(signal['direction_probability']):.4f}")
	if signal.get("reject_reason"):
		parts.append(f"reason={signal['reject_reason']}")
	if "pred" in signal:
		parts.append(f"pred={signal['pred']}")
	if raw_rows is not None:
		parts.append(f"raw_rows={raw_rows}")
	if bars_rows is not None:
		parts.append(f"bars_1m={bars_rows}")
	if candidate_samples is not None:
		parts.append(f"candidates={candidate_samples}")
	if is_trading_hour is not None:
		parts.append(f"market_open={int(bool(is_trading_hour))}")
	if latest_close is not None:
		parts.append(f"close={float(latest_close):.2f}")
	if range150_ok is not None:
		parts.append(f"range150={'🟢✓' if range150_ok else '🔴✗'}")
	if drop15m_ok is not None:
		parts.append(f"drop15m={'🟢✓' if drop15m_ok else '🔴✗'}")
	return " ".join(parts)


def format_signal_status_line(
	*,
	signal: dict[str, object],
	trading_open_now: bool,
	signal_qualified: bool,
	weak_filter_enabled: bool,
	weak_period_block: bool,
	now_utc: Optional[pd.Timestamp] = None,
	cutoff: Optional[float] = None,
) -> str:
	hour_mark = "[OK]" if trading_open_now else "[X]"
	signal_mark = "🟢✓" if signal_qualified else "🔴✗"
	weak_mark = "[X]" if weak_period_block else ("[OK]" if weak_filter_enabled else "[?]")
	lag_mark = "[?]"
	lag_minutes: Optional[float] = None
	if now_utc is not None:
		now_ts = pd.Timestamp(now_utc)
		now_ts = now_ts.tz_convert("UTC") if now_ts.tzinfo else now_ts.tz_localize("UTC")
		signal_ts = pd.Timestamp(signal["signal_bar_time"])
		signal_ts = signal_ts.tz_convert("UTC") if signal_ts.tzinfo else signal_ts.tz_localize("UTC")
		lag_minutes = max((now_ts - signal_ts).total_seconds() / 60.0, 0.0)
		lag_mark = "[OK]" if lag_minutes <= 5.0 else "[X]"
	parts = [
		"SIGNAL STATUS:",
		f"hour={hour_mark}",
		f"signal={signal_mark}",
		f"weak_filter={weak_mark}",
		f"side={signal.get('side', 'flat')}",
		f"prob={float(signal.get('probability', 0.0)):.4f}",
		f"tradable={int(bool(signal.get('tradable', False)))}",
	]
	if signal.get("reject_reason"):
		parts.append(f"reason={signal['reject_reason']}")
	if lag_minutes is not None:
		parts.append(f"freshness={lag_mark}")
		parts.append(f"lag_min={lag_minutes:.1f}")
	if cutoff is not None:
		parts.append(f"cutoff={float(cutoff):.4f}")
	if "pred" in signal:
		parts.append(f"pred={signal['pred']}")
	return " ".join(parts)


def load_weak_period_cells(path: Optional[str]) -> list[dict[str, str]]:
	if not path:
		return []
	p = Path(path)
	if not p.exists():
		return []
	try:
		payload = json.loads(p.read_text(encoding="utf-8"))
	except Exception:
		return []
	raw_cells = payload.get("weak_cells") if isinstance(payload, dict) else payload
	if not isinstance(raw_cells, list):
		return []
	out: list[dict[str, str]] = []
	for item in raw_cells:
		if not isinstance(item, dict):
			continue
		session = str(item.get("session", "")).strip().lower()
		day = str(item.get("day", "")).strip()
		hour = str(item.get("hour", "")).strip()
		if not session or not day or not hour:
			continue
		out.append({"session": session, "day": day, "hour": hour})
	return out


def price_rows_to_frame(prices_data: list[dict[str, object]]) -> pd.DataFrame:
	if not prices_data:
		return pd.DataFrame()
	raw = pd.DataFrame(prices_data)
	if raw.empty or "timestamp" not in raw.columns:
		return pd.DataFrame()
	return prepare_raw_price_frame(raw)


class ExistingSchemaMarketDataStore:
	def __init__(self, logger: logging.Logger):
		self.logger = logger
		self.last_upsert_summaries: list[dict[str, object]] = []

	def sync_from_latest(self, service: IGService, instrument: Price, end_time: datetime) -> dict[str, object]:
		return fetch_and_store_prices_from_latest(service, instrument, end_time=end_time)

	def _instrument_from_snapshot(self, snapshot: dict[str, object]) -> Price:
		epic = str(snapshot.get("epic") or "")
		for instrument in MARKET_DATA_INSTRUMENTS:
			if instrument.epic == epic:
				return instrument
		name = str(snapshot.get("instrument") or "").strip().lower()
		for instrument in MARKET_DATA_INSTRUMENTS:
			if instrument.name.lower() == name:
				return instrument
		raise ValueError(f"Unknown snapshot instrument: epic={epic!r} name={name!r}")

	def upsert_snapshots(self, snapshots: list[dict[str, object]]) -> int:
		if not snapshots:
			self.last_upsert_summaries = []
			return 0
		by_instrument: dict[Price, list[dict[str, object]]] = {}
		for snapshot in snapshots:
			instrument = self._instrument_from_snapshot(snapshot)
			by_instrument.setdefault(instrument, []).append(snapshot_to_price_row(snapshot))
		written = 0
		summaries: list[dict[str, object]] = []
		for instrument, rows in by_instrument.items():
			# MySQL rowcount with ON DUPLICATE KEY UPDATE can be 2 for one logical row update.
			affected_rows = int(insert_prices(rows, instrument))
			logical_written = int(len(rows))
			written += logical_written
			summaries.append(
				{
					"instrument": instrument.name.lower(),
					"table_name": instrument.db_name,
					"logical_written": logical_written,
					"mysql_affected_rows": affected_rows,
				}
			)
		self.last_upsert_summaries = summaries
		return written


class IGPredictionDataCache:
	def __init__(
		self,
		cfg: BotConfig,
		logger: logging.Logger,
		*,
		price_fetcher=fetch_prices,
	):
		self.cfg = cfg
		self.logger = logger
		self.price_fetcher = price_fetcher
		self._service: Optional[IGService] = None
		self._last_bucket: Optional[pd.Timestamp] = None
		self.raw: Optional[pd.DataFrame] = None
		self.last_fetch_summary: dict[str, object] = {}
		self.last_error: Optional[str] = None

	def _service_or_create(self) -> IGService:
		if self._service is None:
			try:
				self._service = IGService(
					api_key=API_CONFIG["api_key"],
					username=API_CONFIG["username"],
					password=API_CONFIG["password"],
					base_url=API_CONFIG["base_url"],
				)
			except Exception as exc:
				raise RuntimeError(
					"IG authentication failed for prediction cache. Check IG_API_KEY, IG_USERNAME, IG_PASSWORD, and IG_BASE_URL in your environment/.env. "
					f"Original error: {exc}"
				) from exc
		return self._service

	def _bootstrap_from_mysql(self, now_utc: pd.Timestamp) -> pd.DataFrame:
		start_date = (now_utc.date() - timedelta(days=max(self.cfg.recent_days, 2))).isoformat()
		end_date = now_utc.date().isoformat()
		raw = DataLoader().load_data(self.cfg.table, start_date=start_date, end_date=end_date)
		prepared = prepare_raw_price_frame(raw)
		max_rows = int(getattr(self.cfg, "prediction_cache_max_rows", 0) or 0)
		if max_rows > 0 and len(prepared) > max_rows:
			prepared = prepared.iloc[-max_rows:].copy()
		self.raw = prepared
		return self.raw

	def _merge_raw(self, incoming: pd.DataFrame, now_utc: pd.Timestamp) -> pd.DataFrame:
		if incoming.empty:
			if self.raw is None:
				self.raw = incoming
			return self.raw if self.raw is not None else incoming
		combined = incoming if self.raw is None or self.raw.empty else pd.concat([self.raw, incoming], axis=0)
		combined = combined[~combined.index.duplicated(keep="last")].sort_index()
		cutoff = now_utc - pd.Timedelta(days=max(self.cfg.recent_days, 2) + 1)
		combined = combined.loc[combined.index >= cutoff]
		max_rows = int(getattr(self.cfg, "prediction_cache_max_rows", 0) or 0)
		if max_rows > 0 and len(combined) > max_rows:
			combined = combined.iloc[-max_rows:].copy()
		self.raw = combined
		return combined

	def maybe_refresh(self, now_ts: Optional[pd.Timestamp] = None, *, force: bool = False) -> Optional[pd.DataFrame]:
		self.last_error = None
		now_utc = pd.Timestamp(datetime.now(UTC) if now_ts is None else now_ts)
		if now_utc.tzinfo is None:
			now_utc = now_utc.tz_localize(UTC)
		else:
			now_utc = now_utc.tz_convert(UTC)
		if not force and not market_data_due(now_utc, self._last_bucket, self.cfg.prediction_poll_second):
			return None
		self._last_bucket = market_data_poll_bucket(now_utc)
		if not instrument_trading_hours_open(Price.Gold, now_utc):
			self.last_fetch_summary = {
				"instrument": "gold",
				"bucket_utc": self._last_bucket.isoformat(),
				"status": "market_closed",
				"cache_rows": int(len(self.raw)) if self.raw is not None else 0,
			}
			return None

		bootstrap_rows = 0
		try:
			if self.raw is None or self.raw.empty:
				bootstrap_rows = int(len(self._bootstrap_from_mysql(now_utc)))
			before_rows = int(len(self.raw)) if self.raw is not None else 0
			last_cached_ts = None if self.raw is None or self.raw.empty else pd.Timestamp(self.raw.index[-1])
			if last_cached_ts is not None:
				last_cached_ts = last_cached_ts.tz_localize(UTC) if last_cached_ts.tzinfo is None else last_cached_ts.tz_convert(UTC)
			# Use only fully closed minute bars to keep live input stable/parity-friendly.
			request_start_ts = now_utc if last_cached_ts is None else (last_cached_ts + pd.Timedelta(minutes=1))
			request_end_ts = now_utc.floor("min") - pd.Timedelta(minutes=1)
			request_start = request_start_ts.to_pydatetime()
			request_end = request_end_ts.to_pydatetime()
			if request_end < request_start:
				fetched_rows = []
			else:
				fetched_rows = self.price_fetcher(
					self._service_or_create(),
					Price.Gold,
					start_time=request_start,
					end_time=request_end,
				)
			incoming = price_rows_to_frame(fetched_rows)
			merged = self._merge_raw(incoming, now_utc)
			after_rows = int(len(merged))
			self.last_fetch_summary = {
				"instrument": "gold",
				"bucket_utc": self._last_bucket.isoformat(),
				"requested_start_utc": pd.Timestamp(request_start).tz_localize(UTC).isoformat() if pd.Timestamp(request_start).tzinfo is None else pd.Timestamp(request_start).tz_convert(UTC).isoformat(),
				"requested_end_utc": pd.Timestamp(request_end).tz_localize(UTC).isoformat() if pd.Timestamp(request_end).tzinfo is None else pd.Timestamp(request_end).tz_convert(UTC).isoformat(),
				"bootstrap_rows": bootstrap_rows,
				"fetched_rows": int(len(fetched_rows)),
				"incoming_rows": int(len(incoming)),
				"appended_rows": max(after_rows - before_rows, 0),
				"cache_rows": after_rows,
				"cache_start_utc": merged.index[0].isoformat() if not merged.empty else None,
				"cache_end_utc": merged.index[-1].isoformat() if not merged.empty else None,
				"prediction_cache_max_rows": int(getattr(self.cfg, "prediction_cache_max_rows", 0) or 0),
			}
			self.logger.info(
				"Prediction cache refresh instrument=gold requested=%s -> %s fetched_rows=%s appended_rows=%s cache_rows=%s",
				self.last_fetch_summary.get("requested_start_utc"),
				self.last_fetch_summary.get("requested_end_utc"),
				self.last_fetch_summary.get("fetched_rows"),
				self.last_fetch_summary.get("appended_rows"),
				self.last_fetch_summary.get("cache_rows"),
			)
			return merged
		except Exception as exc:
			self.last_error = str(exc)
			self.logger.exception("Prediction-cache refresh failed: %s", exc)
			if self.raw is not None and not self.raw.empty:
				self.last_fetch_summary = {
					"instrument": "gold",
					"bucket_utc": self._last_bucket.isoformat() if self._last_bucket is not None else None,
					"status": "using_stale_cache_after_error",
					"error": str(exc),
					"cache_rows": int(len(self.raw)),
					"cache_start_utc": self.raw.index[0].isoformat(),
					"cache_end_utc": self.raw.index[-1].isoformat(),
				}
				return self.raw
			raise


class IGMarketDataCollector:
	def __init__(
		self,
		cfg: BotConfig,
		logger: logging.Logger,
		*,
		store: Optional[ExistingSchemaMarketDataStore] = None,
		snapshot_fetcher=fetch_market_snapshot,
		account_fetcher=fetch_primary_account_summary,
		positions_fetcher=fetch_open_positions,
	):
		self.cfg = cfg
		self.logger = logger
		self.store = store or ExistingSchemaMarketDataStore(logger)
		self.snapshot_fetcher = snapshot_fetcher
		self.account_fetcher = account_fetcher
		self.positions_fetcher = positions_fetcher
		self._service: Optional[IGService] = None
		self._last_bucket: Optional[pd.Timestamp] = None
		self.last_sync_summaries: list[dict[str, object]] = []
		self.last_error: Optional[str] = None

	@staticmethod
	def _safe_float(value: object) -> Optional[float]:
		try:
			if value is None or value == "":
				return None
			return float(value)
		except (TypeError, ValueError):
			return None

	@staticmethod
	def _fmt_money(value: Optional[float], currency: Optional[str]) -> Optional[str]:
		if value is None:
			return None
		suffix = str(currency or "").strip().upper()
		return f"{float(value):.2f}{suffix}" if suffix else f"{float(value):.2f}"

	def _account_status_part(self, service: IGService) -> str:
		try:
			account = dict(self.account_fetcher(service) or {})
		except Exception as exc:
			self.logger.warning("Account summary lookup failed: %s", exc)
			return "acct=unavailable"
		status = str(account.get("status") or "unknown").strip().lower()
		currency = str(account.get("currency") or "").strip().upper() or None
		parts = [f"acct={status}"]
		balance_txt = self._fmt_money(self._safe_float(account.get("balance")), currency)
		equity_txt = self._fmt_money(self._safe_float(account.get("equity")), currency)
		pnl_value = self._safe_float(account.get("profit_loss"))
		pnl_txt = self._fmt_money(pnl_value, currency)
		if balance_txt is not None:
			parts.append(f"bal={balance_txt}")
		if equity_txt is not None:
			parts.append(f"eq={equity_txt}")
		if pnl_txt is not None:
			sign = "+" if pnl_value is not None and pnl_value > 0 else ""
			parts.append(f"pnl={sign}{pnl_txt}")
		return " ".join(parts)

	def _position_status_part(self, service: IGService, snapshot_by_epic: dict[str, dict[str, object]]) -> str:
		try:
			positions = list(self.positions_fetcher(service) or [])
		except Exception as exc:
			self.logger.warning("Open-position lookup failed: %s", exc)
			return "pos=unavailable"
		tracked_epics = {instrument.epic: instrument for instrument in MARKET_DATA_INSTRUMENTS}
		tracked_positions: list[dict[str, object]] = []
		for row in positions:
			if not isinstance(row, dict):
				continue
			market = row.get("market") if isinstance(row.get("market"), dict) else {}
			epic = str(market.get("epic") or "")
			if epic in tracked_epics:
				tracked_positions.append(row)
		if not tracked_positions:
			return "pos=none"
		first = tracked_positions[0]
		pos = first.get("position") if isinstance(first.get("position"), dict) else {}
		market = first.get("market") if isinstance(first.get("market"), dict) else {}
		epic = str(market.get("epic") or "")
		instrument = tracked_epics.get(epic)
		name = instrument.name.lower() if instrument is not None else str(market.get("instrumentName") or epic or "unknown").lower()
		direction = str(pos.get("direction") or "?").upper()
		size = self._safe_float(pos.get("size"))
		entry = self._safe_float(pos.get("level"))
		parts = [f"pos={len(tracked_positions)}", f"{name}:{direction}"]
		if size is not None:
			parts.append(f"size={size:.2f}")
		if entry is not None:
			parts.append(f"entry={entry:.2f}")
		snapshot = snapshot_by_epic.get(epic) or {}
		mark = self._safe_float(snapshot.get("mid"))
		if mark is None:
			bid = self._safe_float(snapshot.get("bid"))
			offer = self._safe_float(snapshot.get("offer"))
			if bid is not None and offer is not None:
				mark = (bid + offer) / 2.0
		if mark is not None and size is not None and entry is not None:
			pnl = (mark - entry) * size if direction == "BUY" else (entry - mark) * size
			parts.append(f"pnl={pnl:+.2f}")
		return " ".join(parts)

	def _service_or_create(self) -> IGService:
		if self._service is None:
			try:
				self._service = IGService(
					api_key=API_CONFIG["api_key"],
					username=API_CONFIG["username"],
					password=API_CONFIG["password"],
					base_url=API_CONFIG["base_url"],
				)
			except Exception as exc:
				raise RuntimeError(
					"IG authentication failed. Check IG_API_KEY, IG_USERNAME, IG_PASSWORD, and IG_BASE_URL in your environment/.env. "
					f"Original error: {exc}"
				) from exc
		return self._service

	def maybe_capture(self, now_ts: Optional[pd.Timestamp] = None, *, force: bool = False) -> list[dict[str, object]]:
		if not self.cfg.market_data_enabled:
			self.last_sync_summaries = []
			self.last_error = None
			return []
		self.last_error = None
		now_utc = pd.Timestamp(datetime.now(UTC) if now_ts is None else now_ts)
		if now_utc.tzinfo is None:
			now_utc = now_utc.tz_localize(UTC)
		else:
			now_utc = now_utc.tz_convert(UTC)
		if not force and not market_data_due(now_utc, self._last_bucket, self.cfg.market_data_poll_second):
			return []
		self._last_bucket = market_data_poll_bucket(now_utc)
		summaries: list[dict[str, object]] = []
		poll_summary_parts: list[str] = []
		snapshot_summary_parts: list[str] = []
		snapshot_by_epic: dict[str, dict[str, object]] = {}
		service = self._service_or_create()
		for instrument in MARKET_DATA_INSTRUMENTS:
			if not instrument_trading_hours_open(instrument, now_utc):
				continue
			summary = self.store.sync_from_latest(service, instrument, now_utc.to_pydatetime())
			record = self.snapshot_fetcher(service, instrument, fetch_time=now_utc.to_pydatetime())
			snapshot_by_epic[str(record.get("epic") or instrument.epic)] = record
			summary["snapshot_market_status"] = record.get("market_status")
			summary["snapshot_fetch_time_utc"] = record.get("fetch_time_utc")
			summary["snapshot_bucket_utc"] = record.get("bucket_minute_utc")
			summary["resulting_period_start_utc"] = summary.get("fetched_period_start_utc") or summary.get("requested_start_utc")
			summary["resulting_period_end_utc"] = summary.get("fetched_period_end_utc") or summary.get("latest_db_before_utc")
			if not snapshot_tradeable(record.get("market_status")):
				summary["snapshot_written"] = 0
				summaries.append(summary)
				poll_summary_parts.append(
					f"{instrument.name.lower()}:rows_added={int(summary.get('inserted_rows') or 0)}"
				)
				snapshot_summary_parts.append(f"{instrument.name.lower()}:snap=0")
				continue
			snapshot_written = self.store.upsert_snapshots([record])
			summary["snapshot_written"] = int(snapshot_written)
			summary["resulting_period_start_utc"] = summary.get("fetched_period_start_utc") or record.get("bucket_minute_utc")
			summary["resulting_period_end_utc"] = record.get("bucket_minute_utc") or summary.get("fetched_period_end_utc")
			summaries.append(summary)
			poll_summary_parts.append(
				f"{instrument.name.lower()}:rows_added={int(summary.get('inserted_rows') or 0)}"
			)
			upsert_summaries = list(getattr(self.store, "last_upsert_summaries", []) or [])
			if upsert_summaries:
				for item in upsert_summaries:
					snapshot_summary_parts.append(
						f"{item.get('instrument')}:snap={int(item.get('logical_written') or 0)} mysql={int(item.get('mysql_affected_rows') or 0)}"
					)
			else:
				snapshot_summary_parts.append(f"{instrument.name.lower()}:snap={int(snapshot_written)}")
		self.last_sync_summaries = summaries
		if poll_summary_parts:
			account_part = self._account_status_part(service)
			position_part = self._position_status_part(service, snapshot_by_epic)
			self.logger.info(
				"MARKET POLL: bucket=%s  %s  %s",
				self._last_bucket.isoformat() if self._last_bucket is not None else "na",
				position_part,
				" ; ".join(poll_summary_parts),
			)
			self.logger.info(
				"ACCOUNT STATUS: bucket=%s  %s",
				self._last_bucket.isoformat() if self._last_bucket is not None else "na",
				account_part,
			)
		else:
			self.logger.info(
				"MARKET SYNC SKIPPED: bucket=%s  all instruments outside trading hours",
				self._last_bucket.isoformat() if self._last_bucket is not None else "na",
			)
		return summaries


class SingleInstanceLock:
	def __init__(self, path: str):
		self.path = path
		self._fh = None

	def __enter__(self):
		Path(self.path).parent.mkdir(parents=True, exist_ok=True)
		self._fh = open(self.path, "w")
		try:
			fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
		except OSError as exc:
			raise RuntimeError(f"Another trading bot instance appears to be running (lock: {self.path})") from exc
		self._fh.write(str(os.getpid()))
		self._fh.flush()
		return self

	def __exit__(self, exc_type, exc, tb):
		if self._fh is not None:
			try:
				fcntl.flock(self._fh, fcntl.LOCK_UN)
			finally:
				self._fh.close()
		try:
			os.remove(self.path)
		except OSError:
			pass


class PaperBroker:
	def __init__(self, cfg: BotConfig, logger: logging.Logger):
		self.cfg = cfg
		self.logger = logger
		self.trade_log_path = PROJECT_ROOT / cfg.trade_log_path
		self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)

	def open_long(
		self,
		*,
		signal_bar_time: pd.Timestamp,
		entry_bar_time: pd.Timestamp,
		entry_time: pd.Timestamp,
		entry_price: float,
		probability: float,
		deal_id: Optional[str] = None,
	) -> PaperPosition:
		normalized_entry_time = pd.Timestamp(entry_time)
		if normalized_entry_time.tzinfo is None:
			normalized_entry_time = normalized_entry_time.tz_localize(UTC)
		else:
			normalized_entry_time = normalized_entry_time.tz_convert(UTC)
		resolved_deal_id = str(deal_id) if deal_id else f"paper-{normalized_entry_time.strftime('%Y%m%dT%H%M%S')}"
		stop_loss = entry_price * (1.0 - self.cfg.stop_loss_pct / 100.0)
		take_profit = entry_price * (1.0 + self.cfg.take_profit_pct / 100.0)
		position = PaperPosition(
			direction="LONG",
			deal_id=resolved_deal_id,
			signal_bar_time=signal_bar_time.isoformat(),
			entry_bar_time=entry_bar_time.isoformat(),
			entry_time=normalized_entry_time.isoformat(),
			entry_price=float(entry_price),
			stop_loss=float(stop_loss),
			take_profit=float(take_profit),
			probability=float(probability),
			size=float(self.cfg.size),
			entry_price_initial=float(entry_price),
			bars_checked=0,
		)
		self.logger.info(
			"PAPER ENTRY deal_id=%s long size=%.2f entry=%.2f stop=%.2f target=%.2f prob=%.4f",
			position.deal_id,
			position.size,
			position.entry_price,
			position.stop_loss,
			position.take_profit,
			position.probability,
		)
		return position

	def update_position(
		self,
		position: PaperPosition,
		bars_15m: pd.DataFrame,
	) -> tuple[Optional[PaperPosition], Optional[dict[str, object]]]:
		if bars_15m.empty:
			return position, None

		entry_bar_time = pd.Timestamp(position.entry_bar_time)
		if entry_bar_time not in bars_15m.index:
			return position, None

		entry_idx = int(bars_15m.index.get_loc(entry_bar_time))
		start_idx = entry_idx + int(position.bars_checked)
		last_idx = min(entry_idx + self.cfg.max_hold_bars - 1, len(bars_15m) - 1)
		if start_idx > last_idx:
			return position, None

		for bar_idx in range(start_idx, last_idx + 1):
			bar = bars_15m.iloc[bar_idx]
			low_bid = float(bar.get("low_bid", bar["low"]))
			high_bid = float(bar.get("high_bid", bar["high"]))
			close_bid = float(bar.get("close_bid", bar["close"]))
			exit_time = bars_15m.index[bar_idx] + BAR_INTERVAL

			if low_bid <= position.stop_loss:
				return None, self._close_trade(position, exit_price=position.stop_loss, exit_time=exit_time, exit_reason="stop_loss")
			if high_bid >= position.take_profit:
				return None, self._close_trade(position, exit_price=position.take_profit, exit_time=exit_time, exit_reason="take_profit")

			position.bars_checked += 1
			if bar_idx == last_idx:
				return None, self._close_trade(position, exit_price=close_bid, exit_time=exit_time, exit_reason="timeout")

		return position, None

	def _close_trade(
		self,
		position: PaperPosition,
		*,
		exit_price: float,
		exit_time: pd.Timestamp,
		exit_reason: str,
	) -> dict[str, object]:
		pnl_usd = (float(exit_price) - position.entry_price) * position.size
		trade = {
			"direction": position.direction,
			"deal_id": position.deal_id,
			"signal_bar_time": position.signal_bar_time,
			"entry_bar_time": position.entry_bar_time,
			"entry_time": position.entry_time,
			"exit_time": exit_time.isoformat(),
			"entry_price": position.entry_price,
			"entry_price_initial": float(position.entry_price_initial) if position.entry_price_initial is not None else position.entry_price,
			"exit_price": float(exit_price),
			"size": position.size,
			"probability": position.probability,
			"bars_held": int(position.bars_checked + 1),
			"exit_reason": exit_reason,
			"pnl_usd": float(pnl_usd),
			"pnl_pct": ((float(exit_price) - position.entry_price) / position.entry_price) * 100.0 if position.entry_price else 0.0,
		}
		self._append_trade(trade)
		self.logger.info(
			"PAPER EXIT reason=%s exit=%.2f pnl=%.2f bars_held=%d",
			exit_reason,
			trade["exit_price"],
			trade["pnl_usd"],
			trade["bars_held"],
		)
		return trade

	def _append_trade(self, row: dict[str, object]) -> None:
		write_header = not self.trade_log_path.exists()
		with self.trade_log_path.open("a", newline="", encoding="utf-8") as fh:
			writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
			if write_header:
				writer.writeheader()
			writer.writerow(row)


class AlphaGoldTradingBot:
	def __init__(self, cfg: BotConfig):
		self.cfg = cfg
		self.logger = self._build_logger(PROJECT_ROOT / cfg.log_path)
		self.state_path = PROJECT_ROOT / cfg.state_path
		self.status_path = PROJECT_ROOT / cfg.status_path
		self.state_path.parent.mkdir(parents=True, exist_ok=True)
		self.status_path.parent.mkdir(parents=True, exist_ok=True)
		self.state = self._load_state()
		self.image_trend = None
		self.model_bundle: Optional[dict[str, Any]] = None
		self.system: Optional[object] = None
		self.model = None
		self.last_raw_cache_info: dict[str, object] = {}
		self.last_best_base_payload_info: dict[str, object] = {}
		self._last_prediction_wait_bucket_utc: Optional[pd.Timestamp] = None
		self.weak_period_cells: list[dict[str, str]] = []
		self.prediction_cache = IGPredictionDataCache(cfg, self.logger)
		if not cfg.market_sync_only:
			if cfg.signal_model_family == "legacy_15m_nextbar":
				model_module = _load_15m_nextbar_module()
				self.system = model_module.UptrendRecognitionSystem15mNextBar(
					threshold_pct=cfg.take_profit_pct,
					max_adverse_low_pct=cfg.stop_loss_pct,
					forward_bars=cfg.max_hold_bars,
				)
				self.system.load_models(str(PROJECT_ROOT / cfg.model_dir))
				if cfg.model_type not in self.system.models:
					raise ValueError(f"Model '{cfg.model_type}' not found in {cfg.model_dir}")
				self.model = self.system.models[cfg.model_type]
			elif cfg.signal_model_family == "best_base_state":
				self.image_trend = _load_image_trend_module()
				bundle_path = PROJECT_ROOT / cfg.signal_model_path
				self.model_bundle = joblib.load(bundle_path)
				if not isinstance(self.model_bundle, dict) or "stage1" not in self.model_bundle or "stage2" not in self.model_bundle:
					raise ValueError(f"Best-base model artifact is invalid: {bundle_path}")
				self.model = self.model_bundle
			else:
				raise ValueError(f"Unsupported --signal-model-family: {cfg.signal_model_family}")
		self.paper_broker = PaperBroker(cfg, self.logger)
		if cfg.mode == "live":
			ig_service = IGService(
				api_key=API_CONFIG["api_key"],
				username=API_CONFIG["username"],
				password=API_CONFIG["password"],
				base_url=API_CONFIG["base_url"],
			)
			broker_adapter = IGLiveBrokerAdapter(
				ig_service,
				instrument=Price.Gold,
				stop_loss_pct=cfg.stop_loss_pct,
				take_profit_pct=cfg.take_profit_pct,
			)
		else:
			broker_adapter = DryRunBrokerAdapter()
		self.execution_engine = ExecutionEngine(broker_adapter)
		self.market_data_collector = IGMarketDataCollector(cfg, self.logger)
		weak_path = (PROJECT_ROOT / cfg.weak_periods_json) if cfg.weak_periods_json and not Path(cfg.weak_periods_json).is_absolute() else Path(cfg.weak_periods_json) if cfg.weak_periods_json else None
		self.weak_period_cells = load_weak_period_cells(str(weak_path)) if weak_path is not None else []
		self._log_model_startup_config()

	def _log_model_startup_config(self) -> None:
		if self.cfg.market_sync_only:
			self.logger.info(
				"BOT STARTUP: mode=%s family=%s market_sync_only=1 size=%.4f",
				self.cfg.mode,
				self.cfg.signal_model_family,
				self.cfg.size,
			)
			return
		if self.cfg.signal_model_family == "best_base_state" and isinstance(self.model_bundle, dict):
			cfg = dict(self.model_bundle.get("config") or {})
			disable_time_filter = bool(cfg.get("disable_time_filter", False))
			blocked_utc = self.model_bundle.get("blocked_utc")
			if blocked_utc is None:
				blocked_utc = cfg.get("blocked_utc")
			if blocked_utc is None and self.image_trend is not None:
				blocked_utc = getattr(self.image_trend, "BLOCKED_UTC_WINDOWS", None)
			self.logger.info(
				"MODEL CONFIG: family=best_base_state path=%s mode=%s size=%.4f timeframe=%s window=%s horizon=%s stage1_min_prob=%s stage2_min_prob=%s use_state_features=%s use_15m_wick_features=%s",
				self.cfg.signal_model_path,
				self.cfg.mode,
				self.cfg.size,
				cfg.get("timeframe", "1min"),
				cfg.get("window", "na"),
				cfg.get("horizon", "na"),
				self.model_bundle.get("stage1_min_prob", cfg.get("stage1_min_prob")),
				self.model_bundle.get("stage2_min_prob", cfg.get("stage2_min_prob")),
				self.model_bundle.get("use_state_features", cfg.get("use_state_features", False)),
				self.model_bundle.get("use_15m_wick_features", cfg.get("use_15m_wick_features", False)),
			)
			self.logger.info(
				"TIME FILTER: enabled=%s disable_time_filter=%s blocked_utc=%s",
				int(not disable_time_filter),
				int(disable_time_filter),
				blocked_utc if blocked_utc is not None else "[]",
			)
			self.logger.info(
				"WEAK FILTER: path=%s loaded_cells=%d",
				self.cfg.weak_periods_json,
				len(self.weak_period_cells),
			)
			if self.weak_period_cells:
				cells_str = ", ".join(
					f"{cell.get('session')}:{cell.get('day')}:{cell.get('hour')}"
					for cell in self.weak_period_cells
				)
				self.logger.info("WEAK FILTER CELLS: %s", cells_str)
			return
		self.logger.info(
			"MODEL CONFIG: family=%s mode=%s size=%.4f model_type=%s model_dir=%s tp_pct=%.4f sl_pct=%.4f max_hold_bars=%d",
			self.cfg.signal_model_family,
			self.cfg.mode,
			self.cfg.size,
			self.cfg.model_type,
			self.cfg.model_dir,
			self.cfg.take_profit_pct,
			self.cfg.stop_loss_pct,
			self.cfg.max_hold_bars,
		)

	def _run_market_data_cycle(self, *, force: bool = False) -> list[dict[str, object]]:
		if not self.cfg.market_data_enabled:
			self.logger.info("Market-data capture disabled; skipping sync.")
			self.market_data_collector.last_sync_summaries = []
			self.market_data_collector.last_error = None
			return []
		try:
			summaries = self.market_data_collector.maybe_capture(pd.Timestamp(datetime.now(UTC)), force=force)
		except Exception as exc:
			self.market_data_collector.last_error = str(exc)
			self.logger.exception("Market-data capture failed: %s", exc)
			if self.cfg.market_sync_only:
				self._save_state()
				raise RuntimeError(str(exc)) from exc
			return []
		if force and not summaries:
			self.logger.info("Market-sync-only run completed with no open instruments or no new market data.")
		return summaries

	def _run_prediction_cycle(self, *, force: bool = False) -> Optional[pd.DataFrame]:
		if self.cfg.signal_model_family != "best_base_state":
			return self._load_recent_raw_data()
		try:
			raw = self.prediction_cache.maybe_refresh(pd.Timestamp(datetime.now(UTC)), force=force)
		except Exception as exc:
			self.prediction_cache.last_error = str(exc)
			self.logger.exception("Prediction-cycle refresh failed: %s", exc)
			return None
		if raw is None or raw.empty:
			cache = self.prediction_cache.raw
			self.last_raw_cache_info = {
				"storage_mode": "prediction_memory_cache",
				"reload_each_cycle": False,
				"table": self.cfg.table,
				"recent_days": int(self.cfg.recent_days),
				"rows": int(len(cache)) if cache is not None else 0,
				"prediction_poll_second": int(self.cfg.prediction_poll_second),
				"prediction_cache_last_bucket_utc": self.prediction_cache._last_bucket.isoformat() if self.prediction_cache._last_bucket is not None else None,
				"prediction_cache_last_error": self.prediction_cache.last_error,
				"prediction_cache_max_rows": int(self.cfg.prediction_cache_max_rows),
			}
			return None
		self.last_raw_cache_info = {
			"storage_mode": "prediction_memory_cache",
			"reload_each_cycle": False,
			"table": self.cfg.table,
			"recent_days": int(self.cfg.recent_days),
			"rows": int(len(raw)),
			"start_utc": raw.index[0].isoformat(),
			"end_utc": raw.index[-1].isoformat(),
			"prediction_poll_second": int(self.cfg.prediction_poll_second),
			"prediction_cache_last_bucket_utc": self.prediction_cache._last_bucket.isoformat() if self.prediction_cache._last_bucket is not None else None,
			"prediction_cache_last_error": self.prediction_cache.last_error,
			"prediction_cache_max_rows": int(self.cfg.prediction_cache_max_rows),
		}
		return raw

	def _is_weak_period_entry(self, ts: pd.Timestamp) -> bool:
		weak_cells = getattr(self, "weak_period_cells", None) or []
		if not weak_cells or self.image_trend is None:
			return False
		checker = getattr(self.image_trend, "_is_weak_period_entry", None)
		if checker is None:
			return False
		try:
			return bool(checker(pd.Timestamp(ts), weak_cells))
		except Exception:
			return False

	def _maybe_adjust_open_position_from_previous_minute(self, raw: pd.DataFrame) -> None:
		position = self.state.open_position
		if position is None or position.target_stop_adjusted or raw.empty:
			return
		entry_ts = pd.Timestamp(position.entry_time)
		if entry_ts.tzinfo is None:
			entry_ts = entry_ts.tz_localize(UTC)
		else:
			entry_ts = entry_ts.tz_convert(UTC)
		latest_raw_ts = pd.Timestamp(raw.index[-1])
		if latest_raw_ts.tzinfo is None:
			latest_raw_ts = latest_raw_ts.tz_localize(UTC)
		else:
			latest_raw_ts = latest_raw_ts.tz_convert(UTC)
		if latest_raw_ts < (entry_ts + pd.Timedelta(minutes=1)):
			return
		entry_rows = raw.loc[raw.index == entry_ts]
		if entry_rows.empty:
			self.logger.info("Post-open TP/SL adjust waiting: entry minute %s not available yet for deal_id=%s", entry_ts.isoformat(), position.deal_id)
			return
		entry_row = entry_rows.iloc[-1]
		anchor_price = float(entry_row.get("openPrice_ask", entry_row.get("openPrice", entry_row.get("closePrice"))))
		if anchor_price <= 0.0:
			self.logger.warning("Post-open TP/SL adjust skipped: invalid anchor price %.6f for deal_id=%s", anchor_price, position.deal_id)
			return
		position.entry_price = anchor_price
		if position.direction == "SHORT":
			position.stop_loss = anchor_price * (1.0 + self.cfg.stop_loss_pct / 100.0)
			position.take_profit = anchor_price * (1.0 - self.cfg.take_profit_pct / 100.0)
		else:
			position.stop_loss = anchor_price * (1.0 - self.cfg.stop_loss_pct / 100.0)
			position.take_profit = anchor_price * (1.0 + self.cfg.take_profit_pct / 100.0)
		position.target_stop_adjusted = True
		position.target_stop_adjusted_at = latest_raw_ts.isoformat()
		self.logger.info(
			"Post-open TP/SL adjusted deal_id=%s anchor_entry=%.4f stop=%.4f target=%.4f adjusted_at=%s",
			position.deal_id,
			position.entry_price,
			position.stop_loss,
			position.take_profit,
			position.target_stop_adjusted_at,
		)

	def _resolve_signal_anchor_price(self, raw: pd.DataFrame, signal: dict[str, object], direction: str) -> Optional[tuple[float, pd.Timestamp]]:
		if raw.empty:
			return None
		signal_entry_time_raw = signal.get("entry_bar_time")
		if signal_entry_time_raw is None:
			entry_row = raw.iloc[-1]
			price = float(entry_row.get("openPrice_ask", entry_row.get("openPrice", entry_row.get("closePrice"))))
			return price, pd.Timestamp(raw.index[-1])
		signal_entry_time = pd.Timestamp(signal_entry_time_raw)
		entry = find_next_entry_minute(raw, signal_entry_time)
		if entry is None:
			entry_row = raw.iloc[-1]
			price = float(entry_row.get("openPrice_ask", entry_row.get("openPrice", entry_row.get("closePrice"))))
			return price, pd.Timestamp(raw.index[-1])
		anchor_ts, anchor_row = entry
		if direction == "SHORT":
			price = float(anchor_row.get("openPrice_bid", anchor_row.get("openPrice", anchor_row.get("closePrice"))))
		else:
			price = float(anchor_row.get("openPrice_ask", anchor_row.get("openPrice", anchor_row.get("closePrice"))))
		return price, pd.Timestamp(anchor_ts)

	def _maybe_dynamic_target_stop_from_signal(self, raw: pd.DataFrame, signal: dict[str, object]) -> None:
		if not self.cfg.dynamic_target_stop_enabled:
			return
		position = self.state.open_position
		if position is None:
			return
		side = str(signal.get("side", "flat"))
		if position.direction == "LONG" and side != "up":
			return
		if position.direction == "SHORT" and side != "down":
			return
		anchor_payload = self._resolve_signal_anchor_price(raw, signal, position.direction)
		if anchor_payload is None:
			return
		anchor_price, anchor_ts = anchor_payload
		if anchor_price <= 0.0:
			return

		if position.direction == "SHORT":
			new_stop = anchor_price * (1.0 + self.cfg.stop_loss_pct / 100.0)
			new_target = anchor_price * (1.0 - self.cfg.take_profit_pct / 100.0)
		else:
			new_stop = anchor_price * (1.0 - self.cfg.stop_loss_pct / 100.0)
			new_target = anchor_price * (1.0 + self.cfg.take_profit_pct / 100.0)

		if abs(position.stop_loss - new_stop) < 1e-9 and abs(position.take_profit - new_target) < 1e-9:
			return

		position.stop_loss = float(new_stop)
		position.take_profit = float(new_target)
		position.target_updates += 1
		position.last_target_signal_time = pd.Timestamp(signal.get("signal_bar_time", anchor_ts)).isoformat()
		position.last_target_price = float(anchor_price)

		adapter = getattr(self.execution_engine, "broker_adapter", None)
		if self.cfg.mode == "live" and adapter is not None and hasattr(adapter, "amend_position_levels"):
			try:
				adapter.amend_position_levels(
					deal_id=position.deal_id,
					stop_level=position.stop_loss,
					limit_level=position.take_profit,
				)
			except Exception as exc:
				self.logger.warning("Live amend failed deal_id=%s error=%s", position.deal_id, exc)

		self.logger.info(
			"Dynamic TP/SL update deal_id=%s side=%s anchor_ts=%s anchor=%.4f stop=%.4f target=%.4f updates=%d",
			position.deal_id,
			position.direction,
			anchor_ts.isoformat(),
			anchor_price,
			position.stop_loss,
			position.take_profit,
			position.target_updates,
		)

	def _maybe_timeout_live_position(self) -> bool:
		"""Close a live position that has exceeded max_hold_minutes. Returns True if closed."""
		if self.cfg.mode != "live" or self.state.open_position is None:
			return False
		if not self.cfg.max_hold_minutes:
			return False
		position = self.state.open_position
		entry_ts = pd.Timestamp(position.entry_time)
		entry_ts = entry_ts.tz_convert("UTC") if entry_ts.tzinfo else entry_ts.tz_localize("UTC")
		now_utc = pd.Timestamp(datetime.now(UTC))
		elapsed_minutes = (now_utc - entry_ts).total_seconds() / 60.0
		if elapsed_minutes < float(self.cfg.max_hold_minutes):
			return False
		self.logger.info(
			"LIVE TIMEOUT deal_id=%s elapsed=%.1fm max_hold_minutes=%.1f — closing position",
			position.deal_id, elapsed_minutes, float(self.cfg.max_hold_minutes),
		)
		adapter = getattr(self.execution_engine, "broker_adapter", None)
		if adapter is None or not hasattr(adapter, "close_position"):
			self.logger.warning("Live timeout close skipped: broker adapter does not support close_position")
			return False
		close_direction = "SELL" if position.direction == "LONG" else "BUY"
		try:
			close_result = adapter.close_position(
				deal_id=position.deal_id,
				direction=close_direction,
				size=float(position.size),
			)
		except Exception as exc:
			self.logger.warning("Live timeout close failed deal_id=%s error=%s", position.deal_id, exc)
			return False
		confirm = close_result.get("confirm") if isinstance(close_result, dict) else None
		exit_price = float(close_result.get("close_level") or position.entry_price)
		exit_time_raw = close_result.get("close_time") if isinstance(close_result, dict) else None
		exit_time = pd.Timestamp(exit_time_raw) if exit_time_raw else now_utc
		broker_reason = None
		if isinstance(confirm, dict):
			broker_reason = str(confirm.get("reason") or "") or None
		self._record_live_close(
			position,
			exit_price=exit_price,
			exit_time=exit_time,
			exit_reason="timeout",
			close_source="bot_max_hold_timeout",
			close_reason_broker=broker_reason,
		)
		self.logger.info(
			"LIVE TIMEOUT CLOSED deal_id=%s exit=%.4f elapsed=%.1fm",
			position.deal_id, exit_price, elapsed_minutes,
		)
		return True

	def _sync_live_open_position(self) -> None:
		if self.cfg.mode != "live" or self.state.open_position is None:
			return
		adapter = getattr(self.execution_engine, "broker_adapter", None)
		if adapter is None or not hasattr(adapter, "get_position_by_deal_id"):
			return
		deal_id = str(self.state.open_position.deal_id)
		if not deal_id:
			return
		try:
			position_snapshot = adapter.get_position_by_deal_id(deal_id)
		except Exception as exc:
			self.logger.warning("Live position lookup failed deal_id=%s error=%s", deal_id, exc)
			return
		if position_snapshot is None:
			now_utc = pd.Timestamp(datetime.now(UTC))
			self.logger.info("Live position no longer open deal_id=%s; clearing local open_position state", deal_id)
			exit_price = float(self.state.open_position.last_target_price or self.state.open_position.entry_price)
			exit_time = now_utc
			close_reason = "position_missing_from_open_positions"
			adapter_close = None
			if hasattr(adapter, "get_closed_trade_by_deal_id"):
				try:
					adapter_close = adapter.get_closed_trade_by_deal_id(deal_id)
				except Exception as exc:
					self.logger.warning("Closed-trade lookup failed deal_id=%s error=%s", deal_id, exc)
			if isinstance(adapter_close, dict):
				if adapter_close.get("exit_price") is not None:
					exit_price = float(adapter_close["exit_price"])
				if adapter_close.get("exit_time"):
					exit_time = pd.Timestamp(adapter_close["exit_time"])
				close_reason = str(adapter_close.get("reason") or close_reason)
			self._record_live_close(
				self.state.open_position,
				exit_price=exit_price,
				exit_time=exit_time,
				exit_reason="broker_closed",
				close_source="broker_closed_detected",
				close_reason_broker=close_reason,
			)

	def _record_live_close(
		self,
		position: PaperPosition,
		*,
		exit_price: float,
		exit_time: pd.Timestamp,
		exit_reason: str,
		close_source: str,
		close_reason_broker: Optional[str] = None,
	) -> None:
		exit_ts = pd.Timestamp(exit_time)
		exit_ts = exit_ts.tz_convert("UTC") if exit_ts.tzinfo else exit_ts.tz_localize("UTC")
		if position.direction == "SHORT":
			pnl_usd = (position.entry_price - float(exit_price)) * position.size
			pnl_pct = ((position.entry_price - float(exit_price)) / position.entry_price) * 100.0 if position.entry_price else 0.0
		else:
			pnl_usd = (float(exit_price) - position.entry_price) * position.size
			pnl_pct = ((float(exit_price) - position.entry_price) / position.entry_price) * 100.0 if position.entry_price else 0.0
		trade = {
			"direction": position.direction,
			"deal_id": position.deal_id,
			"signal_bar_time": position.signal_bar_time,
			"entry_bar_time": position.entry_bar_time,
			"entry_time": position.entry_time,
			"exit_time": exit_ts.isoformat(),
			"entry_price": position.entry_price,
			"entry_price_initial": float(position.entry_price_initial) if position.entry_price_initial is not None else position.entry_price,
			"exit_price": float(exit_price),
			"size": position.size,
			"probability": position.probability,
			"bars_held": int(position.bars_checked),
			"exit_reason": exit_reason,
			"close_source": close_source,
			"close_reason_broker": close_reason_broker,
			"target_updates": int(position.target_updates),
			"last_target_signal_time": position.last_target_signal_time,
			"last_target_price": position.last_target_price,
			"pnl_usd": float(pnl_usd),
			"pnl_pct": float(pnl_pct),
		}
		self.paper_broker._append_trade(trade)
		self.state.last_exit_time = exit_ts.isoformat()
		self.state.last_exit_reason = exit_reason
		self.state.open_position = None

	def _maybe_close_live_position_from_signal(self, signal: dict[str, object], raw: pd.DataFrame) -> bool:
		if self.cfg.mode != "live" or self.state.open_position is None:
			return False
		if not bool(signal.get("tradable", True)):
			return False
		position = self.state.open_position
		side = str(signal.get("side", "flat"))
		if position.direction == "LONG" and side != "down":
			return False
		if position.direction == "SHORT" and side != "up":
			return False
		adapter = getattr(self.execution_engine, "broker_adapter", None)
		if adapter is None or not hasattr(adapter, "close_position"):
			self.logger.warning("Live close skipped: broker adapter does not support close_position")
			return False
		close_direction = "SELL" if position.direction == "LONG" else "BUY"
		try:
			close_result = adapter.close_position(
				deal_id=position.deal_id,
				direction=close_direction,
				size=float(position.size),
			)
		except Exception as exc:
			self.logger.warning("Live active close failed deal_id=%s error=%s", position.deal_id, exc)
			return False
		confirm = close_result.get("confirm") if isinstance(close_result, dict) else None
		exit_price = float(close_result.get("close_level") or position.entry_price)
		exit_time_raw = close_result.get("close_time") if isinstance(close_result, dict) else None
		exit_time = pd.Timestamp(exit_time_raw) if exit_time_raw else pd.Timestamp(datetime.now(UTC))
		broker_reason = None
		if isinstance(confirm, dict):
			broker_reason = str(confirm.get("reason") or "") or None
		self._record_live_close(
			position,
			exit_price=exit_price,
			exit_time=exit_time,
			exit_reason="reverse_signal",
			close_source="bot_active_close",
			close_reason_broker=broker_reason,
		)
		self.logger.info(
			"LIVE CLOSE deal_id=%s reason=reverse_signal side=%s exit=%.4f time=%s broker_reason=%s",
			position.deal_id,
			close_direction,
			exit_price,
			pd.Timestamp(self.state.last_exit_time).isoformat() if self.state.last_exit_time else "unknown",
			broker_reason,
		)
		return True

	def run_once(self) -> None:
		self._run_market_data_cycle(force=self.cfg.market_sync_only)
		if self.cfg.market_sync_only:
			self.logger.info("Market-sync-only mode completed successfully; skipping model load/trading logic.")
			self._save_state()
			return

		if self.cfg.signal_model_family == "best_base_state":
			raw = self._run_prediction_cycle(force=self.cfg.once)
			if raw is None:
				now_utc = pd.Timestamp(datetime.now(UTC)).floor("min")
				last_wait_bucket = getattr(self, "_last_prediction_wait_bucket_utc", None)
				if last_wait_bucket is None or now_utc > last_wait_bucket:
					self.logger.info(
						"Best-base prediction cycle not due yet; waiting for second=%d.",
						self.cfg.prediction_poll_second,
					)
					self._last_prediction_wait_bucket_utc = now_utc
				self._save_state()
				return
		else:
			raw = self._load_recent_raw_data()
		latest_raw_ts = pd.Timestamp(raw.index[-1])
		if self.cfg.signal_model_family == "legacy_15m_nextbar":
			_, complete_bars = build_complete_15m_bars(raw)
			self.logger.info(
				"Loaded raw=%d complete_15m=%d latest_raw=%s latest_complete=%s",
				len(raw),
				len(complete_bars),
				latest_raw_ts.isoformat(),
				complete_bars.index[-1].isoformat() if not complete_bars.empty else "none",
			)
			if len(complete_bars) < MIN_FEATURE_BARS + 1:
				raise ValueError(f"Need at least {MIN_FEATURE_BARS + 1} completed 15m bars, got {len(complete_bars)}")
			signal = self._build_latest_signal_legacy(complete_bars)
			signal_entry_time = signal["signal_bar_time"] + BAR_INTERVAL
		else:
			self.logger.info(
				"Loaded raw=%d latest_raw=%s signal_model_family=%s",
				len(raw),
				latest_raw_ts.isoformat(),
				self.cfg.signal_model_family,
			)
			signal = self._build_latest_signal_best_base(raw)
			signal_entry_time = pd.Timestamp(signal["entry_bar_time"])
			payload_info = dict(getattr(self, "last_best_base_payload_info", {}) or {})
			raw_info = dict(getattr(self, "last_raw_cache_info", {}) or {})
			self.logger.info(
				"BEST_BASE INPUT: source=%s reload_each_cycle=%s recent_days=%s raw_rows=%s bars_1m=%s candidates=%s window=%s horizon=%s pred_history=%s start=%s end=%s",
				raw_info.get("storage_mode", "mysql_recent_window_reload"),
				raw_info.get("reload_each_cycle", True),
				raw_info.get("recent_days", self.cfg.recent_days),
				raw_info.get("rows", len(raw)),
				payload_info.get("bars_rows"),
				payload_info.get("candidate_samples"),
				payload_info.get("window"),
				payload_info.get("horizon"),
				payload_info.get("pred_history_len"),
				raw_info.get("start_utc"),
				raw_info.get("end_utc"),
			)

		now_cycle_utc = pd.Timestamp(datetime.now(UTC))
		signal_trading_open = instrument_trading_hours_open(Price.Gold, now_cycle_utc)
		weak_filter_enabled = bool(getattr(self, "weak_period_cells", None))
		weak_period_block = self._is_weak_period_entry(pd.Timestamp(signal.get("entry_bar_time", signal.get("signal_bar_time"))))
		if weak_period_block:
			signal["tradable"] = False
		signal_qualified = bool(signal.get("tradable", True))
		if self.cfg.signal_model_family == "legacy_15m_nextbar":
			signal_qualified = signal_qualified and float(signal.get("probability", 0.0)) >= self.cfg.probability_cutoff

		if self.state.open_position is not None and self.cfg.mode == "paper":
			if self.cfg.signal_model_family != "legacy_15m_nextbar":
				raise ValueError("Paper mode currently supports only --signal-model-family legacy_15m_nextbar. Use --mode signal_only for the best-base model.")
			self._maybe_adjust_open_position_from_previous_minute(raw)
			self._maybe_dynamic_target_stop_from_signal(raw, signal)
			updated_position, trade = self.paper_broker.update_position(self.state.open_position, complete_bars)
			self.state.open_position = updated_position
			if trade:
				self.state.last_exit_time = str(trade["exit_time"])
				self.state.last_exit_reason = str(trade["exit_reason"])
				self.logger.info("Closed trade: %s", json.dumps(trade, sort_keys=True))
				self._save_state()

		if self.state.open_position is not None and self.cfg.mode == "live":
			if self._maybe_timeout_live_position():
				self._save_state()
				return
			self._sync_live_open_position()
			if self.state.open_position is not None:
				self._maybe_adjust_open_position_from_previous_minute(raw)
				self._maybe_dynamic_target_stop_from_signal(raw, signal)
				if self._maybe_close_live_position_from_signal(signal, raw):
					self._save_state()
					return

		if self.state.open_position is not None:
			self.logger.info("Open position remains active; skipping new entry checks.")
			self._save_state()
			return

		if self.cfg.signal_model_family == "best_base_state":
			payload_info = dict(getattr(self, "last_best_base_payload_info", {}) or {})
			raw_info = dict(getattr(self, "last_raw_cache_info", {}) or {})
			self.logger.info(
				format_best_base_signal_log(
					signal,
					raw_rows=int(raw_info["rows"]) if "rows" in raw_info else None,
					bars_rows=int(payload_info["bars_rows"]) if "bars_rows" in payload_info else None,
					candidate_samples=int(payload_info["candidate_samples"]) if "candidate_samples" in payload_info else None,
					is_trading_hour=signal_trading_open,
					latest_close=float(payload_info["latest_close"]) if payload_info.get("latest_close") is not None else None,
					range150_ok=bool(payload_info["latest_range150_ok"]) if payload_info.get("latest_range150_ok") is not None else None,
					drop15m_ok=bool(payload_info["latest_15m_drop_ok"]) if payload_info.get("latest_15m_drop_ok") is not None else None,
				)
			)
			self.logger.info(
				format_signal_status_line(
					signal=signal,
					trading_open_now=signal_trading_open,
					signal_qualified=signal_qualified,
					weak_filter_enabled=weak_filter_enabled,
					weak_period_block=weak_period_block,
					now_utc=now_cycle_utc,
					cutoff=None,
				)
			)
		else:
			self.logger.info(
				"Latest signal family=%s bar=%s side=%s prob=%.4f cutoff=%.4f market_open=%d",
				self.cfg.signal_model_family,
				signal["signal_bar_time"].isoformat(),
				signal.get("side", "long"),
				signal["probability"],
				self.cfg.probability_cutoff,
				int(signal_trading_open),
			)
			self.logger.info(
				format_signal_status_line(
					signal=signal,
					trading_open_now=signal_trading_open,
					signal_qualified=signal_qualified,
					weak_filter_enabled=weak_filter_enabled,
					weak_period_block=weak_period_block,
					now_utc=now_cycle_utc,
					cutoff=self.cfg.probability_cutoff,
				)
			)

		if not bool(signal.get("tradable", True)):
			self._save_state()
			return

		if self.cfg.signal_model_family == "legacy_15m_nextbar" and signal["probability"] < self.cfg.probability_cutoff:
			self._save_state()
			return

		signal_bar_time = signal["signal_bar_time"]
		trades = load_trade_log(self.paper_broker.trade_log_path)
		block_reason = entry_block_reason(self.cfg, self.state, signal_bar_time, trades)
		if block_reason is not None:
			self.logger.info("Signal qualified but entry blocked: %s", block_reason)
			self._save_state()
			return

		if self.state.last_signal_bar_time == signal_bar_time.isoformat():
			self.logger.info("Signal bar already processed; waiting for the next closed bar.")
			self._save_state()
			return

		entry = find_next_entry_minute(raw, signal_entry_time)
		if entry is None:
			if self.cfg.mode != "signal_only":
				self.logger.info("Signal qualified, but next entry minute is not available yet.")
				self._save_state()
				return
			entry_time = pd.Timestamp(signal_entry_time)
			entry_time = entry_time.tz_convert("UTC") if entry_time.tzinfo else entry_time.tz_localize("UTC")
			entry_row = raw.iloc[-1]
			self.logger.info(
				"Signal-only fallback entry used because next minute is unavailable: entry_time=%s",
				entry_time.isoformat(),
			)
		else:
			entry_time, entry_row = entry
		entry_price = float(entry_row.get("openPrice_ask", entry_row.get("openPrice", entry_row.get("closePrice"))))
		self.state.last_signal_bar_time = signal_bar_time.isoformat()

		if self.cfg.mode == "signal_only":
			engine = getattr(self, "execution_engine", None)
			attempt = engine.handle_signal(
				mode=self.cfg.mode,
				signal_model_family=self.cfg.signal_model_family,
				signal=signal,
				entry_time=entry_time,
				entry_price=entry_price,
				size=self.cfg.size,
			) if engine is not None else None
			self.state.last_execution_attempt = attempt
			self.logger.info(
				"SIGNAL ONLY: side=%s entry_time=%s entry_price=%.2f prob=%.4f family=%s exec_status=%s",
				signal.get("side", "long"),
				entry_time.isoformat(),
				entry_price,
				signal["probability"],
				self.cfg.signal_model_family,
				attempt.get("status") if isinstance(attempt, dict) else "not_available",
			)
			self._save_state()
			return

		if self.cfg.mode == "live":
			engine = getattr(self, "execution_engine", None)
			attempt = engine.handle_signal(
				mode=self.cfg.mode,
				signal_model_family=self.cfg.signal_model_family,
				signal=signal,
				entry_time=entry_time,
				entry_price=entry_price,
				size=self.cfg.size,
			) if engine is not None else None
			self.state.last_execution_attempt = attempt
			if not isinstance(attempt, dict) or not bool(attempt.get("submitted")):
				self.logger.warning("LIVE submit failed or was rejected: %s", attempt)
				self._save_state()
				return
			deal_id = str(attempt.get("deal_id") or attempt.get("broker_order_id") or "")
			if not deal_id:
				self.logger.warning("LIVE submit returned no deal identifier; cannot track open position")
				self._save_state()
				return
			direction = "LONG" if str(signal.get("side", "up")) == "up" else "SHORT"
			if direction == "SHORT":
				stop_loss = entry_price * (1.0 + self.cfg.stop_loss_pct / 100.0)
				take_profit = entry_price * (1.0 - self.cfg.take_profit_pct / 100.0)
			else:
				stop_loss = entry_price * (1.0 - self.cfg.stop_loss_pct / 100.0)
				take_profit = entry_price * (1.0 + self.cfg.take_profit_pct / 100.0)
			entry_bar_time = pd.Timestamp(signal.get("entry_bar_time", entry_time))
			self.state.open_position = PaperPosition(
				direction=direction,
				deal_id=deal_id,
				signal_bar_time=signal_bar_time.isoformat(),
				entry_bar_time=entry_bar_time.isoformat(),
				entry_time=entry_time.isoformat(),
				entry_price=float(entry_price),
				stop_loss=float(stop_loss),
				take_profit=float(take_profit),
				probability=float(signal["probability"]),
				size=float(self.cfg.size),
				entry_price_initial=float(entry_price),
			)
			self.logger.info(
				"LIVE ENTRY submitted deal_id=%s side=%s entry=%.4f stop=%.4f target=%.4f",
				deal_id,
				direction,
				entry_price,
				stop_loss,
				take_profit,
			)
			self._save_state()
			return

		if self.cfg.signal_model_family != "legacy_15m_nextbar":
			raise ValueError("Paper mode currently supports only --signal-model-family legacy_15m_nextbar. Use --mode signal_only for the best-base model.")

		self.state.open_position = self.paper_broker.open_long(
			signal_bar_time=signal_bar_time,
			entry_bar_time=signal_bar_time + BAR_INTERVAL,
			entry_time=entry_time,
			entry_price=entry_price,
			probability=float(signal["probability"]),
			deal_id=self.state.last_execution_attempt.get("deal_id") if isinstance(self.state.last_execution_attempt, dict) else None,
		)
		self._save_state()

	def _load_recent_raw_data(self) -> pd.DataFrame:
		today_utc = datetime.now(UTC).date()
		start_date = (today_utc - timedelta(days=max(self.cfg.recent_days, 2))).isoformat()
		end_date = today_utc.isoformat()
		raw = DataLoader().load_data(self.cfg.table, start_date=start_date, end_date=end_date)
		raw = prepare_raw_price_frame(raw)
		self.last_raw_cache_info = {
			"storage_mode": "mysql_recent_window_reload",
			"reload_each_cycle": True,
			"table": self.cfg.table,
			"recent_days": int(self.cfg.recent_days),
			"rows": int(len(raw)),
			"start_utc": raw.index[0].isoformat(),
			"end_utc": raw.index[-1].isoformat(),
		}
		return raw

	def _build_latest_signal_legacy(self, complete_bars: pd.DataFrame) -> dict[str, object]:
		signal_idx = len(complete_bars) - 1
		feature_payload = self.system.extractor.extract_all_features(complete_bars, signal_idx)
		model_feature_names = list(self.model.feature_names or [])
		if not model_feature_names:
			raise ValueError("Loaded model does not expose feature_names")
		row = {name: feature_payload.get(name, 0.0) for name in model_feature_names}
		X = pd.DataFrame([row], columns=model_feature_names).fillna(0.0)
		probability = float(self.model.predict_proba(X)[0])
		return {
			"signal_bar_time": pd.Timestamp(complete_bars.index[signal_idx]),
			"probability": probability,
			"side": "long",
			"tradable": probability >= self.cfg.probability_cutoff,
			"features": row,
		}

	def _build_live_best_base_samples(self, raw: pd.DataFrame, *, require_future_horizon: bool = False) -> dict[str, object]:
		if self.image_trend is None or self.model_bundle is None:
			raise ValueError("Best-base signal model is not loaded")
		cfg = dict(self.model_bundle.get("config") or {})
		bars = self.image_trend._prepare_ohlcv(raw, str(cfg.get("timeframe", "1min")))
		window = int(cfg.get("window", 150))
		horizon = int(cfg.get("horizon", 25))
		pred_history_len = int(cfg.get("pred_history_len", 150))
		window_15m = int(cfg.get("window_15m", 0))
		if window_15m > 0:
			raise ValueError("Best-base live signal generation currently supports only window_15m=0")
		max_i_exclusive = len(bars) - horizon if require_future_horizon else len(bars)
		empty_payload = {
			"bars": bars,
			"X_flat": np.zeros((0, 0), dtype=np.float64),
			"ts": pd.DatetimeIndex([]),
			"curr": np.zeros(0),
			"entry": np.zeros(0),
			"entry_ts": pd.DatetimeIndex([]),
			"fut": np.zeros(0),
			"fut_ts": pd.DatetimeIndex([]),
			"stage1_extra": None,
		}
		self.last_best_base_payload_info = {
			"bars_rows": int(len(bars)),
			"candidate_samples": 0,
			"window": window,
			"horizon": horizon,
			"pred_history_len": pred_history_len,
			"require_future_horizon": bool(require_future_horizon),
			"latest_close": float(bars["close"].iloc[-1]) if len(bars) else None,
			"latest_range150_ok": None,
			"latest_15m_drop_ok": None,
		}
		if len(bars) < window:
			return empty_payload
		if max_i_exclusive <= window - 1:
			return empty_payload
		tensors: list[np.ndarray] = []
		ts_list: list[pd.Timestamp] = []
		curr_list: list[float] = []
		entry_list: list[float] = []
		entry_ts_list: list[pd.Timestamp] = []
		fut_list: list[float] = []
		fut_ts_list: list[pd.Timestamp] = []
		apply_time_filter = not bool(cfg.get("disable_time_filter", False))
		min_window_range = float(cfg.get("min_window_range", 0.0))
		min_15m_drop = float(cfg.get("min_15m_drop", 0.0))
		min_15m_rise = float(cfg.get("min_15m_rise", 0.0))
		last_bar_wr90_high = cfg.get("last_bar_wr90_high")
		last_bar_wr90_low = cfg.get("last_bar_wr90_low")
		use_15m_wick_features = bool(self.model_bundle.get("use_15m_wick_features", cfg.get("use_15m_wick_features", False)))
		wick_feature_min_range = float(self.model_bundle.get("wick_feature_min_range", cfg.get("wick_feature_min_range", 40.0)))
		wick_feature_min_pct = float(self.model_bundle.get("wick_feature_min_pct", cfg.get("wick_feature_min_pct", 35.0)))
		wick_feature_min_volume = float(self.model_bundle.get("wick_feature_min_volume", cfg.get("wick_feature_min_volume", 3000.0)))
		if len(bars) >= window:
			w_latest = bars.iloc[-window:]
			range150_ok = float(w_latest["high"].max() - w_latest["low"].min()) > min_window_range
			drop15_ok = True
			if min_15m_drop > 0 and len(w_latest) >= 15:
				h_latest = w_latest["high"].to_numpy()
				l_latest = w_latest["low"].to_numpy()
				roll_h_latest = np.lib.stride_tricks.sliding_window_view(h_latest, 15).max(axis=1)
				roll_l_latest = np.lib.stride_tricks.sliding_window_view(l_latest, 15).min(axis=1)
				drop15_ok = bool(((roll_h_latest - roll_l_latest) >= min_15m_drop).any())
			self.last_best_base_payload_info["latest_range150_ok"] = bool(range150_ok)
			self.last_best_base_payload_info["latest_15m_drop_ok"] = bool(drop15_ok)
		for i in range(window - 1, max_i_exclusive):
			signal_ts = pd.Timestamp(bars.index[i])
			if apply_time_filter and self.image_trend._is_blocked(signal_ts):
				continue
			w = bars.iloc[i - window + 1 : i + 1]
			if float(w["high"].max() - w["low"].min()) <= min_window_range:
				continue
			if (min_15m_drop > 0 or min_15m_rise > 0) and len(w) >= 15:
				h = w["high"].to_numpy()
				l = w["low"].to_numpy()
				c = w["close"].to_numpy()
				roll_h = np.lib.stride_tricks.sliding_window_view(h, 15).max(axis=1)
				roll_l = np.lib.stride_tricks.sliding_window_view(l, 15).min(axis=1)
				roll_c0 = np.lib.stride_tricks.sliding_window_view(c, 15)[:, 0]
				drop_ok = False if min_15m_drop <= 0 else bool(((roll_h - roll_l) >= min_15m_drop).any())
				rise_ok = False if min_15m_rise <= 0 else bool(((roll_h - roll_c0) >= min_15m_rise).any())
				if not (drop_ok or rise_ok):
					continue
			if last_bar_wr90_high is not None or last_bar_wr90_low is not None:
				if len(w) < 90:
					continue
				wr_src = w.iloc[-90:]
				wr_high = float(wr_src["high"].max())
				wr_low = float(wr_src["low"].min())
				wr_close = float(wr_src["close"].iloc[-1])
				wr_span = wr_high - wr_low
				if wr_span <= 0.0:
					continue
				wr90_last = -100.0 * ((wr_high - wr_close) / wr_span)
				high_ok = False if last_bar_wr90_high is None else (wr90_last >= float(last_bar_wr90_high))
				low_ok = False if last_bar_wr90_low is None else (wr90_last <= float(last_bar_wr90_low))
				if not (high_ok or low_ok):
					continue
			wick_extra = None
			if use_15m_wick_features:
				wick_flags = list(self.image_trend._last_completed_15m_wick_flags(
					w,
					min_range=wick_feature_min_range,
					min_wick_pct=wick_feature_min_pct,
					min_volume=wick_feature_min_volume,
				))
				wick_extra = self.image_trend._constant_feature_channels(len(w), wick_flags)
			tensors.append(self.image_trend._window_to_image(w, extra_channels=wick_extra))
			ts_list.append(signal_ts)
			curr_close = float(bars["close"].iloc[i])
			curr_list.append(curr_close)
			entry_ts = pd.Timestamp(bars.index[i + 1]) if i + 1 < len(bars) else signal_ts + pd.Timedelta(minutes=1)
			entry_ts_list.append(entry_ts)
			entry_list.append(float(bars["open"].iloc[i + 1]) if i + 1 < len(bars) else curr_close)
			future_ts = pd.Timestamp(bars.index[i + horizon]) if i + horizon < len(bars) else signal_ts + pd.Timedelta(minutes=horizon)
			fut_ts_list.append(future_ts)
			fut_list.append(float(bars["close"].iloc[i + horizon]) if i + horizon < len(bars) else np.nan)
		if not tensors:
			return empty_payload
		X_flat = self.image_trend.flatten_tensors(np.stack(tensors, axis=0))
		ts_idx = pd.DatetimeIndex(ts_list)
		curr = np.asarray(curr_list, dtype=np.float64)
		entry = np.asarray(entry_list, dtype=np.float64)
		entry_ts_idx = pd.DatetimeIndex(entry_ts_list)
		fut = np.asarray(fut_list, dtype=np.float64)
		fut_ts_idx = pd.DatetimeIndex(fut_ts_list)
		stage1_extra = (
			self.image_trend._build_stage1_day_ohl_features(bars, ts_idx, curr)
			if bool(self.model_bundle.get("use_stage1_day_ohl_utc2", False)) else None
		)
		self.last_best_base_payload_info = {
			"bars_rows": int(len(bars)),
			"candidate_samples": int(len(ts_idx)),
			"window": window,
			"horizon": horizon,
			"pred_history_len": pred_history_len,
			"require_future_horizon": bool(require_future_horizon),
			"first_signal_utc": ts_idx[0].isoformat() if len(ts_idx) else None,
			"last_signal_utc": ts_idx[-1].isoformat() if len(ts_idx) else None,
			"latest_close": float(bars["close"].iloc[-1]) if len(bars) else None,
			"latest_range150_ok": self.last_best_base_payload_info.get("latest_range150_ok"),
			"latest_15m_drop_ok": self.last_best_base_payload_info.get("latest_15m_drop_ok"),
		}
		return {
			"bars": bars,
			"X_flat": X_flat,
			"ts": ts_idx,
			"curr": curr,
			"entry": entry,
			"entry_ts": entry_ts_idx,
			"fut": fut,
			"fut_ts": fut_ts_idx,
			"stage1_extra": stage1_extra,
		}

	def _build_best_base_signal_series(self, raw: pd.DataFrame, *, require_future_horizon: bool = False) -> dict[str, object]:
		payload = self._build_live_best_base_samples(raw, require_future_horizon=require_future_horizon)
		X_flat = payload["X_flat"]
		if len(X_flat) == 0:
			return {
				**payload,
				"pred": np.zeros(0, dtype=np.int64),
				"signal_prob": np.zeros(0, dtype=np.float64),
				"trend_prob": np.zeros(0, dtype=np.float64),
				"up_prob": np.full(0, np.nan, dtype=np.float64),
			}
		pm = self._build_best_base_live_predictions(
			X_flat,
			payload["curr"],
			payload["entry"],
			payload["stage1_extra"],
		)
		return {
			**payload,
			"pred": pm["pred"],
			"signal_prob": pm["signal_prob"],
			"trend_prob": pm["trend_prob"],
			"up_prob": pm["up_prob"],
		}

	def _build_best_base_live_predictions(self, X_flat: np.ndarray, curr: np.ndarray, entry: np.ndarray, stage1_extra: Optional[np.ndarray]) -> dict[str, np.ndarray]:
		if self.image_trend is None or self.model_bundle is None:
			raise ValueError("Best-base signal model is not loaded")
		n = int(len(X_flat))
		if n == 0:
			return {
				"pred": np.zeros(0, dtype=np.int64),
				"signal_prob": np.zeros(0, dtype=np.float64),
				"trend_prob": np.zeros(0, dtype=np.float64),
				"up_prob": np.full(0, np.nan, dtype=np.float64),
			}
		cfg = dict(self.model_bundle.get("config") or {})
		m1 = self.model_bundle["stage1"]
		m2 = self.model_bundle["stage2"]
		use_state = bool(self.model_bundle.get("use_state_features", cfg.get("use_state_features", False)))
		stage1_prob = float(self.model_bundle.get("stage1_min_prob", cfg.get("stage1_min_prob", 0.48)))
		stage2_prob = float(self.model_bundle.get("stage2_min_prob", cfg.get("stage2_min_prob", 0.50)))
		stage2_prob_up = self.model_bundle.get("stage2_min_prob_up", cfg.get("stage2_min_prob_up"))
		stage2_prob_down = self.model_bundle.get("stage2_min_prob_down", cfg.get("stage2_min_prob_down"))
		pred = np.full(n, self.image_trend.LABEL_FLAT, dtype=np.int64)
		signal_prob = np.zeros(n, dtype=np.float64)
		trend_prob = np.zeros(n, dtype=np.float64)
		up_prob = np.full(n, np.nan, dtype=np.float64)
		if not use_state:
			pm = self.image_trend.predict_two_stage_details(
				X_flat,
				m1,
				m2,
				stage1_min_prob=stage1_prob,
				stage2_min_prob=stage2_prob,
				stage2_min_prob_up=None if stage2_prob_up is None else float(stage2_prob_up),
				stage2_min_prob_down=None if stage2_prob_down is None else float(stage2_prob_down),
				stage1_min_prob_1m=cfg.get("stage1_min_prob_1m"),
				stage1_min_prob_15m=cfg.get("stage1_min_prob_15m"),
				stage2_min_prob_1m=cfg.get("stage2_min_prob_1m"),
				stage2_min_prob_15m=cfg.get("stage2_min_prob_15m"),
				stage1_extra=stage1_extra,
			)
			return {
				"pred": pm["pred"],
				"signal_prob": pm["signal_prob"],
				"trend_prob": pm["trend_prob"],
				"up_prob": pm["up_prob"],
			}
		hist = int(cfg.get("pred_history_len", 150))
		n_hist_feats = hist * 3
		max_bars = 120
		long_thr = float(cfg.get("long_target_threshold") or cfg.get("trend_threshold", 0.008))
		short_thr = float(cfg.get("short_target_threshold") or cfg.get("trend_threshold", 0.008))
		long_stop = float(cfg.get("long_adverse_limit") or cfg.get("adverse_limit", 15.0))
		short_stop = float(cfg.get("short_adverse_limit") or cfg.get("adverse_limit", 15.0))
		pos = 0
		entry_px = 0.0
		bars_in_pos = 0
		target_hit_flag = 0.0
		stop_hit_flag = 0.0
		for i in range(n):
			state_vec = np.zeros((1, n_hist_feats + 7), dtype=np.float64)
			for lag in range(1, hist + 1):
				j = i - lag
				if j < 0:
					continue
				cls = int(pred[j])
				if cls == self.image_trend.LABEL_RISKY:
					cls = self.image_trend.LABEL_FLAT
				if cls in (self.image_trend.LABEL_DOWN, self.image_trend.LABEL_FLAT, self.image_trend.LABEL_UP):
					state_vec[0, (lag - 1) * 3 + cls] = 1.0
			b = n_hist_feats
			state_vec[0, b + (pos + 1)] = 1.0
			state_vec[0, b + 3] = min(float(bars_in_pos), float(max_bars)) / float(max_bars)
			if pos != 0 and entry_px > 0.0:
				state_vec[0, b + 4] = ((float(curr[i]) - entry_px) * float(pos)) / entry_px
			state_vec[0, b + 5] = target_hit_flag
			state_vec[0, b + 6] = stop_hit_flag
			X_i = np.hstack([X_flat[i : i + 1], state_vec])
			pm = self.image_trend.predict_two_stage_details(
				X_i,
				m1,
				m2,
				stage1_min_prob=stage1_prob,
				stage2_min_prob=stage2_prob,
				stage2_min_prob_up=None if stage2_prob_up is None else float(stage2_prob_up),
				stage2_min_prob_down=None if stage2_prob_down is None else float(stage2_prob_down),
				stage1_min_prob_1m=cfg.get("stage1_min_prob_1m"),
				stage1_min_prob_15m=cfg.get("stage1_min_prob_15m"),
				stage2_min_prob_1m=cfg.get("stage2_min_prob_1m"),
				stage2_min_prob_15m=cfg.get("stage2_min_prob_15m"),
				stage1_extra=None if stage1_extra is None else stage1_extra[i : i + 1],
			)
			pred_i = int(pm["pred"][0])
			pred[i] = pred_i
			signal_prob[i] = float(pm["signal_prob"][0])
			trend_prob[i] = float(pm["trend_prob"][0])
			up_prob_val = pm["up_prob"][0]
			up_prob[i] = float(up_prob_val) if not np.isnan(up_prob_val) else np.nan
			target_hit_flag = 0.0
			stop_hit_flag = 0.0
			if pos != 0 and entry_px > 0.0:
				move_abs = (float(curr[i]) - entry_px) * float(pos)
				target_abs = abs(entry_px) * float(long_thr if pos > 0 else short_thr)
				stop_abs = float(long_stop if pos > 0 else short_stop)
				if move_abs >= target_abs:
					pos = 0
					entry_px = 0.0
					bars_in_pos = 0
					target_hit_flag = 1.0
				elif move_abs <= -stop_abs:
					pos = 0
					entry_px = 0.0
					bars_in_pos = 0
					stop_hit_flag = 1.0
				else:
					bars_in_pos += 1
			if pos == 0 and pred_i in self.image_trend.TREND_LABELS:
				pos = 1 if pred_i == self.image_trend.LABEL_UP else -1
				entry_px = float(entry[i])
				bars_in_pos = 0
		return {"pred": pred, "signal_prob": signal_prob, "trend_prob": trend_prob, "up_prob": up_prob}

	def _build_latest_signal_best_base(self, raw: pd.DataFrame) -> dict[str, object]:
		payload = self._build_best_base_signal_series(raw)
		ts_idx = payload["ts"]
		if len(ts_idx) == 0:
			return {
				"signal_bar_time": pd.Timestamp(raw.index[-1]),
				"entry_bar_time": pd.Timestamp(raw.index[-1]) + pd.Timedelta(minutes=1),
				"probability": 0.0,
				"side": "flat",
				"tradable": False,
			}
		pred = int(payload["pred"][-1])
		prob = float(payload["signal_prob"][-1])
		trend_prob = float(payload["trend_prob"][-1]) if len(payload.get("trend_prob", [])) else 0.0
		up_prob_raw = payload["up_prob"][-1] if len(payload.get("up_prob", [])) else np.nan
		direction_prob = None if pd.isna(up_prob_raw) else max(float(up_prob_raw), 1.0 - float(up_prob_raw))
		side = (
			"up" if pred == self.image_trend.LABEL_UP
			else "down" if pred == self.image_trend.LABEL_DOWN
			else "risk_off" if pred == self.image_trend.LABEL_RISKY
			else "flat"
		)
		reject_reason = None
		display_prob = prob
		if pred not in self.image_trend.TREND_LABELS:
			if direction_prob is not None:
				display_prob = direction_prob
				reject_reason = "risk_off" if pred == getattr(self.image_trend, "LABEL_RISKY", object()) else "stage2_gate"
			else:
				display_prob = trend_prob
				reject_reason = "stage1_gate"
		signal_ts = pd.Timestamp(ts_idx[-1])
		return {
			"signal_bar_time": signal_ts,
			"entry_bar_time": pd.Timestamp(payload["entry_ts"][-1]),
			"probability": float(display_prob),
			"signal_probability": prob,
			"trend_probability": float(trend_prob),
			"direction_probability": None if direction_prob is None else float(direction_prob),
			"side": side,
			"tradable": pred in self.image_trend.TREND_LABELS,
			"reject_reason": reject_reason,
			"pred": pred,
		}

	def _build_logger(self, path: Path) -> logging.Logger:
		path.parent.mkdir(parents=True, exist_ok=True)
		logger = logging.getLogger("alphagold_trading_bot")
		logger.setLevel(logging.INFO)
		logger.handlers.clear()
		formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

		file_handler = logging.FileHandler(path, encoding="utf-8")
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

		stream_handler = logging.StreamHandler(sys.stdout)
		stream_handler.setFormatter(formatter)
		logger.addHandler(stream_handler)
		return logger

	def _load_state(self) -> BotState:
		if not self.state_path.exists():
			return BotState()
		return BotState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))

	def _maybe_emit_scheduled_performance_log(self, trades: pd.DataFrame, now_ts: pd.Timestamp) -> None:
		now_utc = pd.Timestamp(now_ts)
		if now_utc.tzinfo is None:
			now_utc = now_utc.tz_localize(UTC)
		else:
			now_utc = now_utc.tz_convert(UTC)
		bucket = now_utc.floor("min")
		if bucket.minute not in (0, 30):
			return
		bucket_iso = bucket.isoformat()
		if self.state.last_periodic_report_bucket_utc == bucket_iso:
			return

		windows = summarize_trade_windows(trades, now_utc, windows_minutes=(30, 60))
		daily = summarize_daily_trade_log(trades, now_utc)
		weekly = summarize_weekly_trade_log(trades, now_utc)

		w30 = windows.get("30m", {})
		w60 = windows.get("60m", {})
		today = daily.get("today", {}) if isinstance(daily, dict) else {}
		this_week = weekly.get("this_week", {}) if isinstance(weekly, dict) else {}

		self.logger.info(
			"PERF SNAPSHOT bucket=%s 30m(trades=%s pnl=%.2f win=%.1f%%) 60m(trades=%s pnl=%.2f win=%.1f%%) today(trades=%s pnl=%.2f) week(trades=%s pnl=%.2f)",
			bucket_iso,
			w30.get("trades", 0),
			float(w30.get("realized_pnl_usd", 0.0) or 0.0),
			float(w30.get("win_rate_pct", 0.0) or 0.0),
			w60.get("trades", 0),
			float(w60.get("realized_pnl_usd", 0.0) or 0.0),
			float(w60.get("win_rate_pct", 0.0) or 0.0),
			today.get("trades", 0),
			float(today.get("realized_pnl_usd", 0.0) or 0.0),
			this_week.get("trades", 0),
			float(this_week.get("realized_pnl_usd", 0.0) or 0.0),
		)
		self.state.last_periodic_report_bucket_utc = bucket_iso

	def _save_state(self) -> None:
		self.state_path.write_text(json.dumps(self.state.to_dict(), indent=2), encoding="utf-8")
		now_ts = pd.Timestamp(datetime.now(UTC))
		trades = load_trade_log(self.paper_broker.trade_log_path)
		self._maybe_emit_scheduled_performance_log(trades, now_ts)
		status = summarize_trade_log(trades, now_ts)
		collector = self.market_data_collector
		last_bucket = getattr(collector, "_last_bucket", None)
		status["mode"] = self.cfg.mode
		status["signal_model_family"] = self.cfg.signal_model_family
		status["signal_model_path"] = self.cfg.signal_model_path
		status["configured_position_size"] = float(self.cfg.size)
		status["market_sync_only"] = self.cfg.market_sync_only
		status["open_position"] = asdict(self.state.open_position) if self.state.open_position else None
		status["last_signal_bar_time"] = self.state.last_signal_bar_time
		status["last_exit_time"] = self.state.last_exit_time
		status["last_exit_reason"] = self.state.last_exit_reason
		status["last_execution_attempt"] = self.state.last_execution_attempt
		status["max_trades_per_day"] = self.cfg.max_trades_per_day
		status["cooldown_bars_after_exit"] = self.cfg.cooldown_bars_after_exit
		status["market_data_enabled"] = self.cfg.market_data_enabled
		status["market_data_tables"] = list(MARKET_DATA_TABLES)
		status["prediction_poll_second"] = self.cfg.prediction_poll_second
		status["prediction_cache_last_error"] = getattr(self.prediction_cache, "last_error", None)
		status["prediction_cache_last_bucket_utc"] = (
			self.prediction_cache._last_bucket.isoformat()
			if getattr(self.prediction_cache, "_last_bucket", None) is not None else None
		)
		status["prediction_cache_last_summary"] = dict(getattr(self.prediction_cache, "last_fetch_summary", {}) or {})
		status["market_data_poll_second"] = self.cfg.market_data_poll_second
		status["market_data_last_error"] = getattr(collector, "last_error", None)
		status["last_market_data_bucket_utc"] = (
			last_bucket.isoformat()
			if last_bucket is not None else None
		)
		status["market_data_last_sync_summaries"] = list(getattr(collector, "last_sync_summaries", []))
		status["input_data"] = dict(getattr(self, "last_raw_cache_info", {}) or {})
		status["best_base_runtime"] = dict(getattr(self, "last_best_base_payload_info", {}) or {})
		status["performance_windows"] = summarize_trade_windows(trades, now_ts, windows_minutes=(30, 60))
		status["daily_summary"] = summarize_daily_trade_log(trades, now_ts)
		status["weekly_summary"] = summarize_weekly_trade_log(trades, now_ts)
		status["last_periodic_report_bucket_utc"] = self.state.last_periodic_report_bucket_utc
		self.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def build_complete_15m_bars(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	bars_15m = _load_15m_nextbar_module().prepare_gold_data_15m(raw)
	if bars_15m.empty:
		return bars_15m, bars_15m

	latest_raw_ts = pd.Timestamp(raw.index[-1])
	last_bar_start = pd.Timestamp(bars_15m.index[-1])
	last_bar_complete_at = last_bar_start + pd.Timedelta(minutes=14)
	complete_bars = bars_15m.iloc[:-1].copy() if latest_raw_ts < last_bar_complete_at else bars_15m.copy()
	return bars_15m, complete_bars


def find_next_entry_minute(raw: pd.DataFrame, interval_start: pd.Timestamp) -> Optional[tuple[pd.Timestamp, pd.Series]]:
	ts = pd.Timestamp(interval_start)
	interval_start = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
	idx = raw.index.searchsorted(interval_start, side="left")
	if idx >= len(raw):
		return None
	candidate_ts = pd.Timestamp(raw.index[idx])
	if candidate_ts > interval_start + pd.Timedelta(minutes=2):
		return None
	return candidate_ts, raw.iloc[idx]


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="AlphaGold paper trading bot scaffold.")
	p.add_argument("--table", default="gold_prices")
	p.add_argument("--signal-model-family", default=DEFAULT_SIGNAL_MODEL_FAMILY,
		choices=["best_base_state", "legacy_15m_nextbar"])
	p.add_argument("--signal-model-path", default=DEFAULT_BEST_BASE_MODEL_PATH,
		help="Path to best-base no-retrain model artifact (default: runtime/backtest_model_best_base_weak_nostate.joblib).")
	p.add_argument("--model-type", default="gradient_boosting", choices=["random_forest", "gradient_boosting"])
	p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
	p.add_argument("--market-sync-only", action="store_true",
		help="Run only the market-data sync/snapshot task and status output, skipping model/trading logic.")
	p.add_argument("--recent-days", type=int, default=10)
	p.add_argument("--probability-cutoff", type=float, default=0.50)
	p.add_argument("--take-profit-pct", type=float, default=DEFAULT_THRESHOLD_PCT)
	p.add_argument("--stop-loss-pct", type=float, default=DEFAULT_MAX_ADVERSE_LOW_PCT)
	p.add_argument("--max-hold-bars", type=int, default=DEFAULT_FORWARD_BARS)
	p.add_argument("--size", type=float, default=1.0)
	p.add_argument("--mode", default="signal_only", choices=["paper", "signal_only", "live"])
	p.add_argument("--sleep-seconds", type=int, default=5)
	p.add_argument("--state-path", default="runtime/trading_bot_state.json")
	p.add_argument("--status-path", default="runtime/trading_bot_status.json")
	p.add_argument("--trade-log-path", default="runtime/trading_bot_trades.csv")
	p.add_argument("--log-path", default="runtime/trading_bot.log")
	p.add_argument("--lock-path", default="/tmp/alphagold_trading_bot.lock")
	p.add_argument("--max-trades-per-day", type=int, default=0,
		help="Optional cap on paper entries per trading day (NY 17:00 cutoff). 0 disables.")
	p.add_argument("--cooldown-bars-after-exit", type=int, default=0,
		help="Optional number of 15m bars to wait after a paper exit before allowing a new entry. 0 disables.")
	p.add_argument("--disable-market-data-capture", action="store_true",
		help="Disable the :30-per-minute MySQL backup/snapshot task.")
	p.add_argument("--prediction-poll-second", type=int, default=PREDICTION_POLL_SECOND,
		help="Second within each minute when the best-base bot should fetch the latest 1-minute IG data into memory for prediction (default: 5).")
	p.add_argument("--market-data-poll-second", type=int, default=MARKET_DATA_POLL_SECOND,
		help="Second within each minute when the MySQL backup/snapshot task should store latest IG data for later backtest usage (default: 30).")
	p.add_argument("--prediction-cache-max-rows", type=int, default=1200,
		help="Maximum rows kept in in-memory prediction cache (default: 1200).")
	p.add_argument("--weak-periods-json", default=DEFAULT_WEAK_PERIODS_JSON,
		help="Optional weak-period cells JSON to block entries in weak hours (default: runtime/bot_assets/weak-filter.json).")
	p.add_argument("--disable-dynamic-target-stop", action="store_true",
		help="Disable dynamic target/stop updates while a position is open.")
	p.add_argument("--max-hold-minutes", type=float, default=None,
		help="Optional hard timeout in minutes for live positions. Position is force-closed when elapsed >= this value (e.g. 60).")
	p.add_argument("--once", action="store_true")
	return p


def parse_args() -> BotConfig:
	args = build_parser().parse_args()
	if not (0.0 < args.probability_cutoff < 1.0):
		raise ValueError("--probability-cutoff must be in (0, 1)")
	if args.take_profit_pct <= 0.0:
		raise ValueError("--take-profit-pct must be > 0")
	if args.stop_loss_pct <= 0.0:
		raise ValueError("--stop-loss-pct must be > 0")
	if args.max_hold_bars < 1:
		raise ValueError("--max-hold-bars must be >= 1")
	if args.recent_days < 2:
		raise ValueError("--recent-days must be >= 2")
	if args.size <= 0.0:
		raise ValueError("--size must be > 0")
	if args.max_trades_per_day < 0:
		raise ValueError("--max-trades-per-day must be >= 0")
	if args.cooldown_bars_after_exit < 0:
		raise ValueError("--cooldown-bars-after-exit must be >= 0")
	if args.prediction_poll_second < 0 or args.prediction_poll_second > 59:
		raise ValueError("--prediction-poll-second must be within [0, 59]")
	if args.market_data_poll_second < 0 or args.market_data_poll_second > 59:
		raise ValueError("--market-data-poll-second must be within [0, 59]")
	if args.prediction_cache_max_rows < 220:
		raise ValueError("--prediction-cache-max-rows must be >= 220 for best-base window/state features")
	if args.signal_model_family == "best_base_state" and args.mode == "paper":
		raise ValueError("--signal-model-family best_base_state does not support --mode paper; use --mode signal_only or --mode live")
	if args.mode == "live" and args.signal_model_family != "best_base_state":
		raise ValueError("--mode live currently supports only --signal-model-family best_base_state")
	args.market_data_enabled = not bool(args.disable_market_data_capture)
	args.dynamic_target_stop_enabled = not bool(args.disable_dynamic_target_stop)
	if args.market_sync_only and not args.market_data_enabled:
		raise ValueError("--market-sync-only requires market-data capture to be enabled")
	delattr(args, "disable_market_data_capture")
	delattr(args, "disable_dynamic_target_stop")
	return BotConfig(**vars(args))


def main() -> int:
	cfg = parse_args()
	with SingleInstanceLock(cfg.lock_path):
		bot = AlphaGoldTradingBot(cfg)
		if cfg.once:
			bot.run_once()
			return 0

		while True:
			try:
				bot.run_once()
			except Exception as exc:
				bot.logger.exception("Cycle failed: %s", exc)
			time.sleep(aligned_sleep_seconds(pd.Timestamp(datetime.now(UTC)), cfg.sleep_seconds))


if __name__ == "__main__":
	raise SystemExit(main())

