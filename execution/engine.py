from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from brokers.base import BrokerAdapter, OrderRequest

UTC = timezone.utc


class ExecutionEngine:
	"""Execution seam inspired by old_bot.py _execute_trade separation."""

	def __init__(self, broker_adapter: BrokerAdapter):
		self.broker_adapter = broker_adapter

	def handle_signal(
		self,
		*,
		mode: str,
		signal_model_family: str,
		signal: dict[str, Any],
		entry_time: pd.Timestamp,
		entry_price: float,
		size: float,
	) -> dict[str, Any]:
		side = str(signal.get("side", "flat"))
		probability = float(signal.get("probability", 0.0) or 0.0)
		signal_time = pd.Timestamp(signal.get("signal_bar_time", entry_time))
		if signal_time.tzinfo is None:
			signal_time = signal_time.tz_localize("UTC")
		else:
			signal_time = signal_time.tz_convert("UTC")

		entry_ts = pd.Timestamp(entry_time)
		if entry_ts.tzinfo is None:
			entry_ts = entry_ts.tz_localize("UTC")
		else:
			entry_ts = entry_ts.tz_convert("UTC")

		request = OrderRequest(
			symbol="gold",
			side=side,
			size=float(size),
			signal_time_utc=signal_time.isoformat(),
			entry_time_utc=entry_ts.isoformat(),
			entry_price=float(entry_price),
			probability=probability,
			signal_model_family=signal_model_family,
			metadata={
				"mode": mode,
				"tradable": bool(signal.get("tradable", False)),
				"pred": signal.get("pred"),
			},
		)
		result = self.broker_adapter.submit_order(request)
		deal_id = result.deal_id or result.broker_order_id

		return {
			"attempted_at_utc": datetime.now(UTC).isoformat(),
			"mode": mode,
			"signal_model_family": signal_model_family,
			"signal_side": side,
			"signal_probability": probability,
			"signal_tradable": bool(signal.get("tradable", False)),
			"entry_time_utc": entry_ts.isoformat(),
			"entry_price": float(entry_price),
			"size": float(size),
			"submitted": bool(result.submitted),
			"dry_run": bool(result.dry_run),
			"status": result.status,
			"reason": result.reason,
			"deal_id": deal_id,
			"broker_order_id": result.broker_order_id,
			"broker_response": result.raw_response,
		}

