from __future__ import annotations

from typing import Any, Optional

from brokers.base import OrderRequest, OrderResult
from ig_scripts.ig_data_api import (
	IGService,
	Price,
	amend_position_levels,
	close_otc_position,
	get_closed_trade_by_deal_id,
	get_open_position_by_deal_id,
	place_otc_market_order,
)


class IGLiveBrokerAdapter:
	"""Thin live adapter that submits and amends IG OTC positions."""

	def __init__(
		self,
		service: IGService,
		*,
		instrument: Price = Price.Gold,
		stop_loss_pct: float,
		take_profit_pct: float,
	):
		self.service = service
		self.instrument = instrument
		self.stop_loss_pct = float(stop_loss_pct)
		self.take_profit_pct = float(take_profit_pct)

	def submit_order(self, request: OrderRequest) -> OrderResult:
		side = str(request.side).lower()
		if side in {"up", "long", "buy"}:
			direction = "BUY"
		elif side in {"down", "short", "sell"}:
			direction = "SELL"
		else:
			return OrderResult(
				submitted=False,
				dry_run=False,
				status="rejected",
				reason=f"unsupported_side:{request.side}",
			)

		entry_price = float(request.entry_price)
		stop_distance = abs(entry_price) * (self.stop_loss_pct / 100.0)
		limit_distance = abs(entry_price) * (self.take_profit_pct / 100.0)

		try:
			result = place_otc_market_order(
				self.service,
				self.instrument,
				direction,
				size=float(request.size),
				stop_distance=stop_distance,
				limit_distance=limit_distance,
			)
		except Exception as exc:
			return OrderResult(
				submitted=False,
				dry_run=False,
				status="error",
				reason=str(exc),
			)

		deal_id = result.get("deal_id") or result.get("deal_reference")
		status = str(result.get("deal_status") or "submitted")
		submitted = status.upper() in {"ACCEPTED", "OPEN", "SUBMITTED"}

		return OrderResult(
			submitted=submitted,
			dry_run=False,
			status="submitted" if submitted else "rejected",
			reason=None if submitted else str(result.get("reason") or status),
			broker_order_id=result.get("deal_reference"),
			deal_id=deal_id,
			raw_response=result,
		)

	def amend_position_levels(self, *, deal_id: str, stop_level: float, limit_level: float) -> dict[str, Any]:
		return amend_position_levels(
			self.service,
			deal_id,
			stop_level=float(stop_level),
			limit_level=float(limit_level),
		)

	def get_position_by_deal_id(self, deal_id: str) -> Optional[dict[str, Any]]:
		return get_open_position_by_deal_id(self.service, deal_id)

	def close_position(self, *, deal_id: str, direction: str, size: float) -> dict[str, Any]:
		return close_otc_position(
			self.service,
			deal_id=deal_id,
			direction=direction,
			size=float(size),
		)

	def get_closed_trade_by_deal_id(self, deal_id: str) -> Optional[dict[str, Any]]:
		return get_closed_trade_by_deal_id(self.service, deal_id)

