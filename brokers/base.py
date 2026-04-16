from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol


@dataclass
class OrderRequest:
    symbol: str
    side: str
    size: float
    signal_time_utc: str
    entry_time_utc: str
    entry_price: float
    probability: float
    signal_model_family: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    submitted: bool
    dry_run: bool
    status: str
    reason: Optional[str] = None
    broker_order_id: Optional[str] = None
    deal_id: Optional[str] = None
    raw_response: Optional[dict[str, Any]] = None


@dataclass
class PositionSnapshot:
    symbol: str
    side: str
    size: float
    entry_price: float
    opened_at_utc: str
    broker_position_id: Optional[str] = None


class BrokerAdapter(Protocol):
    def submit_order(self, request: OrderRequest) -> OrderResult:
        ...


class DryRunBrokerAdapter:
    """No-op adapter that mirrors old_bot execution intent without sending orders."""

    def submit_order(self, request: OrderRequest) -> OrderResult:
        return OrderResult(
            submitted=False,
            dry_run=True,
            status="dry_run",
            reason="signal_only_mode",
            broker_order_id=None,
            deal_id=None,
            raw_response={
                "received_at_utc": datetime.now(timezone.utc).isoformat(),
                "request": {
                    "symbol": request.symbol,
                    "side": request.side,
                    "size": request.size,
                    "signal_time_utc": request.signal_time_utc,
                    "entry_time_utc": request.entry_time_utc,
                    "entry_price": request.entry_price,
                    "probability": request.probability,
                    "signal_model_family": request.signal_model_family,
                    "metadata": dict(request.metadata),
                },
            },
        )

