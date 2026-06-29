"""Kelly-based Position Sizer.

Computes optimal position size using the Kelly criterion, adapted for:
  - Asymmetric payoffs (TP != SL)
  - Trailing edge estimation (rolling window of recent trades)
  - Drawdown protection (halve Kelly fraction during significant DD)
  - Regime-aware sizing (per-regime Kelly fractions from v15_config)
  - Fixed size fallback

Usage:
    sizer = KellySizer(initial_equity=500.0)
    size = sizer.compute_size(tp=30, sl=25, kelly_fraction=0.13)
    sizer.record_trade(pnl=15.0)   # update trailing edge
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SizerState:
    """Running state of the Kelly sizer for logging/debugging."""
    equity: float
    peak_equity: float
    drawdown_pct: float
    trailing_wr: float
    trailing_pf: float
    trailing_kelly_f: float
    active_fraction: float
    suggested_size: float
    n_trades: int


class KellySizer:
    """Computes position sizes using Kelly criterion with drawdown protection.

    Kelly fraction: f* = (PF * WR - (1 - WR)) / PF
    where PF = avg_win / abs(avg_loss), WR = win_rate

    Half-Kelly (f*/2) is used by default for conservatism.
    """

    def __init__(
        self,
        initial_equity: float = 500.0,
        default_fraction: float = 0.13,
        trailing_window: int = 20,
        min_trades_for_kelly: int = 10,
        max_size: float = 5.0,
        min_size: float = 0.5,
        max_dd_cut: float = 0.25,
    ):
        self.initial_equity = initial_equity
        self.default_fraction = default_fraction
        self.trailing_window = trailing_window
        self.min_trades_for_kelly = min_trades_for_kelly
        self.max_size = max_size
        self.min_size = min_size
        self.max_dd_cut = max_dd_cut

        # Running state
        self._equity: float = initial_equity
        self._peak_equity: float = initial_equity
        self._trades: list[float] = []  # list of PnL values
        self._cumulative_pnl: float = 0.0
        self._current_dd: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_size(
        self,
        tp: float,
        sl: float,
        kelly_fraction: float | None = None,
        regime_id: int | None = None,
    ) -> float:
        """Compute position size for a trade with given TP/SL.

        Args:
            tp: take-profit distance in points
            sl: stop-loss distance in points
            kelly_fraction: override Kelly fraction (e.g., from regime config).
                           If None, uses self.default_fraction.
            regime_id: informational — logged but not used directly (fraction is passed in).

        Returns:
            Position size (number of contracts/units).
        """
        # Fixed size override
        if kelly_fraction is not None and kelly_fraction <= 0:
            return self.min_size

        fraction = kelly_fraction if kelly_fraction is not None else self.default_fraction

        # If not enough trade history, use base fraction * equity / sl
        if len(self._trades) < self.min_trades_for_kelly:
            risk_per_trade = self._equity * fraction
            size = risk_per_trade / max(sl, 1.0)
            return self._clamp_size(size)

        # Compute trailing Kelly fraction
        trailing_kelly = self._compute_trailing_kelly()

        # Apply drawdown protection
        dd_fraction = self._drawdown_scalar()
        active_fraction = min(fraction, trailing_kelly) * dd_fraction

        # Size = (equity * Kelly fraction) / risk_per_trade
        # Risk per trade ≈ sl distance (since that's the max loss if stopped)
        risk_per_trade = self._equity * active_fraction
        size = risk_per_trade / max(sl, 1.0)

        return self._clamp_size(size)

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade's PnL to update trailing edge."""
        self._trades.append(pnl)
        self._cumulative_pnl += pnl
        self._equity = self.initial_equity + self._cumulative_pnl

        # Update peak and drawdown
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        if self._peak_equity > 0:
            self._current_dd = (self._peak_equity - self._equity) / self._peak_equity
        else:
            self._current_dd = 0.0

        # Trim trade history to trailing window
        if len(self._trades) > self.trailing_window * 4:
            self._trades = self._trades[-self.trailing_window * 2:]

        logger.debug(
            "Trade recorded: pnl=%.1f, equity=%.1f, DD=%.1f%%, n_trades=%d",
            pnl, self._equity, self._current_dd * 100, len(self._trades),
        )

    def get_state(self) -> SizerState:
        """Return current sizing state for logging/monitoring."""
        trailing_wr, trailing_pf, trailing_kelly = self._compute_trailing_stats()
        return SizerState(
            equity=round(self._equity, 1),
            peak_equity=round(self._peak_equity, 1),
            drawdown_pct=round(self._current_dd * 100, 1),
            trailing_wr=round(trailing_wr, 3),
            trailing_pf=round(trailing_pf, 2),
            trailing_kelly_f=round(trailing_kelly, 4),
            active_fraction=round(self._active_fraction(), 4),
            suggested_size=round(self.compute_size(30, 25), 2),
            n_trades=len(self._trades),
        )

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def drawdown(self) -> float:
        return self._current_dd

    @property
    def n_trades(self) -> int:
        return len(self._trades)

    def reset_equity(self, new_equity: float) -> None:
        """Reset equity baseline (e.g., on WF cycle boundary)."""
        self.initial_equity = new_equity
        self._equity = new_equity
        self._peak_equity = new_equity
        self._cumulative_pnl = 0.0
        self._current_dd = 0.0
        logger.info("Sizer equity reset to %.1f", new_equity)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_trailing_stats(self) -> tuple[float, float, float]:
        """Compute trailing win rate, profit factor, and Kelly f from recent trades."""
        recent = self._trades[-self.trailing_window:] if len(self._trades) >= self.trailing_window else self._trades

        if len(recent) < 3:
            return 0.5, 1.0, 0.0

        wins = [t for t in recent if t > 0]
        losses = [t for t in recent if t < 0]

        wr = len(wins) / len(recent) if recent else 0.5
        avg_win = sum(wins) / len(wins) if wins else 1.0
        # For losses, use abs value to avoid negative avg
        avg_loss = abs(sum(losses) / len(losses)) if losses else 1.0

        # Guard against degenerate cases
        if avg_loss == 0 or len(wins) == 0:
            pf = 1.0
        else:
            pf = (avg_win * len(wins)) / (avg_loss * len(losses)) if len(losses) > 0 else 999.0

        # Kelly: f* = (PF * WR - (1 - WR)) / PF
        # Clamp to [0, 0.5] range
        kelly = (pf * wr - (1 - wr)) / max(pf, 0.1)
        kelly = max(0.0, min(kelly, 0.5))

        return wr, pf, kelly

    def _compute_trailing_kelly(self) -> float:
        """Get the trailing Kelly f (half-Kelly applied)."""
        _, _, raw_kelly = self._compute_trailing_stats()
        return raw_kelly * 0.5  # half-Kelly

    def _drawdown_scalar(self) -> float:
        """Reduce Kelly fraction based on current drawdown.

        DD < 10%:  full fraction (1.0)
        DD 10-25%: linear reduction from 1.0 → 0.5
        DD > 25%:  0.5 (minimum)
        """
        if self._current_dd < 0.10:
            return 1.0
        elif self._current_dd < self.max_dd_cut:
            # Linear from 1.0 at 10% to 0.5 at max_dd_cut
            t = (self._current_dd - 0.10) / (self.max_dd_cut - 0.10)
            return 1.0 - 0.5 * t
        else:
            return 0.5

    def _active_fraction(self) -> float:
        """Current active Kelly fraction after all adjustments."""
        trailing = self._compute_trailing_kelly()
        dd_scalar = self._drawdown_scalar()
        return min(self.default_fraction, trailing) * dd_scalar

    def _clamp_size(self, size: float) -> float:
        return round(max(self.min_size, min(size, self.max_size)), 2)
