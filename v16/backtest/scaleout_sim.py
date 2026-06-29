"""Bar-by-bar scale-out exit simulation (+5 half, +10 close all, runner lock)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ScaleOutResult:
    pnl: float
    exit_reason: str
    bars_held: int
    scaled_half: bool
    side: int


def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    spread = 0.25
    if "open_ask" not in out.columns:
        for c in ("open", "high", "low", "close"):
            out[f"{c}_ask"] = out[c] + spread
            out[f"{c}_bid"] = out[c] - spread
    if "close_bid" not in out.columns:
        out["close_bid"] = out["close_ask"] - spread
        out["close_ask"] = out.get("close_ask", out["close"] + spread)
    return out


def unrealized_pnl(side: int, entry: float, row: pd.Series) -> float:
    if side == 1:
        return float(row["close_bid"]) - entry
    return entry - float(row["close_ask"])


def exit_price(side: int, row: pd.Series, *, long_exit: bool = True) -> float:
    """long_exit: sell at bid; short cover at ask."""
    if side == 1:
        return float(row["close_bid"] if long_exit else row["close_ask"])
    return float(row["close_ask"] if long_exit else row["close_bid"])


def simulate_scaleout_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: int,
    entry_price: float,
    *,
    first_scale_pnl: float = 5.0,
    first_scale_frac: float = 0.5,
    final_scale_pnl: float = 10.0,
    initial_sl: float = 20.0,
    runner_lock_pnl: float = 5.0,
    horizon: int = 12,
) -> ScaleOutResult:
    """
    Platform-style ladder on 1 unit of risk:
      - Hit +first_scale_pnl: close first_scale_frac
      - Runner stop ratchets to entry + runner_lock_pnl (long) after scale
      - Hit +final_scale_pnl on runner: close remainder
      - Initial stop: initial_sl on full size before scale
    """
    remaining = 1.0
    realized = 0.0
    scaled_half = False
    runner_stop: float | None = None

    start = entry_idx + 1
    end = min(entry_idx + 1 + horizon, len(df))
    if start >= end:
        return ScaleOutResult(0.0, "no_bars", 0, False, side)

    for j in range(start, end):
        row = df.iloc[j]
        bars = j - entry_idx

        if side == 1:
            low_px = float(row["low_bid"])
            high_px = float(row["high_ask"])
            if not scaled_half:
                if low_px <= entry_price - initial_sl:
                    return ScaleOutResult(
                        realized - initial_sl * remaining,
                        "stop_loss",
                        bars,
                        scaled_half,
                        side,
                    )
                if high_px >= entry_price + first_scale_pnl:
                    chunk = first_scale_frac * remaining
                    realized += chunk * first_scale_pnl
                    remaining -= chunk
                    scaled_half = True
                    runner_stop = entry_price + runner_lock_pnl
                if high_px >= entry_price + final_scale_pnl and remaining > 0:
                    realized += remaining * final_scale_pnl
                    return ScaleOutResult(realized, "target_full", bars, scaled_half, side)
            else:
                assert runner_stop is not None
                if low_px <= runner_stop:
                    realized += remaining * (runner_stop - entry_price)
                    return ScaleOutResult(realized, "trail_stop", bars, scaled_half, side)
                if high_px >= entry_price + final_scale_pnl:
                    realized += remaining * final_scale_pnl
                    return ScaleOutResult(realized, "target_full", bars, scaled_half, side)
        else:
            high_px = float(row["high_ask"])
            low_px = float(row["low_bid"])
            if not scaled_half:
                if high_px >= entry_price + initial_sl:
                    return ScaleOutResult(
                        realized - initial_sl * remaining,
                        "stop_loss",
                        bars,
                        scaled_half,
                        side,
                    )
                if low_px <= entry_price - first_scale_pnl:
                    chunk = first_scale_frac * remaining
                    realized += chunk * first_scale_pnl
                    remaining -= chunk
                    scaled_half = True
                    runner_stop = entry_price - runner_lock_pnl
                if low_px <= entry_price - final_scale_pnl and remaining > 0:
                    realized += remaining * final_scale_pnl
                    return ScaleOutResult(realized, "target_full", bars, scaled_half, side)
            else:
                assert runner_stop is not None
                if high_px >= runner_stop:
                    realized += remaining * (entry_price - runner_stop)
                    return ScaleOutResult(realized, "trail_stop", bars, scaled_half, side)
                if low_px <= entry_price - final_scale_pnl:
                    realized += remaining * final_scale_pnl
                    return ScaleOutResult(realized, "target_full", bars, scaled_half, side)

    last = df.iloc[end - 1]
    px = exit_price(side, last)
    pnl_tail = (px - entry_price) * side * remaining
    return ScaleOutResult(realized + pnl_tail, "timeout", end - entry_idx - 1, scaled_half, side)


def label_scaleout_win(res: ScaleOutResult) -> int:
    return int(res.pnl > 0)


def new_scaleout_state(
    entry_price: float,
    side: int,
    entry_ts: pd.Timestamp,
    *,
    horizon: int,
    **kwargs,
) -> dict:
    """Mutable scale-out position state for bar-by-bar simulation."""
    return {
        "side": side,
        "entry_price": entry_price,
        "entry_time": entry_ts,
        "timeout": entry_ts + pd.Timedelta(minutes=horizon),
        "horizon": horizon,
        "remaining": 1.0,
        "realized": 0.0,
        "scaled_half": False,
        "runner_stop": None,
        "scaleout_kw": kwargs,
        "target_updates": 0,
        "stop_updates": 0,
    }


def scaleout_bar_step(state: dict, row: pd.Series) -> ScaleOutResult | None:
    """Advance scale-out state one bar; return result if position closed."""
    side = int(state["side"])
    entry_price = float(state["entry_price"])
    remaining = float(state["remaining"])
    realized = float(state["realized"])
    scaled_half = bool(state["scaled_half"])
    runner_stop = state["runner_stop"]
    kw = state["scaleout_kw"]
    first_scale_pnl = float(kw["first_scale_pnl"])
    first_scale_frac = float(kw["first_scale_frac"])
    final_scale_pnl = float(kw["final_scale_pnl"])
    initial_sl = float(kw["initial_sl"])
    runner_lock_pnl = float(kw["runner_lock_pnl"])

    if side == 1:
        low_px = float(row["low_bid"])
        high_px = float(row["high_ask"])
        if not scaled_half:
            if low_px <= entry_price - initial_sl:
                return ScaleOutResult(realized - initial_sl * remaining, "stop_loss", 0, scaled_half, side)
            if high_px >= entry_price + first_scale_pnl:
                chunk = first_scale_frac * remaining
                realized += chunk * first_scale_pnl
                remaining -= chunk
                scaled_half = True
                runner_stop = entry_price + runner_lock_pnl
            if high_px >= entry_price + final_scale_pnl and remaining > 0:
                realized += remaining * final_scale_pnl
                return ScaleOutResult(realized, "target_full", 0, scaled_half, side)
        else:
            assert runner_stop is not None
            if low_px <= runner_stop:
                realized += remaining * (runner_stop - entry_price)
                return ScaleOutResult(realized, "trail_stop", 0, scaled_half, side)
            if high_px >= entry_price + final_scale_pnl:
                realized += remaining * final_scale_pnl
                return ScaleOutResult(realized, "target_full", 0, scaled_half, side)
    else:
        high_px = float(row["high_ask"])
        low_px = float(row["low_bid"])
        if not scaled_half:
            if high_px >= entry_price + initial_sl:
                return ScaleOutResult(realized - initial_sl * remaining, "stop_loss", 0, scaled_half, side)
            if low_px <= entry_price - first_scale_pnl:
                chunk = first_scale_frac * remaining
                realized += chunk * first_scale_pnl
                remaining -= chunk
                scaled_half = True
                runner_stop = entry_price - runner_lock_pnl
            if low_px <= entry_price - final_scale_pnl and remaining > 0:
                realized += remaining * final_scale_pnl
                return ScaleOutResult(realized, "target_full", 0, scaled_half, side)
        else:
            assert runner_stop is not None
            if high_px >= runner_stop:
                realized += remaining * (entry_price - runner_stop)
                return ScaleOutResult(realized, "trail_stop", 0, scaled_half, side)
            if low_px <= entry_price - final_scale_pnl:
                realized += remaining * final_scale_pnl
                return ScaleOutResult(realized, "target_full", 0, scaled_half, side)

    state["remaining"] = remaining
    state["realized"] = realized
    state["scaled_half"] = scaled_half
    state["runner_stop"] = runner_stop
    return None


def scaleout_timeout_close(state: dict, row: pd.Series) -> ScaleOutResult:
    side = int(state["side"])
    entry_price = float(state["entry_price"])
    remaining = float(state["remaining"])
    realized = float(state["realized"])
    px = exit_price(side, row)
    pnl_tail = (px - entry_price) * side * remaining
    return ScaleOutResult(
        realized + pnl_tail,
        "timeout",
        0,
        bool(state["scaled_half"]),
        side,
    )


def batch_simulate(
    df: pd.DataFrame,
    entries: list[dict[str, Any]],
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for e in entries:
        res = simulate_scaleout_trade(
            df,
            e["entry_idx"],
            e["side"],
            e["entry_price"],
            **kwargs,
        )
        rows.append({**e, **res.__dict__})
    return pd.DataFrame(rows)
