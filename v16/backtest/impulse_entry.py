"""Impulse pattern entry resolution: open, pullback limit, breakout stop."""
from __future__ import annotations

from typing import Any

import pandas as pd


def impulse_bar_range(imp_low: float, imp_high: float) -> float:
    return max(float(imp_high) - float(imp_low), 0.0)


def pullback_limit_price(
    side: int,
    imp_low: float,
    imp_high: float,
    *,
    fraction: float = 0.5,
) -> float:
    """
    Long: retrace from impulse high toward low.
    Short: retrace from impulse low toward high.
    """
    rng = impulse_bar_range(imp_low, imp_high)
    frac = float(fraction)
    if side == 1:
        return float(imp_high) - frac * rng
    return float(imp_low) + frac * rng


def breakout_trigger_price(
    side: int,
    imp_low: float,
    imp_high: float,
    *,
    buffer_pts: float = 0.0,
) -> float:
    buf = float(buffer_pts)
    if side == 1:
        return float(imp_high) + buf
    return float(imp_low) - buf


def entry_config(cfg: dict | None) -> dict[str, Any]:
    from v16.config import v16_config

    c = cfg or v16_config.MOMENTUM_15M_HOLD
    pb = dict(c.get("entry_pullback", {}))
    bo = dict(c.get("entry_breakout", {}))
    fc = dict(c.get("entry_fill", {}))
    return {
        "mode": c.get("entry_mode", "open"),
        "pullback_fraction": float(pb.get("fraction", 0.5)),
        "pullback_timeout": int(pb.get("timeout_minutes", 5)),
        "pullback_cancel_on_stop": bool(pb.get("cancel_on_stop_touch", True)),
        "breakout_buffer_pts": float(bo.get("buffer_pts", 0.0)),
        "breakout_timeout": int(bo.get("timeout_minutes", 10)),
        "breakout_fill": str(bo.get("fill", "next_open")),  # trigger | next_open
        "max_entry_gap_pts": bo.get("max_entry_gap_pts"),  # None = off; for next_open only
        "max_close_dist_pts": bo.get("max_close_dist_pts"),  # break-bar close within N pt of trigger
        "max_pre_break_close_dist_pts": bo.get("max_pre_break_close_dist_pts"),  # bar before break: close near trigger
        "fill_mode": str(fc.get("mode", "ideal")),  # ideal | conservative | pessimistic
        "slippage_pts": float(fc.get("slippage_pts", 0.25)),
        "intrabar_stop_first": bool(fc.get("intrabar_stop_first", True)),
        "cancel_on_stop_during_wait": bool(fc.get("cancel_on_stop_during_wait", False)),
    }


def _entry_slippage(side: int, price: float, *, fill_mode: str, slippage_pts: float) -> float:
    slip = float(slippage_pts)
    if fill_mode == "ideal":
        return float(price)
    if fill_mode == "conservative":
        slip *= 0.5
    if side == 1:
        return float(price) + slip
    return float(price) - slip


def _open_entry_price(
    side: int,
    j: int,
    open_ask: Any,
    open_bid: Any,
    *,
    fill_mode: str,
    slippage_pts: float,
) -> float:
    base = float(open_ask[j] if side == 1 else open_bid[j])
    return _entry_slippage(side, base, fill_mode=fill_mode, slippage_pts=slippage_pts)


def _breakout_entry_price(
    side: int,
    j: int,
    trigger: float,
    open_ask: Any,
    open_bid: Any,
    *,
    fill_mode: str,
    slippage_pts: float,
) -> float:
    if fill_mode == "ideal":
        base = float(trigger)
    else:
        base = max(float(trigger), float(open_ask[j])) if side == 1 else min(float(trigger), float(open_bid[j]))
    return _entry_slippage(side, base, fill_mode=fill_mode, slippage_pts=slippage_pts)


def _stop_touched_during_wait(
    side: int,
    j: int,
    low_bid: Any,
    high_ask: Any,
    imp_low: float,
    imp_high: float,
) -> bool:
    if side == 1:
        return float(low_bid[j]) <= float(imp_low)
    return float(high_ask[j]) >= float(imp_high)


def _intrabar_stop_before_fill(
    side: int,
    j: int,
    low_bid: Any,
    high_ask: Any,
    imp_low: float,
    imp_high: float,
) -> bool:
    """On the fill bar, assume stop trades before breakout if both touched."""
    return _stop_touched_during_wait(side, j, low_bid, high_ask, imp_low, imp_high)


def build_resolved_entry_table(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    *,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Map each impulse signal to a concrete fill (or drop if timeout / cancelled)."""
    if signal_table.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for ts, row in signal_table.iterrows():
        sig_idx = int(df.index.get_loc(ts))
        side_i = int(row["side"])
        imp_low = float(row["impulse_low"])
        imp_high = float(row["impulse_high"])
        resolved = resolve_impulse_entry(
            df,
            sig_idx,
            side_i,
            imp_low,
            imp_high,
            cfg=cfg,
        )
        if resolved is None:
            continue
        entry_idx, ep, style = resolved
        rows.append(
            {
                "signal_ts": ts,
                "entry_idx": entry_idx,
                "entry_time": df.index[entry_idx],
                "entry_price": ep,
                "entry_style": style,
                "side": side_i,
                "impulse_low": imp_low,
                "impulse_high": imp_high,
                "slot_low": float(row.get("slot_low", imp_low)),
                "slot_high": float(row.get("slot_high", imp_high)),
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("signal_ts").sort_values("entry_idx")
    return out[~out.index.duplicated(keep="first")]


def resolve_impulse_entry(
    df: pd.DataFrame,
    signal_idx: int,
    side: int,
    imp_low: float,
    imp_high: float,
    *,
    cfg: dict | None = None,
) -> tuple[int, float, str] | None:
    """
    Return (entry_bar_idx, entry_price, entry_style) or None if not filled / skipped.

    entry_bar_idx is the 1m bar index where the fill occurs (same as position sim).
    """
    ec = entry_config(cfg)
    mode = ec["mode"]
    fill_mode = ec["fill_mode"]
    slippage = ec["slippage_pts"]
    n = len(df)
    if signal_idx >= n - 2:
        return None

    low_bid = df["low_bid"].to_numpy(dtype=float)
    high_ask = df["high_ask"].to_numpy(dtype=float)
    close_bid = df["close_bid"].to_numpy(dtype=float)
    close_ask = df["close_ask"].to_numpy(dtype=float)
    open_ask = df["open_ask"].to_numpy(dtype=float)
    open_bid = df["open_bid"].to_numpy(dtype=float)

    def _close_near_trigger(bar_j: int, trig: float) -> bool:
        max_cd = ec.get("max_close_dist_pts")
        if max_cd is None:
            return True
        mid_close = 0.5 * (float(close_bid[bar_j]) + float(close_ask[bar_j]))
        return abs(mid_close - float(trig)) <= float(max_cd)

    def _pre_break_close_near_trigger(break_j: int, trig: float) -> bool:
        """Bar before break must have closed near the trigger (compression at level)."""
        max_pd = ec.get("max_pre_break_close_dist_pts")
        if max_pd is None:
            return True
        if break_j < 1:
            return False
        pj = break_j - 1
        mid_close = 0.5 * (float(close_bid[pj]) + float(close_ask[pj]))
        return abs(mid_close - float(trig)) <= float(max_pd)

    cancel_wait = ec["cancel_on_stop_during_wait"] or fill_mode == "pessimistic"

    if mode == "open":
        j = signal_idx
        if j >= n:
            return None
        ep = _open_entry_price(
            side, j, open_ask, open_bid, fill_mode=fill_mode, slippage_pts=slippage
        )
        return j, ep, "open"

    if mode == "pullback":
        limit = pullback_limit_price(side, imp_low, imp_high, fraction=ec["pullback_fraction"])
        end = min(signal_idx + ec["pullback_timeout"], n - 1)
        for j in range(signal_idx + 1, end + 1):
            if ec["pullback_cancel_on_stop"] or cancel_wait:
                if _stop_touched_during_wait(side, j, low_bid, high_ask, imp_low, imp_high):
                    return None
            if side == 1:
                if low_bid[j] <= limit:
                    ep = _entry_slippage(side, limit, fill_mode=fill_mode, slippage_pts=slippage)
                    return j, ep, "pullback"
            else:
                if high_ask[j] >= limit:
                    ep = _entry_slippage(side, limit, fill_mode=fill_mode, slippage_pts=slippage)
                    return j, ep, "pullback"
        return None

    if mode == "breakout":
        trigger = breakout_trigger_price(
            side, imp_low, imp_high, buffer_pts=ec["breakout_buffer_pts"]
        )
        breakout_fill = ec["breakout_fill"]
        end = min(signal_idx + ec["breakout_timeout"], n - 1)
        for j in range(signal_idx + 1, end + 1):
            if cancel_wait and _stop_touched_during_wait(side, j, low_bid, high_ask, imp_low, imp_high):
                return None
            broke = (side == 1 and high_ask[j] >= trigger) or (side == -1 and low_bid[j] <= trigger)
            if not broke:
                continue
            if not _pre_break_close_near_trigger(j, trigger):
                continue
            if not _close_near_trigger(j, trigger):
                continue
            if ec["intrabar_stop_first"] and fill_mode == "pessimistic":
                if _intrabar_stop_before_fill(side, j, low_bid, high_ask, imp_low, imp_high):
                    return None
            if breakout_fill == "next_open":
                ej = j + 1
                if ej >= n:
                    return None
                ep = _open_entry_price(
                    side, ej, open_ask, open_bid, fill_mode=fill_mode, slippage_pts=slippage
                )
                max_gap = ec.get("max_entry_gap_pts")
                if max_gap is not None:
                    gap = abs(ep - float(trigger))
                    if gap > float(max_gap):
                        continue
                return ej, ep, "breakout_next_open"
            ep = _breakout_entry_price(
                side,
                j,
                trigger,
                open_ask,
                open_bid,
                fill_mode=fill_mode,
                slippage_pts=slippage,
            )
            return j, ep, "breakout"
        return None

    raise ValueError(f"Unknown entry_mode: {mode}")
