"""v15-style single-position simulation with same-direction refresh."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_same_direction_upgrade(
    active_pos: dict,
    mid: float,
    side: int,
    now_ts: pd.Timestamp,
    tp: float,
    sl: float,
    horizon: int,
    *,
    mode: str = "entry",
    upgrade_stop: bool = False,
) -> None:
    """Extend horizon and trail target on a new same-direction signal (v15 pattern default)."""
    if mode == "entry":
        pos_tp = float(active_pos.get("tp", tp))
        pos_h = int(active_pos.get("horizon", horizon))
        active_pos["timeout"] = now_ts + pd.Timedelta(minutes=pos_h)
        active_pos["target_updates"] = active_pos.get("target_updates", 0) + 1
        new_target = mid + pos_tp if side == 1 else mid - pos_tp
        if (side == 1 and new_target > active_pos["target"]) or (side == -1 and new_target < active_pos["target"]):
            active_pos["target"] = new_target
    elif mode == "global":
        active_pos["timeout"] = now_ts + pd.Timedelta(minutes=horizon)
        active_pos["target_updates"] = active_pos.get("target_updates", 0) + 1
        new_target = mid + tp if side == 1 else mid - tp
        if (side == 1 and new_target > active_pos["target"]) or (side == -1 and new_target < active_pos["target"]):
            active_pos["target"] = new_target
    elif mode == "exec":
        pos_tp = max(float(active_pos.get("tp", tp)), float(tp))
        pos_sl = max(float(active_pos.get("sl", sl)), float(sl))
        pos_h = max(int(active_pos.get("horizon", horizon)), int(horizon))
        active_pos["tp"] = pos_tp
        active_pos["sl"] = pos_sl
        active_pos["horizon"] = pos_h
        active_pos["timeout"] = now_ts + pd.Timedelta(minutes=pos_h)
        active_pos["target_updates"] = active_pos.get("target_updates", 0) + 1
        new_target = mid + pos_tp if side == 1 else mid - pos_tp
        if (side == 1 and new_target > active_pos["target"]) or (side == -1 and new_target < active_pos["target"]):
            active_pos["target"] = new_target
        if upgrade_stop:
            new_stop = mid - pos_sl if side == 1 else mid + pos_sl
            if (side == 1 and new_stop > active_pos["stop"]) or (side == -1 and new_stop < active_pos["stop"]):
                active_pos["stop"] = new_stop
                active_pos["stop_updates"] = active_pos.get("stop_updates", 0) + 1


def simulate_single_position(
    df: pd.DataFrame,
    signals: pd.Series,
    *,
    side: int = -1,
    tp: float,
    sl: float,
    horizon: int,
    same_dir_refresh: str = "entry",
    upgrade_stop: bool = False,
) -> pd.DataFrame:
    """
    Walk 1m bars with at most one open position.

    signals: bool Series aligned to df.index — True on signal bar (entry next bar open).
    Matches v15 pattern behaviour: same-direction signals refresh target + horizon, no overlap.
    """
    if side not in (1, -1):
        raise ValueError("side must be 1 or -1")

    sig_arr = signals.reindex(df.index, fill_value=False).to_numpy(dtype=bool)
    signal_idxs = np.flatnonzero(sig_arr)
    n = len(df) - 1
    if n <= 0:
        return pd.DataFrame()

    mid = df["mid"].to_numpy(dtype=float)
    low_bid = df["low_bid"].to_numpy(dtype=float)
    high_ask = df["high_ask"].to_numpy(dtype=float)
    close_bid = df["close_bid"].to_numpy(dtype=float)
    close_ask = df["close_ask"].to_numpy(dtype=float)
    open_ask = df["open_ask"].to_numpy(dtype=float)
    open_bid = df["open_bid"].to_numpy(dtype=float)
    index = df.index

    trades: list[dict] = []
    active_pos: dict | None = None
    sig_ptr = 0
    i = 0

    while i < n:
        if active_pos is None:
            while sig_ptr < len(signal_idxs) and signal_idxs[sig_ptr] < i:
                sig_ptr += 1
            if sig_ptr >= len(signal_idxs):
                break
            i = int(signal_idxs[sig_ptr])

        now_ts = index[i]
        sig = bool(sig_arr[i])

        if active_pos:
            s = active_pos["side"]
            exit_info = None
            if s == 1:
                if low_bid[i] <= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif high_ask[i] >= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif now_ts >= active_pos["timeout"]:
                    exit_info = (close_bid[i], "timeout")
            else:
                if high_ask[i] >= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif low_bid[i] <= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif now_ts >= active_pos["timeout"]:
                    exit_info = (close_ask[i], "timeout")

            if exit_info:
                px, reason = exit_info
                pnl = (px - active_pos["entry_price"]) * s
                trades.append(
                    {
                        **active_pos,
                        "exit_time": now_ts,
                        "exit_price": px,
                        "exit_reason": reason,
                        "pnl": pnl,
                        "win": pnl > 0,
                    }
                )
                active_pos = None

        if active_pos and sig and side == active_pos["side"] and same_dir_refresh != "none":
            apply_same_direction_upgrade(
                active_pos,
                float(mid[i]),
                side,
                now_ts,
                tp,
                sl,
                horizon,
                mode=same_dir_refresh,
                upgrade_stop=upgrade_stop,
            )

        if active_pos is None and sig:
            ep = float(open_ask[i + 1] if side == 1 else open_bid[i + 1])
            entry_ts = index[i + 1]
            active_pos = {
                "side": side,
                "signal_ts": now_ts,
                "entry_time": entry_ts,
                "entry_price": ep,
                "stop": ep - sl if side == 1 else ep + sl,
                "target": ep + tp if side == 1 else ep - tp,
                "timeout": entry_ts + pd.Timedelta(minutes=horizon),
                "tp": tp,
                "sl": sl,
                "horizon": horizon,
                "target_updates": 0,
                "stop_updates": 0,
            }

        i += 1

    return pd.DataFrame(trades)


def simulate_position_sided_scaleout(
    df: pd.DataFrame,
    side_signals: pd.Series,
    *,
    scaleout_kw: dict,
    same_dir_refresh: str = "entry",
) -> pd.DataFrame:
    """Single position with v16 scale-out exit (+5 half, +10 all, SL20, horizon)."""
    side_arr = side_signals.reindex(df.index, fill_value=0).astype(int).to_numpy()
    signal_idxs = np.flatnonzero(side_arr != 0)
    n = len(df) - 1
    if n <= 0:
        return pd.DataFrame()

    horizon = int(scaleout_kw.get("horizon", scaleout_kw.get("horizon_minutes", 10)))
    kw = {k: v for k, v in scaleout_kw.items() if k not in ("horizon", "horizon_minutes")}
    index = df.index
    open_ask = df["open_ask"].to_numpy(dtype=float)
    open_bid = df["open_bid"].to_numpy(dtype=float)

    from v16.backtest.scaleout_sim import new_scaleout_state, scaleout_bar_step, scaleout_timeout_close

    trades: list[dict] = []
    active_pos: dict | None = None
    sig_ptr = 0
    i = 0

    while i < n:
        if active_pos is None:
            while sig_ptr < len(signal_idxs) and signal_idxs[sig_ptr] < i:
                sig_ptr += 1
            if sig_ptr >= len(signal_idxs):
                break
            i = int(signal_idxs[sig_ptr])

        now_ts = index[i]
        sig_side = int(side_arr[i])
        row = df.iloc[i]

        if active_pos:
            if now_ts >= active_pos["timeout"]:
                res = scaleout_timeout_close(active_pos, row)
            else:
                res = scaleout_bar_step(active_pos, row)

            if res is not None:
                trades.append(
                    {
                        "signal_ts": active_pos["signal_ts"],
                        "entry_time": active_pos["entry_time"],
                        "entry_price": active_pos["entry_price"],
                        "side": active_pos["side"],
                        "exit_time": now_ts,
                        "exit_reason": res.exit_reason,
                        "pnl": res.pnl,
                        "scaled_half": res.scaled_half,
                        "win": res.pnl > 0,
                        "target_updates": active_pos.get("target_updates", 0),
                    }
                )
                active_pos = None
            elif now_ts >= active_pos["timeout"]:
                pass  # handled above

        if active_pos and sig_side != 0 and sig_side == active_pos["side"] and same_dir_refresh != "none":
            active_pos["timeout"] = now_ts + pd.Timedelta(minutes=int(active_pos["horizon"]))
            active_pos["target_updates"] = active_pos.get("target_updates", 0) + 1

        if active_pos is None and sig_side != 0:
            ep = float(open_ask[i + 1] if sig_side == 1 else open_bid[i + 1])
            entry_ts = index[i + 1]
            active_pos = new_scaleout_state(ep, sig_side, entry_ts, horizon=horizon, **kw)
            active_pos["signal_ts"] = now_ts

        i += 1

    return pd.DataFrame(trades)


def _stop_price_for_row(side: int, row: pd.Series, *, stop_mode: str) -> tuple[float, float]:
    """Return (stop_price, range_ref) for SL/TP sizing."""
    if stop_mode == "slot_15m":
        slo = float(row["slot_low"])
        shi = float(row["slot_high"])
        stop = slo if side == 1 else shi
        return stop, shi - slo
    slo = float(row["impulse_low"])
    shi = float(row["impulse_high"])
    stop = slo if side == 1 else shi
    return stop, shi - slo


def _tpsl_from_stop(
    side: int,
    entry_price: float,
    stop_price: float,
    *,
    tp_multiple: float,
    min_sl_pts: float,
    max_sl_pts: float,
) -> tuple[float, float, float, float] | None:
    """TP = tp_multiple × SL distance; stop at stop_price."""
    if side == 1:
        sl_pts = float(entry_price) - float(stop_price)
        target = float(entry_price) + tp_multiple * sl_pts
    else:
        sl_pts = float(stop_price) - float(entry_price)
        target = float(entry_price) - tp_multiple * sl_pts

    if sl_pts < min_sl_pts or sl_pts > max_sl_pts:
        return None
    return sl_pts, tp_multiple * sl_pts, float(stop_price), target


def _tpsl_fixed_tp(
    side: int,
    entry_price: float,
    stop_price: float,
    *,
    tp_pts: float,
    min_sl_pts: float,
    max_sl_pts: float,
) -> tuple[float, float, float, float] | None:
    """SL at stop_price; TP = fixed tp_pts from entry."""
    tp_pts = float(tp_pts)
    if side == 1:
        sl_pts = float(entry_price) - float(stop_price)
        target = float(entry_price) + tp_pts
    else:
        sl_pts = float(stop_price) - float(entry_price)
        target = float(entry_price) - tp_pts

    if sl_pts < min_sl_pts or sl_pts > max_sl_pts:
        return None
    return sl_pts, tp_pts, float(stop_price), target


def _entry_stop_price(
    row: pd.Series,
    side: int,
    *,
    stop_mode: str,
) -> float:
    if str(stop_mode) == "slot_15m":
        stop, _ = _stop_price_for_row(side, row, stop_mode=stop_mode)
        return stop
    return float(row["impulse_low"]) if side == 1 else float(row["impulse_high"])


def _entry_tpsl_levels(
    row: pd.Series,
    side: int,
    entry_price: float,
    *,
    tp_multiple: float,
    tp_mode: str = "multiple",
    tp_pts: float | None = None,
    min_sl_pts: float,
    max_sl_pts: float,
    stop_mode: str = "impulse_bar",
) -> tuple[float, float, float, float] | None:
    stop = _entry_stop_price(row, side, stop_mode=str(stop_mode))
    if str(tp_mode) == "fixed_pts" and tp_pts is not None:
        return _tpsl_fixed_tp(
            side,
            entry_price,
            stop,
            tp_pts=float(tp_pts),
            min_sl_pts=min_sl_pts,
            max_sl_pts=max_sl_pts,
        )
    return _tpsl_from_stop(
        side,
        entry_price,
        stop,
        tp_multiple=tp_multiple,
        min_sl_pts=min_sl_pts,
        max_sl_pts=max_sl_pts,
    )


def _impulse_tpsl(
    side: int,
    entry_price: float,
    imp_low: float,
    imp_high: float,
    *,
    tp_multiple: float,
    min_sl_pts: float,
    max_sl_pts: float,
) -> tuple[float, float, float, float] | None:
    """SL at impulse bar low (long) / high (short); TP = tp_multiple × SL distance."""
    stop = float(imp_low) if side == 1 else float(imp_high)
    return _tpsl_from_stop(
        side, entry_price, stop, tp_multiple=tp_multiple, min_sl_pts=min_sl_pts, max_sl_pts=max_sl_pts
    )


def simulate_impulse_stop_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: int,
    entry_price: float,
    imp_low: float,
    imp_high: float,
    *,
    tp_multiple: float = 3.0,
    horizon: int = 120,
    min_sl_pts: float = 1.0,
    max_sl_pts: float = 80.0,
    stop_price: float | None = None,
    cfg: dict | None = None,
    signal_side: pd.Series | None = None,
) -> dict:
    """Single-trade stop exit (for ML labels). stop_price overrides impulse H/L when set."""
    is_cfg = (cfg or {}).get("impulse_stop", {})
    tp_multiple = float(is_cfg.get("tp_multiple", tp_multiple))
    tp_mode = str(is_cfg.get("tp_mode", "multiple"))
    tp_pts_cfg = is_cfg.get("tp_pts")
    horizon = int(is_cfg.get("horizon", horizon))
    min_sl_pts = float(is_cfg.get("min_sl_pts", min_sl_pts))
    max_sl_pts = float(is_cfg.get("max_sl_pts", max_sl_pts))
    struct_exit_min = float(is_cfg.get("exit_on_structure_change_min_pnl", 0.0))
    tp_enabled = bool(is_cfg.get("tp_enabled", True))
    reverse_exit = bool(is_cfg.get("exit_on_reverse_signal", False))
    reverse_exit_min = float(is_cfg.get("exit_on_reverse_signal_min_pnl", -1e9))
    struct_trend = _structure_trend_arr(df, cfg)
    entry_struct_trend = int(struct_trend[entry_idx]) if struct_trend is not None else 0
    side_arr = None
    if signal_side is not None and not signal_side.empty:
        side_arr = signal_side.reindex(df.index).fillna(0).astype(int).to_numpy()

    if stop_price is not None:
        if str(tp_mode) == "fixed_pts" and tp_pts_cfg is not None:
            levels = _tpsl_fixed_tp(
                side,
                entry_price,
                stop_price,
                tp_pts=float(tp_pts_cfg),
                min_sl_pts=min_sl_pts,
                max_sl_pts=max_sl_pts,
            )
        else:
            levels = _tpsl_from_stop(
                side,
                entry_price,
                stop_price,
                tp_multiple=tp_multiple,
                min_sl_pts=min_sl_pts,
                max_sl_pts=max_sl_pts,
            )
    else:
        stop = float(imp_low) if side == 1 else float(imp_high)
        if str(tp_mode) == "fixed_pts" and tp_pts_cfg is not None:
            levels = _tpsl_fixed_tp(
                side,
                entry_price,
                stop,
                tp_pts=float(tp_pts_cfg),
                min_sl_pts=min_sl_pts,
                max_sl_pts=max_sl_pts,
            )
        else:
            levels = _impulse_tpsl(
                side,
                entry_price,
                imp_low,
                imp_high,
                tp_multiple=tp_multiple,
                min_sl_pts=min_sl_pts,
                max_sl_pts=max_sl_pts,
            )
    if levels is None:
        return {"pnl": 0.0, "exit_reason": "invalid_sl", "win": 0, "sl": 0.0, "tp": 0.0}

    sl_pts, tp_pts, stop, target = levels
    start = entry_idx + 1
    end = min(entry_idx + 1 + horizon, len(df))
    if start >= end:
        return {"pnl": 0.0, "exit_reason": "no_bars", "win": 0, "sl": sl_pts, "tp": tp_pts}

    for j in range(start, end):
        row = df.iloc[j]
        if side == 1:
            if float(row["low_bid"]) <= stop:
                return {"pnl": float(stop - entry_price), "exit_reason": "stop_loss", "win": 0, "sl": sl_pts, "tp": tp_pts}
            if tp_enabled and float(row["high_ask"]) >= target:
                return {"pnl": float(target - entry_price), "exit_reason": "target_hit", "win": 1, "sl": sl_pts, "tp": tp_pts}
            if struct_trend is not None:
                sc = _structure_change_exit(
                    side,
                    entry_price,
                    float(row["close_bid"]),
                    float(row["close_ask"]),
                    cur_trend=int(struct_trend[j]),
                    entry_trend=entry_struct_trend,
                    min_pnl=struct_exit_min,
                )
                if sc is not None:
                    px, reason = sc
                    pnl = px - entry_price
                    return {"pnl": pnl, "exit_reason": reason, "win": int(pnl > 0), "sl": sl_pts, "tp": tp_pts}
            if reverse_exit and side_arr is not None:
                rs = _reverse_signal_exit(
                    side,
                    entry_price,
                    float(row["close_bid"]),
                    float(row["close_ask"]),
                    sig_side=int(side_arr[j]),
                    min_pnl=reverse_exit_min,
                )
                if rs is not None:
                    px, reason = rs
                    pnl = px - entry_price
                    return {"pnl": pnl, "exit_reason": reason, "win": int(pnl > 0), "sl": sl_pts, "tp": tp_pts}
            if j == end - 1:
                px = float(row["close_bid"])
                pnl = px - entry_price
                return {"pnl": pnl, "exit_reason": "timeout", "win": int(pnl > 0), "sl": sl_pts, "tp": tp_pts}
        else:
            if float(row["high_ask"]) >= stop:
                return {"pnl": float(entry_price - stop), "exit_reason": "stop_loss", "win": 0, "sl": sl_pts, "tp": tp_pts}
            if tp_enabled and float(row["low_bid"]) <= target:
                return {"pnl": float(entry_price - target), "exit_reason": "target_hit", "win": 1, "sl": sl_pts, "tp": tp_pts}
            if struct_trend is not None:
                sc = _structure_change_exit(
                    side,
                    entry_price,
                    float(row["close_bid"]),
                    float(row["close_ask"]),
                    cur_trend=int(struct_trend[j]),
                    entry_trend=entry_struct_trend,
                    min_pnl=struct_exit_min,
                )
                if sc is not None:
                    px, reason = sc
                    pnl = entry_price - px
                    return {"pnl": pnl, "exit_reason": reason, "win": int(pnl > 0), "sl": sl_pts, "tp": tp_pts}
            if reverse_exit and side_arr is not None:
                rs = _reverse_signal_exit(
                    side,
                    entry_price,
                    float(row["close_bid"]),
                    float(row["close_ask"]),
                    sig_side=int(side_arr[j]),
                    min_pnl=reverse_exit_min,
                )
                if rs is not None:
                    px, reason = rs
                    pnl = entry_price - px
                    return {"pnl": pnl, "exit_reason": reason, "win": int(pnl > 0), "sl": sl_pts, "tp": tp_pts}
            if j == end - 1:
                px = float(row["close_ask"])
                pnl = entry_price - px
                return {"pnl": pnl, "exit_reason": "timeout", "win": int(pnl > 0), "sl": sl_pts, "tp": tp_pts}

    return {"pnl": 0.0, "exit_reason": "no_bars", "win": 0, "sl": sl_pts, "tp": tp_pts}


def _structure_trend_arr(df: pd.DataFrame, cfg: dict | None) -> np.ndarray | None:
    """Per-1m struct_trend for structure-change exit; None if disabled."""
    is_cfg = (cfg or {}).get("impulse_stop", {})
    if not is_cfg.get("exit_on_structure_change"):
        return None
    from v16.backtest.impulse_features import structure_kwargs
    from v16.structure.swing_zigzag import build_structure_context

    skw = structure_kwargs(cfg)
    if not skw:
        return None
    struct = build_structure_context(df, **skw)
    if struct.empty or "struct_trend" not in struct.columns:
        return np.zeros(len(df), dtype=int)
    return struct["struct_trend"].reindex(df.index).ffill().fillna(0).astype(int).to_numpy()


def _structure_change_exit(
    side: int,
    entry_price: float,
    close_bid: float,
    close_ask: float,
    *,
    cur_trend: int,
    entry_trend: int,
    min_pnl: float,
) -> tuple[float, str] | None:
    """Exit at market if 15m trend label changed and trade is in profit."""
    if int(cur_trend) == int(entry_trend):
        return None
    px = float(close_bid) if side == 1 else float(close_ask)
    pnl = (px - float(entry_price)) * side
    if pnl > float(min_pnl):
        return px, "structure_change"
    return None


def _reverse_signal_exit(
    side: int,
    entry_price: float,
    close_bid: float,
    close_ask: float,
    *,
    sig_side: int,
    min_pnl: float = -1e9,
) -> tuple[float, str] | None:
    """Exit at market on opposite-direction impulse signal (v15 close_on_reverse)."""
    if int(sig_side) == 0 or int(sig_side) != -int(side):
        return None
    px = float(close_bid) if side == 1 else float(close_ask)
    pnl = (px - float(entry_price)) * side
    if pnl > float(min_pnl):
        return px, "reverse_signal"
    return None


def simulate_position_impulse_stop(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    *,
    tp_multiple: float = 3.0,
    horizon: int = 60,
    min_sl_pts: float = 1.0,
    max_sl_pts: float = 80.0,
    same_dir_refresh: str = "entry",
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Single position; per-trade SL = entry to impulse bar low/high, TP = tp_multiple × SL.

    signal_table from build_signal_table (indexed by signal_ts).
    Entry timing via cfg entry_mode: open | pullback | breakout (v16/backtest/impulse_entry.py).
    """
    if signal_table.empty:
        return pd.DataFrame()

    is_cfg = (cfg or {}).get("impulse_stop", {})
    tp_multiple = float(is_cfg.get("tp_multiple", tp_multiple))
    tp_mode = str(is_cfg.get("tp_mode", "multiple"))
    tp_pts = is_cfg.get("tp_pts")
    horizon = int(is_cfg.get("horizon", horizon))
    min_sl_pts = float(is_cfg.get("min_sl_pts", min_sl_pts))
    max_sl_pts = float(is_cfg.get("max_sl_pts", max_sl_pts))

    from v16.backtest.impulse_entry import build_resolved_entry_table
    from v16.structure.filter import apply_structure_gate

    signal_table = apply_structure_gate(df, signal_table, cfg=cfg)
    if signal_table.empty:
        return pd.DataFrame()

    entries = build_resolved_entry_table(df, signal_table, cfg=cfg)
    if entries.empty:
        return pd.DataFrame()

    dlf = (cfg or {}).get("day_level_entry_filter") or {}
    if dlf.get("enabled") and dlf.get("offset") is not None:
        from v16.backtest.day_levels import filter_entries_by_day_level

        entries = filter_entries_by_day_level(df, entries, float(dlf["offset"]))
        if entries.empty:
            return pd.DataFrame()

    entry_by_idx: dict[int, pd.Series] = {}
    for ts, row in entries.iterrows():
        entry_by_idx[int(row["entry_idx"])] = row

    signal_side = pd.Series(0, index=df.index, dtype=int)
    signal_side.loc[signal_table.index] = signal_table["side"].astype(int)
    side_arr = signal_side.to_numpy(dtype=int)
    n = len(df) - 1
    if n <= 0:
        return pd.DataFrame()

    low_bid = df["low_bid"].to_numpy(dtype=float)
    high_ask = df["high_ask"].to_numpy(dtype=float)
    close_bid = df["close_bid"].to_numpy(dtype=float)
    close_ask = df["close_ask"].to_numpy(dtype=float)
    index = df.index
    struct_trend = _structure_trend_arr(df, cfg)
    is_cfg = (cfg or {}).get("impulse_stop", {})
    struct_exit_min = float(is_cfg.get("exit_on_structure_change_min_pnl", 0.0))
    tp_enabled = bool(is_cfg.get("tp_enabled", True))
    reverse_exit = bool(is_cfg.get("exit_on_reverse_signal", False))
    reverse_exit_min = float(is_cfg.get("exit_on_reverse_signal_min_pnl", -1e9))
    reverse_raw = bool(is_cfg.get("exit_on_reverse_signal_raw", True))
    same_dir_refresh = str((cfg or {}).get("same_dir_refresh", same_dir_refresh))

    reverse_side_arr = side_arr
    if reverse_exit and reverse_raw:
        from v16.patterns.momentum_15m_hold import build_signal_table as _build_signals

        raw_signals = _build_signals(df, cfg=cfg)
        reverse_side = pd.Series(0, index=df.index, dtype=int)
        if not raw_signals.empty:
            reverse_side.loc[raw_signals.index] = raw_signals["side"].astype(int)
        reverse_side_arr = reverse_side.to_numpy(dtype=int)

    trades: list[dict] = []
    active_pos: dict | None = None
    i = 0

    while i < n:
        now_ts = index[i]
        sig_side = int(side_arr[i])

        if active_pos:
            s = active_pos["side"]
            exit_info = None
            if s == 1:
                if low_bid[i] <= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif tp_enabled and high_ask[i] >= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif struct_trend is not None:
                    sc = _structure_change_exit(
                        s,
                        active_pos["entry_price"],
                        close_bid[i],
                        close_ask[i],
                        cur_trend=int(struct_trend[i]),
                        entry_trend=int(active_pos["entry_struct_trend"]),
                        min_pnl=struct_exit_min,
                    )
                    if sc is not None:
                        exit_info = sc
                if exit_info is None and now_ts >= active_pos["timeout"]:
                    exit_info = (close_bid[i], "timeout")
            else:
                if high_ask[i] >= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif tp_enabled and low_bid[i] <= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif struct_trend is not None:
                    sc = _structure_change_exit(
                        s,
                        active_pos["entry_price"],
                        close_bid[i],
                        close_ask[i],
                        cur_trend=int(struct_trend[i]),
                        entry_trend=int(active_pos["entry_struct_trend"]),
                        min_pnl=struct_exit_min,
                    )
                    if sc is not None:
                        exit_info = sc
                if exit_info is None and now_ts >= active_pos["timeout"]:
                    exit_info = (close_ask[i], "timeout")

            if exit_info:
                px, reason = exit_info
                pnl = (px - active_pos["entry_price"]) * s
                trades.append(
                    {
                        **active_pos,
                        "exit_time": now_ts,
                        "exit_price": px,
                        "exit_reason": reason,
                        "pnl": pnl,
                        "win": pnl > 0,
                    }
                )
                active_pos = None

        if active_pos and reverse_exit:
            s = active_pos["side"]
            rs = _reverse_signal_exit(
                s,
                active_pos["entry_price"],
                close_bid[i],
                close_ask[i],
                sig_side=int(reverse_side_arr[i]),
                min_pnl=reverse_exit_min,
            )
            if rs is not None:
                px, reason = rs
                pnl = (px - active_pos["entry_price"]) * s
                trades.append(
                    {
                        **active_pos,
                        "exit_time": now_ts,
                        "exit_price": px,
                        "exit_reason": reason,
                        "pnl": pnl,
                        "win": pnl > 0,
                    }
                )
                active_pos = None

        if active_pos and sig_side != 0 and sig_side == active_pos["side"] and same_dir_refresh != "none":
            active_pos["timeout"] = now_ts + pd.Timedelta(minutes=int(active_pos["horizon"]))
            active_pos["target_updates"] = active_pos.get("target_updates", 0) + 1

        if active_pos is None and i in entry_by_idx:
            row = entry_by_idx[i]
            sig_side_i = int(row["side"])
            ep = float(row["entry_price"])
            stop_mode = str((cfg or {}).get("impulse_stop", {}).get("stop_mode", "impulse_bar"))
            levels = _entry_tpsl_levels(
                row,
                sig_side_i,
                ep,
                tp_multiple=tp_multiple,
                tp_mode=tp_mode,
                tp_pts=tp_pts,
                min_sl_pts=min_sl_pts,
                max_sl_pts=max_sl_pts,
                stop_mode=stop_mode,
            )
            if levels is not None:
                sl_pts, tp_pts, stop, target = levels
                entry_ts = index[i]
                active_pos = {
                    "side": sig_side_i,
                    "signal_ts": row.name,
                    "entry_time": entry_ts,
                    "entry_price": ep,
                    "entry_style": row.get("entry_style", "open"),
                    "stop": stop,
                    "target": target,
                    "sl": sl_pts,
                    "tp": tp_pts,
                    "horizon": horizon,
                    "timeout": entry_ts + pd.Timedelta(minutes=horizon),
                    "impulse_low": float(row["impulse_low"]),
                    "impulse_high": float(row["impulse_high"]),
                    "slot_low": float(row.get("slot_low", row["impulse_low"])),
                    "slot_high": float(row.get("slot_high", row["impulse_high"])),
                    "stop_mode": stop_mode,
                    "entry_struct_trend": int(struct_trend[i]) if struct_trend is not None else 0,
                    "target_updates": 0,
                }

        i += 1

    return pd.DataFrame(trades)


def _retrace_day_target_levels(
    side: int,
    entry_price: float,
    imp_low: float,
    imp_high: float,
    day_low: float,
    day_high: float,
    *,
    retrace_sl_pct: float,
    target_offset: float,
    min_sl_pts: float,
    max_sl_pts: float,
) -> tuple[float, float, float, float] | None:
    """
    SL distance = retrace_sl_pct × impulse bar range.
    Long target = day_low + target_offset; short target = day_high - target_offset.
    """
    rng = max(float(imp_high) - float(imp_low), 0.0)
    sl_pts = float(retrace_sl_pct) * rng
    if sl_pts < min_sl_pts or sl_pts > max_sl_pts:
        return None

    if side == 1:
        stop = float(entry_price) - sl_pts
        target = float(day_low) + float(target_offset)
        if stop >= entry_price or target <= entry_price:
            return None
    else:
        stop = float(entry_price) + sl_pts
        target = float(day_high) - float(target_offset)
        if stop <= entry_price or target >= entry_price:
            return None

    tp_pts = abs(target - entry_price)
    return sl_pts, tp_pts, stop, target


def simulate_position_retrace_day_target(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    *,
    retrace_sl_pct: float = 1.0,
    target_offset: float = 80.0,
    horizon: int = 120,
    min_sl_pts: float = 1.0,
    max_sl_pts: float = 80.0,
    same_dir_refresh: str = "entry",
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Single position; SL = retrace_sl_pct × impulse range; TP at day low + offset (long)
    or day high - offset (short). Uses resolved entries from cfg (breakout / open / pullback).
    """
    if signal_table.empty:
        return pd.DataFrame()

    from v16.backtest.day_levels import attach_day_levels
    from v16.backtest.impulse_entry import build_resolved_entry_table, impulse_bar_range
    from v16.structure.filter import apply_structure_gate

    if "day_low_rolling" not in df.columns or "day_high_rolling" not in df.columns:
        df = attach_day_levels(df)

    signal_table = apply_structure_gate(df, signal_table, cfg=cfg)
    if signal_table.empty:
        return pd.DataFrame()

    entries = build_resolved_entry_table(df, signal_table, cfg=cfg)
    if entries.empty:
        return pd.DataFrame()

    entry_by_idx: dict[int, pd.Series] = {}
    for ts, row in entries.iterrows():
        entry_by_idx[int(row["entry_idx"])] = row

    n = len(df) - 1
    if n <= 0:
        return pd.DataFrame()

    low_bid = df["low_bid"].to_numpy(dtype=float)
    high_ask = df["high_ask"].to_numpy(dtype=float)
    close_bid = df["close_bid"].to_numpy(dtype=float)
    close_ask = df["close_ask"].to_numpy(dtype=float)
    day_low = df["day_low_rolling"].to_numpy(dtype=float)
    day_high = df["day_high_rolling"].to_numpy(dtype=float)
    index = df.index

    trades: list[dict] = []
    active_pos: dict | None = None
    i = 0

    while i < n:
        now_ts = index[i]

        if active_pos:
            s = active_pos["side"]
            exit_info = None
            if s == 1:
                if low_bid[i] <= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif high_ask[i] >= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif now_ts >= active_pos["timeout"]:
                    exit_info = (close_bid[i], "timeout")
            else:
                if high_ask[i] >= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif low_bid[i] <= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif now_ts >= active_pos["timeout"]:
                    exit_info = (close_ask[i], "timeout")

            if exit_info:
                px, reason = exit_info
                pnl = (px - active_pos["entry_price"]) * s
                trades.append(
                    {
                        **active_pos,
                        "exit_time": now_ts,
                        "exit_price": px,
                        "exit_reason": reason,
                        "pnl": pnl,
                        "win": pnl > 0,
                    }
                )
                active_pos = None

        if active_pos and i in entry_by_idx and int(entry_by_idx[i]["side"]) == active_pos["side"]:
            if same_dir_refresh != "none":
                active_pos["timeout"] = now_ts + pd.Timedelta(minutes=int(active_pos["horizon"]))

        if active_pos is None and i in entry_by_idx:
            row = entry_by_idx[i]
            sig_side_i = int(row["side"])
            ep = float(row["entry_price"])
            imp_low = float(row["impulse_low"])
            imp_high = float(row["impulse_high"])
            levels = _retrace_day_target_levels(
                sig_side_i,
                ep,
                imp_low,
                imp_high,
                float(day_low[i]),
                float(day_high[i]),
                retrace_sl_pct=retrace_sl_pct,
                target_offset=target_offset,
                min_sl_pts=min_sl_pts,
                max_sl_pts=max_sl_pts,
            )
            if levels is not None:
                sl_pts, tp_pts, stop, target = levels
                entry_ts = index[i]
                active_pos = {
                    "side": sig_side_i,
                    "signal_ts": row.name,
                    "entry_time": entry_ts,
                    "entry_price": ep,
                    "entry_style": row.get("entry_style", "open"),
                    "stop": stop,
                    "target": target,
                    "sl": sl_pts,
                    "tp": tp_pts,
                    "impulse_range": impulse_bar_range(imp_low, imp_high),
                    "horizon": horizon,
                    "timeout": entry_ts + pd.Timedelta(minutes=horizon),
                    "impulse_low": imp_low,
                    "impulse_high": imp_high,
                    "target_updates": 0,
                }

        i += 1

    return pd.DataFrame(trades)


def simulate_position_sided(
    df: pd.DataFrame,
    side_signals: pd.Series,
    *,
    tp: float,
    sl: float,
    horizon: int,
    same_dir_refresh: str = "entry",
    upgrade_stop: bool = False,
) -> pd.DataFrame:
    """
    Single position; side_signals is per-bar 1 (long), -1 (short), or 0 (flat).
    Same-direction refresh when a new signal matches the open side.
    """
    side_arr = side_signals.reindex(df.index, fill_value=0).astype(int).to_numpy()
    signal_idxs = np.flatnonzero(side_arr != 0)
    n = len(df) - 1
    if n <= 0:
        return pd.DataFrame()

    mid = df["mid"].to_numpy(dtype=float)
    low_bid = df["low_bid"].to_numpy(dtype=float)
    high_ask = df["high_ask"].to_numpy(dtype=float)
    close_bid = df["close_bid"].to_numpy(dtype=float)
    close_ask = df["close_ask"].to_numpy(dtype=float)
    open_ask = df["open_ask"].to_numpy(dtype=float)
    open_bid = df["open_bid"].to_numpy(dtype=float)
    index = df.index

    trades: list[dict] = []
    active_pos: dict | None = None
    sig_ptr = 0
    i = 0

    while i < n:
        if active_pos is None:
            while sig_ptr < len(signal_idxs) and signal_idxs[sig_ptr] < i:
                sig_ptr += 1
            if sig_ptr >= len(signal_idxs):
                break
            i = int(signal_idxs[sig_ptr])

        now_ts = index[i]
        sig_side = int(side_arr[i])

        if active_pos:
            s = active_pos["side"]
            exit_info = None
            if s == 1:
                if low_bid[i] <= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif high_ask[i] >= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif now_ts >= active_pos["timeout"]:
                    exit_info = (close_bid[i], "timeout")
            else:
                if high_ask[i] >= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif low_bid[i] <= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif now_ts >= active_pos["timeout"]:
                    exit_info = (close_ask[i], "timeout")

            if exit_info:
                px, reason = exit_info
                pnl = (px - active_pos["entry_price"]) * s
                trades.append(
                    {
                        **active_pos,
                        "exit_time": now_ts,
                        "exit_price": px,
                        "exit_reason": reason,
                        "pnl": pnl,
                        "win": pnl > 0,
                    }
                )
                active_pos = None

        if active_pos and sig_side != 0 and sig_side == active_pos["side"] and same_dir_refresh != "none":
            apply_same_direction_upgrade(
                active_pos,
                float(mid[i]),
                sig_side,
                now_ts,
                tp,
                sl,
                horizon,
                mode=same_dir_refresh,
                upgrade_stop=upgrade_stop,
            )

        if active_pos is None and sig_side != 0:
            ep = float(open_ask[i + 1] if sig_side == 1 else open_bid[i + 1])
            entry_ts = index[i + 1]
            active_pos = {
                "side": sig_side,
                "signal_ts": now_ts,
                "entry_time": entry_ts,
                "entry_price": ep,
                "stop": ep - sl if sig_side == 1 else ep + sl,
                "target": ep + tp if sig_side == 1 else ep - tp,
                "timeout": entry_ts + pd.Timedelta(minutes=horizon),
                "tp": tp,
                "sl": sl,
                "horizon": horizon,
                "target_updates": 0,
                "stop_updates": 0,
            }

        i += 1

    return pd.DataFrame(trades)
