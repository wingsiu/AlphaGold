"""
impulse_1m_15m — first 1m bar >=5pt in 15m slot; enter next slot in impulse direction.

Rule:
  1. First 1m bar in the 15m slot with |body| >= min_move_pts
  2. Wait for slot close → signal at entry_minute of next 15m slot (default 0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v16.backtest.features import session_mask
from v16.config import v16_config


def _cfg() -> dict:
    return v16_config.MOMENTUM_15M_HOLD


def _slot_impulse(
    slot_df: pd.DataFrame,
    *,
    min_move: float,
    change_mode: str = "body",
    reverse_impulse: bool = False,
) -> dict | None:
    """
    First 1m bar in slot with |body| (or range) >= min_move.
    Returns impulse metadata dict or None.
    """
    if slot_df.empty:
        return None

    body = (slot_df["close_ask"] - slot_df["open_ask"]).astype(float)
    rng = (slot_df["high_ask"] - slot_df["low_bid"]).astype(float)
    if change_mode == "range":
        hit = rng >= min_move
    elif change_mode == "body":
        hit = body.abs() >= min_move
    else:
        hit = (body.abs() >= min_move) | (rng >= min_move)
    if not hit.any():
        return None

    i = int(np.flatnonzero(hit.values)[0])
    row = slot_df.iloc[i]
    b = float(body.iloc[i])
    side = 1 if b >= 0 else -1
    if reverse_impulse:
        side = -side
    return {
        "side": side,
        "impulse_low": float(row["low_bid"]),
        "impulse_high": float(row["high_ask"]),
        "impulse_open_ask": float(row["open_ask"]),
        "impulse_open_bid": float(row["open_bid"]),
        "impulse_body": b,
        "impulse_body_abs": abs(b),
        "impulse_range": float(rng.iloc[i]),
        "impulse_volume": float(row["volume"]),
        "impulse_minute": i,
        "bars_after_impulse": max(0, len(slot_df) - 1 - i),
        "slot_low": float(slot_df["low_bid"].min()),
        "slot_high": float(slot_df["high_ask"].max()),
    }

def build_signal_table(
    df: pd.DataFrame,
    *,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Per-entry rows: side, impulse bar low/high (bid/ask), signal timestamp.
    Entry is next bar after signal_ts (see position sim).
    """
    c = cfg or _cfg()
    min_move = float(c.get("min_move_pts", 5.0))
    change_mode = c.get("change_mode", "body")
    entry_min = int(c.get("entry_minute_in_slot", 0))
    sessions = tuple(c.get("sessions", ("london", "ny")))

    rows: list[dict] = []
    slot_id = df.index.floor("15min")
    in_sess = session_mask(df.index, sessions)

    for sid, slot_df in df.groupby(slot_id, sort=True):
        imp = _slot_impulse(
            slot_df,
            min_move=min_move,
            change_mode=change_mode,
            reverse_impulse=bool(c.get("reverse_impulse", False)),
        )
        if imp is None:
            continue
        min_vol = c.get("min_impulse_volume")
        if min_vol is not None and float(imp.get("impulse_volume", 0.0)) < float(min_vol):
            continue
        min_after = c.get("min_bars_after_impulse")
        if min_after is not None and int(imp.get("bars_after_impulse", 0)) < int(min_after):
            continue
        entry_ts = sid + pd.Timedelta(minutes=15 + entry_min)
        if entry_ts not in df.index or not in_sess.loc[entry_ts]:
            continue
        rows.append({"signal_ts": entry_ts, **imp})

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("signal_ts").sort_index()
    return out[~out.index.duplicated(keep="first")]


def build_side_signals(
    df: pd.DataFrame,
    *,
    cfg: dict | None = None,
) -> pd.Series:
    """Side signal on chosen minute of slot after prior slot's first 5pt 1m impulse."""
    table = build_signal_table(df, cfg=cfg)
    sides = pd.Series(0, index=df.index, dtype=int)
    if not table.empty:
        sides.loc[table.index] = table["side"].astype(int)
    return sides


def count_signals(side_signals: pd.Series) -> dict:
    return {
        "long": int((side_signals == 1).sum()),
        "short": int((side_signals == -1).sum()),
        "total": int((side_signals != 0).sum()),
    }


def build_labeled_set(
    df: pd.DataFrame,
    *,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Label each impulse signal (win = pnl > 0). Uses scaleout or fixed execution from cfg."""
    c = cfg or _cfg()
    label_mode = c.get("ml_label_mode", "scaleout")
    rows: list[dict] = []

    if label_mode == "impulse_stop":
        from v16.backtest.position_sim import simulate_impulse_stop_trade, _stop_price_for_row
        from v16.structure.filter import apply_structure_gate

        is_cfg = c.get("impulse_stop", {})
        tp_r = float(is_cfg.get("tp_multiple", 3.0))
        horizon = int(is_cfg.get("horizon", 120))
        min_sl = float(is_cfg.get("min_sl_pts", 1.0))
        max_sl = float(is_cfg.get("max_sl_pts", 80.0))
        stop_mode = str(is_cfg.get("stop_mode", "impulse_bar"))
        signals = build_signal_table(df, cfg=c)
        if c.get("structure", {}).get("gate", {}).get("enabled", False):
            signals = apply_structure_gate(df, signals, cfg=c)
        raw_signals = build_signal_table(df, cfg=c)
        signal_side = pd.Series(0, index=df.index, dtype=int)
        signal_side.loc[signals.index] = signals["side"].astype(int)
        reverse_side = pd.Series(0, index=df.index, dtype=int)
        reverse_side.loc[raw_signals.index] = raw_signals["side"].astype(int)
        use_raw_reverse = bool(is_cfg.get("exit_on_reverse_signal_raw", True))
        exit_side = reverse_side if use_raw_reverse and is_cfg.get("exit_on_reverse_signal") else signal_side
        for ts, row in signals.iterrows():
            sig_idx = int(df.index.get_loc(ts))
            side_i = int(row["side"])
            from v16.backtest.impulse_entry import resolve_impulse_entry

            resolved = resolve_impulse_entry(
                df,
                sig_idx,
                side_i,
                float(row["impulse_low"]),
                float(row["impulse_high"]),
                cfg=c,
            )
            if resolved is None:
                continue
            entry_idx, ep, entry_style = resolved
            if entry_idx >= len(df) - horizon - 1:
                continue
            stop_px = None
            if stop_mode == "slot_15m":
                stop_px, _ = _stop_price_for_row(side_i, row, stop_mode=stop_mode)
            res = simulate_impulse_stop_trade(
                df,
                entry_idx,
                side_i,
                ep,
                float(row["impulse_low"]),
                float(row["impulse_high"]),
                tp_multiple=tp_r,
                horizon=horizon,
                min_sl_pts=min_sl,
                max_sl_pts=max_sl,
                stop_price=stop_px,
                cfg=c,
                signal_side=exit_side,
            )
            if res["exit_reason"] == "invalid_sl":
                continue
            imp_rng = float(row["impulse_high"]) - float(row["impulse_low"])
            rows.append(
                {
                    "signal_ts": ts,
                    "entry_idx": entry_idx,
                    "entry_style": entry_style,
                    "side": side_i,
                    "pnl": res["pnl"],
                    "win": int(res["pnl"] > 0),
                    "exit_reason": res["exit_reason"],
                    "sl_pts": res["sl"],
                    "tp_pts": res["tp"],
                    "impulse_bar_range": imp_rng,
                    "impulse_body": float(row["impulse_body"]),
                    "impulse_body_abs": float(row["impulse_body_abs"]),
                    "impulse_range": float(row["impulse_range"]),
                    "impulse_volume": float(row.get("impulse_volume", 0.0)),
                    "impulse_minute": int(row["impulse_minute"]),
                    "bars_after_impulse": int(row["bars_after_impulse"]),
                    "impulse_low": float(row["impulse_low"]),
                    "impulse_high": float(row["impulse_high"]),
                }
            )
    elif label_mode == "execution":
        from v16.backtest.fixed_tpsl_sim import simulate_fixed_tpsl

        ex = c.get("execution", {"tp": 25.0, "sl": 35.0, "horizon": 60})
        tp, sl, horizon = float(ex["tp"]), float(ex["sl"]), int(ex["horizon"])
        sides = build_side_signals(df, cfg=c)
        for ts, side in sides[sides != 0].items():
            sig_idx = int(df.index.get_loc(ts))
            entry_idx = sig_idx + 1
            if entry_idx >= len(df) - horizon - 1:
                continue
            side_i = int(side)
            ep = float(df.iloc[entry_idx]["open_ask"] if side_i == 1 else df.iloc[entry_idx]["open_bid"])
            res = simulate_fixed_tpsl(df, entry_idx, side_i, ep, tp=tp, sl=sl, horizon=horizon)
            rows.append(
                {
                    "signal_ts": ts,
                    "entry_idx": entry_idx,
                    "side": side_i,
                    "pnl": res.pnl,
                    "win": int(res.pnl > 0),
                    "exit_reason": res.exit_reason,
                }
            )
    else:
        from v16.backtest.scaleout_sim import simulate_scaleout_trade

        so = dict(c.get("scaleout", v16_config.EXIT_CONFIG))
        horizon = int(so.pop("horizon_minutes", so.pop("horizon", 10)))
        kw = {**so, "horizon": horizon}
        sides = build_side_signals(df, cfg=c)
        for ts, side in sides[sides != 0].items():
            sig_idx = int(df.index.get_loc(ts))
            entry_idx = sig_idx + 1
            if entry_idx >= len(df) - horizon - 1:
                continue
            side_i = int(side)
            ep = float(df.iloc[entry_idx]["open_ask"] if side_i == 1 else df.iloc[entry_idx]["open_bid"])
            res = simulate_scaleout_trade(df, entry_idx, side_i, ep, **kw)
            rows.append(
                {
                    "signal_ts": ts,
                    "entry_idx": entry_idx,
                    "side": side_i,
                    "pnl": res.pnl,
                    "win": int(res.pnl > 0),
                    "exit_reason": res.exit_reason,
                    "scaled_half": int(res.scaled_half),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("signal_ts").sort_index()
