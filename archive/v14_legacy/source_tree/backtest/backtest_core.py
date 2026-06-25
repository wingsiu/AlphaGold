"""
AlphaGold v13 — Reusable Backtest Simulation Core
=================================================
Standalone module exposing `simulate_v13_core` so it can be imported by
`daily_reconciliation.py`, parameter sweeps, etc. without executing the
full backtest script in `backtest.py` (which parses sys.argv at import).
"""
import numpy as np
import pandas as pd

from config.v13_config import EXECUTION_CONFIG


def _exec_params_from_row(row, df_test, tp: float, sl: float, horizon_minutes: int) -> tuple[float, float, int]:
    """Read per-bar exec_* overrides when present (pattern router)."""
    cols = df_test.columns
    use_tp = float(row["exec_tp"]) if "exec_tp" in cols and pd.notna(row.get("exec_tp")) else float(tp)
    use_sl = float(row["exec_sl"]) if "exec_sl" in cols and pd.notna(row.get("exec_sl")) else float(sl)
    use_h = int(row["exec_horizon"]) if "exec_horizon" in cols and pd.notna(row.get("exec_horizon")) else int(horizon_minutes)
    return use_tp, use_sl, use_h


def _apply_same_direction_upgrade(
    active_pos: dict,
    row,
    side: int,
    now_ts,
    tp: float,
    sl: float,
    horizon_minutes: int,
    df_test,
    *,
    mode: str = "entry",
    upgrade_stop: bool = False,
) -> None:
    """
    Same-direction refresh while a position is open.

    mode "entry"  — 2398 baseline: extend timeout and trail target using entry TP/H only.
    mode "global" — extend timeout and trail target using global tp/horizon fallbacks.
    mode "exec"   — max(exec_*) from new signal; optional stop trail via upgrade_stop.
    """
    if mode == "entry":
        pos_tp = float(active_pos.get("tp", tp))
        pos_h = int(active_pos.get("horizon", horizon_minutes))
        active_pos["timeout"] = now_ts + pd.Timedelta(minutes=pos_h)
        active_pos["target_updates"] = active_pos.get("target_updates", 0) + 1
        new_target = row["close"] + pos_tp if side == 1 else row["close"] - pos_tp
        if (side == 1 and new_target > active_pos["target"]) or (side == -1 and new_target < active_pos["target"]):
            active_pos["target"] = new_target
    elif mode == "global":
        active_pos["timeout"] = now_ts + pd.Timedelta(minutes=horizon_minutes)
        active_pos["target_updates"] = active_pos.get("target_updates", 0) + 1
        new_target = row["close"] + tp if side == 1 else row["close"] - tp
        if (side == 1 and new_target > active_pos["target"]) or (side == -1 and new_target < active_pos["target"]):
            active_pos["target"] = new_target
    elif mode == "exec":
        sig_tp, sig_sl, sig_h = _exec_params_from_row(row, df_test, tp, sl, horizon_minutes)
        pos_tp = max(float(active_pos.get("tp", tp)), sig_tp)
        pos_sl = max(float(active_pos.get("sl", sl)), sig_sl)
        pos_h = max(int(active_pos.get("horizon", horizon_minutes)), sig_h)

        active_pos["tp"] = pos_tp
        active_pos["sl"] = pos_sl
        active_pos["horizon"] = pos_h
        active_pos["timeout"] = now_ts + pd.Timedelta(minutes=pos_h)
        active_pos["target_updates"] = active_pos.get("target_updates", 0) + 1

        new_target = row["close"] + pos_tp if side == 1 else row["close"] - pos_tp
        if (side == 1 and new_target > active_pos["target"]) or (side == -1 and new_target < active_pos["target"]):
            active_pos["target"] = new_target

        if upgrade_stop:
            new_stop = row["close"] - pos_sl if side == 1 else row["close"] + pos_sl
            if (side == 1 and new_stop > active_pos["stop"]) or (side == -1 and new_stop < active_pos["stop"]):
                active_pos["stop"] = new_stop
                active_pos["stop_updates"] = active_pos.get("stop_updates", 0) + 1
    else:
        return

    if "matched_pattern" in df_test.columns and pd.notna(row.get("matched_pattern")):
        active_pos["matched_pattern"] = row["matched_pattern"]


def simulate_v13_core(df_test, tp, sl, horizon_minutes, config=None, weak_period_cells=None):
    """
    Extracted trade simulation loop for parameter sweeping.
    Expects df_test to already have side_signal, s1_prob, s2_prob columns.
    Uses real bid/ask columns when available, else synthetic spread.

    weak_period_cells: optional list of {session, day, hour} dicts — skip new
    entries when the signal bar timestamp falls in a blocked slot.
    """
    from xgboost_filter_model.time_slot_filter import is_blocked_entry
    if config is None:
        from config.v13_config import EXECUTION_CONFIG as config
    
    _spread = config.get("spread_default", 0.25)
    if 'open_ask' not in df_test.columns:
        if 'openPrice_ask' in df_test.columns:
            df_test = df_test.copy()
            df_test['open_ask']  = df_test['openPrice_ask']
            df_test['open_bid']  = df_test['openPrice_bid']
            df_test['close_ask'] = df_test['closePrice_ask']
            df_test['close_bid'] = df_test['closePrice_bid']
            df_test['high_ask']  = df_test['highPrice_ask']
            df_test['low_bid']   = df_test['lowPrice_bid']
        else:
            df_test = df_test.copy()
            df_test['open_ask']  = df_test['open']  + _spread
            df_test['open_bid']  = df_test['open']  - _spread
            df_test['close_ask'] = df_test['close'] + _spread
            df_test['close_bid'] = df_test['close'] - _spread
            df_test['high_ask']  = df_test['high']  + _spread
            df_test['low_bid']   = df_test['low']   - _spread

    s2_base      = config.get("s2_threshold", 0.55)
    s2_increment = config.get("s2_loss_increment", 0.01)
    s2_max       = config.get("s2_max_threshold", 0.70)
    close_on_reverse = config.get("close_on_reverse", True)
    same_dir_refresh = config.get("same_dir_refresh", "entry")
    upgrade_stop = config.get("upgrade_stop", False)

    all_trades, active_pos = [], None
    consecutive_losses = 0
    
    for i in range(len(df_test) - 1):
        row      = df_test.iloc[i]
        next_row = df_test.iloc[i + 1]
        now_ts   = row.name
        sig      = int(row['side_signal'])
        reverse_flip_sig = None

        if active_pos:
            s = active_pos['side']
            exit_info = None
            if s == 1:
                if   row['low_bid']  <= active_pos['stop']:    exit_info = (active_pos['stop'],   'stop_loss')
                elif row['high_ask'] >= active_pos['target']:  exit_info = (active_pos['target'], 'target_hit')
                elif now_ts          >= active_pos['timeout']: exit_info = (row['close_bid'],     'timeout')
            else:
                if   row['high_ask'] >= active_pos['stop']:    exit_info = (active_pos['stop'],   'stop_loss')
                elif row['low_bid']  <= active_pos['target']:  exit_info = (active_pos['target'], 'target_hit')
                elif now_ts          >= active_pos['timeout']: exit_info = (row['close_ask'],     'timeout')
            if exit_info:
                px, reason = exit_info
                pnl = (px - active_pos['entry_price']) * s
                all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': px,
                                    'exit_reason': reason, 'pnl': pnl})
                consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
                active_pos = None

        if active_pos:
            s = active_pos['side']
            if close_on_reverse and sig != 0 and sig == -s:
                px  = row['close_bid'] if s == 1 else row['close_ask']
                pnl = (px - active_pos['entry_price']) * s
                all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': px,
                                    'exit_reason': 'reverse_signal', 'pnl': pnl})
                consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
                active_pos = None
                reverse_flip_sig = sig
            elif sig == s and same_dir_refresh != "none":
                _apply_same_direction_upgrade(
                    active_pos, row, s, now_ts, tp, sl, horizon_minutes, df_test,
                    mode=same_dir_refresh,
                    upgrade_stop=upgrade_stop,
                )

        if active_pos is None and sig != 0:
            if is_blocked_entry(now_ts, weak_period_cells):
                continue
            dynamic_s2 = min(s2_max, s2_base + consecutive_losses * s2_increment)
            s2_p = row["s2_prob"]
            if reverse_flip_sig is not None and sig == reverse_flip_sig:
                passes = True
            else:
                passes = (sig == 1 and s2_p >= dynamic_s2) or (sig == -1 and s2_p <= (1.0 - dynamic_s2))
            if passes:
                ep = next_row["open_ask"] if sig == 1 else next_row["open_bid"]
                use_tp, use_sl, use_h = _exec_params_from_row(row, df_test, tp, sl, horizon_minutes)
                active_pos = {
                    "side": sig,
                    "entry_time": next_row.name,
                    "entry_price": ep,
                    "stop":    ep - use_sl if sig == 1 else ep + use_sl,
                    "target":  ep + use_tp if sig == 1 else ep - use_tp,
                    "timeout": next_row.name + pd.Timedelta(minutes=use_h),
                    "target_updates": 0,
                    "stop_updates": 0,
                    "s1_prob": row["s1_prob"],
                    "s2_prob": s2_p,
                    "tp": use_tp,
                    "sl": use_sl,
                    "horizon": use_h,
                }
                if "matched_pattern" in df_test.columns and pd.notna(row.get("matched_pattern")):
                    active_pos["matched_pattern"] = row["matched_pattern"]

    return all_trades


def simulate_hybrid_core(
    df_test,
    tp,
    sl,
    horizon_minutes,
    config=None,
    pattern_config=None,
    weak_period_cells=None,
):
    """
    Pattern-first hybrid simulation.

    - New entry: pattern_side if set, else energetic_side (energetic bars only).
    - Open pattern position: managed by pattern signals + pattern_config rules.
    - Open energetic position: managed by energetic signals + config (global TP/SL/H).
    """
    from xgboost_filter_model.time_slot_filter import is_blocked_entry

    if config is None:
        from config.v13_config import EXECUTION_CONFIG as config
    if pattern_config is None:
        pattern_config = config

    _spread = config.get("spread_default", 0.25)
    if "open_ask" not in df_test.columns:
        if "openPrice_ask" in df_test.columns:
            df_test = df_test.copy()
            df_test["open_ask"] = df_test["openPrice_ask"]
            df_test["open_bid"] = df_test["openPrice_bid"]
            df_test["close_ask"] = df_test["closePrice_ask"]
            df_test["close_bid"] = df_test["closePrice_bid"]
            df_test["high_ask"] = df_test["highPrice_ask"]
            df_test["low_bid"] = df_test["lowPrice_bid"]
        else:
            df_test = df_test.copy()
            df_test["open_ask"] = df_test["open"] + _spread
            df_test["open_bid"] = df_test["open"] - _spread
            df_test["close_ask"] = df_test["close"] + _spread
            df_test["close_bid"] = df_test["close"] - _spread
            df_test["high_ask"] = df_test["high"] + _spread
            df_test["low_bid"] = df_test["low"] - _spread

    s2_base = config.get("s2_threshold", 0.55)
    s2_increment = config.get("s2_loss_increment", 0.01)
    s2_max = config.get("s2_max_threshold", 0.70)

    en_close_rev = config.get("close_on_reverse", True)
    en_refresh = config.get("same_dir_refresh", "global")
    en_upgrade_stop = config.get("upgrade_stop", False)

    pat_close_rev = pattern_config.get("close_on_reverse", False)
    pat_refresh = pattern_config.get("same_dir_refresh", "entry")
    pat_upgrade_stop = pattern_config.get("upgrade_stop", False)

    all_trades, active_pos = [], None
    consecutive_losses = 0

    for i in range(len(df_test) - 1):
        row = df_test.iloc[i]
        next_row = df_test.iloc[i + 1]
        now_ts = row.name
        pat_sig = int(row.get("pattern_side", 0) or 0)
        en_sig = int(row.get("energetic_side", 0) or 0)
        reverse_flip_sig = None

        if active_pos:
            s = active_pos["side"]
            exit_info = None
            if s == 1:
                if row["low_bid"] <= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif row["high_ask"] >= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif now_ts >= active_pos["timeout"]:
                    exit_info = (row["close_bid"], "timeout")
            else:
                if row["high_ask"] >= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif row["low_bid"] <= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif now_ts >= active_pos["timeout"]:
                    exit_info = (row["close_ask"], "timeout")
            if exit_info:
                px, reason = exit_info
                pnl = (px - active_pos["entry_price"]) * s
                all_trades.append(
                    {**active_pos, "exit_time": now_ts, "exit_price": px, "exit_reason": reason, "pnl": pnl}
                )
                consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
                active_pos = None

        # Pattern has absolute priority: close an open energetic trade when a pattern
        # signal fires so pattern entries are never blocked by fallback positions.
        if active_pos and active_pos.get("source") == "energetic" and pat_sig != 0:
            s = active_pos["side"]
            px = row["close_bid"] if s == 1 else row["close_ask"]
            pnl = (px - active_pos["entry_price"]) * s
            all_trades.append(
                {
                    **active_pos,
                    "exit_time": now_ts,
                    "exit_price": px,
                    "exit_reason": "pattern_priority",
                    "pnl": pnl,
                }
            )
            consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
            active_pos = None

        if active_pos:
            s = active_pos["side"]
            source = active_pos.get("source", "pattern")
            sig = pat_sig if source == "pattern" else en_sig
            close_rev = pat_close_rev if source == "pattern" else en_close_rev
            refresh_mode = pat_refresh if source == "pattern" else en_refresh
            up_stop = pat_upgrade_stop if source == "pattern" else en_upgrade_stop
            use_tp = tp if source == "energetic" else float(active_pos.get("tp", tp))
            use_sl = sl if source == "energetic" else float(active_pos.get("sl", sl))
            use_h = horizon_minutes if source == "energetic" else int(active_pos.get("horizon", horizon_minutes))

            if close_rev and sig != 0 and sig == -s:
                px = row["close_bid"] if s == 1 else row["close_ask"]
                pnl = (px - active_pos["entry_price"]) * s
                all_trades.append(
                    {
                        **active_pos,
                        "exit_time": now_ts,
                        "exit_price": px,
                        "exit_reason": "reverse_signal",
                        "pnl": pnl,
                    }
                )
                consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
                active_pos = None
                reverse_flip_sig = sig
            elif sig == s and refresh_mode != "none":
                _apply_same_direction_upgrade(
                    active_pos,
                    row,
                    s,
                    now_ts,
                    use_tp,
                    use_sl,
                    use_h,
                    df_test,
                    mode=refresh_mode,
                    upgrade_stop=up_stop,
                )

        if active_pos is None:
            if pat_sig != 0:
                if is_blocked_entry(now_ts, weak_period_cells):
                    continue
                ep = next_row["open_ask"] if pat_sig == 1 else next_row["open_bid"]
                use_tp, use_sl, use_h = _exec_params_from_row(row, df_test, tp, sl, horizon_minutes)
                active_pos = {
                    "side": pat_sig,
                    "source": "pattern",
                    "entry_time": next_row.name,
                    "entry_price": ep,
                    "stop": ep - use_sl if pat_sig == 1 else ep + use_sl,
                    "target": ep + use_tp if pat_sig == 1 else ep - use_tp,
                    "timeout": next_row.name + pd.Timedelta(minutes=use_h),
                    "target_updates": 0,
                    "stop_updates": 0,
                    "s1_prob": row.get("s1_prob"),
                    "s2_prob": row.get("s2_prob"),
                    "tp": use_tp,
                    "sl": use_sl,
                    "horizon": use_h,
                }
                if "matched_pattern" in df_test.columns and pd.notna(row.get("matched_pattern")):
                    active_pos["matched_pattern"] = row["matched_pattern"]
            elif en_sig != 0:
                if is_blocked_entry(now_ts, weak_period_cells):
                    continue
                # Block energetic entry when a pattern position is already open.
                # This mirrors simulate_hybrid_two_pass's busy-mask logic.
                pattern_open = any(
                    t["source"] == "pattern"
                    and pd.Timestamp(t["entry_time"]) <= now_ts <= pd.Timestamp(t["exit_time"])
                    for t in all_trades
                )
                if pattern_open:
                    continue
                dynamic_s2 = min(s2_max, s2_base + consecutive_losses * s2_increment)
                s2_p = row.get("energetic_s2_prob")
                if reverse_flip_sig is not None and en_sig == reverse_flip_sig:
                    passes = True
                elif pd.isna(s2_p):
                    passes = False
                else:
                    passes = (en_sig == 1 and s2_p >= dynamic_s2) or (
                        en_sig == -1 and s2_p <= (1.0 - dynamic_s2)
                    )
                if passes:
                    ep = next_row["open_ask"] if en_sig == 1 else next_row["open_bid"]
                    active_pos = {
                        "side": en_sig,
                        "source": "energetic",
                        "entry_time": next_row.name,
                        "entry_price": ep,
                        "stop": ep - sl if en_sig == 1 else ep + sl,
                        "target": ep + tp if en_sig == 1 else ep - tp,
                        "timeout": next_row.name + pd.Timedelta(minutes=horizon_minutes),
                        "target_updates": 0,
                        "stop_updates": 0,
                        "s1_prob": row.get("energetic_s1_prob"),
                        "s2_prob": s2_p,
                        "tp": tp,
                        "sl": sl,
                        "horizon": horizon_minutes,
                        "matched_pattern": "energetic",
                    }

    return all_trades


def _ts_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _build_pattern_busy_mask(pattern_trades, index):
    """O(trades * log bars) mask — True while a pattern position is open."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    busy = np.zeros(len(idx), dtype=bool)
    for t in pattern_trades:
        et = _ts_utc(t["entry_time"])
        xt = _ts_utc(t["exit_time"])
        i0 = int(idx.searchsorted(et, side="left"))
        i1 = int(idx.searchsorted(xt, side="left"))
        if i0 < i1:
            busy[i0:i1] = True
    return busy


def _build_pattern_entry_flags(pattern_trades, index):
    """Boolean mask on index — True on bars where a pattern trade opens."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    pos = idx.get_indexer([_ts_utc(t["entry_time"]) for t in pattern_trades])
    flags = np.zeros(len(idx), dtype=bool)
    flags[pos[pos >= 0]] = True
    return flags


def simulate_hybrid_two_pass(
    df_test,
    pattern_trades,
    tp,
    sl,
    horizon_minutes,
    config=None,
    weak_period_cells=None,
):
    """
    Two-pass hybrid: pattern leg is identical to pattern-only sim; energetic
    fallback only enters when flat (no pattern position) and no pattern signal.
    """
    from xgboost_filter_model.time_slot_filter import is_blocked_entry

    if config is None:
        config = EXECUTION_CONFIG

    df = df_test.copy()
    busy = _build_pattern_busy_mask(pattern_trades, df.index)
    pat_entry = _build_pattern_entry_flags(pattern_trades, df.index)
    next_pat_entry = np.roll(pat_entry, -1)
    next_pat_entry[-1] = False

    pat_side = df["pattern_side"].fillna(0).astype(int).to_numpy()
    en_side = df.get("energetic_side", pd.Series(0, index=df.index)).fillna(0).astype(int).to_numpy()
    filtered = en_side.copy()
    filtered[(pat_side != 0) | busy | next_pat_entry] = 0
    df["side_signal"] = filtered

    s2_base = config.get("s2_threshold", 0.55)
    s2_increment = config.get("s2_loss_increment", 0.01)
    s2_max = config.get("s2_max_threshold", 0.70)
    close_on_reverse = config.get("close_on_reverse", True)
    same_dir_refresh = config.get("same_dir_refresh", "global")
    upgrade_stop = config.get("upgrade_stop", False)

    _spread = config.get("spread_default", 0.25)
    if "open_ask" not in df.columns:
        if "openPrice_ask" in df.columns:
            df["open_ask"] = df["openPrice_ask"]
            df["open_bid"] = df["openPrice_bid"]
            df["close_ask"] = df["closePrice_ask"]
            df["close_bid"] = df["closePrice_bid"]
            df["high_ask"] = df["highPrice_ask"]
            df["low_bid"] = df["lowPrice_bid"]
        else:
            df["open_ask"] = df["open"] + _spread
            df["open_bid"] = df["open"] - _spread
            df["close_ask"] = df["close"] + _spread
            df["close_bid"] = df["close"] - _spread
            df["high_ask"] = df["high"] + _spread
            df["low_bid"] = df["low"] - _spread

    en_trades, active_pos = [], None
    consecutive_losses = 0

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        now_ts = row.name
        sig = int(row.get("side_signal", 0) or 0)
        reverse_flip_sig = None
        pattern_open = bool(busy[i])

        if active_pos and pattern_open:
            s = active_pos["side"]
            px = row["close_bid"] if s == 1 else row["close_ask"]
            pnl = (px - active_pos["entry_price"]) * s
            en_trades.append(
                {
                    **active_pos,
                    "exit_time": now_ts,
                    "exit_price": px,
                    "exit_reason": "pattern_priority",
                    "pnl": pnl,
                }
            )
            consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
            active_pos = None

        if active_pos:
            s = active_pos["side"]
            exit_info = None
            if s == 1:
                if row["low_bid"] <= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif row["high_ask"] >= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif now_ts >= active_pos["timeout"]:
                    exit_info = (row["close_bid"], "timeout")
            else:
                if row["high_ask"] >= active_pos["stop"]:
                    exit_info = (active_pos["stop"], "stop_loss")
                elif row["low_bid"] <= active_pos["target"]:
                    exit_info = (active_pos["target"], "target_hit")
                elif now_ts >= active_pos["timeout"]:
                    exit_info = (row["close_ask"], "timeout")
            if exit_info:
                px, reason = exit_info
                pnl = (px - active_pos["entry_price"]) * s
                en_trades.append(
                    {**active_pos, "exit_time": now_ts, "exit_price": px, "exit_reason": reason, "pnl": pnl}
                )
                consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
                active_pos = None

        if active_pos:
            s = active_pos["side"]
            if close_on_reverse and sig != 0 and sig == -s:
                px = row["close_bid"] if s == 1 else row["close_ask"]
                pnl = (px - active_pos["entry_price"]) * s
                en_trades.append(
                    {
                        **active_pos,
                        "exit_time": now_ts,
                        "exit_price": px,
                        "exit_reason": "reverse_signal",
                        "pnl": pnl,
                    }
                )
                consecutive_losses = 0 if pnl > 0 else consecutive_losses + 1
                active_pos = None
                reverse_flip_sig = sig
            elif sig == s and same_dir_refresh != "none":
                _apply_same_direction_upgrade(
                    active_pos, row, s, now_ts, tp, sl, horizon_minutes, df,
                    mode=same_dir_refresh, upgrade_stop=upgrade_stop,
                )

        if active_pos is None and sig != 0:
            if is_blocked_entry(now_ts, weak_period_cells):
                continue
            if pattern_open:
                continue
            if next_pat_entry[i]:
                continue
            dynamic_s2 = min(s2_max, s2_base + consecutive_losses * s2_increment)
            s2_p = row.get("energetic_s2_prob")
            if reverse_flip_sig is not None and sig == reverse_flip_sig:
                passes = True
            elif pd.isna(s2_p):
                passes = False
            else:
                passes = (sig == 1 and s2_p >= dynamic_s2) or (sig == -1 and s2_p <= (1.0 - dynamic_s2))
            if passes:
                ep = next_row["open_ask"] if sig == 1 else next_row["open_bid"]
                active_pos = {
                    "side": sig,
                    "source": "energetic",
                    "entry_time": next_row.name,
                    "entry_price": ep,
                    "stop": ep - sl if sig == 1 else ep + sl,
                    "target": ep + tp if sig == 1 else ep - tp,
                    "timeout": next_row.name + pd.Timedelta(minutes=horizon_minutes),
                    "target_updates": 0,
                    "stop_updates": 0,
                    "s1_prob": row.get("energetic_s1_prob"),
                    "s2_prob": s2_p,
                    "tp": tp,
                    "sl": sl,
                    "horizon": horizon_minutes,
                    "matched_pattern": "energetic",
                }

    for t in pattern_trades:
        t["source"] = "pattern"
    for t in en_trades:
        t["source"] = "energetic"
        if "matched_pattern" not in t:
            t["matched_pattern"] = "energetic"

    return pattern_trades + en_trades
