"""v16 entry candidates: burst and/or 15m-open fade."""
from __future__ import annotations

import pandas as pd

from v16.backtest.features import session_mask
from v16.backtest.scaleout_sim import simulate_scaleout_trade
from v16.config import v16_config


def _cfg() -> dict:
    return v16_config.SIGNAL_CONFIG


def burst_mask(feats: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    cfg = _cfg()
    in_sess = session_mask(index, tuple(cfg["sessions"]))
    burst = (
        (feats["range_1"] >= cfg["min_range_pts"])
        & (feats["vol_ratio"] >= cfg["min_volume_ratio"])
        & (feats["body_abs"] >= cfg["min_body_pts"])
    )
    stride = int(cfg["sample_stride"])
    stride_ok = pd.Series([i % stride == 0 for i in range(len(index))], index=index)
    return in_sess & burst & stride_ok


def fade_15m_mask(feats: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Fade large prior 15m bar in first N minutes of new slot."""
    cfg = _cfg()
    in_sess = session_mask(index, tuple(cfg["sessions"]))
    open_mins = set(cfg.get("fade_open_minutes", (0, 1, 2, 3, 4)))
    in_open = feats["minute_in_15m"].astype(int).isin(open_mins)
    large = (feats["prev_15m_body_abs"] >= cfg["fade_min_prev_body_pts"]) & (
        feats["prev_15m_range"] >= cfg.get("fade_min_prev_range_pts", 0)
    )
    return in_sess & in_open & large & feats["prev_15m_dir"].abs().ge(1)


def _dip_base_long(feats: pd.DataFrame, index: pd.DatetimeIndex, cfg: dict) -> pd.Series:
    in_sess = session_mask(index, tuple(cfg["sessions"]))
    slot_dn = feats["slot_down"] > 0
    below = feats["dip_from_slot_open"] >= cfg["dip_min_below_open_pts"]
    early = feats["minute_in_15m"].astype(int) < int(cfg["dip_max_minute_in_slot"])
    m = in_sess & slot_dn & below & early
    if cfg.get("dip_require_slot_low", True):
        m = m & (feats["slot_low_dip"] >= cfg["dip_min_slot_low_pts"])
    return m


def _apply_prev_15m_filters(feats: pd.DataFrame, m: pd.Series, cfg: dict, *, side: str) -> pd.Series:
    """Optional prev / prev2 body & range filters (side='long' | 'short')."""
    if side == "long":
        if cfg.get("dip_require_prev_down", True):
            m = m & (feats["prev_15m_dir"] < 0)
        if cfg.get("dip_require_two_prev_down", False):
            m = m & (feats["two_prev_15m_down"] > 0)
        min_b = float(cfg.get("dip_min_prev_body_pts", 0))
        if min_b > 0:
            m = m & (feats["prev_15m_body"] <= -min_b)
        min_r = float(cfg.get("dip_min_prev_range_pts", 0))
        if min_r > 0:
            m = m & (feats["prev_15m_range"] >= min_r)
        min_b2 = float(cfg.get("dip_min_prev2_body_pts", 0))
        if min_b2 > 0:
            m = m & (feats["prev2_15m_body"] <= -min_b2)
        min_r2 = float(cfg.get("dip_min_prev2_range_pts", 0))
        if min_r2 > 0:
            m = m & (feats["prev2_15m_range"] >= min_r2)
    else:
        if cfg.get("dip_require_prev_up", True):
            m = m & (feats["prev_15m_dir"] > 0)
        if cfg.get("dip_require_two_prev_up", False):
            m = m & (feats["two_prev_15m_up"] > 0)
        min_b = float(cfg.get("dip_min_prev_body_pts", 0))
        if min_b > 0:
            m = m & (feats["prev_15m_body"] >= min_b)
        min_r = float(cfg.get("dip_min_prev_range_pts", 0))
        if min_r > 0:
            m = m & (feats["prev_15m_range"] >= min_r)
        min_b2 = float(cfg.get("dip_min_prev2_body_pts", 0))
        if min_b2 > 0:
            m = m & (feats["prev2_15m_body"] >= min_b2)
        min_r2 = float(cfg.get("dip_min_prev2_range_pts", 0))
        if min_r2 > 0:
            m = m & (feats["prev2_15m_range"] >= min_r2)
    return m


def dip_long_15m_mask(feats: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """
    Long after 15m down context + intrabar dip:
      - optional: 1 or 2 prior completed 15m bars down
      - active slot down, price dip below slot open
      - optional prev/prev2 body & range filters
    """
    cfg = _cfg()
    m = _dip_base_long(feats, index, cfg)
    return _apply_prev_15m_filters(feats, m, cfg, side="long")


def dip_short_15m_mask(feats: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Short: prior 15m up (+ optional 2nd) + slot up + rip above slot open."""
    cfg = _cfg()
    in_sess = session_mask(index, tuple(cfg["sessions"]))
    slot_up = feats["slot_up"] > 0
    rip = (-feats["dip_from_slot_open"]) >= cfg.get("dip_short_min_above_open_pts", 5.0)
    early = feats["minute_in_15m"].astype(int) < int(cfg["dip_max_minute_in_slot"])
    m = in_sess & slot_up & rip & early
    slot_high_rip = feats["slot_high"] - feats["slot_open"]
    if cfg.get("dip_require_slot_high", True):
        m = m & (slot_high_rip >= cfg.get("dip_short_min_slot_high_pts", 10.0))
    short_cfg = {**cfg, "dip_require_prev_up": True}
    return _apply_prev_15m_filters(feats, m, short_cfg, side="short")


def dip_pool_mask(feats: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    return _dip_base_long(feats, index, _cfg())


def candidate_mask(feats: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    mode = _cfg().get("mode", "burst")
    if mode == "burst":
        return burst_mask(feats, index)
    if mode == "fade_15m":
        return fade_15m_mask(feats, index)
    if mode == "dip_long_15m":
        return dip_long_15m_mask(feats, index)
    if mode == "dip_short_15m":
        return dip_short_15m_mask(feats, index)
    if mode == "dip_pool":
        return dip_pool_mask(feats, index)
    if mode == "both":
        return burst_mask(feats, index) | fade_15m_mask(feats, index)
    raise ValueError(f"Unknown signal mode: {mode}")


def fade_side(feats: pd.DataFrame, ts: pd.Timestamp) -> int:
    """Counter prior 15m direction: prev up → short, prev down → long."""
    d = int(feats.loc[ts, "prev_15m_dir"])
    return -1 if d > 0 else 1


def _exit_kwargs() -> dict:
    ec = v16_config.EXIT_CONFIG
    return {
        "first_scale_pnl": ec["first_scale_pnl"],
        "first_scale_frac": ec["first_scale_frac"],
        "final_scale_pnl": ec["final_scale_pnl"],
        "initial_sl": ec["initial_sl"],
        "runner_lock_pnl": ec["runner_lock_pnl"],
        "horizon": ec["horizon_minutes"],
    }


def build_labeled_set(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    *,
    exit_overrides: dict | None = None,
) -> pd.DataFrame:
    mask = candidate_mask(feats, df.index)
    horizon = v16_config.EXIT_CONFIG["horizon_minutes"]
    mode = _cfg().get("mode", "burst")
    kw = _exit_kwargs()
    if exit_overrides:
        kw.update(exit_overrides)
    rows: list[dict] = []

    for i in range(len(df) - horizon - 2):
        ts = df.index[i]
        if not mask.iloc[i]:
            continue
        nxt = df.iloc[i + 1]
        entry_idx = i + 1
        ep_long = float(nxt["open_ask"])
        ep_short = float(nxt["open_bid"])
        long_res = simulate_scaleout_trade(df, entry_idx, 1, ep_long, **kw)
        short_res = simulate_scaleout_trade(df, entry_idx, -1, ep_short, **kw)

        fade = fade_side(feats, ts) if mode in ("fade_15m", "both") else None
        fixed_side = 1 if mode == "dip_long_15m" else (-1 if mode == "dip_short_15m" else None)
        rows.append(
            {
                "signal_ts": ts,
                "entry_idx": entry_idx,
                "minute_in_15m": int(feats.loc[ts, "minute_in_15m"]),
                "prev_15m_body": float(feats.loc[ts, "prev_15m_body"]),
                "dip_from_slot_open": float(feats.loc[ts, "dip_from_slot_open"]),
                "slot_low_dip": float(feats.loc[ts, "slot_low_dip"]),
                "fade_side": fade,
                "fixed_side": fixed_side,
                "long_pnl": long_res.pnl,
                "short_pnl": short_res.pnl,
                "long_win": int(long_res.pnl > 0),
                "short_win": int(short_res.pnl > 0),
                "best_side": 1 if long_res.pnl >= short_res.pnl else -1,
                "best_pnl": max(long_res.pnl, short_res.pnl),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("signal_ts").sort_index()
