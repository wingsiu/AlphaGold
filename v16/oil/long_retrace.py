"""Oil long retrace patterns — 15m symmetric + gold-style dip long 15m."""
from __future__ import annotations

import numpy as np
import pandas as pd

from v16.backtest.features import session_mask
from v16.config.oil_config import LONG_RETRACE_15M, OIL_DIP_LONG_15M


def enrich_d15_wicks(d15: pd.DataFrame) -> pd.DataFrame:
    """Upper wick column for retrace short / long retrace."""
    d = d15.copy()
    d["uw"] = d["high"] - np.maximum(d["open"], d["close_ask"])
    return d


def enrich_d15_long_retrace(d15: pd.DataFrame) -> pd.DataFrame:
    """Add day-high distance features (mirror of cad/Dlow on ret leg)."""
    d = enrich_d15_wicks(d15)
    ny = d.index.tz_convert("America/New_York")
    d["Dhigh"] = d.groupby(ny.date)["high"].cummax()
    d["cah"] = d["Dhigh"] - d["close_ask"]
    d["h_dhigh"] = d["Dhigh"] - d["high"]
    d["l_dhigh"] = d["Dhigh"] - d["low"]
    return d


def long_retrace_15m_exits(d15: pd.DataFrame, struct_frame: pd.DataFrame | None = None, cfg: dict | None = None) -> list[dict]:
    """Long: extended below day high + green bounce bar."""
    from v16.oil.structure_feats import apply_long_structure_gate

    c = cfg or LONG_RETRACE_15M
    d = enrich_d15_long_retrace(d15)
    mask = (
        (d["cah"] > c["dhigh"])
        & (d["avg_r3"] > c["rng"])
        & (d["bc"] > c["chg"])
        & (d["uw"] < c["wick"])
        & d["ins"]
    )
    if struct_frame is not None:
        mask = apply_long_structure_gate(d15, struct_frame, mask)
    return [{"idx": i} for i in range(len(d)) if mask.iloc[i]]


def oil_dip_long_15m_mask(feats: pd.DataFrame, index: pd.DatetimeIndex, cfg: dict | None = None) -> pd.Series:
    """Gold v16 dip_long_15m — oil-scaled thresholds."""
    c = cfg or OIL_DIP_LONG_15M
    in_sess = session_mask(index, tuple(c.get("sessions", ("ny",))))
    slot_dn = feats["slot_down"] > 0
    below = feats["dip_from_slot_open"] >= float(c["dip_min_below_open_pts"])
    early = feats["minute_in_15m"].astype(int) < int(c["dip_max_minute_in_slot"])
    m = in_sess & slot_dn & below & early
    if c.get("dip_require_slot_low", True):
        m = m & (feats["slot_low_dip"] >= float(c["dip_min_slot_low_pts"]))
    if c.get("dip_require_prev_down", True):
        m = m & (feats["prev_15m_dir"] < 0)
    if c.get("dip_require_two_prev_down", False):
        m = m & (feats["two_prev_15m_down"] > 0)
    min_b = float(c.get("dip_min_prev_body_pts", 0))
    if min_b > 0:
        m = m & (feats["prev_15m_body"] <= -min_b)
    min_r = float(c.get("dip_min_prev_range_pts", 0))
    if min_r > 0:
        m = m & (feats["prev_15m_range"] >= min_r)
    return m
