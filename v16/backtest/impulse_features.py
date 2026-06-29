"""Impulse ML feature column definitions (v16 + v15 dip + structure + explicit impulse)."""
from __future__ import annotations

import pandas as pd

from v16.backtest.features import dip_ml_feature_columns

# Core v16 point features at signal time
V16_BASE_COLUMNS = (
    "side",
    "prev_15m_body",
    "prev_15m_range",
    "prev_15m_dir",
    "prev_15m_body_abs",
    "prev_15m_open",
    "prev_15m_high",
    "prev_15m_low",
    "prev2_15m_body",
    "prev2_15m_range",
    "prev2_15m_dir",
    "minute_in_15m",
    "range_1",
    "body",
    "body_abs",
    "close_loc",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "range_3",
    "range_5",
    "range_10",
    "range_20",
    "vol_ratio",
    "range_expansion",
    "atr_14",
    "dist_ema_20",
    "dist_ema_50",
    "dist_ema_100",
    "ema_20_slope",
    "ema_50_slope",
    "trend_strength",
    "above_ema_20",
    "above_ema_50",
    "lon_hour",
    "ny_hour",
    "dow",
)

# v15 dip_short_rip ML extras not in V16_BASE (slot context)
V15_DIP_EXTRA = (
    "two_prev_15m_down",
    "two_prev_15m_up",
    "dip_from_slot_open",
    "slot_rip_pts",
    "slot_low_dip",
    "slot_down",
    "slot_up",
    "prev2_15m_body_abs",
)

# 15m zigzag structure (v16/structure/swing_zigzag.py)
STRUCTURE_COLUMNS = (
    "struct_trend",
    "struct_last_kind",
    "struct_dist_pts",
    "struct_leg_pts",
    "struct_prior_leg_pts",
    "struct_pullback_pct",
    "struct_leg_age_15m",
    "struct_hh",
    "struct_hl",
    "struct_lh",
    "struct_ll",
    "struct_aligned",
)

# Explicit impulse bar metadata (from build_signal_table / labeled set)
IMPULSE_EXPLICIT = (
    "impulse_body",
    "impulse_body_abs",
    "impulse_range",
    "impulse_volume",
    "impulse_minute",
    "bars_after_impulse",
    "impulse_body_atr",
    "impulse_range_atr",
    "impulse_aligned",
    "impulse_bar_range",
    "sl_pts",
    "tp_pts",
    "sl_to_impulse_range",
)


def impulse_ml_feature_columns(
    feats: pd.DataFrame,
    labeled: pd.DataFrame | None = None,
    *,
    include_structure: bool = True,
) -> list[str]:
    """Union of v16 base, v15 dip ML, structure, and explicit impulse columns."""
    labeled_cols = set(labeled.columns) if labeled is not None else set()
    dip = dip_ml_feature_columns(feats)
    struct_cols = list(STRUCTURE_COLUMNS) if include_structure else []
    merged = list(
        dict.fromkeys(
            list(V16_BASE_COLUMNS)
            + [c for c in dip if c not in V16_BASE_COLUMNS]
            + list(V15_DIP_EXTRA)
            + struct_cols
            + list(IMPULSE_EXPLICIT)
        )
    )
    has_structure = "struct_trend" in feats.columns
    return [
        c
        for c in merged
        if c == "side"
        or c in feats.columns
        or c in labeled_cols
        or (include_structure and has_structure and c in STRUCTURE_COLUMNS)
    ]


def structure_kwargs(cfg: dict | None = None) -> dict:
    """Resolve structure params from pattern config."""
    from v16.config import v16_config

    sc = dict((cfg or v16_config.MOMENTUM_15M_HOLD).get("structure", {}))
    if not sc.get("enabled", True):
        return {}
    return {
        "rule": sc.get("rule", "15min"),
        "atr_mult": float(sc.get("atr_mult", 3.0)),
        "atr_period": int(sc.get("atr_period", 14)),
    }


def attach_structure_features(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    *,
    rule: str = "15min",
    atr_mult: float = 3.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Merge 15m zigzag structure columns onto per-1m feature frame (idempotent)."""
    if "struct_trend" in feats.columns:
        return feats
    from v16.structure.swing_zigzag import build_structure_context

    struct = build_structure_context(
        df,
        rule=rule,
        atr_mult=atr_mult,
        atr_period=atr_period,
    )
    if struct.empty:
        return feats
    out = feats.copy()
    for c in struct.columns:
        out[c] = struct[c]
    return out.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)


def enrich_structure_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Side × trend alignment flags for impulse ML."""
    out = frame.copy()
    if "struct_trend" in out.columns and "side" in out.columns:
        out["struct_aligned"] = out["side"].astype(float) * out["struct_trend"].astype(float)
    return out.fillna(0.0)


def enrich_impulse_derived(
    frame: pd.DataFrame,
    feats: pd.DataFrame,
    ts: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Add ATR-normalized impulse fields computed at signal time."""
    out = enrich_structure_derived(frame)
    atr = feats.loc[ts, "atr_14"].replace(0, pd.NA).astype(float)
    if "impulse_body" in out.columns:
        out["impulse_body_atr"] = out["impulse_body"] / atr.values
        if "side" in out.columns:
            out["impulse_aligned"] = out["impulse_body"] * out["side"]
    if "impulse_range" in out.columns:
        out["impulse_range_atr"] = out["impulse_range"] / atr.values
    if "sl_pts" in out.columns and "impulse_bar_range" in out.columns:
        out["sl_to_impulse_range"] = out["sl_pts"] / out["impulse_bar_range"].replace(0, pd.NA)
    return out.fillna(0.0)
