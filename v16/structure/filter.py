"""Filter impulse signals by 15m zigzag structure (no lookahead)."""
from __future__ import annotations

import pandas as pd

from v16.structure.swing_zigzag import build_structure_context


def structure_gate_config(cfg: dict | None) -> dict:
    from v16.config import v16_config

    c = cfg or v16_config.MOMENTUM_15M_HOLD
    sc = dict(c.get("structure", {}))
    gate = dict(sc.get("gate", {}))
    if not sc.get("enabled", True):
        return {"enabled": False}
    return {
        "enabled": bool(gate.get("enabled", False)),
        "require_with_trend": bool(gate.get("require_with_trend", True)),
        "max_leg_age_15m": gate.get("max_leg_age_15m"),
        "rule": sc.get("rule", "15min"),
        "atr_mult": float(sc.get("atr_mult", 3.0)),
        "atr_period": int(sc.get("atr_period", 14)),
    }


def apply_structure_gate(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    *,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Drop signals that fail structure gate at signal_ts.

    require_with_trend: keep only side × struct_trend > 0 (HH/HL or LH/LL).
    max_leg_age_15m: keep only fresh legs (optional int).
    """
    if signal_table.empty:
        return signal_table

    gc = structure_gate_config(cfg)
    if not gc["enabled"]:
        return signal_table

    struct = build_structure_context(
        df,
        rule=gc["rule"],
        atr_mult=gc["atr_mult"],
        atr_period=gc["atr_period"],
    )
    if struct.empty:
        return signal_table.iloc[0:0]

    ctx = struct.reindex(signal_table.index)
    keep = pd.Series(True, index=signal_table.index)
    sides = signal_table["side"].astype(int)

    if gc["require_with_trend"]:
        aligned = sides * ctx["struct_trend"].fillna(0).astype(int)
        keep &= aligned > 0

    max_age = gc.get("max_leg_age_15m")
    if max_age is not None:
        keep &= ctx["struct_leg_age_15m"].fillna(999).astype(int) <= int(max_age)

    return signal_table.loc[keep].copy()
