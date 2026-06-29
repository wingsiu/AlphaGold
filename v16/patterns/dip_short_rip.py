"""
dip_short_rip — v16 pattern lane.

Copied from the v15 integration spec (not added to production v15):
  prev 15m up, slot up, price ≥5 above slot open, minute < 10 → short
  ML filter p ≥ 0.55; execution from v16_config.DIP_SHORT_RIP
"""
from __future__ import annotations

import pandas as pd

from v16.backtest.features import dip_ml_feature_columns, session_mask
from v16.backtest.fixed_tpsl_sim import simulate_fixed_tpsl
from v16.config import v16_config

PATTERN = {
    "name": "dip_short_rip",
    "direction": "short",
    "priority": 0,
    "router": [
        ("prev_15m_dir", ">=", 1.0),
        ("slot_up", ">=", 1.0),
        ("slot_rip_pts", ">=", 5.0),
        ("minute_in_15m", "<", 10.0),
    ],
    "ml_prob": 0.70,
    "execution": {"tp": 35.0, "sl": 35.0, "horizon": 45},
}


def _cfg() -> dict:
    return v16_config.DIP_SHORT_RIP


def resolve_execution(cfg: dict | None = None, *, mechanical: bool = False) -> dict:
    c = cfg or _cfg()
    if mechanical:
        return c.get("execution_mechanical", c.get("execution", PATTERN["execution"]))
    return c.get("execution", PATTERN["execution"])


def _apply_rule(feats: pd.DataFrame, feat: str, op: str, val: float) -> pd.Series:
    col = feats[feat]
    if op == ">=":
        return col >= val
    if op == ">":
        return col > val
    if op == "<=":
        return col <= val
    if op == "<":
        return col < val
    if op == "==":
        return col == val
    raise ValueError(f"Unknown op: {op}")


def router_mask(
    feats: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    cfg: dict | None = None,
) -> pd.Series:
    """Mechanical router — matches v15 dip_short_rip pattern rules."""
    c = cfg or _cfg()
    sessions = tuple(c.get("sessions", ("london", "ny")))
    m = session_mask(index, sessions)
    for feat, op, val in c.get("router", PATTERN["router"]):
        m = m & _apply_rule(feats, feat, op, val)
    if c.get("dip_require_two_prev_up", False):
        m = m & (feats["two_prev_15m_up"] > 0)
    min_b = float(c.get("dip_min_prev_body_pts", 0))
    if min_b > 0:
        m = m & (feats["prev_15m_body"] >= min_b)
    return m


def build_labeled_set(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    *,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Label pool rows for walk-forward ML (short_win column)."""
    c = cfg or _cfg()
    source = c.get("ml_label_source", "execution")

    if source == "scaleout":
        from v16.backtest.signals import build_labeled_set as scaleout_labels
        from v16.research.profit_hunt import signal_cfg

        pool_cfg = {
            "mode": "dip_short_15m",
            "dip_require_two_prev_up": c.get("dip_require_two_prev_up", False),
            "dip_min_prev_body_pts": c.get("dip_min_prev_body_pts", 0.0),
            "dip_max_minute_in_slot": 10,
            "dip_short_min_above_open_pts": 5.0,
            "dip_require_slot_high": False,
        }
        with signal_cfg(pool_cfg):
            labeled = scaleout_labels(df, feats)
        rip = router_mask(feats, df.index, cfg=c)
        return labeled[rip.reindex(labeled.index, fill_value=False)]

    ex = c.get("execution", PATTERN["execution"])
    tp = float(ex["tp"])
    sl = float(ex["sl"])
    horizon = int(ex["horizon"])

    mask = router_mask(feats, df.index, cfg=c)
    rows: list[dict] = []

    for i in range(len(df) - horizon - 2):
        ts = df.index[i]
        if not mask.iloc[i]:
            continue
        entry_idx = i + 1
        ep = float(df.iloc[entry_idx]["open_bid"])
        res = simulate_fixed_tpsl(df, entry_idx, -1, ep, tp=tp, sl=sl, horizon=horizon)
        rows.append(
            {
                "signal_ts": ts,
                "entry_idx": entry_idx,
                "short_pnl": res.pnl,
                "short_win": int(res.pnl > 0),
                "exit_reason": res.exit_reason,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("signal_ts").sort_index()


def feature_columns(feats: pd.DataFrame) -> list[str]:
    cols = dip_ml_feature_columns(feats)
    if "slot_rip_pts" in feats.columns and "slot_rip_pts" not in cols:
        cols.append("slot_rip_pts")
    return cols
