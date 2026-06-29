"""Oil v16 pattern signal collectors."""
from __future__ import annotations

import pandas as pd

from oil.signal_engine import (
    RET_FEATS,
    SI_FEATS,
    WR_FEATS,
    compute_si_features,
    sim_si_fixed,
)
from v16.config.oil_config import OIL_RIP_SHORT, RETRACE, RET_SHORT, SHORT_IMPULSE, WR90
from v16.config.oil_config import RET_SHORT_FEATS
from v16.oil.long_retrace import enrich_d15_wicks
from v16.oil.structure_feats import apply_long_structure_gate, apply_short_structure_gate


def wr90_cluster_exits(d15: pd.DataFrame) -> list[dict]:
    cfg = WR90
    in_s = d15["ins"]
    o = (d15["wr"] < cfg["entry"]) & in_s
    sigs, ie, cv, bc = [], False, 0.0, 0
    for i in range(len(d15)):
        if o.iloc[i]:
            if not ie:
                cv, bc = 0.0, 0
            ie = True
            cv += d15["volume"].iloc[i]
            bc += 1
        elif ie:
            if i < len(d15) - 1 and in_s.iloc[i] and cv >= cfg["cv"] and bc >= cfg["ep_min"]:
                sigs.append({"idx": i})
            ie, cv, bc = False, 0.0, 0
    return sigs


def retrace_exits(d15: pd.DataFrame, struct_frame: pd.DataFrame | None = None) -> list[dict]:
    """Long: red bar pullback when extended above day low."""
    cfg = RETRACE
    mask = (
        (d15["cad"] > cfg["dlow"])
        & (d15["avg_r3"] > cfg["rng"])
        & (d15["bc"] < cfg["chg"])
        & (d15["wb"] < cfg["wick"])
        & d15["ins"]
    )
    if struct_frame is not None:
        mask = apply_long_structure_gate(d15, struct_frame, mask)
    return [{"idx": i} for i in range(len(d15)) if mask.iloc[i]]


def retrace_short_exits(d15: pd.DataFrame, struct_frame: pd.DataFrame | None = None) -> list[dict]:
    """Short: fade green extension bar when elevated above day low."""
    cfg = RET_SHORT
    d = enrich_d15_wicks(d15)
    mask = (
        (d["cad"] > cfg["dlow"])
        & (d["avg_r3"] > cfg["rng"])
        & (d["bc"] > cfg["chg"])
        & (d["uw"] < cfg["wick"])
        & d["ins"]
    )
    if struct_frame is not None:
        mask = apply_short_structure_gate(d15, struct_frame, mask)
    return [{"idx": i} for i in range(len(d)) if mask.iloc[i]]


def short_impulse_records(d1m: pd.DataFrame, struct_frame_1m: pd.DataFrame | None = None) -> list[dict]:
    cfg = SHORT_IMPULSE
    d1m_s = compute_si_features(d1m)
    si_mask = (
        (d1m_s["prev_change"] < cfg["change_max"])
        & (d1m_s["prev2_change"] < 10.0)
        & (d1m_s["prev2_change"] > -14.0)
        & (d1m_s["prev_lower_wick"] < 35.0)
        & (d1m_s["prev_volume"] > cfg["vol_min"])
        & d1m_s["ny_hour"]
        & (d1m_s["up_count3_15min"] != -3)
        & (d1m_s["dist_day_high"] < 180.0)
    )
    if struct_frame_1m is not None and "struct_trend" in struct_frame_1m.columns:
        st = struct_frame_1m["struct_trend"].reindex(d1m_s.index).fillna(0)
        si_mask = si_mask & (st <= 0)

    si_sigs = sorted(d1m_s.index[si_mask].tolist())
    recs, in_si, si_ex = [], False, -1
    for sig in si_sigs:
        ei = d1m_s.index.get_loc(sig)
        if ei + cfg["max_bars"] >= len(d1m_s):
            continue
        if in_si and ei <= si_ex:
            continue
        ep = d1m_s.iloc[ei]["close_bid"]
        ex_price, bars, reason = sim_si_fixed(ei, ep, d1m_s)
        recs.append(
            {
                "entry_idx": sig,
                "exit_ts": d1m_s.index[ei + bars],
                "pnl": ep - ex_price,
                "reason": reason,
                "entry_price": ep,
                "exit_price": ex_price,
                "row": d1m_s.iloc[ei],
            }
        )
        in_si, si_ex = True, ei + bars
    return recs


def rip_short_router(feats: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    from v16.backtest.features import session_mask

    cfg = OIL_RIP_SHORT
    m = session_mask(index, tuple(cfg["sessions"]))
    for feat, op, val in cfg["router"]:
        col = feats[feat]
        if op == ">=":
            m = m & (col >= val)
        elif op == "<":
            m = m & (col < val)
    return m


def rip_short_labeled(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    *,
    tp: float,
    sl: float,
    horizon: int,
) -> pd.DataFrame:
    from v16.backtest.fixed_tpsl_sim import simulate_fixed_tpsl

    router = rip_short_router(feats, df.index)
    rows = []
    for ts in df.index[router]:
        ei = df.index.get_loc(ts)
        if ei + 1 >= len(df):
            continue
        entry_idx = ei + 1
        ep = float(df.iloc[entry_idx]["close_bid"])
        res = simulate_fixed_tpsl(df, entry_idx, -1, ep, tp=tp, sl=sl, horizon=horizon)
        rows.append(
            {
                "signal_ts": ts,
                "entry_idx": entry_idx,
                "side": -1,
                "short_pnl": res.pnl,
                "short_win": int(res.pnl > 0),
            }
        )
    return pd.DataFrame(rows).set_index("signal_ts") if rows else pd.DataFrame()


FEAT_MAP = {
    "wr90": WR_FEATS,
    "ret": RET_FEATS,
    "ret_short": RET_SHORT_FEATS,
    "si": SI_FEATS,
}
