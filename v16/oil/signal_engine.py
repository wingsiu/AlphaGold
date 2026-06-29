"""v16 oil signal engine — shared by replay bot and parity check.

Collect ML signals, merge by entry time (matches combined_run backtest).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from oil.signal_engine import (
    OilSignalState,
    build_15m,
    compute_si_features,
    d15_through_completed,
    init_wr90_cluster_state,
    latest_completed_15m,
    should_evaluate_15m_bar,
    mark_15m_evaluated,
    sim_si_fixed,
)
from v16.oil.merge import LEG_ENTRY_PRIORITY
from v16.config.oil_config import (
    LONG_RETRACE_15M,
    LONG_RETRACE_15M_FEATS,
    OIL_LEG_MODELS,
    RETRACE,
    RET_SHORT,
    RET_SHORT_FEATS,
    SHORT_IMPULSE,
    STRUCTURE_GATE,
    WR90,
)
from v16.oil.patterns import (
    FEAT_MAP,
    retrace_exits,
    retrace_short_exits,
    short_impulse_records,
    wr90_cluster_exits,
)
from v16.oil.long_retrace import enrich_d15_long_retrace, enrich_d15_wicks, long_retrace_15m_exits
from v16.oil.sim_15m import sim_15m_long, sim_15m_short
from v16.oil.struct_hold import sim_wr90_struct_hold
from v16.oil.structure_feats import apply_long_structure_gate, apply_short_structure_gate, structure_on_d15
from v16.oil.wf_ml import score_oil_leg

# Merge tie-break (must match combined_run.py)
LEG_ENTRY_PRIORITY.setdefault("ret_short", 0)
LEG_ENTRY_PRIORITY.setdefault("oil_retrace_short", 0)
LEG_ENTRY_PRIORITY.setdefault("long_retrace", 2)
LEG_ENTRY_PRIORITY.setdefault("oil_long_retrace", 2)

LEG_TO_TYPE = {
    "wr90": "wr90",
    "ret": "ret",
    "ret_short": "ret_short",
    "long_ret": "long_retrace",
    "si": "short_impulse",
}

ENTRY_PRIORITY = ("si", "ret_short", "wr90", "ret", "long_ret")


@dataclass
class V16Decision:
    leg: str
    entry_ts: pd.Timestamp
    prob: Optional[float]
    would_enter: bool
    reason: str
    side: int = 1
    detail: str = ""


def _struct_row(struct_d15: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    if struct_d15 is None or struct_d15.empty:
        return None
    if ts in struct_d15.index:
        return struct_d15.loc[ts]
    return struct_d15.reindex([ts]).iloc[0]


def _ml_cfg(leg: str) -> dict:
    return OIL_LEG_MODELS[leg]


def evaluate_wr90(
    d15: pd.DataFrame,
    st: OilSignalState,
    flat: bool,
    struct_d15: pd.DataFrame,
) -> V16Decision:
    from oil.signal_engine import detect_wr90_cluster

    latest = d15.index[-1]
    if not flat:
        return V16Decision("wr90", latest, None, False, "in_trade", 1)
    wr = detect_wr90_cluster(d15, st)
    if not wr:
        return V16Decision("wr90", latest, None, False, "no_cluster", 1)
    idx, cv, ep = wr
    entry_ts = d15.index[idx]
    cfg = _ml_cfg("wr90")
    sr = _struct_row(struct_d15, entry_ts)
    prob = score_oil_leg("wr90", entry_ts, d15.iloc[idx], FEAT_MAP["wr90"], cfg["model"], sr)
    if prob is None:
        return V16Decision("wr90", entry_ts, None, False, "no_model", 1)
    ok = prob >= cfg["ml_th"]
    return V16Decision(
        "wr90", entry_ts, prob, ok, "pass" if ok else "below_threshold", 1, f"CV={cv:.0f} Ep={ep}"
    )


def evaluate_ret(d15: pd.DataFrame, flat: bool, struct_d15: pd.DataFrame) -> V16Decision:
    latest = d15.index[-1]
    if not flat:
        return V16Decision("ret", latest, None, False, "in_trade", 1)
    cfg_r = RETRACE
    mask = (
        (d15["cad"] > cfg_r["dlow"])
        & (d15["avg_r3"] > cfg_r["rng"])
        & (d15["bc"] < cfg_r["chg"])
        & (d15["wb"] < cfg_r["wick"])
        & d15["ins"]
    )
    if STRUCTURE_GATE.get("enabled"):
        mask = apply_long_structure_gate(d15, struct_d15, mask)
    if not bool(mask.iloc[-1]):
        return V16Decision("ret", latest, None, False, "no_signal", 1)
    cfg = _ml_cfg("ret")
    sr = _struct_row(struct_d15, latest)
    prob = score_oil_leg("ret", latest, d15.iloc[-1], FEAT_MAP["ret"], cfg["model"], sr)
    if prob is None:
        return V16Decision("ret", latest, None, False, "no_model", 1)
    ok = prob >= cfg["ml_th"]
    return V16Decision("ret", latest, prob, ok, "pass" if ok else "below_threshold", 1)


def evaluate_ret_short(
    d15: pd.DataFrame,
    d15_w: pd.DataFrame,
    flat: bool,
    struct_d15: pd.DataFrame,
) -> V16Decision:
    latest = d15.index[-1]
    if not flat:
        return V16Decision("ret_short", latest, None, False, "in_trade", -1)
    cfg_r = RET_SHORT
    mask = (
        (d15_w["cad"] > cfg_r["dlow"])
        & (d15_w["avg_r3"] > cfg_r["rng"])
        & (d15_w["bc"] > cfg_r["chg"])
        & (d15_w["uw"] < cfg_r["wick"])
        & d15_w["ins"]
    )
    if STRUCTURE_GATE.get("enabled"):
        mask = apply_short_structure_gate(d15, struct_d15, mask)
    if not bool(mask.iloc[-1]):
        return V16Decision("ret_short", latest, None, False, "no_signal", -1)
    cfg = _ml_cfg("ret_short")
    sr = _struct_row(struct_d15, latest)
    prob = score_oil_leg("ret_short", latest, d15_w.iloc[-1], RET_SHORT_FEATS, cfg["model"], sr)
    if prob is None:
        return V16Decision("ret_short", latest, None, False, "no_model", -1)
    ok = prob >= cfg["ml_th"]
    return V16Decision("ret_short", latest, prob, ok, "pass" if ok else "below_threshold", -1)


def evaluate_long_ret(
    d15_lr: pd.DataFrame,
    flat: bool,
    struct_d15: pd.DataFrame,
) -> V16Decision:
    latest = d15_lr.index[-1]
    if not flat:
        return V16Decision("long_ret", latest, None, False, "in_trade", 1)
    cfg_l = LONG_RETRACE_15M
    mask = (
        (d15_lr["cah"] > cfg_l["dhigh"])
        & (d15_lr["avg_r3"] > cfg_l["rng"])
        & (d15_lr["bc"] > cfg_l["chg"])
        & (d15_lr["uw"] < cfg_l["wick"])
        & d15_lr["ins"]
    )
    if STRUCTURE_GATE.get("enabled"):
        mask = apply_long_structure_gate(d15_lr, struct_d15, mask)
    if not bool(mask.iloc[-1]):
        return V16Decision("long_ret", latest, None, False, "no_signal", 1)
    cfg = _ml_cfg("long_ret")
    sr = _struct_row(struct_d15, latest)
    prob = score_oil_leg("long_ret", latest, d15_lr.iloc[-1], LONG_RETRACE_15M_FEATS, cfg["model"], sr)
    if prob is None:
        return V16Decision("long_ret", latest, None, False, "no_model", 1)
    ok = prob >= cfg["ml_th"]
    return V16Decision("long_ret", latest, prob, ok, "pass" if ok else "below_threshold", 1)


def evaluate_si_at(
    ts: pd.Timestamp,
    flat: bool,
    si_entries: set[pd.Timestamp],
    d1m_feats: pd.DataFrame,
) -> V16Decision:
    """SI entry at 1m bar — uses same records as backtest short_impulse_records."""
    if not flat:
        return V16Decision("si", ts, None, False, "in_trade", -1)
    if ts not in si_entries:
        return V16Decision("si", ts, None, False, "not_on_bar", -1)
    cfg = _ml_cfg("si")
    if ts not in d1m_feats.index:
        return V16Decision("si", ts, None, False, "no_bar", -1)
    prob = score_oil_leg("si", ts, d1m_feats.loc[ts], FEAT_MAP["si"], cfg["model"])
    if prob is None:
        return V16Decision("si", ts, None, False, "no_model", -1)
    ok = prob >= cfg["ml_th"]
    return V16Decision("si", ts, prob, ok, "pass" if ok else "below_threshold", -1)


def simulate_trade(
    dec: V16Decision,
    d1m: pd.DataFrame,
    d15: pd.DataFrame,
    d15_w: pd.DataFrame,
    d15_lr: pd.DataFrame,
    *,
    wr90_exit: str = "struct_hold",
) -> dict:
    """Run leg exit sim; return trade dict matching combined_run format."""
    entry = pd.Timestamp(dec.entry_ts)
    typ = LEG_TO_TYPE[dec.leg]
    cfg_ml = _ml_cfg(dec.leg)

    if dec.leg == "si":
        feats = compute_si_features(d1m)
        ei = feats.index.get_loc(entry)
        ep = float(feats.iloc[ei]["close_bid"])
        ex_p, bars, reason = sim_si_fixed(ei, ep, feats)
        exit_ts = feats.index[ei + bars]
        pnl = ep - ex_p
        return {
            "entry": entry,
            "exit": exit_ts,
            "pnl": pnl,
            "reason": reason,
            "type": typ,
            "side": -1,
            "_leg": dec.leg,
            "_prob": dec.prob,
        }

    if dec.leg == "wr90" and wr90_exit == "struct_hold":
        idx = d15.index.get_loc(entry)
        tr = sim_wr90_struct_hold(d15, d1m, [idx])
        if not tr:
            raise RuntimeError("wr90 struct_hold sim failed")
        return {**tr[0], "_leg": "wr90", "_prob": dec.prob}

    idx = d15.index.get_loc(entry)
    sig = [{"idx": idx}]
    if dec.leg == "ret":
        _, tr, _ = sim_15m_long(d15, sig, RETRACE["tp"], RETRACE["sl"], "ret")
    elif dec.leg == "ret_short":
        _, tr, _ = sim_15m_short(d15_w, sig, RET_SHORT["tp"], RET_SHORT["sl"], "ret_short")
    elif dec.leg == "long_ret":
        _, tr, _ = sim_15m_long(d15_lr, sig, LONG_RETRACE_15M["tp"], LONG_RETRACE_15M["sl"], "long_retrace")
    elif dec.leg == "wr90":
        _, tr, _ = sim_15m_long(d15, sig, WR90["tp"], WR90["sl"], "wr90")
    else:
        raise ValueError(dec.leg)
    t = tr[0]
    return {
        "entry": entry,
        "exit": t["exit"],
        "pnl": t["pnl"],
        "reason": t["reason"],
        "type": typ,
        "side": dec.side,
        "_leg": dec.leg,
        "_prob": dec.prob,
    }


def _merge_sort_key(dec: V16Decision) -> tuple:
    entry = pd.Timestamp(dec.entry_ts).tz_convert("UTC")
    typ = LEG_TO_TYPE.get(dec.leg, dec.leg)
    return (entry, LEG_ENTRY_PRIORITY.get(typ, 9))


def _collect_candidates(
    d15_full: pd.DataFrame,
    d15_w: pd.DataFrame,
    d15_lr: pd.DataFrame,
    d1m_feats: pd.DataFrame,
    struct_d15: pd.DataFrame,
    struct_1m: pd.DataFrame | None,
    si_entries: set[pd.Timestamp],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[V16Decision]:
    """All ML-passing signals in window — same entries as combined_run pre-merge."""
    sf = struct_d15 if STRUCTURE_GATE.get("enabled") and len(struct_d15) else None
    out: list[V16Decision] = []

    for sig in wr90_cluster_exits(d15_full):
        ts = d15_full.index[sig["idx"]]
        if not (start <= ts <= end):
            continue
        cfg = _ml_cfg("wr90")
        prob = score_oil_leg("wr90", ts, d15_full.iloc[sig["idx"]], FEAT_MAP["wr90"], cfg["model"], _struct_row(struct_d15, ts))
        if prob is not None and prob >= cfg["ml_th"]:
            out.append(V16Decision("wr90", ts, prob, True, "pass", 1))

    for sig in retrace_exits(d15_full, sf):
        ts = d15_full.index[sig["idx"]]
        if not (start <= ts <= end):
            continue
        cfg = _ml_cfg("ret")
        prob = score_oil_leg("ret", ts, d15_full.iloc[sig["idx"]], FEAT_MAP["ret"], cfg["model"], _struct_row(struct_d15, ts))
        if prob is not None and prob >= cfg["ml_th"]:
            out.append(V16Decision("ret", ts, prob, True, "pass", 1))

    for sig in retrace_short_exits(d15_full, sf):
        ts = d15_full.index[sig["idx"]]
        if not (start <= ts <= end):
            continue
        cfg = _ml_cfg("ret_short")
        prob = score_oil_leg("ret_short", ts, d15_w.iloc[sig["idx"]], RET_SHORT_FEATS, cfg["model"], _struct_row(struct_d15, ts))
        if prob is not None and prob >= cfg["ml_th"]:
            out.append(V16Decision("ret_short", ts, prob, True, "pass", -1))

    for sig in long_retrace_15m_exits(d15_full, sf):
        ts = d15_full.index[sig["idx"]]
        if not (start <= ts <= end):
            continue
        cfg = _ml_cfg("long_ret")
        prob = score_oil_leg("long_ret", ts, d15_lr.iloc[sig["idx"]], LONG_RETRACE_15M_FEATS, cfg["model"], _struct_row(struct_d15, ts))
        if prob is not None and prob >= cfg["ml_th"]:
            out.append(V16Decision("long_ret", ts, prob, True, "pass", 1))

    for ts in sorted(si_entries):
        if not (start <= ts <= end):
            continue
        cfg = _ml_cfg("si")
        if ts not in d1m_feats.index:
            continue
        prob = score_oil_leg("si", ts, d1m_feats.loc[ts], FEAT_MAP["si"], cfg["model"])
        if prob is not None and prob >= cfg["ml_th"]:
            out.append(V16Decision("si", ts, prob, True, "pass", -1))

    return out


def _merge_candidates_by_entry(
    candidates: list[V16Decision],
    d1m: pd.DataFrame,
    d15: pd.DataFrame,
    d15_w: pd.DataFrame,
    d15_lr: pd.DataFrame,
    *,
    wr90_exit: str = "struct_hold",
) -> list[dict]:
    """Single-slot merge by entry time — matches merge_single_position + backtest."""
    taken: list[dict] = []
    busy_until: Optional[pd.Timestamp] = None
    for dec in sorted(candidates, key=_merge_sort_key):
        entry = pd.Timestamp(dec.entry_ts).tz_convert("UTC")
        if busy_until is not None and entry < busy_until:
            continue
        trade = simulate_trade(dec, d1m, d15, d15_w, d15_lr, wr90_exit=wr90_exit)
        taken.append(trade)
        busy_until = pd.Timestamp(trade["exit"]).tz_convert("UTC")
    return taken


def replay_portfolio(
    d1m: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    warmup_days: int = 90,
    wr90_exit: str = "struct_hold",
) -> list[dict]:
    """Replay portfolio — collect ML signals, merge by entry time (matches backtest)."""
    warmup_start = start - pd.Timedelta(days=warmup_days)
    window = d1m[(d1m.index >= warmup_start) & (d1m.index <= end)].copy()
    if window.empty:
        return []

    d15_full = build_15m(window)
    struct_d15 = structure_on_d15(window, d15_full) if STRUCTURE_GATE.get("enabled") else pd.DataFrame()
    d15_w = enrich_d15_wicks(d15_full)
    d15_lr = enrich_d15_long_retrace(d15_full)
    d1m_feats = compute_si_features(window)

    from v16.structure.swing_zigzag import build_structure_context

    struct_1m = (
        build_structure_context(window, rule="15min", atr_mult=3.0, atr_period=14)
        if STRUCTURE_GATE.get("enabled")
        else None
    )
    si_entries = {pd.Timestamp(r["entry_idx"]).tz_convert("UTC") for r in short_impulse_records(window, struct_1m)}

    candidates = _collect_candidates(
        d15_full, d15_w, d15_lr, d1m_feats, struct_d15, struct_1m, si_entries, start, end
    )
    return _merge_candidates_by_entry(
        candidates, window, d15_full, d15_w, d15_lr, wr90_exit=wr90_exit
    )


def replay_entries(
    d1m: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    **kwargs,
) -> list[dict]:
    """Alias — returns merged portfolio trades."""
    return replay_portfolio(d1m, start, end, **kwargs)


def _pick_live_winner(candidates: list[V16Decision]) -> V16Decision | None:
    """Single-slot winner at same entry minute — matches backtest merge priority."""
    passing = [c for c in candidates if c.would_enter]
    if not passing:
        return None
    passing.sort(key=_merge_sort_key)
    return passing[0]


def evaluate_minute_v16(
    cached: pd.DataFrame,
    st: OilSignalState,
    *,
    submit: bool = False,
    flat: bool | None = None,
) -> dict[str, Any]:
    """Live minute evaluation — v16 five-leg portfolio with entry-time merge."""
    from oil.signal_engine import (
        apply_entry,
        latest_completed_15m,
        mark_15m_evaluated,
        should_evaluate_15m_bar,
    )
    from v16.structure.swing_zigzag import build_structure_context

    from v16.oil.structure_feats import _normalize_1m_for_structure

    cached = _normalize_1m_for_structure(cached)
    d15, completed_15m = d15_through_completed(cached)
    d15_w = enrich_d15_wicks(d15) if len(d15) else pd.DataFrame()
    d15_lr = enrich_d15_long_retrace(d15) if len(d15) else pd.DataFrame()
    struct_d15 = structure_on_d15(cached, d15) if STRUCTURE_GATE.get("enabled") and len(d15) else pd.DataFrame()
    struct_1m = (
        build_structure_context(cached, rule="15min", atr_mult=3.0, atr_period=14)
        if STRUCTURE_GATE.get("enabled")
        else None
    )
    d1m_feats = compute_si_features(cached)
    si_entries = {pd.Timestamp(r["entry_idx"]).tz_convert("UTC") for r in short_impulse_records(cached, struct_1m)}
    latest_bar = cached.index[-1]
    is_flat = st.open_deal_id is None if flat is None else flat

    out: dict[str, Any] = {
        "latest_1m": str(latest_bar),
        "latest_15m": str(completed_15m) if completed_15m is not None else None,
        "min15": str(latest_completed_15m(latest_bar)),
        "flat": is_flat,
        "wr90_cluster": {"in_cluster": st.wr90_in_cluster, "cv": st.wr90_cv, "ep": st.wr90_bc},
    }

    si = evaluate_si_at(latest_bar, is_flat, si_entries, d1m_feats)
    out["si"] = si

    wr = ret = ret_short = long_ret = None
    candidates: list[V16Decision] = []
    if si.would_enter:
        candidates.append(si)

    if completed_15m is not None and len(d15) > 0 and should_evaluate_15m_bar(completed_15m, st):
        wr = evaluate_wr90(d15, st, is_flat, struct_d15)
        ret = evaluate_ret(d15, is_flat, struct_d15)
        ret_short = evaluate_ret_short(d15, d15_w, is_flat, struct_d15)
        long_ret = evaluate_long_ret(d15_lr, is_flat, struct_d15)
        mark_15m_evaluated(completed_15m, st)
        for dec in (ret_short, wr, ret, long_ret):
            if dec and dec.would_enter:
                candidates.append(dec)

    out["wr90"] = wr
    out["ret"] = ret
    out["ret_short"] = ret_short
    out["long_ret"] = long_ret

    winner = _pick_live_winner(candidates) if is_flat else None
    out["winner"] = winner
    if winner and submit and is_flat:
        apply_entry(winner, st)
    return out
