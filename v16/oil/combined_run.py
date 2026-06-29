"""Run full v16 oil portfolio — WR90 + ret + ret_short + long_ret + SI, single-slot merge."""
from __future__ import annotations

import numpy as np
import pandas as pd

from oil.signal_engine import SI_FEATS, build_15m
from v16.oil.merge import LEG_ENTRY_PRIORITY, merge_single_position

# Merge tie-break priorities (extend defaults)
LEG_ENTRY_PRIORITY.setdefault("oil_rip_short", 0)
LEG_ENTRY_PRIORITY.setdefault("rip", 0)
LEG_ENTRY_PRIORITY.setdefault("ret_short", 0)
LEG_ENTRY_PRIORITY.setdefault("oil_retrace_short", 0)
LEG_ENTRY_PRIORITY.setdefault("long_retrace", 2)
LEG_ENTRY_PRIORITY.setdefault("oil_long_retrace", 2)
from v16.config.oil_config import (
    LONG_RETRACE_15M,
    LONG_RETRACE_15M_FEATS,
    OIL_LEG_MODELS,
    OIL_RIP_SHORT,
    RETRACE,
    RET_SHORT,
    SHORT_IMPULSE,
    STRUCTURE_GATE,
    WR90,
    WR90_STRUCT_HOLD,
)
from v16.backtest.features import build_features, dip_ml_feature_columns
from v16.backtest.fixed_tpsl_sim import simulate_fixed_tpsl
from v16.backtest.position_sim import simulate_single_position
from v16.data.load_oil import load_oil_1m
from v16.oil.long_retrace import enrich_d15_long_retrace, enrich_d15_wicks, long_retrace_15m_exits
from v16.oil.patterns import (
    FEAT_MAP,
    retrace_exits,
    retrace_short_exits,
    rip_short_labeled,
    rip_short_router,
    short_impulse_records,
    wr90_cluster_exits,
)
from v16.oil.struct_hold import sim_wr90_struct_hold
from v16.oil.structure_feats import structure_on_d15
from v16.oil.wf_ml import filter_trades_by_ml, walk_forward_oil_leg
from v16.structure.swing_zigzag import build_structure_context


def _run_wr90(
    d15: pd.DataFrame,
    d1m_v16: pd.DataFrame,
    struct_d15: pd.DataFrame,
    *,
    wr90_exit: str = "fixed_tpsl",
    model_cfg: dict | None = None,
) -> list[dict]:
    from v16.oil.sim_15m import sim_15m_long as _sim15

    cfg = model_cfg or OIL_LEG_MODELS["wr90"]
    sigs = wr90_cluster_exits(d15)
    pnls, tr, pr = walk_forward_oil_leg(
        d15,
        sigs,
        WR90["tp"],
        WR90["sl"],
        FEAT_MAP["wr90"],
        "wr90",
        model_name=cfg["model"],
        ml_th=cfg["ml_th"],
        struct_frame=struct_d15,
        side="long",
    )
    passed_idx = [i for i in range(len(tr)) if i < len(pr) and pr[i] >= cfg["ml_th"]]
    if wr90_exit == "struct_hold":
        _, _, m = _sim15(d15, sigs, WR90["tp"], WR90["sl"], "wr90")
        ml_entries = [sigs[m[i]]["idx"] for i in passed_idx if i < len(m)]
        return sim_wr90_struct_hold(d15, d1m_v16, ml_entries)
    return filter_trades_by_ml(pnls, tr, pr, cfg["ml_th"], "wr90")


def _run_retrace(d15: pd.DataFrame, struct_d15: pd.DataFrame, model_cfg: dict | None = None) -> list[dict]:
    cfg = model_cfg or OIL_LEG_MODELS["ret"]
    sf = struct_d15 if STRUCTURE_GATE.get("enabled") else None
    sigs = retrace_exits(d15, sf)
    pnls, tr, pr = walk_forward_oil_leg(
        d15,
        sigs,
        RETRACE["tp"],
        RETRACE["sl"],
        FEAT_MAP["ret"],
        "ret",
        model_name=cfg["model"],
        ml_th=cfg["ml_th"],
        struct_frame=struct_d15 if sf is not None else None,
        side="long",
    )
    out = filter_trades_by_ml(pnls, tr, pr, cfg["ml_th"], "ret")
    for t in out:
        t["type"] = "ret"
        t["side"] = 1
    return out


def _run_retrace_short(d15: pd.DataFrame, struct_d15: pd.DataFrame, model_cfg: dict | None = None) -> list[dict]:
    cfg = model_cfg or OIL_LEG_MODELS["ret_short"]
    sf = struct_d15 if STRUCTURE_GATE.get("enabled") else None
    d = enrich_d15_wicks(d15)
    sigs = retrace_short_exits(d15, sf)
    pnls, tr, pr = walk_forward_oil_leg(
        d,
        sigs,
        RET_SHORT["tp"],
        RET_SHORT["sl"],
        FEAT_MAP["ret_short"],
        "ret_short",
        model_name=cfg["model"],
        ml_th=cfg["ml_th"],
        struct_frame=struct_d15 if sf is not None else None,
        side="short",
        stype="ret_short",
    )
    out = filter_trades_by_ml(pnls, tr, pr, cfg["ml_th"], "ret_short")
    for t in out:
        t["type"] = "ret_short"
        t["side"] = -1
    return out


def _run_long_retrace(d15: pd.DataFrame, struct_d15: pd.DataFrame, model_cfg: dict | None = None) -> list[dict]:
    cfg = model_cfg or OIL_LEG_MODELS["long_ret"]
    sf = struct_d15 if STRUCTURE_GATE.get("enabled") else None
    d15e = enrich_d15_long_retrace(d15)
    sigs = long_retrace_15m_exits(d15, sf)
    pnls, tr, pr = walk_forward_oil_leg(
        d15e,
        sigs,
        LONG_RETRACE_15M["tp"],
        LONG_RETRACE_15M["sl"],
        LONG_RETRACE_15M_FEATS,
        "long_ret",
        model_name=cfg["model"],
        ml_th=cfg["ml_th"],
        struct_frame=struct_d15 if sf is not None else None,
        side="long",
        stype="long_retrace",
    )
    out = filter_trades_by_ml(pnls, tr, pr, cfg["ml_th"], "long_ret")
    for t in out:
        t["type"] = "long_retrace"
        t["side"] = 1
    return out


def _run_si(d1m: pd.DataFrame, d1m_v16: pd.DataFrame, model_cfg: dict | None = None) -> list[dict]:
    cfg = model_cfg or OIL_LEG_MODELS["si"]
    struct_1m = (
        build_structure_context(d1m_v16, rule="15min", atr_mult=3.0, atr_period=14)
        if STRUCTURE_GATE.get("enabled")
        else None
    )
    recs = short_impulse_records(d1m, struct_1m)
    if not recs:
        return []

    from v16.oil.wf_ml import wf_test_windows
    from v16.config.oil_config import OIL_ML_CONFIG
    from v16.oil.wf_ml import _feature_matrix, _fit_model, _predict_proba, model_path
    import joblib

    # Build pseudo-sigs for feature matrix helper
    y = np.array([1.0 if r["pnl"] > 0 else 0.0 for r in recs])
    tdates = pd.DatetimeIndex([r["entry_idx"] for r in recs])
    X_all = np.array([[float(r["row"].get(f, 0)) for f in SI_FEATS] for r in recs])

    cfg_ml = OIL_ML_CONFIG
    rf = cfg_ml["retrain_freq"]
    rd = int(cfg_ml["retrain_days"])
    tdays = int(cfg_ml["train_days"])
    min_rows = int(cfg_ml["min_train_rows"])
    wf_start = pd.Timestamp(cfg_ml["wf_start"], tz="UTC")
    test_start = max(tdates.min() + pd.Timedelta(days=tdays), wf_start)
    pr = np.zeros(len(recs))

    for w_start, w_end in wf_test_windows(test_start, tdates.max(), retrain_freq=rf, retrain_days=rd):
        train_mask = tdates < w_start
        test_mask = (tdates >= w_start) & (tdates < w_end)
        if train_mask.sum() < min_rows or test_mask.sum() == 0:
            continue
        X_tr, y_tr = X_all[train_mask], y[train_mask]
        w_idx = np.where(y_tr == 1)[0]
        l_idx = np.where(y_tr == 0)[0]
        nm = min(len(w_idx), len(l_idx))
        if nm < 5:
            continue
        rng = np.random.RandomState(42 + int(w_start.strftime("%Y%m%d")))
        bal = np.concatenate([rng.choice(w_idx, nm, 0), rng.choice(l_idx, nm, 0)])
        model = _fit_model(cfg["model"], X_tr[bal], y_tr[bal])
        joblib.dump(model, model_path("si", w_start.strftime("%Y-%m-%d")))
        probs = _predict_proba(model, cfg["model"], X_all[test_mask])
        for j, idx in enumerate(np.where(test_mask)[0]):
            pr[idx] = probs[j]

    out = []
    for i, r in enumerate(recs):
        if pr[i] >= cfg["ml_th"]:
            out.append(
                {
                    "entry": r["entry_idx"],
                    "exit": r["exit_ts"],
                    "pnl": r["pnl"],
                    "reason": r["reason"],
                    "type": "short_impulse",
                    "side": -1,
                    "_leg": "si",
                    "_prob": float(pr[i]),
                }
            )
    return out


def _run_rip_short(d1m_v16: pd.DataFrame, feats: pd.DataFrame, model_cfg: dict | None = None) -> list[dict]:
    cfg = model_cfg or OIL_LEG_MODELS["rip"]
    rip_cfg = OIL_RIP_SHORT
    ex = rip_cfg["execution"]
    tp, sl, h = float(ex["tp"]), float(ex["sl"]), int(ex["horizon"])
    labeled = rip_short_labeled(d1m_v16, feats, tp=tp, sl=sl, horizon=h)
    if labeled.empty:
        return []

    feat_cols = [c for c in dip_ml_feature_columns(feats) if c in feats.columns]
    from v16.oil.wf_ml import wf_test_windows
    from v16.config.oil_config import OIL_ML_CONFIG
    from v16.oil.wf_ml import _fit_model, _predict_proba, model_path
    import joblib

    X_all = feats.loc[labeled.index, feat_cols].to_numpy(dtype=float)
    y = labeled["short_win"].to_numpy(dtype=float)
    tdates = labeled.index
    cfg_ml = OIL_ML_CONFIG
    test_start = max(
        tdates.min() + pd.Timedelta(days=int(cfg_ml["train_days"])),
        pd.Timestamp(cfg_ml["wf_start"], tz="UTC"),
    )
    pr_map: dict[pd.Timestamp, float] = {}

    for w_start, w_end in wf_test_windows(
        test_start,
        tdates.max(),
        retrain_freq=cfg_ml["retrain_freq"],
        retrain_days=int(cfg_ml["retrain_days"]),
    ):
        train_mask = tdates < w_start
        test_mask = (tdates >= w_start) & (tdates < w_end)
        if train_mask.sum() < cfg_ml["min_train_rows"] or test_mask.sum() == 0:
            continue
        X_tr, y_tr = X_all[train_mask], y[train_mask]
        w_idx = np.where(y_tr == 1)[0]
        l_idx = np.where(y_tr == 0)[0]
        nm = min(len(w_idx), len(l_idx))
        if nm < 5:
            continue
        rng = np.random.RandomState(42 + int(w_start.strftime("%Y%m%d")))
        bal = np.concatenate([rng.choice(w_idx, nm, 0), rng.choice(l_idx, nm, 0)])
        model = _fit_model(cfg["model"], X_tr[bal], y_tr[bal])
        joblib.dump(model, model_path("rip", w_start.strftime("%Y-%m-%d")))
        probs = _predict_proba(model, cfg["model"], X_all[test_mask])
        for ts, prob in zip(tdates[test_mask], probs):
            pr_map[ts] = float(prob)

    trades = []
    router = rip_short_router(feats, d1m_v16.index)
    for ts in labeled.index:
        prob = pr_map.get(ts)
        if prob is None or prob < cfg["ml_th"]:
            continue
        if not bool(router.loc[ts]):
            continue
        row = labeled.loc[ts]
        ei = int(row["entry_idx"])
        ep = float(d1m_v16.iloc[ei]["close_bid"])
        res = simulate_fixed_tpsl(d1m_v16, ei, -1, ep, tp=tp, sl=sl, horizon=h)
        trades.append(
            {
                "entry": d1m_v16.index[ei],
                "exit": d1m_v16.index[min(ei + res.bars_held, len(d1m_v16) - 1)],
                "pnl": res.pnl,
                "reason": res.exit_reason,
                "type": "oil_rip_short",
                "side": -1,
                "_leg": "rip",
                "_prob": prob,
            }
        )
    return trades


def run_oil_v16_combined(
    start: str,
    end: str,
    *,
    wr90_exit: str = "fixed_tpsl",
    include_rip: bool | None = None,
    leg_models: dict | None = None,
) -> tuple[list[dict], dict]:
    """Load data, run all lanes, merge single slot. Returns (trades, leg_stats)."""
    d1m_v16 = load_oil_1m(start, end)
    d1m = d1m_v16.copy()  # legacy columns present
    d15 = build_15m(d1m)
    struct_d15 = structure_on_d15(d1m_v16, d15)
    feats = build_features(d1m_v16)

    models = leg_models or OIL_LEG_MODELS
    all_raw: list[dict] = []
    leg_stats: dict = {}

    wr = _run_wr90(d15, d1m_v16, struct_d15, wr90_exit=wr90_exit, model_cfg=models.get("wr90"))
    all_raw.extend(wr)
    leg_stats["wr90"] = {"trades": len(wr), "pnl": sum(t["pnl"] for t in wr)}

    ret = _run_retrace(d15, struct_d15, model_cfg=models.get("ret"))
    all_raw.extend(ret)
    leg_stats["ret"] = {"trades": len(ret), "pnl": sum(t["pnl"] for t in ret)}

    ret_s = _run_retrace_short(d15, struct_d15, model_cfg=models.get("ret_short"))
    all_raw.extend(ret_s)
    leg_stats["ret_short"] = {"trades": len(ret_s), "pnl": sum(t["pnl"] for t in ret_s)}

    lret = _run_long_retrace(d15, struct_d15, model_cfg=models.get("long_ret"))
    all_raw.extend(lret)
    leg_stats["long_ret"] = {"trades": len(lret), "pnl": sum(t["pnl"] for t in lret)}

    si = _run_si(d1m, d1m_v16, model_cfg=models.get("si"))
    all_raw.extend(si)
    leg_stats["si"] = {"trades": len(si), "pnl": sum(t["pnl"] for t in si)}

    if include_rip is None:
        from v16.config.oil_config import BACKTEST

        include_rip = bool(BACKTEST.get("include_rip_short", False))
    if include_rip:
        rip = _run_rip_short(d1m_v16, feats, model_cfg=models.get("rip"))
        all_raw.extend(rip)
        leg_stats["rip"] = {"trades": len(rip), "pnl": sum(t["pnl"] for t in rip)}

    raw_n = len(all_raw)
    merged = merge_single_position(all_raw)
    leg_stats["_raw"] = raw_n
    leg_stats["_merged"] = len(merged)
    leg_stats["_combined_pnl"] = sum(t["pnl"] for t in merged)
    return merged, leg_stats


def leg_stats_table(leg_stats: dict) -> str:
    lines = ["Leg stats (pre-merge):"]
    for leg in ("wr90", "ret", "ret_short", "long_ret", "si", "rip"):
        if leg in leg_stats:
            s = leg_stats[leg]
            lines.append(f"  {leg:5s}  {s['trades']:4d}t  PnL={s['pnl']:+.1f}")
    lines.append(
        f"  Combined: {leg_stats.get('_merged', 0)}t (from {leg_stats.get('_raw', 0)})  "
        f"PnL={leg_stats.get('_combined_pnl', 0):+.1f}"
    )
    return "\n".join(lines)
