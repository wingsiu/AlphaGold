#!/usr/bin/env python3
"""Standalone oil long retrace research — 15m symmetric + gold dip-long 15m.

  PYTHONPATH=. python3 v16/research/oil_long_retrace_backtest.py [start] [end]
  PYTHONPATH=. python3 v16/research/oil_long_retrace_backtest.py --sweep-dip
"""
from __future__ import annotations

import sys
from pathlib import Path

from v16._paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from oil.signal_engine import build_15m
from v15.backtest.backtest_oil import sim_full
from v16.backtest.features import build_features, dip_ml_feature_columns
from v16.backtest.fixed_tpsl_sim import simulate_fixed_tpsl
from v16.backtest.position_sim import simulate_single_position
from v16.config.oil_config import (
    BACKTEST,
    LONG_RETRACE_15M,
    LONG_RETRACE_15M_FEATS,
    OIL_DIP_LONG_15M,
    OIL_ML_CONFIG,
)
from v16.data.load_oil import load_oil_1m
from v16.oil.long_retrace import enrich_d15_long_retrace, long_retrace_15m_exits, oil_dip_long_15m_mask
from v16.oil.structure_feats import structure_on_d15
from v16.oil.wf_ml import filter_trades_by_ml, walk_forward_oil_leg

MODELS = ("xgb", "et", "lgb", "hgb")
ML_TH = [0.50, 0.55, 0.60, 0.65]


def _stats(name: str, trades: list[dict]) -> dict:
    if not trades:
        print(f"\n{name}: 0 trades")
        return {"trades": 0, "pnl": 0.0, "wr": 0.0}
    pnl = sum(t["pnl"] for t in trades)
    wr = sum(1 for t in trades if t["pnl"] > 0) / len(trades) * 100
    print(f"\n{name}")
    print(f"  Trades   : {len(trades)}")
    print(f"  Net PnL  : {pnl:+.1f}")
    print(f"  Win rate : {wr:.1f}%")
    print(f"  Avg/trade: {pnl / len(trades):+.2f}")
    reasons: dict[str, list[float]] = {}
    for t in trades:
        reasons.setdefault(str(t.get("reason", "?")), []).append(t["pnl"])
    for r, ps in sorted(reasons.items(), key=lambda x: -abs(sum(x[1]))):
        print(f"    {r:14s}: {len(ps):4d}  PnL={sum(ps):+.1f}  WR={sum(1 for x in ps if x > 0) / len(ps) * 100:.0f}%")
    return {"trades": len(trades), "pnl": pnl, "wr": wr}


def run_long_retrace_15m(d15: pd.DataFrame, struct_d15: pd.DataFrame) -> None:
    cfg = LONG_RETRACE_15M
    d15e = enrich_d15_long_retrace(d15)
    sigs = long_retrace_15m_exits(d15)
    print(f"\n{'=' * 60}")
    print("  A) LONG RETRACE 15m (mirror of prod ret leg)")
    print(f"  cah>{cfg['dhigh']} rng>{cfg['rng']} bc>{cfg['chg']} uw<{cfg['wick']}")
    print(f"  Raw signals: {len(sigs)}")

    pnls, tr, _ = sim_full(d15e, sigs, cfg["tp"], cfg["sl"], "long_retrace")
    _stats("Mechanical (all signals)", tr)

    print("\n  14D ML model search:")
    best = None
    for model in MODELS:
        if model == "lgb":
            try:
                import lightgbm  # noqa: F401
            except ImportError:
                continue
        try:
            pnls, tr, pr = walk_forward_oil_leg(
                d15e,
                sigs,
                cfg["tp"],
                cfg["sl"],
                LONG_RETRACE_15M_FEATS,
                "long_ret",
                model_name=model,
                struct_frame=struct_d15,
                save_models=True,
            )
        except Exception as e:
            print(f"    {model}: skip ({e})")
            continue
        for th in ML_TH:
            idx = [i for i in range(len(tr)) if i < len(pr) and pr[i] >= th]
            if len(idx) < 5:
                continue
            pnl = sum(pnls[i] for i in idx)
            wr = sum(1 for i in idx if pnls[i] > 0) / len(idx) * 100
            print(f"    {model:4s} ML>={th:.2f}  {len(idx):4d}t  PnL={pnl:+8.1f}  WR={wr:.0f}%")
            if best is None or pnl > best[0]:
                best = (pnl, model, th, len(idx), wr)

    if best:
        pnl, model, th, n, wr = best
        print(f"\n  Best: {model} ML>={th:.2f}  →  {n}t  PnL={pnl:+.1f}  WR={wr:.0f}%")
        pnls, tr, pr = walk_forward_oil_leg(
            d15e, sigs, cfg["tp"], cfg["sl"], LONG_RETRACE_15M_FEATS, "long_ret",
            model_name=model, struct_frame=struct_d15, save_models=True,
        )
        ml_tr = filter_trades_by_ml(pnls, tr, pr, th, "long_ret")
        _stats(f"ML best ({model} ≥{th:.2f})", ml_tr)
        out = PROJECT_ROOT / "runtime/oil_long_retrace_15m_trades.csv"
        pd.DataFrame(ml_tr).to_csv(out, index=False)
        print(f"  CSV: {out}")


def run_dip_long_15m(df: pd.DataFrame, feats: pd.DataFrame, *, sweep: bool = False) -> None:
    cfg = OIL_DIP_LONG_15M
    ex = cfg["execution"]
    tp, sl, h = float(ex["tp"]), float(ex["sl"]), int(ex["horizon"])

    print(f"\n{'=' * 60}")
    print("  B) DIP LONG 15m (gold v16 dip_long_15m port)")
    print(
        f"  prev 15m DOWN | slot DOWN | dip>={cfg['dip_min_below_open_pts']} | "
        f"slot_low_dip>={cfg['dip_min_slot_low_pts']} | minute<{cfg['dip_max_minute_in_slot']}"
    )

    def _mech_trades(c: dict) -> list[dict]:
        mask = oil_dip_long_15m_mask(feats, df.index, c)
        sigs = simulate_single_position(
            df,
            mask,
            side=1,
            tp=tp,
            sl=sl,
            horizon=h,
            same_dir_refresh="entry",
        )
        if sigs.empty:
            return []
        out = []
        for _, row in sigs.iterrows():
            out.append(
                {
                    "entry": row.get("entry_ts", row.get("signal_ts")),
                    "exit": row.get("exit_ts"),
                    "pnl": float(row["pnl"]),
                    "reason": row.get("exit_reason", "?"),
                    "type": "dip_long_15m",
                    "side": 1,
                }
            )
        return out

    if sweep:
        print("\n  Dip threshold sweep (mechanical):")
        rows = []
        for dip in [0.30, 0.40, 0.50, 0.60, 0.80]:
            for slot_low in [0.25, 0.40, 0.50]:
                c = {**cfg, "dip_min_below_open_pts": dip, "dip_min_slot_low_pts": slot_low}
                tr = _mech_trades(c)
                pnl = sum(t["pnl"] for t in tr)
                rows.append({"dip": dip, "slot_low": slot_low, "trades": len(tr), "pnl": pnl})
                print(f"    dip>={dip:.2f} slot_low>={slot_low:.2f}  {len(tr):4d}t  PnL={pnl:+8.1f}")
        pd.DataFrame(rows).to_csv(PROJECT_ROOT / "runtime/oil_dip_long_sweep.csv", index=False)
        return

    tr = _mech_trades(cfg)
    n_mask = int(oil_dip_long_15m_mask(feats, df.index, cfg).sum())
    print(f"  Signal bars: {n_mask}")
    _stats("Mechanical (single position)", tr)

    # ML on labeled dip-long pool
    from v16.oil.wf_ml import _fit_model, _predict_proba, wf_test_windows

    labeled_rows = []
    router = oil_dip_long_15m_mask(feats, df.index, cfg)
    feat_cols = [c for c in dip_ml_feature_columns(feats) if c in feats.columns]
    for ts in df.index[router]:
        ei = df.index.get_loc(ts)
        if ei + 1 >= len(df):
            continue
        entry_idx = ei + 1
        ep = float(df.iloc[entry_idx]["open_ask"])
        res = simulate_fixed_tpsl(df, entry_idx, 1, ep, tp=tp, sl=sl, horizon=h)
        labeled_rows.append({"ts": ts, "entry_idx": entry_idx, "win": int(res.pnl > 0), "pnl": res.pnl})
    if len(labeled_rows) < 80:
        print("  Too few dip-long labels for ML.")
        return

    lab = pd.DataFrame(labeled_rows).set_index("ts")
    X_all = feats.loc[lab.index, feat_cols].to_numpy(dtype=float)
    y = lab["win"].to_numpy(dtype=float)
    tdates = lab.index
    cfg_ml = OIL_ML_CONFIG
    test_start = max(
        tdates.min() + pd.Timedelta(days=int(cfg_ml["train_days"])),
        pd.Timestamp(cfg_ml["wf_start"], tz="UTC"),
    )

    print("\n  14D ML model search:")
    best = None
    for model in MODELS:
        if model == "lgb":
            try:
                import lightgbm  # noqa: F401
            except ImportError:
                continue
        pr = np.zeros(len(lab))
        try:
            for w_start, w_end in wf_test_windows(
                test_start, tdates.max(),
                retrain_freq=cfg_ml["retrain_freq"], retrain_days=int(cfg_ml["retrain_days"]),
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
                m = _fit_model(model, X_tr[bal], y_tr[bal])
                probs = _predict_proba(m, model, X_all[test_mask])
                for j, idx in enumerate(np.where(test_mask)[0]):
                    pr[idx] = probs[j]
        except Exception as e:
            print(f"    {model}: skip ({e})")
            continue
        pnls = lab["pnl"].tolist()
        for th in ML_TH:
            idx = [i for i in range(len(pnls)) if pr[i] >= th]
            if len(idx) < 5:
                continue
            pnl = sum(pnls[i] for i in idx)
            wr = sum(1 for i in idx if pnls[i] > 0) / len(idx) * 100
            print(f"    {model:4s} ML>={th:.2f}  {len(idx):4d}t  PnL={pnl:+8.1f}  WR={wr:.0f}%")
            if best is None or pnl > best[0]:
                best = (pnl, model, th)

    if best and best[0] > 0:
        print(f"\n  Best dip-long ML: {best[1]} ≥{best[2]:.2f}  PnL={best[0]:+.1f}")

    out = PROJECT_ROOT / "runtime/oil_dip_long_15m_trades.csv"
    if tr:
        pd.DataFrame(tr).to_csv(out, index=False)
        print(f"  CSV: {out}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    start = args[0] if args else BACKTEST["default_start"]
    end = args[1] if len(args) > 1 else BACKTEST["default_end"]

    print("=" * 72)
    print("  OIL LONG RETRACE — standalone research")
    print(f"  Period: {start} → {end}")
    print("=" * 72)

    d1m_v16 = load_oil_1m(start, end)
    d15 = build_15m(d1m_v16.copy())
    struct_d15 = structure_on_d15(d1m_v16, d15)
    feats = build_features(d1m_v16)

    run_long_retrace_15m(d15, struct_d15)
    run_dip_long_15m(d1m_v16, feats, sweep="--sweep-dip" in flags)


if __name__ == "__main__":
    main()
