#!/usr/bin/env python3
"""Phase A — model search for v16 oil legs (14D WF).

  PYTHONPATH=. python3 v16/research/oil_v16_ml_model_search.py [start] [end]
"""
from __future__ import annotations

import sys
from pathlib import Path

from v16._paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from oil.signal_engine import SI_FEATS, build_15m
from v16.config.oil_config import BACKTEST, OIL_ML_CONFIG, OIL_LEG_MODELS, RETRACE, WR90
from v16.data.load_oil import load_oil_1m
from v16.oil.patterns import FEAT_MAP, retrace_exits, short_impulse_records, wr90_cluster_exits
from v16.oil.structure_feats import structure_on_d15
from v16.oil.wf_ml import walk_forward_oil_leg

MODELS = ("xgb", "et", "hgb", "lgb")
ML_THRESHOLDS = {
    "wr90": [0.50, 0.55, 0.60],
    "ret": [0.55, 0.60, 0.65],
}


def _score(pnls: list, pr: np.ndarray, th: float) -> tuple[int, float, float]:
    idx = [i for i in range(len(pnls)) if i < len(pr) and pr[i] >= th]
    if not idx:
        return 0, 0.0, 0.0
    sel = [pnls[i] for i in idx]
    wr = sum(1 for x in sel if x > 0) / len(sel) * 100
    return len(sel), sum(sel), wr


def main() -> None:
    args = sys.argv[1:]
    start = args[0] if args else BACKTEST["default_start"]
    end = args[1] if len(args) > 1 else BACKTEST["default_end"]

    print("Loading data...")
    d1m_v16 = load_oil_1m(start, end)
    d1m = d1m_v16.copy()
    d15 = build_15m(d1m)
    struct_d15 = structure_on_d15(d1m_v16, d15)

    rows: list[dict] = []

    for leg, sig_fn, tp, sl, feats in [
        ("wr90", wr90_cluster_exits, WR90["tp"], WR90["sl"], FEAT_MAP["wr90"]),
        ("ret", lambda d: retrace_exits(d, struct_d15), RETRACE["tp"], RETRACE["sl"], FEAT_MAP["ret"]),
    ]:
        sigs = sig_fn(d15)
        print(f"\n[{leg}] {len(sigs)} cluster/signal exits")
        for model in MODELS:
            if model == "lgb":
                try:
                    import lightgbm  # noqa: F401
                except ImportError:
                    continue
            try:
                pnls, _, pr = walk_forward_oil_leg(
                    d15,
                    sigs,
                    tp,
                    sl,
                    feats,
                    leg,
                    model_name=model,
                    struct_frame=struct_d15,
                    save_models=False,
                )
            except Exception as e:
                print(f"  {model}: skip ({e})")
                continue
            for th in ML_THRESHOLDS.get(leg, [0.55, 0.60]):
                n, pnl, wr = _score(pnls, pr, th)
                rows.append(
                    {
                        "leg": leg,
                        "model": model,
                        "ml_th": th,
                        "trades": n,
                        "pnl": pnl,
                        "wr": wr,
                    }
                )
                print(f"  {model:4s} ML>={th:.2f}  {n:4d}t  PnL={pnl:+8.1f}  WR={wr:.0f}%")

    # SI quick search
    recs = short_impulse_records(d1m)
    print(f"\n[si] {len(recs)} raw records")
    if len(recs) >= 30:
        from v16.oil.wf_ml import wf_test_windows
        from v16.oil.wf_ml import _fit_model, _predict_proba

        y = np.array([1.0 if r["pnl"] > 0 else 0.0 for r in recs])
        tdates = pd.DatetimeIndex([r["entry_idx"] for r in recs])
        X_all = np.array([[float(r["row"].get(f, 0)) for f in SI_FEATS] for r in recs])
        pnls = [r["pnl"] for r in recs]
        cfg = OIL_ML_CONFIG
        test_start = max(
            tdates.min() + pd.Timedelta(days=int(cfg["train_days"])),
            pd.Timestamp(cfg["wf_start"], tz="UTC"),
        )
        for model in MODELS:
            if model == "lgb":
                try:
                    import lightgbm  # noqa: F401
                except ImportError:
                    continue
            pr = np.zeros(len(recs))
            try:
                for w_start, w_end in wf_test_windows(
                    test_start,
                    tdates.max(),
                    retrain_freq=cfg["retrain_freq"],
                    retrain_days=int(cfg["retrain_days"]),
                ):
                    train_mask = tdates < w_start
                    test_mask = (tdates >= w_start) & (tdates < w_end)
                    if train_mask.sum() < cfg["min_train_rows"] or test_mask.sum() == 0:
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
                print(f"  {model}: skip ({e})")
                continue
            for th in [0.50, 0.55, 0.60]:
                n, pnl, wr = _score(pnls, pr, th)
                rows.append({"leg": "si", "model": model, "ml_th": th, "trades": n, "pnl": pnl, "wr": wr})
                print(f"  {model:4s} ML>={th:.2f}  {n:4d}t  PnL={pnl:+8.1f}  WR={wr:.0f}%")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n  No model search results.")
        return
    out = PROJECT_ROOT / BACKTEST["model_search_csv"]
    df.to_csv(out, index=False)

    print(f"\n{'=' * 60}")
    print("  Best per leg:")
    for leg in df["leg"].unique():
        sub = df[df["leg"] == leg].sort_values("pnl", ascending=False).iloc[0]
        print(
            f"  {leg:5s}  {sub['model']} ML>={sub['ml_th']:.2f}  "
            f"{int(sub['trades'])}t  PnL={sub['pnl']:+.1f}"
        )
        OIL_LEG_MODELS[leg]["model"] = str(sub["model"])
        OIL_LEG_MODELS[leg]["ml_th"] = float(sub["ml_th"])
    print(f"\n  CSV: {out}")
    print(f"  Suggested OIL_LEG_MODELS update: {OIL_LEG_MODELS}")


if __name__ == "__main__":
    main()
