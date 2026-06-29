#!/usr/bin/env python3
"""Compare WR90 Option 1 (strict/prod) vs Option 2 (relaxed). Safe: Option 2 models -> wr90_option2/."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import joblib
import numpy as np
import pandas as pd
from v15.backtest import backtest_oil as bt
import oil.wf_ml as wf

OPT2_DIR = REPO / "v15" / "oil" / "wf_models" / "wr90_option2"
OPT1 = dict(entry=-80, cv=15000, ep=3, tp=80, sl=30, ml=0.55)
OPT2 = dict(entry=-70, cv=3000, ep=3, tp=60, sl=20, ml=0.65)
START, END = "2024-01-01", "2026-06-30"

_orig_model_path = wf.model_path


def _model_path(stype: str, month: str) -> Path:
    if stype == "wr90":
        return OPT2_DIR / f"{month}.joblib"
    return _orig_model_path(stype, month)


def wr90_sigs(d15, entry, cv, ep):
    in_s = d15["ins"]
    o = (d15["wr"] < entry) & in_s
    sigs, ie, c, b = [], False, 0.0, 0
    for i in range(len(d15)):
        if o.iloc[i]:
            if not ie:
                c, b = 0.0, 0
            ie = True
            c += d15["volume"].iloc[i]
            b += 1
        elif ie:
            if i < len(d15) - 1 and in_s.iloc[i] and c >= cv and b >= ep:
                sigs.append({"idx": i})
            ie, c, b = False, 0.0, 0
    return sigs


def wr90_trades(d15, cfg, use_opt2_models=False):
    sigs = wr90_sigs(d15, cfg["entry"], cfg["cv"], cfg["ep"])
    if use_opt2_models:
        bt._model_path = _model_path
        wf.model_path = _model_path
    else:
        bt._model_path = _orig_model_path
        wf.model_path = _orig_model_path
    res = bt.train_ml(d15, sigs, cfg["tp"], cfg["sl"], bt.WR_FEATS, "wr90", cfg["ml"])
    bt._model_path = _orig_model_path
    wf.model_path = _orig_model_path
    if not res:
        return [], sigs
    pnls, tr, pr = res
    out = [{**tr[i], "_leg": "wr90"} for i in range(len(tr)) if pr[i] >= cfg["ml"]]
    return out, sigs, pnls, pr


def ret_si_trades(d1m, d15):
    other = []
    mask = (
        (d15["cad"] > bt.RET_DLOW)
        & (d15["avg_r3"] > bt.RET_RNG)
        & (d15["bc"] < bt.RET_CHG)
        & (d15["wb"] < bt.RET_WICK)
        & d15["ins"]
    )
    sigs_r = [{"idx": i} for i in range(len(d15)) if mask.iloc[i]]
    res_r = bt.train_ml(d15, sigs_r, bt.RET_TP, bt.RET_SL, bt.RET_FEATS, "ret", bt.RET_ML_TH)
    if res_r:
        _, tr_r, pr_r = res_r
        for i, t in enumerate(tr_r):
            if pr_r[i] >= bt.RET_ML_TH:
                other.append({**t, "_leg": "ret"})

    d1m_s = bt.compute_si_features(d1m)
    si_mask = (
        (d1m_s["prev_change"] < bt.SI_CHANGE_MAX)
        & (d1m_s["prev2_change"] < 10.0)
        & (d1m_s["prev2_change"] > -14.0)
        & (d1m_s["prev_lower_wick"] < 35.0)
        & (d1m_s["prev_volume"] > bt.SI_VOL_MIN)
        & d1m_s["ny_hour"]
        & (d1m_s["up_count3_15min"] != -3)
        & (d1m_s["dist_day_high"] < 180.0)
    )
    si_sigs = sorted(d1m_s.index[si_mask].tolist())
    si_recs, in_si, si_ex = [], False, -1
    for sig in si_sigs:
        ei = d1m_s.index.get_loc(sig)
        if ei + bt.SI_MAX_B >= len(d1m_s):
            continue
        if in_si and ei <= si_ex:
            continue
        ep = d1m_s.iloc[ei]["close_bid"]
        ex_price, bars, reason = bt.sim_si_fixed(ei, ep, d1m_s)
        si_recs.append(
            {
                "entry_idx": sig,
                "exit_ts": d1m_s.index[ei + bars],
                "pnl": ep - ex_price,
                "reason": reason,
                "entry_price": ep,
                "exit_price": ex_price,
            }
        )
        in_si, si_ex = True, ei + bars
    if si_recs:
        ds = pd.DatetimeIndex([r["entry_idx"] for r in si_recs])
        ms = sorted(set(d.to_period("M") for d in ds))
        X_all = np.array(
            [[float(d1m_s.loc[r["entry_idx"]].get(f, 0)) for f in bt.SI_FEATS] for r in si_recs]
        )
        sp = np.zeros(len(si_recs))
        for tm in ms:
            saved = _orig_model_path("si", str(tm))
            if saved.exists():
                tst = np.array([d.to_period("M") == tm for d in ds])
                prib = joblib.load(saved).predict_proba(X_all[tst])[:, 1]
                for j, idx in enumerate(np.where(tst)[0]):
                    sp[idx] = prib[j]
        for i, r in enumerate(si_recs):
            if sp[i] >= bt.SI_PROB:
                other.append(
                    {
                        "entry": r["entry_idx"],
                        "exit": r["exit_ts"],
                        "pnl": r["pnl"],
                        "reason": r["reason"],
                        "type": "short_impulse",
                        "side": -1,
                        "_leg": "si",
                    }
                )
    return other


def stats(trades):
    if not trades:
        return 0, 0.0, 0.0
    pnl = sum(t["pnl"] for t in trades)
    wr = sum(1 for t in trades if t["pnl"] > 0) / len(trades) * 100
    return len(trades), pnl, wr


def report(label, cfg, sigs_n, wr_tr, merged, other):
    wr = [t for t in merged if t.get("_leg") == "wr90"]
    nt, pnl, wr_pct = stats(merged)
    wn, wp, ww = stats(wr)
    rn, rp, _ = stats([t for t in merged if t.get("_leg") == "ret"])
    sn, sp, _ = stats([t for t in merged if t.get("_leg") == "si"])
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(
        f"  WR<{cfg['entry']} CV>={cfg['cv']} Ep>={cfg['ep']}  "
        f"TP={cfg['tp']}/SL={cfg['sl']}  ML>={cfg['ml']}"
    )
    print(f"  Cluster exits: {sigs_n}  WR90 pre-merge: {len(wr_tr)}")
    print(f"  Combined (1-slot): {nt}t  PnL={pnl:+.1f}  WR={wr_pct:.1f}%")
    print(f"  WR90 in combined:  {wn}t  PnL={wp:+.1f}  WR={ww:.1f}%")
    print(f"  Ret in combined:   {rn}t  PnL={rp:+.1f}")
    print(f"  SI in combined:    {sn}t  PnL={sp:+.1f}")
    return nt, pnl, wr_pct, wn, wp, ww


def main():
    print("Loading data...")
    d1m = bt.load(START, END)
    d15 = bt.build_15m(d1m)
    other = ret_si_trades(d1m, d15)
    print(f"Ret+SI legs (fixed): {len(other)} trades pre-merge")

    print("\n[Option 1] prod models...")
    wr1, sigs1, _, _ = wr90_trades(d15, OPT1, use_opt2_models=False)
    m1 = bt.merge_single_position(other + wr1)
    s1 = report("Option 1 (strict, current prod)", OPT1, len(sigs1), wr1, m1, other)

    OPT2_DIR.mkdir(parents=True, exist_ok=True)
    need_train = len(list(OPT2_DIR.glob("*.joblib"))) < 20
    if need_train:
        print("\n[Option 2] training models -> wr90_option2/ (prod untouched)...")
        sigs2 = wr90_sigs(d15, OPT2["entry"], OPT2["cv"], OPT2["ep"])
        bt._model_path = _model_path
        wf.model_path = _model_path
        bt.train_ml(d15, sigs2, OPT2["tp"], OPT2["sl"], bt.WR_FEATS, "wr90", OPT2["ml"])
        bt._model_path = _orig_model_path
        wf.model_path = _orig_model_path
    else:
        print("\n[Option 2] using cached wr90_option2 models...")

    wr2, sigs2, pnls2, pr2 = wr90_trades(d15, OPT2, use_opt2_models=True)
    print("\n  ML threshold sweep (Option 2, retrained models):")
    print(f"  {'ML':>6} {'trades':>6} {'pnl':>8} {'wr':>5}")
    best = None
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        idx = [i for i in range(len(pnls2)) if pr2[i] >= th]
        if len(idx) < 5:
            continue
        pnl = sum(pnls2[i] for i in idx)
        wr = sum(1 for i in idx if pnls2[i] > 0) / len(idx) * 100
        print(f"  {th:>6.2f} {len(idx):6d} {pnl:+8.1f} {wr:4.0f}%")
        if best is None or pnl > best[0]:
            best = (pnl, th)
    opt2_run = {**OPT2, "ml": best[1]}
    print(f"\n  Best Option 2 ML: {best[1]:.2f} (WR90 isolation PnL {best[0]:+.1f})")
    wr2f, _, _, _ = wr90_trades(d15, opt2_run, use_opt2_models=True)
    m2 = bt.merge_single_position(other + wr2f)
    s2 = report(f"Option 2 (relaxed, ML>={best[1]:.2f})", opt2_run, len(sigs2), wr2f, m2, other)

    print(f"\n{'=' * 60}")
    print("  HEAD-TO-HEAD (single-slot, 2024-01 -> 2026-06-30)")
    print(f"{'=' * 60}")
    print(f"  {'':28s} {'Opt1':>12s} {'Opt2':>12s}")
    print(f"  {'Combined trades':28s} {s1[0]:12d} {s2[0]:12d}")
    print(f"  {'Combined PnL':28s} {s1[1]:+12.1f} {s2[1]:+12.1f}")
    print(f"  {'Combined WR':28s} {s1[2]:11.1f}% {s2[2]:11.1f}%")
    print(f"  {'WR90 trades':28s} {s1[3]:12d} {s2[3]:12d}")
    print(f"  {'WR90 PnL':28s} {s1[4]:+12.1f} {s2[4]:+12.1f}")
    print(f"  {'WR90 WR':28s} {s1[5]:11.1f}% {s2[5]:11.1f}%")
    winner = "Option 1" if s1[1] > s2[1] else "Option 2"
    print(f"\n  Winner combined: {winner} ({abs(s1[1] - s2[1]):+.1f} pts)")


if __name__ == "__main__":
    main()
