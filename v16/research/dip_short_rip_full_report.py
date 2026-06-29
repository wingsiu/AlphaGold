#!/usr/bin/env python3
"""Full dip_short_rip backtest report — mechanical + ML, anomaly checks."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from v16._paths import PROJECT_ROOT
from v16.backtest.dip_short_rip_run import run_dip_short_rip
from v16.backtest.features import build_features, session_mask
from v16.config import v16_config
from v16.data.load_gold import load_gold_1m
from v16.patterns.dip_short_rip import resolve_execution, router_mask


def _enrich_trades(tdf: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    if tdf.empty:
        return tdf
    out = tdf.copy()
    out["signal_ts"] = pd.to_datetime(out["signal_ts"], utc=True)
    out = out.set_index("signal_ts", drop=False)
    for c in ("minute_in_15m", "slot_rip_pts", "prev_15m_body"):
        if c in feats.columns:
            out[c] = feats[c].reindex(out.index)
    out["month"] = out["signal_ts"].dt.to_period("M").astype(str)
    out["hour_lon"] = out["signal_ts"].dt.tz_convert("Europe/London").dt.hour
    return out.reset_index(drop=True)


def _equity_stats(pnl: pd.Series) -> dict:
    eq = pnl.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    return {
        "max_dd": float(dd.min()),
        "final_eq": float(eq.iloc[-1]) if len(eq) else 0.0,
        "peak_eq": float(peak.max()) if len(peak) else 0.0,
    }


def _streaks(wins: pd.Series) -> tuple[int, int]:
    max_w = max_l = cur_w = cur_l = 0
    for w in wins:
        if w:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _anomaly_checks(tdf: pd.DataFrame, ex: dict) -> list[str]:
    flags: list[str] = []
    tp, sl = float(ex["tp"]), float(ex["sl"])
    if tdf.empty:
        flags.append("EMPTY: no trades")
        return flags

    dup = tdf["signal_ts"].duplicated().sum()
    if dup:
        flags.append(f"DUP_TS: {dup} duplicate signal timestamps")

    if "stop_updates" in tdf.columns:
        off = tdf[(tdf["exit_reason"] == "stop_loss") & (tdf["stop_updates"] == 0) & (tdf["pnl"].abs().sub(sl).abs() > 0.51)]
    else:
        off = tdf[(tdf["exit_reason"] == "stop_loss") & (tdf["pnl"].abs().sub(sl).abs() > 0.51)]
    if len(off):
        flags.append(f"BAD_SL_PNL: {len(off)} stop_loss rows not ~= -{sl}")

    weird = tdf[~tdf["exit_reason"].isin(("target_hit", "stop_loss", "timeout", "no_bars"))]
    if len(weird):
        flags.append(f"UNKNOWN_EXIT: {weird['exit_reason'].unique().tolist()}")

    if (tdf["entry_price"] <= 0).any():
        flags.append("BAD_ENTRY: non-positive entry prices")

    if tdf["pnl"].abs().max() > max(tp, sl) + 50:
        flags.append(f"HUGE_PNL: max |pnl|={tdf['pnl'].abs().max():.1f}")

    idx_counts = tdf["entry_time"].value_counts() if "entry_time" in tdf.columns else pd.Series(dtype=int)
    if len(idx_counts) and (idx_counts > 1).any():
        flags.append(f"OVERLAP_ENTRY_TIME: {(idx_counts > 1).sum()} duplicate entries")

    return flags


def _print_lane(name: str, tdf: pd.DataFrame, ex: dict) -> None:
    print("\n" + "=" * 88)
    print(f"  {name}")
    print("=" * 88)
    if tdf.empty:
        print("  (no trades)")
        return

    pnl = tdf["pnl"]
    eq = _equity_stats(pnl)
    max_w, max_l = _streaks(tdf["win"])

    print(f"\n  Period       : {tdf['signal_ts'].min()} → {tdf['signal_ts'].max()}")
    print(f"  Exit params  : TP{ex['tp']:.0f} / SL{ex['sl']:.0f} / H{ex['horizon']}")
    print(f"  Trades       : {len(tdf)}")
    print(f"  Wins / Loss  : {int(tdf['win'].sum())} / {int((~tdf['win']).sum())}")
    print(f"  Win rate     : {tdf['win'].mean()*100:.2f}%")
    print(f"  Net PnL      : {pnl.sum():+.2f} pts")
    print(f"  Avg / Median : {pnl.mean():+.3f} / {pnl.median():+.3f}")
    print(f"  Std / Sharpe*: {pnl.std():.3f} / {(pnl.mean()/pnl.std()*np.sqrt(len(pnl))):.2f}" if pnl.std() > 0 else "")
    print(f"  Best / Worst : {pnl.max():+.2f} / {pnl.min():+.2f}")
    print(f"  PnL p10/p90  : {pnl.quantile(0.1):+.2f} / {pnl.quantile(0.9):+.2f}")
    print(f"  Max drawdown : {eq['max_dd']:+.2f} pts")
    print(f"  Peak equity  : {eq['peak_eq']:+.2f} pts")
    print(f"  Max win/loss streak: {max_w} / {max_l}")

    print("\n  Exit reason breakdown:")
    for reason, g in tdf.groupby("exit_reason", sort=False):
        print(
            f"    {reason:12s}  n={len(g):5d}  "
            f"WR={g['win'].mean()*100:5.1f}%  "
            f"net={g['pnl'].sum():+9.1f}  avg={g['pnl'].mean():+.3f}"
        )

    print("\n  Monthly PnL:")
    monthly = tdf.groupby("month").agg(trades=("pnl", "count"), net=("pnl", "sum"), wr=("win", "mean"))
    for m, r in monthly.iterrows():
        print(f"    {m}  tr={int(r['trades']):4d}  WR={r['wr']*100:5.1f}%  net={r['net']:+.1f}")

    neg_months = (monthly["net"] < 0).sum()
    print(f"  Losing months: {neg_months} / {len(monthly)}")

    print("\n  By minute in 15m slot:")
    by_min = tdf.groupby("minute_in_15m").agg(n=("pnl", "count"), net=("pnl", "sum"), wr=("win", "mean"))
    for m, r in by_min.iterrows():
        print(f"    min {m:2d}  n={int(r['n']):4d}  WR={r['wr']*100:5.1f}%  net={r['net']:+.1f}")

    print("\n  By London hour:")
    by_h = tdf.groupby("hour_lon").agg(n=("pnl", "count"), net=("pnl", "sum")).sort_values("net", ascending=False)
    for h, r in by_h.head(8).iterrows():
        print(f"    {int(h):02d}:xx  n={int(r['n']):4d}  net={r['net']:+.1f}")
    print("    ... worst hours:")
    for h, r in by_h.tail(3).iterrows():
        print(f"    {int(h):02d}:xx  n={int(r['n']):4d}  net={r['net']:+.1f}")

    if "p_short" in tdf.columns and tdf["p_short"].notna().any():
        print("\n  ML score buckets:")
        tdf2 = tdf.dropna(subset=["p_short"]).copy()
        tdf2["bucket"] = pd.cut(tdf2["p_short"], bins=[0.55, 0.6, 0.65, 0.7, 0.8, 1.0])
        for b, g in tdf2.groupby("bucket", observed=True):
            print(f"    {b}  n={len(g):4d}  WR={g['win'].mean()*100:.1f}%  net={g['pnl'].sum():+.1f}")

    flags = _anomaly_checks(tdf, ex)
    print("\n  Anomaly checks:")
    if flags:
        for f in flags:
            print(f"    ⚠ {f}")
    else:
        print("    ✓ no anomalies detected")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="dip_short_rip full report")
    parser.add_argument("start", nargs="?", default="2025-06-01")
    parser.add_argument("end", nargs="?", default="2026-06-25")
    parser.add_argument("--ml-prob", type=float, default=None)
    args = parser.parse_args()

    bt_start, bt_end = args.start, args.end
    cfg = v16_config.DIP_SHORT_RIP.copy()
    ml_p = float(args.ml_prob if args.ml_prob is not None else cfg["ml_prob"])
    cfg["ml_prob"] = ml_p

    print("#" * 88)
    print(f"  dip_short_rip FULL REPORT  |  {bt_start} → {bt_end}")
    print(f"  router: prev15m up + slot up + rip>=5 + min<10 | sessions={cfg['sessions']}")
    print(f"  single position | same_dir_refresh={cfg.get('same_dir_refresh', 'entry')}")
    print(f"  ML: p>={ml_p} labels={cfg['ml_label_source']}")
    print("#" * 88)

    df = load_gold_1m(bt_start, bt_end)
    feats = build_features(df)
    pool = int(router_mask(feats, df.index, cfg=cfg).sum())
    in_sess = int(session_mask(df.index, cfg["sessions"]).sum())
    print(f"\nData bars: {len(df):,}  |  session bars: {in_sess:,}  |  router pool: {pool:,}")

    mech = _enrich_trades(run_dip_short_rip(df, feats, cfg, mechanical=True), feats)
    ml = _enrich_trades(run_dip_short_rip(df, feats, cfg, mechanical=False, ml_prob=ml_p), feats)

    ex_m = resolve_execution(cfg, mechanical=True)
    ex_ml = resolve_execution(cfg, mechanical=False)
    _print_lane("MECHANICAL (no ML)", mech, ex_m)
    _print_lane("ML FILTERED", ml, ex_ml)

    # overlap
    if not mech.empty and not ml.empty:
        ml_ts = set(ml["signal_ts"])
        mech_ts = set(mech["signal_ts"])
        overlap = len(ml_ts & mech_ts)
        print("\n" + "=" * 88)
        print("  CROSS-LANE")
        print("=" * 88)
        print(f"  Raw router signals: {pool:,}")
        print(f"  Executed trades (no overlap): {len(mech)} mech / {len(ml)} ML")
        if "target_updates" in mech.columns:
            print(f"  Target refreshes: mech={int(mech['target_updates'].sum())} ML={int(ml['target_updates'].sum())}")

    out_m = PROJECT_ROOT / "runtime" / "v16_dip_short_rip_mech_trades.csv"
    out_ml = PROJECT_ROOT / "runtime" / "v16_dip_short_rip_trades.csv"
    mech.to_csv(out_m, index=False)
    ml.to_csv(out_ml, index=False)
    print(f"\nSaved mechanical -> {out_m}")
    print(f"Saved ML         -> {out_ml}")


if __name__ == "__main__":
    main()
