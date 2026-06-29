"""Oil live vs backtest parity checker.

Run: python3 _check_oil_parity.py [YYYY-MM-DD] [YYYY-MM-DD]
Without dates: checks yesterday (HKT).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from oil.signal_engine import (
    SignalDecision,
    load_mysql_bars,
    replay_entries,
    replay_leg_entries,
    LONG_WR_ML_TH,
    RET_ML_TH,
    SI_PROB,
)

OUTPUT_PATH = REPO / "runtime" / "oil_parity_latest.txt"
CSV_PATH = REPO / "runtime" / "oil_combined_backtest_trades.csv"

LEG_MAP = {
    "wr90_long": "wr90",
    "oil_retrace": "ret",
    "short_impulse": "si",
    "wr90": "wr90",
    "ret": "ret",
    "si": "si",
}


@dataclass
class Entry:
    leg: str
    ts: pd.Timestamp
    prob: Optional[float] = None
    source: str = ""

    def key(self) -> tuple:
        return self.leg, self.ts.floor("min")


def _hkt_day_bounds(day: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC bounds for one HKT calendar day."""
    start_hkt = pd.Timestamp(f"{day} 00:00:00", tz="Asia/Hong_Kong")
    end_hkt = start_hkt + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    return start_hkt.tz_convert("UTC"), end_hkt.tz_convert("UTC")


def _decisions_to_entries(decisions: list[SignalDecision]) -> list[Entry]:
    return [
        Entry(d.leg, pd.Timestamp(d.entry_ts).tz_convert("UTC"), d.prob, "live_replay")
        for d in decisions if d.would_enter
    ]


def _ml_trades_window(d1m: pd.DataFrame) -> list[dict]:
    """All ML-filtered trades in loaded window (legs may overlap before merge)."""
    from v15.backtest import backtest_oil as bt

    d15 = bt.build_15m(d1m)
    d1m_s = bt.compute_si_features(d1m)
    trades: list[dict] = []

    in_s = d15["ins"]
    o = (d15["wr"] < bt.LONG_ENTRY) & in_s
    sigs_w = []
    ie, cv, bc = False, 0.0, 0
    for i in range(len(d15)):
        if o.iloc[i]:
            if not ie:
                cv, bc = 0.0, 0
            ie = True
            cv += d15["volume"].iloc[i]
            bc += 1
        elif ie:
            ebi = i
            if ebi < len(d15) - 1 and in_s.iloc[ebi] and cv >= bt.LONG_CV and bc >= bt.LONG_EP_MIN:
                sigs_w.append({"idx": ebi})
            ie, cv, bc = False, 0.0, 0

    res_w = bt.train_ml(d15, sigs_w, bt.LONG_TP, bt.LONG_SL, bt.WR_FEATS, "wr90", bt.LONG_WR_ML_TH)
    if res_w:
        _, tr_w, probas_w = res_w
        for ti, tr in enumerate(tr_w):
            if ti < len(probas_w) and probas_w[ti] >= bt.LONG_WR_ML_TH:
                trades.append({**tr, "_prob": float(probas_w[ti]), "_leg": "wr90"})

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
        _, tr_r, probas_r = res_r
        for ti, tr in enumerate(tr_r):
            if ti < len(probas_r) and probas_r[ti] >= bt.RET_ML_TH:
                trades.append({**tr, "_prob": float(probas_r[ti]), "_leg": "ret"})

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
    si_recs = []
    in_si, si_ex = False, -1
    for sig in si_sigs:
        ei = d1m_s.index.get_loc(sig)
        if ei + bt.SI_MAX_B >= len(d1m_s):
            continue
        if in_si and ei <= si_ex:
            continue
        ep = d1m_s.iloc[ei]["close_bid"]
        ex_price, bars, reason = bt.sim_si_fixed(ei, ep, d1m_s)
        si_recs.append({
            "entry_idx": sig,
            "exit_ts": d1m_s.index[ei + bars],
            "pnl": ep - ex_price,
            "reason": reason,
            "entry_price": ep,
            "exit_price": ex_price,
        })
        in_si, si_ex = True, ei + bars

    if si_recs:
        from oil.wf_ml import model_path
        import joblib
        import numpy as np

        ds = pd.DatetimeIndex([r["entry_idx"] for r in si_recs])
        ms = sorted(set(d.to_period("M") for d in ds))
        X_all = np.array(
            [[float(d1m_s.loc[r["entry_idx"]].get(f, 0)) for f in bt.SI_FEATS] for r in si_recs]
        )
        sp = np.zeros(len(si_recs))
        for tm in ms:
            month_str = str(tm)
            tst = np.array([d.to_period("M") == tm for d in ds])
            saved = model_path("si", month_str)
            if saved.exists():
                model = joblib.load(saved)
                prib = model.predict_proba(X_all[tst])[:, 1]
                for j, idx in enumerate(np.where(tst)[0]):
                    sp[idx] = prib[j]
        for i, r in enumerate(si_recs):
            if sp[i] >= bt.SI_PROB:
                trades.append({
                    "entry": r["entry_idx"],
                    "exit": r["exit_ts"],
                    "pnl": r["pnl"],
                    "reason": r["reason"],
                    "type": "short_impulse",
                    "side": -1,
                    "entry_price": r["entry_price"],
                    "exit_price": r["exit_price"],
                    "_prob": float(sp[i]),
                    "_leg": "si",
                })

    return bt.merge_single_position(trades)


def backtest_entries_for_day(day: str, d1m: pd.DataFrame) -> list[Entry]:
    """ML-filtered backtest entries for one HKT day (single slot, no cross-leg overlap)."""
    day_start, day_end = _hkt_day_bounds(day)
    entries: list[Entry] = []
    for tr in _ml_trades_window(d1m):
        ts = pd.Timestamp(tr["entry"]).tz_convert("UTC")
        if not (day_start <= ts <= day_end):
            continue
        entries.append(Entry(tr["_leg"], ts, tr.get("_prob"), "backtest"))
    return sorted(entries, key=lambda e: e.ts)


def csv_entries_for_day(day: str) -> list[Entry]:
    if not CSV_PATH.exists():
        return []
    tdf = pd.read_csv(CSV_PATH)
    tdf["entry"] = pd.to_datetime(tdf["entry"], utc=True)
    day_start, day_end = _hkt_day_bounds(day)
    mask = (tdf["entry"] >= day_start) & (tdf["entry"] <= day_end)
    out = []
    for _, r in tdf[mask].iterrows():
        leg = LEG_MAP.get(str(r.get("type", "")), str(r.get("type", "")))
        out.append(Entry(leg, r["entry"], source="csv"))
    return sorted(out, key=lambda e: e.ts)


def compare_day(day: str, d1m: pd.DataFrame) -> dict:
    day_start, day_end = _hkt_day_bounds(day)
    live_combined = _decisions_to_entries(replay_entries(d1m, day_start, day_end))
    live_by_leg: list[Entry] = []
    for leg in ('wr90', 'ret', 'si'):
        live_by_leg.extend(_decisions_to_entries(replay_leg_entries(d1m, day_start, day_end, leg)))
    live_by_leg = sorted(live_by_leg, key=lambda e: e.ts)
    bt_entries = backtest_entries_for_day(day, d1m)
    csv_entries = csv_entries_for_day(day)

    # Single-slot: combined replay vs merged backtest (matches live IG).
    live_keys = {e.key() for e in live_combined}
    bt_keys = {e.key() for e in bt_entries}

    matched = live_keys & bt_keys
    missed = bt_keys - live_keys
    extra = live_keys - bt_keys

    per_leg_keys = {e.key() for e in live_by_leg}
    per_leg_extra = per_leg_keys - bt_keys
    per_leg_miss = bt_keys - per_leg_keys

    return {
        "day": day,
        "live": live_by_leg,
        "live_combined": live_combined,
        "backtest": bt_entries,
        "csv": csv_entries,
        "matched": matched,
        "missed": missed,
        "extra": extra,
        "per_leg_miss": per_leg_miss,
        "per_leg_extra": per_leg_extra,
        "ok": len(missed) == 0 and len(extra) == 0,
    }


def format_report(results: list[dict]) -> str:
    lines = [
        "=" * 72,
        "  OIL PARITY CHECK — single-slot replay vs backtest (no cross-leg overlap)",
        f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Thresholds: WR90≥{LONG_WR_ML_TH}  Ret≥{RET_ML_TH}  SI≥{SI_PROB}",
        "=" * 72,
        "",
    ]
    all_ok = True
    for r in results:
        status = "✓ MATCH" if r["ok"] else "✗ MISMATCH"
        if not r["ok"]:
            all_ok = False
        lines.append(f"--- {r['day']} ({status}) ---")
        lines.append(f"  Backtest (1 slot): {len(r['backtest'])}")
        lines.append(f"  Live replay (1 slot): {len(r['live_combined'])}")
        if r["live"]:
            lines.append(f"  Per-leg replay (diag): {len(r['live'])}")
        if r["csv"]:
            lines.append(f"  CSV (cached BT)  : {len(r['csv'])}")
        lines.append(f"  Matched          : {len(r['matched'])}")

        if r["missed"]:
            lines.append("  MISS (backtest has, live replay lacks):")
            for leg, ts in sorted(r["missed"]):
                hkt = pd.Timestamp(ts).tz_convert("Asia/Hong_Kong").strftime("%H:%M HKT")
                lines.append(f"    {leg:5s} @ {hkt}")

        if r["extra"]:
            lines.append("  EXTRA (live replay has, backtest lacks):")
            for leg, ts in sorted(r["extra"]):
                hkt = pd.Timestamp(ts).tz_convert("Asia/Hong_Kong").strftime("%H:%M HKT")
                lines.append(f"    {leg:5s} @ {hkt}")

        if r["ok"] and r["backtest"]:
            lines.append("  Entries:")
            for e in r["backtest"]:
                hkt = e.ts.tz_convert("Asia/Hong_Kong").strftime("%H:%M HKT")
                prob = f" prob={e.prob:.3f}" if e.prob is not None else ""
                lines.append(f"    {e.leg:5s} @ {hkt}{prob}")
        elif r["ok"]:
            lines.append("  (no entries — both agree)")
        if r.get("per_leg_miss") or r.get("per_leg_extra"):
            if r.get("per_leg_miss"):
                lines.append("  Note — per-leg replay would MISS:")
                for leg, ts in sorted(r["per_leg_miss"]):
                    hkt = pd.Timestamp(ts).tz_convert("Asia/Hong_Kong").strftime("%H:%M HKT")
                    lines.append(f"    {leg:5s} @ {hkt}")
            if r.get("per_leg_extra"):
                lines.append("  Note — per-leg replay would EXTRA:")
                for leg, ts in sorted(r["per_leg_extra"]):
                    hkt = pd.Timestamp(ts).tz_convert("Asia/Hong_Kong").strftime("%H:%M HKT")
                    lines.append(f"    {leg:5s} @ {hkt}")
        lines.append("")

    lines.append("=" * 72)
    lines.append(f"  OVERALL: {'ALL DAYS MATCH ✓' if all_ok else 'MISMATCHES FOUND — check live bot'}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Quick checks:")
    lines.append("  cat runtime/oil_bot_health.json     # live bot status (updated each minute)")
    lines.append("  tail -30 runtime/oil_live_bot.log     # recent scoring")
    lines.append("  python3 _check_oil_parity.py          # re-run parity")
    return "\n".join(lines)


def run_parity(start_day: Optional[str] = None, end_day: Optional[str] = None) -> str:
    if not end_day:
        yesterday_hkt = (datetime.now(timezone.utc) + timedelta(hours=8) - timedelta(days=1)).date()
        end_day = str(yesterday_hkt)
    if not start_day:
        start_day = end_day

    days = pd.date_range(start_day, end_day, freq="D").strftime("%Y-%m-%d").tolist()
    end_dt = _hkt_day_bounds(days[-1])[1].to_pydatetime() + timedelta(hours=1)
    d1m = load_mysql_bars(warmup_days=90, end=end_dt)

    results = [compare_day(d, d1m) for d in days]
    report = format_report(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    sd = sys.argv[1] if len(sys.argv) > 1 else None
    ed = sys.argv[2] if len(sys.argv) > 2 else sd
    print(run_parity(sd, ed))
