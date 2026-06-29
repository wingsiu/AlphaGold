"""v16 oil replay vs combined_run backtest parity."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from v16.config.oil_config import BACKTEST, OIL_LEG_MODELS
from v16.data.load_oil import load_oil_1m
from v16.oil.combined_run import run_oil_v16_combined
from v16.oil.signal_engine import replay_portfolio

OUTPUT_PATH = REPO / "runtime" / "oil_v16_parity_latest.txt"
CSV_PATH = REPO / BACKTEST["trades_csv"]


@dataclass
class TradeKey:
    leg: str
    entry: pd.Timestamp
    pnl: float

    @classmethod
    def from_trade(cls, t: dict) -> "TradeKey":
        leg = str(t.get("_leg", t.get("type", "?")))
        entry = pd.Timestamp(t["entry"]).tz_convert("UTC").floor("min")
        return cls(leg, entry, round(float(t["pnl"]), 2))

    def key(self) -> tuple:
        return self.leg, self.entry


def _data_load_start(start: str) -> str:
    """Load enough history for 14D WF training (matches full backtest window)."""
    from v16.config.oil_config import BACKTEST, OIL_ML_CONFIG

    s = pd.Timestamp(start)
    warm = s - pd.Timedelta(days=int(OIL_ML_CONFIG["train_days"]) + 30)
    floor = pd.Timestamp(BACKTEST.get("default_start", "2024-01-01"))
    wf = pd.Timestamp(OIL_ML_CONFIG["wf_start"])
    return str(min(warm, floor, wf).date())


def _filter_trades(trades: list[dict], start: str, end: str) -> list[dict]:
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    out = []
    for t in trades:
        entry = pd.Timestamp(t["entry"]).tz_convert("UTC")
        if t0 <= entry <= t1:
            out.append(t)
    return out


def _load_d1m(start: str, end: str) -> pd.DataFrame:
    return load_oil_1m(_data_load_start(start), end)


def _backtest_trades(
    start: str,
    end: str,
    *,
    wr90_exit: str = "struct_hold",
    use_csv: bool = True,
) -> tuple[list[dict], dict | None]:
    """Backtest reference trades — from CSV if fresh, else run combined_run."""
    if use_csv and CSV_PATH.exists():
        tdf = pd.read_csv(CSV_PATH)
        tdf["entry"] = pd.to_datetime(tdf["entry"], utc=True)
        trades = tdf.to_dict("records")
        filtered = _filter_trades(trades, start, end)
        if filtered:
            return filtered, None
    data_start = _data_load_start(start)
    merged, leg_stats = run_oil_v16_combined(data_start, end, wr90_exit=wr90_exit)
    return _filter_trades(merged, start, end), leg_stats


def compare_period(
    start: str,
    end: str,
    *,
    wr90_exit: str = "struct_hold",
    use_csv: bool = True,
) -> dict:
    """Run replay + backtest on same window; compare merged trades."""
    print(f"Loading data ({_data_load_start(start)} → {end})...", flush=True)
    d1m = _load_d1m(start, end)
    t_start = pd.Timestamp(start, tz="UTC")
    t_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)

    print(f"Replay portfolio ({start} → {end})...", flush=True)
    replay_trades = replay_portfolio(d1m, t_start, t_end, wr90_exit=wr90_exit)
    replay_trades = _filter_trades(replay_trades, start, end)
    print(f"  Replay done: {len(replay_trades)} trades", flush=True)

    print("Loading backtest reference...", flush=True)
    bt_trades, leg_stats = _backtest_trades(start, end, wr90_exit=wr90_exit, use_csv=use_csv)
    if leg_stats is None and use_csv:
        print(f"  (from {CSV_PATH})", flush=True)

    print(f"  Backtest ref: {len(bt_trades)} trades", flush=True)

    replay_keys = {TradeKey.from_trade(t).key(): t for t in replay_trades}
    bt_keys = {TradeKey.from_trade(t).key(): t for t in bt_trades}

    matched = set(replay_keys) & set(bt_keys)
    replay_only = set(replay_keys) - set(bt_keys)
    bt_only = set(bt_keys) - set(replay_keys)

    pnl_replay = sum(t["pnl"] for t in replay_trades)
    pnl_bt = sum(t["pnl"] for t in bt_trades)

    pnl_mismatch = []
    for k in matched:
        pr = round(float(replay_keys[k]["pnl"]), 2)
        pb = round(float(bt_keys[k]["pnl"]), 2)
        if abs(pr - pb) > 0.05:
            pnl_mismatch.append((k, pr, pb))

    return {
        "start": start,
        "end": end,
        "wr90_exit": wr90_exit,
        "replay_n": len(replay_trades),
        "bt_n": len(bt_trades),
        "replay_pnl": pnl_replay,
        "bt_pnl": pnl_bt,
        "matched": len(matched),
        "replay_only": replay_only,
        "bt_only": bt_only,
        "pnl_mismatch": pnl_mismatch,
        "leg_stats": leg_stats,
        "replay_trades": replay_trades,
        "bt_trades": bt_trades,
        "ok": not replay_only and not bt_only and not pnl_mismatch,
    }


def _leg_summary(trades: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in trades:
        leg = str(t.get("_leg", t.get("type", "?")))
        out.setdefault(leg, {"n": 0, "pnl": 0.0})
        out[leg]["n"] += 1
        out[leg]["pnl"] += float(t["pnl"])
    return out


def format_report(r: dict) -> str:
    lines = [
        "=" * 72,
        "  OIL v16 PARITY — replay vs combined_run backtest",
        f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Period: {r['start']} → {r['end']}  |  WR90 exit: {r['wr90_exit']}",
        f"  Models: {OIL_LEG_MODELS}",
        "=" * 72,
        "",
        f"  Replay:   {r['replay_n']:4d} trades  PnL={r['replay_pnl']:+.1f}",
        f"  Backtest: {r['bt_n']:4d} trades  PnL={r['bt_pnl']:+.1f}",
        f"  Matched entries: {r['matched']}",
        "",
    ]

    if r["ok"]:
        lines.append("  ✓ FULL MATCH — replay and backtest agree")
    else:
        lines.append("  ✗ MISMATCH")
        if r["bt_only"]:
            lines.append(f"\n  Backtest-only ({len(r['bt_only'])}):")
            for leg, entry in sorted(r["bt_only"])[:30]:
                lines.append(f"    {leg:12s} @ {entry}")
            if len(r["bt_only"]) > 30:
                lines.append(f"    ... +{len(r['bt_only']) - 30} more")
        if r["replay_only"]:
            lines.append(f"\n  Replay-only ({len(r['replay_only'])}):")
            for leg, entry in sorted(r["replay_only"])[:30]:
                lines.append(f"    {leg:12s} @ {entry}")
            if len(r["replay_only"]) > 30:
                lines.append(f"    ... +{len(r['replay_only']) - 30} more")
        if r["pnl_mismatch"]:
            lines.append(f"\n  PnL mismatch on matched entries ({len(r['pnl_mismatch'])}):")
            for (leg, entry), pr, pb in r["pnl_mismatch"][:20]:
                lines.append(f"    {leg:12s} @ {entry}  replay={pr:+.1f}  bt={pb:+.1f}")

    lines.append("\n--- Replay leg breakdown ---")
    for leg, s in sorted(_leg_summary(r["replay_trades"]).items()):
        lines.append(f"  {leg:12s}  {s['n']:4d}t  PnL={s['pnl']:+.1f}")

    lines.append("\n--- Backtest leg breakdown (merged) ---")
    for leg, s in sorted(_leg_summary(r["bt_trades"]).items()):
        lines.append(f"  {leg:12s}  {s['n']:4d}t  PnL={s['pnl']:+.1f}")

    if r.get("leg_stats"):
        lines.append("\n--- Backtest pre-merge ---")
        for leg in ("wr90", "ret", "ret_short", "long_ret", "si"):
            if leg in r["leg_stats"]:
                st = r["leg_stats"][leg]
                lines.append(f"  {leg:12s}  {st['trades']:4d}t  PnL={st['pnl']:+.1f}")

    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def run_parity(
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    wr90_exit: str = "struct_hold",
) -> str:
    start = start or BACKTEST["default_start"]
    end = end or BACKTEST["default_end"]
    r = compare_period(start, end, wr90_exit=wr90_exit)
    report = format_report(r)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    if r["replay_trades"]:
        pd.DataFrame(r["replay_trades"]).to_csv(REPO / "runtime" / "oil_v16_replay_trades.csv", index=False)
    return report


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    sd = args[0] if args else None
    ed = args[1] if len(args) > 1 else sd
    wr90_exit = "fixed_tpsl" if "--fixed-tpsl" in flags else "struct_hold"
    print(run_parity(sd, ed, wr90_exit=wr90_exit))
