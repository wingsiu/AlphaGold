"""v16 gold replay vs combined_run backtest parity."""
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

from v16.config.gold_config import BACKTEST, GOLD_TRAIN_START
from v16.gold.combined_run import run_gold_v16_combined, save_combined_trades
from v16.gold.signal_engine import replay_portfolio

OUTPUT_PATH = REPO / BACKTEST["parity_csv"]
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
    s = pd.Timestamp(start)
    warm = s - pd.Timedelta(days=400)
    floor = pd.Timestamp(GOLD_TRAIN_START)
    return str(min(warm, floor).date())


def _filter_trades(trades: list[dict], start: str, end: str) -> list[dict]:
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    return [
        t for t in trades
        if t0 <= pd.Timestamp(t["entry"]).tz_convert("UTC") <= t1
    ]


def _backtest_trades(start: str, end: str, *, use_csv: bool = True) -> tuple[list[dict], dict | None]:
    if use_csv and CSV_PATH.exists():
        tdf = pd.read_csv(CSV_PATH)
        tdf["entry"] = pd.to_datetime(tdf["entry"], utc=True)
        filtered = _filter_trades(tdf.to_dict("records"), start, end)
        if filtered:
            return filtered, None
    ds = _data_load_start(start)
    merged, leg_stats = run_gold_v16_combined(ds, end, oos_start=start, verbose=True)
    save_combined_trades(merged)
    return _filter_trades(merged, start, end), leg_stats


def compare_period(start: str, end: str, *, use_csv: bool = True) -> dict:
    print(f"Replay portfolio ({start} → {end})...", flush=True)
    replay_trades = replay_portfolio(start, end, data_start=_data_load_start(start), verbose=True)
    replay_trades = _filter_trades(replay_trades, start, end)
    print(f"  Replay done: {len(replay_trades)} trades", flush=True)

    print("Loading backtest reference...", flush=True)
    bt_trades, leg_stats = _backtest_trades(start, end, use_csv=use_csv)
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
        "  GOLD v16 PARITY — replay vs combined_run backtest",
        f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Period: {r['start']} → {r['end']}",
        "  Stack: hybrid patterns + energetic + momentum + dip short",
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
                lines.append(f"    {leg:20s} @ {entry}")
        if r["replay_only"]:
            lines.append(f"\n  Replay-only ({len(r['replay_only'])}):")
            for leg, entry in sorted(r["replay_only"])[:30]:
                lines.append(f"    {leg:20s} @ {entry}")
        if r["pnl_mismatch"]:
            lines.append(f"\n  PnL mismatch ({len(r['pnl_mismatch'])}):")
            for (leg, entry), pr, pb in r["pnl_mismatch"][:20]:
                lines.append(f"    {leg:20s} @ {entry}  replay={pr:+.1f}  bt={pb:+.1f}")

    lines.append("\n--- Replay leg breakdown ---")
    for leg, s in sorted(_leg_summary(r["replay_trades"]).items()):
        lines.append(f"  {leg:20s}  {s['n']:4d}t  PnL={s['pnl']:+.1f}")

    lines.append("\n--- Backtest leg breakdown ---")
    for leg, s in sorted(_leg_summary(r["bt_trades"]).items()):
        lines.append(f"  {leg:20s}  {s['n']:4d}t  PnL={s['pnl']:+.1f}")

    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def run_parity(start: Optional[str] = None, end: Optional[str] = None) -> str:
    start = start or BACKTEST["default_start"]
    end = end or BACKTEST["default_end"]
    r = compare_period(start, end)
    report = format_report(r)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    if r["replay_trades"]:
        pd.DataFrame(r["replay_trades"]).to_csv(REPO / "runtime/gold_v16_replay_trades.csv", index=False)
    return report


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    sd = args[0] if args else None
    ed = args[1] if len(args) > 1 else sd
    print(run_parity(sd, ed))
