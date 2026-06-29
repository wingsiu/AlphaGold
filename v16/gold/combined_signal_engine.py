"""Gold v16 combined portfolio — hybrid + momentum + dip (live + replay)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from v16.config import v16_config
from v16.config.gold_config import DIP_SHORT, GOLD_TRAIN_START, MOMENTUM
from v16.data.load_gold import load_gold_1m
from v16.gold.merge import GOLD_LEG_PRIORITY, merge_gold_trades
from v16.gold.v16_legs import run_v16_legs
from xgboost_filter_model.hybrid_live import LiveSignal

SPREAD_DEFAULT = 0.25


@dataclass
class GoldV16Decision:
    leg: str  # pattern | energetic | v16_momentum | v16_dip_short
    entry_ts: pd.Timestamp
    side: int
    probability: float = 0.0
    would_enter: bool = True
    reason: str = "pass"
    pattern_name: str | None = None
    tp: float | None = None
    sl: float | None = None
    horizon: int | None = None
    exit_mode: str = "fixed_tpsl"  # fixed_tpsl | struct_hold
    impulse_stop: float | None = None
    entry_struct_trend: int | None = None
    detail: str = ""


@dataclass
class GoldV16LegCache:
    """Precomputed v16 momentum/dip entry times (pre-merge leg signals)."""

    df_end: pd.Timestamp | None = None
    df_len: int = 0
    last_refresh: pd.Timestamp | None = None
    momentum: dict[str, dict] = field(default_factory=dict)  # entry_iso -> meta
    dip: dict[str, dict] = field(default_factory=dict)

    def is_stale(self, df: pd.DataFrame, *, max_age_hours: float = 6.0) -> bool:
        if df.empty or self.df_end is None or self.last_refresh is None:
            return True
        age_h = (pd.Timestamp.now(tz="UTC") - self.last_refresh).total_seconds() / 3600.0
        return age_h >= max_age_hours

    def refresh(self, df: pd.DataFrame, *, oos_start: str | pd.Timestamp | None = None) -> None:
        if df.empty:
            return
        df = ensure_bid_ask_df(df)
        oos = pd.Timestamp(oos_start or (df.index.max() - pd.Timedelta(days=90)))
        if oos.tzinfo is None:
            oos = oos.tz_localize("UTC")
        else:
            oos = oos.tz_convert("UTC")
        mom_trades, dip_trades = run_v16_legs(df, oos)
        self.momentum = {}
        self.dip = {}
        for t in mom_trades:
            ts = pd.Timestamp(t["entry"]).tz_convert("UTC").floor("min")
            self.momentum[ts.isoformat()] = t
        for t in dip_trades:
            ts = pd.Timestamp(t["entry"]).tz_convert("UTC").floor("min")
            self.dip[ts.isoformat()] = t
        self.df_end = df.index.max()
        self.df_len = len(df)
        self.last_refresh = pd.Timestamp.now(tz="UTC")

    def momentum_at(self, ts: pd.Timestamp) -> dict | None:
        return self.momentum.get(pd.Timestamp(ts).tz_convert("UTC").floor("min").isoformat())

    def dip_at(self, ts: pd.Timestamp) -> dict | None:
        return self.dip.get(pd.Timestamp(ts).tz_convert("UTC").floor("min").isoformat())


def ensure_bid_ask_df(df: pd.DataFrame, spread: float = SPREAD_DEFAULT) -> pd.DataFrame:
    """Add bid/ask columns when live cache only has mid OHLC."""
    if "open_ask" in df.columns:
        out = df.copy()
    else:
        out = pd.DataFrame(index=df.index)
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                out[col] = df[col].astype(float)
        out["open_ask"] = out["open"] + spread
        out["open_bid"] = out["open"] - spread
        out["high_ask"] = out["high"] + spread
        out["high_bid"] = out["high"] - spread
        out["low_ask"] = out["low"] + spread
        out["low_bid"] = out["low"] - spread
        out["close_ask"] = out["close"] + spread
        out["close_bid"] = out["close"] - spread
        out["volume"] = df["volume"].astype(float) if "volume" in df.columns else 0.0
    out["mid"] = (out["close_ask"] + out["close_bid"]) / 2.0
    out["spread"] = out["close_ask"] - out["close_bid"]
    return out.sort_index()


def leg_priority(leg: str, pattern_name: str | None = None) -> int:
    if leg == "pattern" and pattern_name:
        return int(GOLD_LEG_PRIORITY.get(pattern_name, GOLD_LEG_PRIORITY.get("pattern", 10)))
    return int(GOLD_LEG_PRIORITY.get(leg, 9))


def _merge_sort_key(dec: GoldV16Decision) -> tuple:
    entry = pd.Timestamp(dec.entry_ts).tz_convert("UTC")
    return (entry, leg_priority(dec.leg, dec.pattern_name))


def decision_from_live_signal(sig: LiveSignal, entry_ts: pd.Timestamp) -> GoldV16Decision:
    return GoldV16Decision(
        leg=sig.source,
        entry_ts=pd.Timestamp(entry_ts).tz_convert("UTC"),
        side=int(sig.side),
        probability=float(sig.probability or 0.0),
        would_enter=True,
        pattern_name=sig.pattern_name,
        tp=float(sig.tp),
        sl=float(sig.sl),
        horizon=int(sig.horizon),
        exit_mode="fixed_tpsl",
    )


def decision_from_momentum_trade(trade: dict, entry_ts: pd.Timestamp) -> GoldV16Decision:
    side = int(trade.get("side", 1))
    cfg = v16_config.MOMENTUM_V16_WINNER_PRECLOSE
    horizon = int(MOMENTUM.get("horizon", 720))
    return GoldV16Decision(
        leg="v16_momentum",
        entry_ts=pd.Timestamp(entry_ts).tz_convert("UTC"),
        side=side,
        probability=float(MOMENTUM.get("ml_prob", 0.5)),
        would_enter=True,
        tp=None,
        sl=float(cfg.get("impulse_stop", {}).get("min_sl_pts", 25)),
        horizon=horizon,
        exit_mode="struct_hold",
        detail="preclose struct-hold",
    )


def decision_from_dip_trade(trade: dict, entry_ts: pd.Timestamp) -> GoldV16Decision:
    ex = v16_config.DIP_SHORT_RIP.get("execution", {})
    return GoldV16Decision(
        leg="v16_dip_short",
        entry_ts=pd.Timestamp(entry_ts).tz_convert("UTC"),
        side=-1,
        probability=float(DIP_SHORT.get("ml_prob", 0.7)),
        would_enter=True,
        tp=float(ex.get("tp", 35)),
        sl=float(ex.get("sl", 35)),
        horizon=int(ex.get("horizon", 45)),
        exit_mode="fixed_tpsl",
    )


def pick_combined_winner(candidates: list[GoldV16Decision]) -> GoldV16Decision | None:
    passing = [c for c in candidates if c.would_enter]
    if not passing:
        return None
    passing.sort(key=_merge_sort_key)
    return passing[0]


def gold_decision_to_live_signal(dec: GoldV16Decision) -> LiveSignal:
    tp = float(dec.tp or 0.0)
    sl = float(dec.sl or 25.0)
    horizon = int(dec.horizon or 720)
    if dec.exit_mode == "struct_hold":
        tp = float(MOMENTUM.get("horizon", 720))  # wide limit placeholder; exits via struct/horizon
        sl = sl
        horizon = horizon
    return LiveSignal(
        source=dec.leg,
        side=int(dec.side),
        tp=tp,
        sl=sl,
        horizon=horizon,
        probability=float(dec.probability),
        pattern_name=dec.pattern_name if dec.leg == "pattern" else dec.leg,
    )


def collect_minute_candidates(
    latest_ts: pd.Timestamp,
    *,
    pat_sig: LiveSignal | None,
    en_sig: LiveSignal | None,
    leg_cache: GoldV16LegCache,
    flat: bool,
) -> list[GoldV16Decision]:
    if not flat:
        return []
    ts = pd.Timestamp(latest_ts).tz_convert("UTC")
    out: list[GoldV16Decision] = []
    if pat_sig:
        out.append(decision_from_live_signal(pat_sig, ts))
    elif en_sig:
        out.append(decision_from_live_signal(en_sig, ts))
    mom = leg_cache.momentum_at(ts)
    if mom:
        out.append(decision_from_momentum_trade(mom, ts))
    dip = leg_cache.dip_at(ts)
    if dip:
        out.append(decision_from_dip_trade(dip, ts))
    return out


def evaluate_combined_minute(
    df: pd.DataFrame,
    latest_ts: pd.Timestamp,
    *,
    pat_sig: LiveSignal | None,
    en_sig: LiveSignal | None,
    leg_cache: GoldV16LegCache,
    flat: bool,
) -> dict[str, Any]:
    """Evaluate all v16 gold legs at one bar; return winner + leg flags."""
    if leg_cache.is_stale(df):
        leg_cache.refresh(df)
    candidates = collect_minute_candidates(
        latest_ts, pat_sig=pat_sig, en_sig=en_sig, leg_cache=leg_cache, flat=flat
    )
    winner = pick_combined_winner(candidates) if flat else None
    ts = pd.Timestamp(latest_ts).tz_convert("UTC")
    return {
        "latest_ts": str(ts),
        "flat": flat,
        "candidates": candidates,
        "winner": winner,
        "has_momentum": leg_cache.momentum_at(ts) is not None,
        "has_dip": leg_cache.dip_at(ts) is not None,
    }


def check_momentum_struct_exit(
    df: pd.DataFrame,
    latest_ts: pd.Timestamp,
    *,
    side: int,
    entry_price: float,
    entry_struct_trend: int | None,
    min_pnl: float = -1e9,
) -> tuple[bool, str]:
    """Return (should_close, reason) for struct-hold momentum exit."""
    from v16.backtest.position_sim import _structure_change_exit, _structure_trend_arr

    cfg = copy.deepcopy(v16_config.MOMENTUM_V16_WINNER_PRECLOSE)
    df = ensure_bid_ask_df(df)
    if latest_ts not in df.index:
        return False, ""
    i = df.index.get_loc(latest_ts)
    struct = _structure_trend_arr(df, cfg)
    if struct is None or entry_struct_trend is None:
        return False, ""
    close_bid = float(df.iloc[i]["close_bid"])
    close_ask = float(df.iloc[i]["close_ask"])
    hit = _structure_change_exit(
        int(side),
        float(entry_price),
        close_bid,
        close_ask,
        cur_trend=int(struct[i]),
        entry_trend=int(entry_struct_trend),
        min_pnl=min_pnl,
    )
    if hit is not None:
        return True, "structure_change"
    return False, ""


def replay_portfolio(
    start: str,
    end: str,
    *,
    data_start: str | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Replay combined portfolio — delegates to combined_run (matches backtest merge)."""
    from v16.gold.combined_run import run_gold_v16_combined

    ds = data_start or _data_load_start(start)
    merged, _ = run_gold_v16_combined(ds, end, oos_start=start, verbose=verbose)
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    return [
        t for t in merged
        if t0 <= pd.Timestamp(t["entry"]).tz_convert("UTC") <= t1
    ]


def replay_portfolio_from_df(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    hybrid_trades: list[dict] | None = None,
) -> list[dict]:
    """Replay from loaded 1m data — hybrid + v16 legs + merge."""
    from v16.gold.hybrid_legs import run_hybrid_legs

    df = ensure_bid_ask_df(df)
    oos = pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end).tz_convert("UTC") + pd.Timedelta(hours=23, minutes=59)
    hybrid = hybrid_trades
    if hybrid is None:
        hybrid = run_hybrid_legs(str(oos.date()), str(end_ts.date()), verbose=False)
    hybrid = [
        t for t in hybrid
        if oos <= pd.Timestamp(t["entry"]).tz_convert("UTC") <= end_ts
    ]
    mom_trades, dip_trades = run_v16_legs(df, oos)
    raw = hybrid + mom_trades + dip_trades
    return merge_gold_trades(raw)


def _data_load_start(start: str) -> str:
    s = pd.Timestamp(start)
    warm = s - pd.Timedelta(days=400)
    floor = pd.Timestamp(GOLD_TRAIN_START)
    return str(min(warm, floor).date())


def load_replay_data(start: str, end: str) -> pd.DataFrame:
    return load_gold_1m(_data_load_start(start), end)
