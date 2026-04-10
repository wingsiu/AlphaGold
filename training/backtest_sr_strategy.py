#!/usr/bin/env python3
"""Walk-forward Support/Resistance strategy backtest — hybrid design.

Architecture (no lookahead bias):
  S/R levels : built from Databento trade ticks (volume profile peaks).
               Each test day uses ONLY ticks from the prior `lookback_days` window.
  Price bars : loaded from MySQL 1-minute gold_prices table (253k rows).
               Used for entry/exit simulation — much denser than Databento ticks.

Entry logic (mean-reversion):
  Long  – 1m bar low touches support within buffer AND close >= level + reject_margin
  Short – 1m bar high touches resistance within buffer AND close <= level - reject_margin

Exit logic:
  TP   – high (long) or low (short) reaches entry ± tp_span_frac × span
         span = nearest paired level distance; fallback = tp_span_frac × yesterday's bar range
  SL   – low  (long) or high (short) reaches entry ∓ sl_pts  (fixed dollar stop)
  Timeout – max_hold_bars bars elapsed since entry

Usage:
  python3 training/backtest_sr_strategy.py \\
    --trades-csv training/l2/trades_gc_c0_20250520_20260204.csv \\
    --table gold_prices \\
    --start-date 2025-05-20 \\
    --end-date   2026-02-04 \\
    --out-csv training/l2/backtest_sr_trades_gc_c0.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import DataLoader
from databento_l2_sr import compute_volume_profile, sr_from_volume_profile


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_tick_trades(trades_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(trades_csv, usecols=["ts_event", "price", "size"])
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, format="mixed", errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"]  = pd.to_numeric(df["size"],  errors="coerce")
    df = df.dropna(subset=["ts_event", "price", "size"]).sort_values("ts_event").reset_index(drop=True)
    return df


def load_price_bars(table: str, start_date: str, end_date: str) -> pd.DataFrame:
    raw = DataLoader().load_data(table, start_date=start_date, end_date=end_date)
    df = pd.DataFrame({
        "ts":     pd.to_datetime(raw["timestamp"], unit="ms", utc=True),
        "open":   raw["openPrice"].astype(float),
        "high":   raw["highPrice"].astype(float),
        "low":    raw["lowPrice"].astype(float),
        "close":  raw["closePrice"].astype(float),
        "volume": raw["lastTradedVolume"].astype(float),
    }).set_index("ts").sort_index()
    df = df.dropna(subset=["open","high","low","close"])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Level builder
# ──────────────────────────────────────────────────────────────────────────────

def build_levels(
    window_ticks: pd.DataFrame,
    tick_size: float,
    top_n: int,
    merge_ticks: int,
    min_prom_pct: float,
) -> pd.DataFrame:
    if len(window_ticks) < 50:
        return pd.DataFrame(columns=["price","type"])
    profile = compute_volume_profile(window_ticks, price_col="price", size_col="size", tick_size=tick_size)
    if profile.empty:
        return pd.DataFrame(columns=["price","type"])
    return sr_from_volume_profile(profile, tick_size=tick_size, top_n=top_n,
                                  merge_ticks=merge_ticks, min_prom_pct=min_prom_pct)


# ──────────────────────────────────────────────────────────────────────────────
# Trade dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    side:          str
    entry_time:    pd.Timestamp
    exit_time:     pd.Timestamp
    entry_price:   float
    exit_price:    float
    level_price:   float
    level_span:    float          # resistance − support used to size the TP
    hold_bars:     int
    reason:        str
    pnl_points:    float
    pnl_pct:       float


# ──────────────────────────────────────────────────────────────────────────────
# Core backtest
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(
    tick_trades:    pd.DataFrame,
    price_bars:     pd.DataFrame,
    *,
    lookback_days:  int   = 20,
    tick_size:      float = 0.10,
    top_n:          int   = 15,
    merge_ticks:    int   = 30,
    min_prom_pct:   float = 0.02,
    touch_buffer:   float = 0.15,   # $ buffer for touch detection
    reject_margin:  float = 0.05,   # $ bar-close must exceed level by this
    tp_span_frac:   float = 0.60,   # TP = entry ± tp_span_frac × (resistance − support)
    sl_pts:         float = 17.0,   # SL = fixed dollar distance from entry
    max_hold_bars:  int   = 30,
    max_trades_per_level_per_day: int = 1,
) -> list[Trade]:
    out: list[Trade] = []

    bar_vals = price_bars[["open","high","low","close"]].to_numpy(dtype="float64")
    bar_idx  = price_bars.index

    unique_days = pd.Series(bar_idx.normalize().unique()).sort_values().reset_index(drop=True)
    if len(unique_days) <= lookback_days:
        return out

    # Pre-compute daily high-low range from bar data (used as TP fallback)
    daily_range: dict[pd.Timestamp, float] = (
        price_bars.groupby(price_bars.index.normalize())
        .apply(lambda g: float(g["high"].max() - g["low"].min()))
        .to_dict()
    )

    # Cache levels per day to avoid recomputing every bar
    levels_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    # Track which level was traded today to cap trades_per_level
    daily_trades: dict[pd.Timestamp, dict[float, int]] = {}

    i = 0   # bar pointer (global)
    in_trade = False
    entry_price = 0.0
    entry_side  = ""
    entry_level = 0.0
    entry_span  = 0.0
    entry_bar_i = 0
    tp_price = sl_price = 0.0

    while i < len(price_bars):
        bar_time = bar_idx[i]
        day = bar_time.normalize()

        # Skip days in the warmup period
        day_pos = (unique_days == day).argmax()
        if unique_days[day_pos] != day or day_pos < lookback_days:
            i += 1
            continue

        # Build / cache levels for this day
        if day not in levels_cache:
            train_end   = day
            train_start = unique_days[day_pos - lookback_days]
            ts_start = pd.Timestamp(train_start).tz_convert("UTC") if getattr(train_start, "tzinfo", None) else pd.Timestamp(train_start, tz="UTC")
            ts_end   = pd.Timestamp(train_end  ).tz_convert("UTC") if getattr(train_end,   "tzinfo", None) else pd.Timestamp(train_end,   tz="UTC")
            mask = (tick_trades["ts_event"] >= ts_start) & \
                   (tick_trades["ts_event"] <  ts_end)
            window_ticks = tick_trades.loc[mask, ["price","size"]]
            levels_cache[day] = build_levels(
                window_ticks, tick_size=tick_size, top_n=top_n,
                merge_ticks=merge_ticks, min_prom_pct=min_prom_pct,
            )
            daily_trades[day] = {}

        levels = levels_cache[day]
        traded_today = daily_trades[day]

        o, h, l, c = bar_vals[i]

        # ── Exit open trade ───────────────────────────────────────────────
        if in_trade:
            exit_price = None
            reason     = None

            if entry_side == "long":
                if l <= sl_price and h >= tp_price:
                    exit_price, reason = sl_price, "sl_both"
                elif l <= sl_price:
                    exit_price, reason = sl_price, "sl"
                elif h >= tp_price:
                    exit_price, reason = tp_price, "tp"
            else:
                if h >= sl_price and l <= tp_price:
                    exit_price, reason = sl_price, "sl_both"
                elif h >= sl_price:
                    exit_price, reason = sl_price, "sl"
                elif l <= tp_price:
                    exit_price, reason = tp_price, "tp"

            if exit_price is None and (i - entry_bar_i) >= max_hold_bars:
                exit_price, reason = c, "timeout"

            if exit_price is not None:
                pnl_pts = (exit_price - entry_price) if entry_side == "long" else (entry_price - exit_price)
                out.append(Trade(
                    side=entry_side,
                    entry_time=bar_idx[entry_bar_i],
                    exit_time=bar_time,
                    entry_price=entry_price,
                    exit_price=float(exit_price),
                    level_price=entry_level,
                    level_span=entry_span,
                    hold_bars=i - entry_bar_i,
                    reason=str(reason),
                    pnl_points=float(pnl_pts),
                    pnl_pct=float(pnl_pts / entry_price * 100.0) if entry_price else 0.0,
                ))
                in_trade = False
            i += 1
            continue

        # ── Look for entry ────────────────────────────────────────────────
        if levels.empty:
            i += 1
            continue

        supports    = levels.loc[levels["type"] == "support",    "price"].tolist()
        resistances = levels.loc[levels["type"] == "resistance", "price"].tolist()

        signal_side  = None
        signal_level = None

        # Long: low touched near support AND close confirms rejection up
        for lv in sorted(supports, reverse=True):
            count = traded_today.get(float(lv), 0)
            if count >= max_trades_per_level_per_day:
                continue
            if l <= (lv + touch_buffer) and c >= (lv + reject_margin):
                signal_side  = "long"
                signal_level = float(lv)
                break

        # Short: high touched near resistance AND close confirms rejection down
        if signal_side is None:
            for lv in sorted(resistances):
                count = traded_today.get(float(lv), 0)
                if count >= max_trades_per_level_per_day:
                    continue
                if h >= (lv - touch_buffer) and c <= (lv - reject_margin):
                    signal_side  = "short"
                    signal_level = float(lv)
                    break

        if signal_side is not None:
            # ── Compute level span ────────────────────────────────────────
            if signal_side == "long":
                paired_r = sorted([r for r in resistances if r > signal_level])
                span = (paired_r[0] - signal_level) if paired_r else 0.0
            else:
                paired_s = sorted([s for s in supports if s < signal_level], reverse=True)
                span = (signal_level - paired_s[0]) if paired_s else 0.0

            # Fallback: no paired level → use 0.6 × yesterday's bar range
            if span <= 0.0:
                prev_day = unique_days[day_pos - 1] if day_pos > 0 else None
                span = tp_span_frac * daily_range.get(prev_day, 0.0) if prev_day is not None else 0.0

            if span <= 0.0:
                i += 1
                continue

            entry_price = c
            entry_side  = signal_side
            entry_level = signal_level
            entry_span  = span
            entry_bar_i = i
            traded_today[signal_level] = traded_today.get(signal_level, 0) + 1

            # TP = entry ± tp_span_frac × span;  SL = entry ∓ sl_pts (fixed $)
            if signal_side == "long":
                tp_price = entry_price + tp_span_frac * span
                sl_price = entry_price - sl_pts
            else:
                tp_price = entry_price - tp_span_frac * span
                sl_price = entry_price + sl_pts
            in_trade = True

        i += 1

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def summarise(trades: list[Trade]) -> dict:
    if not trades:
        return dict(trades=0, win_rate=0.0, total_points=0.0, avg_points=0.0,
                    avg_hold_bars=0.0, max_win=0.0, max_loss=0.0,
                    max_drawdown=0.0, profit_factor=0.0)
    pnl   = pd.Series([t.pnl_points for t in trades], dtype="float64")
    holds = pd.Series([t.hold_bars  for t in trades], dtype="float64")
    eq    = pnl.cumsum()
    dd    = (eq - eq.cummax()).min()
    wins  = pnl[pnl > 0].sum()
    loss  = pnl[pnl < 0].abs().sum()
    pf    = float(wins / loss) if loss > 0 else float("inf")
    return dict(
        trades=float(len(trades)),
        win_rate=float((pnl > 0).mean() * 100.0),
        total_points=float(pnl.sum()),
        avg_points=float(pnl.mean()),
        avg_hold_bars=float(holds.mean()),
        max_win=float(pnl.max()),
        max_loss=float(pnl.min()),
        max_drawdown=float(dd),
        profit_factor=float(pf),
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward SR backtest (Databento levels + MySQL bars)")

    # Data sources
    p.add_argument("--trades-csv",  required=True,          help="Databento trades CSV")
    p.add_argument("--table",       default="gold_prices",  help="MySQL price table")
    p.add_argument("--start-date",  default="2025-05-20")
    p.add_argument("--end-date",    default="2026-02-04")
    p.add_argument("--out-csv",     default="training/l2/backtest_sr_trades_gc_c0.csv")

    # Level building
    p.add_argument("--lookback-days",  type=int,   default=20)
    p.add_argument("--tick-size",      type=float, default=0.10)
    p.add_argument("--top-n",          type=int,   default=15)
    p.add_argument("--merge-ticks",    type=int,   default=30)
    p.add_argument("--min-prom-pct",   type=float, default=0.02)

    # Entry
    p.add_argument("--touch-buffer",   type=float, default=0.15,  help="$ distance for touch")
    p.add_argument("--reject-margin",  type=float, default=0.05,  help="$ bar close must exceed level")
    p.add_argument("--max-trades-per-level-per-day", type=int, default=2)

    # Exit
    p.add_argument("--tp-span-frac",  type=float, default=0.60,
                   help="TP = entry ± frac × (resistance − support)")
    p.add_argument("--sl-pts",        type=float, default=17.0,
                   help="SL = fixed dollar distance from entry")
    p.add_argument("--max-hold-bars",  type=int,   default=30)

    return p


def main() -> int:
    args = build_parser().parse_args()

    print("Loading Databento tick trades …")
    tick_trades = load_tick_trades(Path(args.trades_csv))
    print(f"  {len(tick_trades):,} ticks  ({tick_trades['ts_event'].min().date()} → {tick_trades['ts_event'].max().date()})")

    print("Loading 1m price bars from MySQL …")
    price_bars = load_price_bars(args.table, args.start_date, args.end_date)
    print(f"  {len(price_bars):,} bars  ({price_bars.index.min()} → {price_bars.index.max()})")

    print("Running walk-forward backtest …")
    backtest_trades = run_backtest(
        tick_trades, price_bars,
        lookback_days  = args.lookback_days,
        tick_size      = args.tick_size,
        top_n          = args.top_n,
        merge_ticks    = args.merge_ticks,
        min_prom_pct   = args.min_prom_pct,
        touch_buffer   = args.touch_buffer,
        reject_margin  = args.reject_margin,
        tp_span_frac   = args.tp_span_frac,
        sl_pts         = args.sl_pts,
        max_hold_bars  = args.max_hold_bars,
        max_trades_per_level_per_day = args.max_trades_per_level_per_day,
    )

    out_df = pd.DataFrame([t.__dict__ for t in backtest_trades])
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    stats = summarise(backtest_trades)
    print("\n── Backtest Results ──────────────────────────────────────────")
    print(f"  Trades:            {int(stats['trades'])}")
    print(f"  Win rate:          {stats['win_rate']:.2f}%")
    print(f"  Total points:      {stats['total_points']:.2f}")
    print(f"  Avg pts/trade:     {stats['avg_points']:.4f}")
    print(f"  Profit factor:     {stats['profit_factor']:.3f}")
    print(f"  Max drawdown:      {stats['max_drawdown']:.2f} pts")
    print(f"  Avg hold bars:     {stats['avg_hold_bars']:.1f}")
    print(f"  Max win:           {stats['max_win']:.2f}")
    print(f"  Max loss:          {stats['max_loss']:.2f}")
    print(f"  Saved:             {out_path}")

    if not out_df.empty:
        print("\n── Breakdown by side ─────────────────────────────────────────")
        print(out_df.groupby("side")["pnl_points"].agg(["count","mean","sum"]).to_string())
        print("\n── Breakdown by exit reason ──────────────────────────────────")
        print(out_df.groupby("reason")["pnl_points"].agg(["count","mean","sum"]).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

