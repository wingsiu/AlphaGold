#!/usr/bin/env python3
"""Fresh re-analysis for spot gold using a new rule-based method.

Method: Volatility-adjusted breakout with trend filter (15-minute bars)
- Breakout trigger: close breaks prior N-bar high/low.
- Trend filter: EMA fast above/below EMA slow.
- Risk model: ATR-based take-profit and stop-loss.
- Execution: enter next bar open (ask for long, bid for short),
  exit on TP/SL intrabar checks with bid/ask-aware logic.

Validation protocol:
- Chronological split: first 70% train, last 30% test.
- Grid-search parameters on train only.
- Report out-of-sample test stats and save trades CSV.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import DataLoader


@dataclass(frozen=True)
class Params:
    lookback: int
    ema_fast: int
    ema_slow: int
    atr_period: int
    tp_atr: float
    sl_atr: float
    max_hold: int


def _prepare_bars_15m(raw: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
    df = pd.DataFrame(
        {
            "open": raw["openPrice"].astype(float).to_numpy(),
            "high": raw["highPrice"].astype(float).to_numpy(),
            "low": raw["lowPrice"].astype(float).to_numpy(),
            "close": raw["closePrice"].astype(float).to_numpy(),
            "open_ask": raw["openPrice_ask"].astype(float).to_numpy(),
            "high_ask": raw["highPrice_ask"].astype(float).to_numpy(),
            "low_ask": raw["lowPrice_ask"].astype(float).to_numpy(),
            "close_ask": raw["closePrice_ask"].astype(float).to_numpy(),
            "open_bid": raw["openPrice_bid"].astype(float).to_numpy(),
            "high_bid": raw["highPrice_bid"].astype(float).to_numpy(),
            "low_bid": raw["lowPrice_bid"].astype(float).to_numpy(),
            "close_bid": raw["closePrice_bid"].astype(float).to_numpy(),
            "volume": raw["lastTradedVolume"].fillna(0.0).astype(float).to_numpy(),
        },
        index=idx,
    ).sort_index()

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "open_ask": "first",
        "high_ask": "max",
        "low_ask": "min",
        "close_ask": "last",
        "open_bid": "first",
        "high_bid": "max",
        "low_bid": "min",
        "close_bid": "last",
        "volume": "sum",
    }
    out = df.resample("15min", label="left", closed="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close", "open_ask", "open_bid"])  # keep complete bars
    return out


def _add_features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=p.ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=p.ema_slow, adjust=False).mean()

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(p.atr_period).mean()

    out["roll_high"] = out["high"].shift(1).rolling(p.lookback).max()
    out["roll_low"] = out["low"].shift(1).rolling(p.lookback).min()
    out["atr_pct"] = out["atr"] / out["close"]

    return out


def _run_backtest(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    bars = _add_features(df, p)
    min_idx = max(p.lookback, p.ema_slow, p.atr_period) + 2

    trades: list[dict] = []
    in_pos = False
    side = ""
    entry_i = -1
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0

    for i in range(min_idx, len(bars) - 1):
        row = bars.iloc[i]
        next_row = bars.iloc[i + 1]

        if not in_pos:
            # Restrict to liquid UTC hours to reduce noisy fills.
            hour = bars.index[i].hour
            if hour < 6 or hour > 20:
                continue

            if pd.isna(row["roll_high"]) or pd.isna(row["roll_low"]) or pd.isna(row["atr"]):
                continue

            vol_ok = 0.0005 <= float(row["atr_pct"]) <= 0.008
            long_signal = bool(
                vol_ok
                and row["close"] > row["roll_high"]
                and row["ema_fast"] > row["ema_slow"]
            )
            short_signal = bool(
                vol_ok
                and row["close"] < row["roll_low"]
                and row["ema_fast"] < row["ema_slow"]
            )

            if not (long_signal or short_signal):
                continue

            in_pos = True
            entry_i = i + 1
            side = "long" if long_signal else "short"
            atr = float(row["atr"])

            if side == "long":
                entry_price = float(next_row["open_ask"])
                tp_price = entry_price + p.tp_atr * atr
                sl_price = max(entry_price - p.sl_atr * atr, 0.0)
            else:
                entry_price = float(next_row["open_bid"])
                tp_price = max(entry_price - p.tp_atr * atr, 0.0)
                sl_price = entry_price + p.sl_atr * atr
            continue

        # Manage open position.
        hold_bars = i - entry_i + 1
        exit_reason = ""
        exit_price = 0.0

        if side == "long":
            low_bid = float(row["low_bid"])
            high_bid = float(row["high_bid"])
            if low_bid <= sl_price:
                exit_reason = "stop_loss"
                exit_price = sl_price
            elif high_bid >= tp_price:
                exit_reason = "take_profit"
                exit_price = tp_price
        else:
            high_ask = float(row["high_ask"])
            low_ask = float(row["low_ask"])
            if high_ask >= sl_price:
                exit_reason = "stop_loss"
                exit_price = sl_price
            elif low_ask <= tp_price:
                exit_reason = "take_profit"
                exit_price = tp_price

        if not exit_reason and hold_bars >= p.max_hold:
            exit_reason = "timeout"
            exit_price = float(row["close_bid"] if side == "long" else row["close_ask"])

        if not exit_reason:
            continue

        pnl = (exit_price - entry_price) if side == "long" else (entry_price - exit_price)
        trades.append(
            {
                "side": side,
                "entry_idx": entry_i,
                "exit_idx": i,
                "entry_time": bars.index[entry_i],
                "exit_time": bars.index[i],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_usd": pnl,
                "bars_held": hold_bars,
                "exit_reason": exit_reason,
                "lookback": p.lookback,
                "ema_fast": p.ema_fast,
                "ema_slow": p.ema_slow,
                "atr_period": p.atr_period,
                "tp_atr": p.tp_atr,
                "sl_atr": p.sl_atr,
                "max_hold": p.max_hold,
            }
        )

        in_pos = False
        side = ""

    return pd.DataFrame(trades)


def _daily_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "total_pnl": 0.0,
            "n_trades": 0.0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
            "avg_day": 0.0,
            "median_day": 0.0,
            "positive_days": 0.0,
            "worst_day": 0.0,
            "best_day": 0.0,
            "max_dd": 0.0,
        }

    t = trades.copy()
    t["day"] = pd.to_datetime(t["entry_time"], utc=True).dt.floor("D")
    daily = t.groupby("day")["pnl_usd"].sum().sort_index()

    equity = daily.cumsum()
    peak = equity.cummax()
    dd = equity - peak

    return {
        "total_pnl": float(t["pnl_usd"].sum()),
        "n_trades": float(len(t)),
        "win_rate": float((t["pnl_usd"] > 0).mean() * 100.0),
        "avg_trade": float(t["pnl_usd"].mean()),
        "avg_day": float(daily.mean()),
        "median_day": float(daily.median()),
        "positive_days": float((daily > 0).mean() * 100.0),
        "worst_day": float(daily.min()),
        "best_day": float(daily.max()),
        "max_dd": float(dd.min()),
    }


def _objective(stats: dict[str, float]) -> float:
    # Reward daily expectancy and consistency; penalize deep drawdown.
    return stats["avg_day"] + 0.05 * stats["positive_days"] + 0.01 * stats["win_rate"] + 0.001 * stats["max_dd"]


def _param_grid() -> list[Params]:
    grid = []
    for lookback, atr_period, tp_atr, sl_atr, max_hold in itertools.product(
        [16, 24, 32],
        [10, 14],
        [1.6, 2.0, 2.4],
        [1.0, 1.3, 1.6],
        [6, 10, 14],
    ):
        if tp_atr <= sl_atr:
            continue
        grid.append(
            Params(
                lookback=lookback,
                ema_fast=20,
                ema_slow=60,
                atr_period=atr_period,
                tp_atr=tp_atr,
                sl_atr=sl_atr,
                max_hold=max_hold,
            )
        )
    return grid


def _print_stats(label: str, stats: dict[str, float]) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Trades: {int(stats['n_trades'])}")
    print(f"Win rate: {stats['win_rate']:.2f}%")
    print(f"Avg PnL/trade: ${stats['avg_trade']:.2f}")
    print(f"Total PnL (1 unit): ${stats['total_pnl']:.2f}")
    print(f"Avg daily PnL: ${stats['avg_day']:.2f}")
    print(f"Median daily PnL: ${stats['median_day']:.2f}")
    print(f"Positive days: {stats['positive_days']:.2f}%")
    print(f"Worst day: ${stats['worst_day']:.2f}")
    print(f"Best day: ${stats['best_day']:.2f}")
    print(f"Max daily-drawdown from equity peak: ${stats['max_dd']:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh gold re-analysis with a new breakout method.")
    parser.add_argument("--table", default="gold_prices")
    parser.add_argument("--start-date", default="2025-05-20")
    parser.add_argument("--end-date", default="2026-04-10")
    parser.add_argument("--out-csv", default="training/backtest_trades_new_method.csv")
    args = parser.parse_args()

    print("Loading raw data from MySQL...")
    raw = DataLoader().load_data(table_name=args.table, start_date=args.start_date, end_date=args.end_date)
    if raw.empty:
        raise ValueError("No data loaded from DB for requested range.")

    bars = _prepare_bars_15m(raw)
    print(f"Prepared 15m bars: {len(bars):,}")

    split_idx = int(len(bars) * 0.7)
    train = bars.iloc[:split_idx].copy()
    test = bars.iloc[split_idx:].copy()
    print(f"Train bars: {len(train):,}  |  Test bars: {len(test):,}")

    print("Searching parameters on train split...")
    best_params: Params | None = None
    best_score = -1e18
    best_train_stats: dict[str, float] | None = None

    for p in _param_grid():
        train_trades = _run_backtest(train, p)
        stats = _daily_stats(train_trades)
        if stats["n_trades"] < 40:
            continue
        score = _objective(stats)
        if score > best_score:
            best_score = score
            best_params = p
            best_train_stats = stats

    if best_params is None or best_train_stats is None:
        raise RuntimeError("No viable parameter set found. Try widening date range.")

    print("\nBest parameters found on train:")
    print(best_params)
    _print_stats("Train performance (optimized)", best_train_stats)

    print("\nEvaluating out-of-sample test split with fixed params...")
    test_trades = _run_backtest(test, best_params)
    test_stats = _daily_stats(test_trades)
    _print_stats("Test performance (out-of-sample)", test_stats)

    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test_trades.to_csv(out_path, index=False)
    print(f"\nSaved OOS test trades: {out_path}")

    if test_stats["avg_day"] > 0:
        units = int(np.ceil(40.0 / test_stats["avg_day"]))
        proj = units * test_stats["avg_day"]
        worst = units * test_stats["worst_day"]
        print(f"\nTarget check ($40/day):")
        print(f"- Required units (based on test avg): {units}")
        print(f"- Projected avg daily PnL: ${proj:.2f}")
        print(f"- Projected worst observed day: ${worst:.2f}")
    else:
        print("\nTarget check ($40/day): Not feasible on out-of-sample test (non-positive avg day).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

