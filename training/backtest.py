#!/usr/bin/env python3
"""Simple bias-aware backtest for the ML uptrend model.

Strategy:
- Build features up to each candle using existing extractor logic.
- Generate uptrend signals from a selected trained model.
- Enter long on next bar open after a signal.
- Exit by take-profit / stop-loss / max holding bars.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import sys

import pandas as pd

# Allow running as a script from training/ while importing project modules.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import DataLoader
try:
    from training.training import prepare_gold_data
except Exception:
    from training import prepare_gold_data
try:
    from ml_alpha_model import UptrendRecognitionSystem
except ModuleNotFoundError:
    from training.ml_alpha_model import UptrendRecognitionSystem


@dataclass
class BacktestConfig:
    table: str
    start_date: str
    end_date: str
    model_type: str
    model_dir: str
    signals_file: Optional[str]
    take_profit_pct: float
    take_profit_usd: Optional[float]
    stop_loss_pct: float
    stop_loss_usd: Optional[float]
    max_hold_bars: int
    cutoff: Optional[float]
    optimize_cutoff: bool
    cutoff_min: float
    cutoff_max: float
    cutoff_step: float
    allow_overlap: bool
    session_filter: bool
    config_file: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest ML uptrend signals.")
    parser.add_argument("--table", default="gold_prices")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--model-type", default="random_forest", choices=["random_forest", "gradient_boosting"])
    parser.add_argument("--model-dir", default="training/ml_models")
    parser.add_argument("--signals-file", default=None, help="Optional cached signals CSV (e.g. training/test_period_signals.csv)")
    parser.add_argument("--take-profit-pct", type=float, default=0.30, help="Percent gain target, e.g. 0.30")
    parser.add_argument("--take-profit-usd", type=float, default=None, help="Absolute take-profit distance in USD per 1 unit trade, e.g. 10")
    parser.add_argument("--stop-loss-pct", type=float, default=0.20, help="Percent loss stop, e.g. 0.20 (ignored when --stop-loss-usd is set)")
    parser.add_argument("--stop-loss-usd", type=float, default=None, help="Absolute stop-loss distance in USD per 1 unit trade, e.g. 10")
    parser.add_argument("--max-hold-bars", type=int, default=30)
    parser.add_argument("--cutoff", type=float, default=None, help="Optional probability cutoff override")
    parser.add_argument("--optimize-cutoff", action="store_true", help="Optimize cutoff on the selected backtest window")
    parser.add_argument("--cutoff-min", type=float, default=0.05)
    parser.add_argument("--cutoff-max", type=float, default=0.90)
    parser.add_argument("--cutoff-step", type=float, default=0.05)
    parser.add_argument("--allow-overlap", action="store_true", help="Allow multiple concurrent entries")
    parser.add_argument("--session-filter", action="store_true", help="Apply session day/hour filters from ml_config.json")
    parser.add_argument("--config-file", default="ml_config.json", help="Path to config JSON with time_filters")
    parser.add_argument("--out", default="training/backtest_trades.csv")
    return parser


def _load_time_filters(config_file: str) -> dict[str, dict[str, set[int]]]:
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_filters = payload.get("time_filters")
    if not isinstance(raw_filters, dict):
        raise ValueError("Missing or invalid 'time_filters' in config file")

    normalized: dict[str, dict[str, set[int]]] = {"asia": {}, "ny": {}}
    for session in ("asia", "ny"):
        session_map = raw_filters.get(session, {})
        if not isinstance(session_map, dict):
            continue
        for day_name, hours in session_map.items():
            if not isinstance(hours, list):
                continue
            normalized[session][str(day_name)] = {
                int(hour) for hour in hours if isinstance(hour, (int, float)) and 0 <= int(hour) <= 23
            }

    return normalized


def _build_allowed_entry_indices(df: pd.DataFrame, cfg: BacktestConfig) -> Optional[set[int]]:
    if not cfg.session_filter:
        return None

    time_filters = _load_time_filters(cfg.config_file)
    idx = pd.DatetimeIndex(df.index)
    idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

    idx_hkt = idx_utc.tz_convert("Asia/Hong_Kong")
    idx_ny = idx_utc.tz_convert("America/New_York")
    trading_day_name = (idx_hkt - pd.Timedelta(hours=6)).day_name()

    allowed: set[int] = set()
    for i in range(len(df)):
        day = str(trading_day_name[i])
        if day == "Sunday":
            continue

        hour_hkt = int(idx_hkt.hour[i])
        hour_ny = int(idx_ny.hour[i])

        if 6 <= hour_hkt <= 19:
            if hour_hkt in time_filters.get("asia", {}).get(day, set()):
                allowed.add(i)
            continue

        if 6 <= hour_ny <= 17:
            if hour_ny in time_filters.get("ny", {}).get(day, set()):
                allowed.add(i)

    return allowed


def _load_prices(cfg: BacktestConfig) -> pd.DataFrame:
    loader = DataLoader()
    raw = loader.load_data(cfg.table, cfg.start_date, cfg.end_date)
    df = prepare_gold_data(raw)

    # Keep ask/bid OHLC for execution realism (long: buy at ask, sell at bid).
    for side in ("ask", "bid"):
        for base in ("open", "high", "low", "close"):
            raw_col = f"{base}Price_{side}"
            out_col = f"{base}_{side}"
            if raw_col in raw.columns:
                df[out_col] = raw[raw_col].to_numpy()
            else:
                # Fallback to mid if side price is unavailable.
                df[out_col] = df[base].to_numpy()

    if len(df) < 200:
        raise ValueError("Not enough rows for backtest. Increase date range.")
    return df


def _generate_signals(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    if cfg.signals_file:
        cached = pd.read_csv(cfg.signals_file)
        prob_col = "rf_probability" if cfg.model_type == "random_forest" else "gb_probability"
        if prob_col not in cached.columns:
            if "probability" in cached.columns:
                prob_col = "probability"
            else:
                raise ValueError(f"Missing probability column in cached file: {prob_col}")

        if "timestamp" in cached.columns:
            ref = pd.DataFrame({"timestamp": df.index.astype(str), "idx": range(len(df))})
            merged = cached.merge(ref, on="timestamp", how="inner", suffixes=("", "_mapped"))
            signals = merged[["idx", prob_col]].copy()
        elif "idx" in cached.columns:
            signals = cached[["idx", prob_col]].copy()
        else:
            raise ValueError("Cached signals file must contain either 'timestamp' or 'idx'.")

        signals.columns = ["idx", "probability"]
        signals["idx"] = signals["idx"].astype(int)
        signals["probability"] = pd.to_numeric(signals["probability"], errors="coerce").fillna(0.0)
        signals = signals.sort_values("idx").drop_duplicates(subset=["idx"], keep="last").reset_index(drop=True)
        return signals

    system = UptrendRecognitionSystem()
    system.load_models(cfg.model_dir)
    if cfg.model_type not in system.models:
        raise ValueError(f"Model '{cfg.model_type}' not found in {cfg.model_dir}")

    feature_df = system.extractor.create_feature_dataframe(df)
    if feature_df.empty:
        raise ValueError("Feature extraction returned no rows.")

    model = system.models[cfg.model_type]
    X = feature_df[system.extractor.feature_names].fillna(0)
    proba = model.predict_proba(X)
    signals = feature_df[["idx"]].copy()
    signals["probability"] = proba
    return signals


def _run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: BacktestConfig,
    cutoff: float,
    allowed_entry_indices: Optional[set[int]] = None,
) -> pd.DataFrame:
    trades = []
    next_entry_allowed_idx = 0
    signal_probs = signals.groupby("idx")["probability"].max().to_dict()

    for row in signals.itertuples(index=False):
        signal_idx = int(row.idx)
        if float(row.probability) < cutoff:
            continue

        entry_idx = signal_idx + 1
        if entry_idx >= len(df):
            continue
        if allowed_entry_indices is not None and entry_idx not in allowed_entry_indices:
            continue
        if not cfg.allow_overlap and entry_idx < next_entry_allowed_idx:
            continue

        # Long entry executes at ask; exits execute at bid.
        entry_price = float(df["open_ask"].iloc[entry_idx])
        entry_time = df.index[entry_idx]
        end_idx = min(entry_idx + cfg.max_hold_bars, len(df) - 1)

        def resolve_tp_from_ask(price_ask: float) -> float:
            if cfg.take_profit_usd is not None:
                return price_ask + cfg.take_profit_usd
            return price_ask * (1.0 + cfg.take_profit_pct / 100.0)

        tp_price = resolve_tp_from_ask(entry_price)
        if cfg.stop_loss_usd is not None:
            sl_price = max(entry_price - cfg.stop_loss_usd, 0.0)
        else:
            sl_price = entry_price * (1.0 - cfg.stop_loss_pct / 100.0)

        exit_idx = end_idx
        exit_price = float(df["close_bid"].iloc[end_idx])
        exit_reason = "timeout"

        for i in range(entry_idx, end_idx + 1):
            low_i = float(df["low_bid"].iloc[i])
            high_i = float(df["high_bid"].iloc[i])

            # Conservative tie-break: stop-loss first when both hit in same bar.
            if low_i <= sl_price:
                exit_idx = i
                exit_price = sl_price
                exit_reason = "stop_loss"
                break
            if high_i >= tp_price:
                exit_idx = i
                exit_price = tp_price
                exit_reason = "take_profit"
                break

            # Dynamic target adjustment: if a fresh qualifying signal appears mid-trade,
            # raise TP so strong continuation moves are not capped too early.
            if i != entry_idx and float(signal_probs.get(i, 0.0)) >= cutoff:
                refreshed_tp = resolve_tp_from_ask(float(df["close_ask"].iloc[i]))
                if refreshed_tp > tp_price:
                    tp_price = refreshed_tp

        pnl_usd = exit_price - entry_price  # One unit per trade.
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
        trades.append(
            {
                "signal_idx": signal_idx,
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "entry_time": entry_time,
                "exit_time": df.index[exit_idx],
                "probability": float(row.probability),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_pct,
                "bars_held": exit_idx - entry_idx + 1,
                "exit_reason": exit_reason,
            }
        )

        if not cfg.allow_overlap:
            next_entry_allowed_idx = exit_idx + 1

    return pd.DataFrame(trades)


def _optimize_cutoff(df: pd.DataFrame, signals: pd.DataFrame, cfg: BacktestConfig) -> float:
    allowed_entry_indices = _build_allowed_entry_indices(df, cfg)
    step = max(cfg.cutoff_step, 0.001)
    values = []
    current = cfg.cutoff_min
    print("Sweeping cutoffs...")
    while current <= cfg.cutoff_max + 1e-12:
        cutoff_value = float(round(current, 4))
        trades = _run_backtest(
            df,
            signals,
            cfg,
            cutoff=cutoff_value,
            allowed_entry_indices=allowed_entry_indices,
        )
        total_profit = float(trades["pnl_usd"].sum()) if not trades.empty else 0.0
        trade_count = len(trades)
        values.append((cutoff_value, total_profit, trade_count))
        print(f"  cutoff={cutoff_value:.2f} -> trades={trade_count}, profit=${total_profit:.2f}")
        current += step

    best_cutoff, best_profit, best_trades = max(values, key=lambda x: x[1])

    print("=" * 80)
    print("CUTOFF OPTIMIZATION")
    print("=" * 80)
    print(f"Best cutoff: {best_cutoff:.2f}")
    print(f"Best total profit (USD): ${best_profit:.2f}")
    print(f"Trades at best cutoff: {best_trades}")

    top_rows = sorted(values, key=lambda x: x[1], reverse=True)[:5]
    print("Top cutoffs by total USD profit:")
    for cutoff, profit, count in top_rows:
        print(f"  cutoff={cutoff:.2f} -> profit=${profit:.2f}, trades={count}")

    return best_cutoff


def _print_summary(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("No trades generated in the selected period.")
        return

    total = len(trades)
    wins = int((trades["pnl_pct"] > 0).sum())
    win_rate = wins / total * 100.0
    avg_pnl_usd = float(trades["pnl_usd"].mean())
    total_pnl_usd = float(trades["pnl_usd"].sum())

    # Simple compounded equity from percentage returns.
    equity = trades["pnl_usd"].cumsum()
    final_equity = float(equity.iloc[-1])

    daily = trades.copy()
    daily["entry_day"] = pd.to_datetime(daily["entry_time"]).dt.date
    trades_per_day = daily.groupby("entry_day").size()

    print("=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)
    print(f"Trades: {total}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average PnL per trade (USD, 1 unit): ${avg_pnl_usd:.4f}")
    print(f"Total profit (USD, 1 unit): ${total_pnl_usd:.2f}")
    print(f"Cumulative net PnL (USD): ${final_equity:.2f}")
    print(f"Avg trades/day: {trades_per_day.mean():.2f}")
    print(f"Max trades/day: {int(trades_per_day.max())}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.take_profit_usd is None and args.take_profit_pct <= 0:
        parser.error("--take-profit-pct must be > 0 when --take-profit-usd is not set")
    if args.take_profit_usd is not None and args.take_profit_usd <= 0:
        parser.error("--take-profit-usd must be > 0")
    if args.stop_loss_pct <= 0:
        parser.error("--stop-loss-pct must be > 0")
    if args.stop_loss_usd is not None and args.stop_loss_usd <= 0:
        parser.error("--stop-loss-usd must be > 0")

    cfg = BacktestConfig(
        table=args.table,
        start_date=args.start_date,
        end_date=args.end_date,
        model_type=args.model_type,
        model_dir=args.model_dir,
        signals_file=args.signals_file,
        take_profit_pct=args.take_profit_pct,
        take_profit_usd=args.take_profit_usd,
        stop_loss_pct=args.stop_loss_pct,
        stop_loss_usd=args.stop_loss_usd,
        max_hold_bars=args.max_hold_bars,
        cutoff=args.cutoff,
        optimize_cutoff=args.optimize_cutoff,
        cutoff_min=args.cutoff_min,
        cutoff_max=args.cutoff_max,
        cutoff_step=args.cutoff_step,
        allow_overlap=args.allow_overlap,
        session_filter=args.session_filter,
        config_file=args.config_file,
    )

    print("Loading data...")
    df = _load_prices(cfg)
    print(f"Loaded candles: {len(df)}")

    print("Generating model signals...")
    signals = _generate_signals(df, cfg)
    print(f"Signal rows: {len(signals)}")

    allowed_entry_indices = _build_allowed_entry_indices(df, cfg)
    if allowed_entry_indices is not None:
        print(f"Session filter enabled from {cfg.config_file}")
        print(f"Allowed entry bars: {len(allowed_entry_indices)} / {len(df)}")

    if cfg.optimize_cutoff:
        cutoff = _optimize_cutoff(df, signals, cfg)
    else:
        cutoff = cfg.cutoff if cfg.cutoff is not None else 0.5

    if cfg.take_profit_usd is not None:
        print(f"Using take-profit mode: ${cfg.take_profit_usd:.2f} absolute distance")
    else:
        print(f"Using take-profit mode: {cfg.take_profit_pct:.2f}%")
    if cfg.stop_loss_usd is not None:
        print(f"Using stop-loss mode: ${cfg.stop_loss_usd:.2f} absolute distance")
    else:
        print(f"Using stop-loss mode: {cfg.stop_loss_pct:.2f}%")
    print(f"Using cutoff: {cutoff:.2f}")
    print("Running backtest...")
    trades = _run_backtest(df, signals, cfg, cutoff=cutoff, allowed_entry_indices=allowed_entry_indices)
    _print_summary(trades)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_path, index=False)
    print(f"Saved trades: {out_path}")


if __name__ == "__main__":
    main()

