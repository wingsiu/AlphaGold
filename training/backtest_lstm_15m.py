#!/usr/bin/env python3
"""Backtest runner for 15-minute LSTM signals."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import DataLoader
try:
    from training.ml_alpha_model_15m_nextbar import prepare_gold_data_15m
    from training.ml_lstm_15m import (
        DEFAULT_FORWARD_BARS,
        DEFAULT_MAX_ADVERSE_LOW_PCT,
        DEFAULT_THRESHOLD_PCT,
        apply_standardization,
        load_lstm_artifacts,
    )
    from training.ml_alpha_model import CandlePatternExtractor
except ModuleNotFoundError:
    from ml_alpha_model_15m_nextbar import prepare_gold_data_15m
    from ml_lstm_15m import (
        DEFAULT_FORWARD_BARS,
        DEFAULT_MAX_ADVERSE_LOW_PCT,
        DEFAULT_THRESHOLD_PCT,
        apply_standardization,
        load_lstm_artifacts,
    )
    from ml_alpha_model import CandlePatternExtractor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = Path(__file__).resolve().parent


@dataclass
class BacktestConfig:
    table: str
    start_date: str
    end_date: str
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest 15-minute LSTM signals.")
    parser.add_argument("--table", default="gold_prices")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--model-dir", default="training/ml_models_lstm_15m")
    parser.add_argument("--signals-file", default=None)
    parser.add_argument("--take-profit-pct", type=float, default=DEFAULT_THRESHOLD_PCT)
    parser.add_argument("--take-profit-usd", type=float, default=None)
    parser.add_argument("--stop-loss-pct", type=float, default=DEFAULT_MAX_ADVERSE_LOW_PCT)
    parser.add_argument("--stop-loss-usd", type=float, default=None)
    parser.add_argument("--max-hold-bars", type=int, default=DEFAULT_FORWARD_BARS)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--optimize-cutoff", action="store_true")
    parser.add_argument("--cutoff-min", type=float, default=0.05)
    parser.add_argument("--cutoff-max", type=float, default=0.50)
    parser.add_argument("--cutoff-step", type=float, default=0.05)
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--out", default="training/backtest_trades_lstm_15m.csv")
    return parser


def load_prices(cfg: BacktestConfig) -> pd.DataFrame:
    raw = DataLoader().load_data(cfg.table, start_date=cfg.start_date, end_date=cfg.end_date)
    df = prepare_gold_data_15m(raw)
    if len(df) < 150:
        raise ValueError("Not enough 15-minute rows for backtest.")
    return df


def _resolve_signals_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    candidates = [PROJECT_ROOT / path, TRAINING_DIR / path, TRAINING_DIR / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Signals file not found: {path_text}. Tried cwd-relative and project/training-relative paths."
    )


def generate_signals(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    if cfg.signals_file:
        sig_path = _resolve_signals_path(cfg.signals_file)
        cached = pd.read_csv(sig_path)
        prob_col = "lstm_probability" if "lstm_probability" in cached.columns else "probability"

        if "timestamp" in cached.columns:
            ref = pd.DataFrame({"timestamp": df.index.astype(str), "idx": range(len(df))})
            merged = cached.merge(ref, on="timestamp", how="inner", suffixes=("", "_mapped"))
            if "idx" not in merged.columns and "idx_mapped" in merged.columns:
                merged["idx"] = merged["idx_mapped"]
            signals = merged[["idx", prob_col]].copy()
        elif "idx" in cached.columns:
            signals = cached[["idx", prob_col]].copy()
        else:
            raise ValueError("Cached signals file must contain either 'timestamp' or 'idx'.")

        signals.columns = ["idx", "probability"]
        signals["idx"] = signals["idx"].astype(int)
        signals["probability"] = pd.to_numeric(signals["probability"], errors="coerce").fillna(0.0)
        return signals.sort_values("idx").drop_duplicates(subset=["idx"], keep="last").reset_index(drop=True)

    artifacts = load_lstm_artifacts(cfg.model_dir)
    model = artifacts["model"]
    mean = artifacts["mean"]
    std = artifacts["std"]
    feature_names = artifacts["feature_names"]
    seq_len = artifacts["sequence_length"]

    extractor = CandlePatternExtractor()
    feature_df = extractor.create_feature_dataframe(df)
    if feature_df.empty:
        raise ValueError("No features extracted for LSTM inference.")

    feature_block = feature_df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    idx_values = feature_df["idx"].astype(int).to_numpy()

    X_seq = []
    seq_idx = []
    for end_pos in range(seq_len - 1, len(feature_df)):
        start_pos = end_pos - seq_len + 1
        X_seq.append(feature_block[start_pos: end_pos + 1])
        seq_idx.append(int(idx_values[end_pos]))

    X = np.asarray(X_seq, dtype=np.float32)
    X = apply_standardization(X, mean, std)
    probs = model.predict(X, verbose=0).reshape(-1)

    return pd.DataFrame({"idx": seq_idx, "probability": probs})


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, cfg: BacktestConfig, cutoff: float) -> pd.DataFrame:
    trades: list[dict[str, object]] = []
    next_entry_allowed_idx = 0

    for signal_idx, probability in zip(
        signals["idx"].astype(int).to_numpy(),
        signals["probability"].astype(float).to_numpy(),
    ):
        if probability < cutoff:
            continue

        entry_idx = signal_idx + 1
        if entry_idx >= len(df):
            continue
        if not cfg.allow_overlap and entry_idx < next_entry_allowed_idx:
            continue

        entry_price = float(df["open_ask"].iloc[entry_idx])
        end_idx = min(entry_idx + max(cfg.max_hold_bars - 1, 0), len(df) - 1)

        tp_price = entry_price + cfg.take_profit_usd if cfg.take_profit_usd is not None else entry_price * (1.0 + cfg.take_profit_pct / 100.0)
        sl_price = (
            max(entry_price - cfg.stop_loss_usd, 0.0)
            if cfg.stop_loss_usd is not None
            else entry_price * (1.0 - cfg.stop_loss_pct / 100.0)
        )

        exit_idx = end_idx
        exit_price = float(df["close_bid"].iloc[end_idx])
        exit_reason = "timeout"

        for i in range(entry_idx, end_idx + 1):
            low_bid = float(df["low_bid"].iloc[i])
            high_bid = float(df["high_bid"].iloc[i])
            if low_bid <= sl_price:
                exit_idx = i
                exit_price = sl_price
                exit_reason = "stop_loss"
                break
            if high_bid >= tp_price:
                exit_idx = i
                exit_price = tp_price
                exit_reason = "take_profit"
                break

        trades.append(
            {
                "signal_idx": signal_idx,
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "entry_time": df.index[entry_idx],
                "exit_time": df.index[exit_idx],
                "probability": probability,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_usd": exit_price - entry_price,
                "pnl_pct": ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0.0,
                "bars_held": exit_idx - entry_idx + 1,
                "exit_reason": exit_reason,
            }
        )

        if not cfg.allow_overlap:
            next_entry_allowed_idx = exit_idx + 1

    return pd.DataFrame(trades)


def optimize_cutoff(df: pd.DataFrame, signals: pd.DataFrame, cfg: BacktestConfig) -> float:
    step = max(cfg.cutoff_step, 0.001)
    current = cfg.cutoff_min
    values: list[tuple[float, float, int]] = []

    print("Sweeping cutoffs...")
    while current <= cfg.cutoff_max + 1e-12:
        cutoff_value = float(round(current, 4))
        trades = run_backtest(df, signals, cfg, cutoff_value)
        total_profit = float(trades["pnl_usd"].sum()) if not trades.empty else 0.0
        values.append((cutoff_value, total_profit, len(trades)))
        print(f"  cutoff={cutoff_value:.2f} -> trades={len(trades)}, profit=${total_profit:.2f}")
        current += step

    best_cutoff, best_profit, best_trades = max(values, key=lambda item: item[1])
    print("=" * 80)
    print("CUTOFF OPTIMIZATION")
    print("=" * 80)
    print(f"Best cutoff: {best_cutoff:.2f}")
    print(f"Best total profit (USD): ${best_profit:.2f}")
    print(f"Trades at best cutoff: {best_trades}")
    return best_cutoff


def print_summary(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("No trades generated in the selected period.")
        return

    total = len(trades)
    wins = int((trades["pnl_usd"] > 0).sum())
    win_rate = wins / total * 100.0
    avg_pnl = float(trades["pnl_usd"].mean())
    total_pnl = float(trades["pnl_usd"].sum())

    print("=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)
    print(f"Trades: {total}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average PnL per trade (USD, 1 unit): ${avg_pnl:.4f}")
    print(f"Total profit (USD, 1 unit): ${total_pnl:.2f}")
    print(trades["exit_reason"].value_counts().to_string())


def main() -> int:
    args = build_parser().parse_args()

    if args.take_profit_usd is None and args.take_profit_pct <= 0:
        raise ValueError("--take-profit-pct must be > 0 when --take-profit-usd is not set")
    if args.stop_loss_usd is None and args.stop_loss_pct <= 0:
        raise ValueError("--stop-loss-pct must be > 0 when --stop-loss-usd is not set")

    cfg = BacktestConfig(
        table=args.table,
        start_date=args.start_date,
        end_date=args.end_date,
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
    )

    print("Loading and aggregating raw data to 15-minute candles...")
    df = load_prices(cfg)
    print(f"Loaded 15-minute candles: {len(df):,}")

    print("Generating LSTM signals...")
    signals = generate_signals(df, cfg)
    print(f"Signal rows: {len(signals):,}")

    cutoff = optimize_cutoff(df, signals, cfg) if cfg.optimize_cutoff else (cfg.cutoff if cfg.cutoff is not None else 0.5)

    print(f"Using cutoff: {cutoff:.2f}")
    print("Running backtest...")
    trades = run_backtest(df, signals, cfg, cutoff=cutoff)
    print_summary(trades)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_path, index=False)
    print(f"Saved trades: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

