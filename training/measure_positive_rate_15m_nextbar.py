#!/usr/bin/env python3
"""Measure next-bar positive rates on 15-minute candles."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import DataLoader
try:
    from training.ml_alpha_model_15m_nextbar import (
        DEFAULT_FORWARD_BARS,
        DEFAULT_MAX_ADVERSE_LOW_PCT,
        DEFAULT_THRESHOLD_PCT,
        prepare_gold_data_15m,
    )
except ModuleNotFoundError:
    from ml_alpha_model_15m_nextbar import (
        DEFAULT_FORWARD_BARS,
        DEFAULT_MAX_ADVERSE_LOW_PCT,
        DEFAULT_THRESHOLD_PCT,
        prepare_gold_data_15m,
    )


def parse_thresholds(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure positive rates for 15m next-bar-high targets.")
    parser.add_argument("--table", default="gold_prices")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--thresholds", default=f"0.20,0.30,{DEFAULT_THRESHOLD_PCT:.2f},0.70")
    parser.add_argument("--max-adverse-low-pct", type=float, default=DEFAULT_MAX_ADVERSE_LOW_PCT)
    parser.add_argument("--forward-bars", type=int, default=DEFAULT_FORWARD_BARS)
    parser.add_argument("--out", default="training/positive_rate_15m_nextbar.csv")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)

    raw = DataLoader().load_data(args.table, start_date=args.start_date, end_date=args.end_date)
    df = prepare_gold_data_15m(raw)
    forward = max(1, int(args.forward_bars))
    shifted_high = pd.concat([df["high"].shift(-h) for h in range(1, forward + 1)], axis=1)
    shifted_low = pd.concat([df["low"].shift(-h) for h in range(1, forward + 1)], axis=1)
    max_future_high = shifted_high.max(axis=1)
    min_future_low = shifted_low.min(axis=1)

    gain_pct = ((max_future_high - df["close"]) / df["close"]) * 100.0
    drawdown_pct = ((min_future_low - df["close"]) / df["close"]) * 100.0
    valid_mask = gain_pct.notna() & drawdown_pct.notna()
    valid_gain = gain_pct[valid_mask]
    valid_drawdown = drawdown_pct[valid_mask]

    rows: list[dict[str, float | int]] = []
    for thr in thresholds:
        hit = (valid_gain >= thr) & (valid_drawdown >= -args.max_adverse_low_pct)
        rows.append(
            {
                "threshold_pct": thr,
                "hits": int(hit.sum()),
                "samples": int(len(valid_gain)),
                "positive_rate_pct": float(hit.mean() * 100.0),
            }
        )

    summary = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    print(f"15m candles: {len(df):,}")
    print(
        "Target rule: max(high over next "
        f"{forward} bars) >= +threshold and min(low over next {forward} bars) >= -{args.max_adverse_low_pct:.2f}%"
    )
    print(f"valid samples (forward window available): {len(valid_gain):,}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

