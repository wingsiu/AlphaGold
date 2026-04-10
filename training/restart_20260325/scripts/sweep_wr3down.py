#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data import DataLoader


def backtest(df: pd.DataFrame, sig: pd.Series, hold: int, stop: float) -> tuple[float, int]:
    i = 0
    n = len(df)
    pnl = 0.0
    trades = 0
    while i < n - 1:
        if not (bool(sig.iat[i]) and bool(df['eligible'].iat[i])):
            i += 1
            continue
        entry = float(df['close'].iat[i])
        stop_px = entry + stop
        max_i = min(n - 1, i + hold)
        exit_i = max_i
        exit_px = float(df['close'].iat[max_i])
        for j in range(i + 1, max_i + 1):
            if float(df['high'].iat[j]) >= stop_px:
                exit_i = j
                exit_px = stop_px
                break
        pnl += entry - exit_px
        trades += 1
        i = exit_i + 1
    return pnl, trades


def main() -> int:
    raw = DataLoader().load_data('gold_prices', start_date='2025-05-20', end_date='2026-02-04')
    df = pd.DataFrame(
        {
            'ts': pd.to_datetime(raw['timestamp'], unit='ms', utc=True),
            'open': raw['openPrice'].astype(float),
            'high': raw['highPrice'].astype(float),
            'low': raw['lowPrice'].astype(float),
            'close': raw['closePrice'].astype(float),
        }
    ).sort_values('ts').reset_index(drop=True)

    wr_lookback = 90
    hh = df['high'].rolling(wr_lookback, min_periods=wr_lookback).max()
    ll = df['low'].rolling(wr_lookback, min_periods=wr_lookback).min()
    rng = hh - ll
    df['wr'] = (-100.0 * (hh - df['close']) / rng).where(rng != 0)
    df['change_1m'] = df['close'] - df['close'].shift(1)
    df['change_3bar_total'] = df['change_1m'] + df['change_1m'].shift(1) + df['change_1m'].shift(2)
    df['down'] = df['close'] < df['open']
    df['three_down'] = df['down'] & df['down'].shift(1) & df['down'].shift(2)

    payload = json.loads(Path('ml_config.json').read_text(encoding='utf-8'))
    nyraw = payload.get('time_filters', {}).get('ny', {})
    ny = {str(d): {int(h) for h in hrs if isinstance(h, (int, float))} for d, hrs in nyraw.items() if isinstance(hrs, list)}
    ts_ny = pd.to_datetime(df['ts'], utc=True).dt.tz_convert('America/New_York')
    df['ny_day'] = ts_ny.dt.day_name()
    df['ny_hour'] = ts_ny.dt.hour
    df['eligible'] = [h in ny.get(d, set()) for d, h in zip(df['ny_day'], df['ny_hour'])]

    rows: list[tuple[float, int, int, int, int, int, float]] = []
    for wr_min in (-90, -80, -70, -60):
        for wr_max in (-30, -20, -10):
            if wr_min >= wr_max:
                continue
            wr_mask = df['wr'].between(wr_min, wr_max, inclusive='both')
            for ch in (-3, -5, -7, -10):
                sig = (df['three_down'] & (df['change_3bar_total'] < ch) & wr_mask).fillna(False)
                for hold in (10, 15, 20, 30):
                    for stop in (4, 5, 6, 7, 8):
                        pnl, tr = backtest(df, sig, hold, stop)
                        rows.append((pnl, tr, wr_min, wr_max, ch, hold, stop))

    rows.sort(key=lambda x: x[0], reverse=True)
    out = Path('training/restart_20260325/reports/wr3down_param_sweep_top20.csv')
    top = pd.DataFrame(
        rows[:20],
        columns=['total_profit_usd', 'trades', 'wr_min', 'wr_max', 'min3chg', 'hold', 'stop'],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(out, index=False)

    print('top20 total_profit_usd,trades,wr_min,wr_max,min3chg,hold,stop')
    for r in rows[:20]:
        print(f'{r[0]:.2f},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}')
    print(f'saved {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

