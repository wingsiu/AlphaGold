#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data import DataLoader


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Backtest WR+3down signal with fixed hold and stop.')
    p.add_argument('--start-date', default='2025-05-20')
    p.add_argument('--end-date', default='2026-02-04')
    p.add_argument('--wr-lookback', type=int, default=90)
    p.add_argument('--wr-min', type=float, default=-80.0)
    p.add_argument('--wr-max', type=float, default=-20.0)
    p.add_argument('--min-total-change', type=float, default=-5.0, help='3-bar total close-to-close change threshold (signal when change_3bar_total < this value)')
    p.add_argument('--hold-bars', type=int, default=15)
    p.add_argument('--stop-usd', type=float, default=5.0)
    p.add_argument('--tp-usd', type=float, default=0.0, help='0 disables TP')
    p.add_argument('--session-filter', choices=['off', 'ny_broad', 'ny_config'], default='ny_config')
    p.add_argument('--config-file', default='ml_config.json')
    p.add_argument('--out-csv', default='training/restart_20260325/reports/wr3down_bt_trades.csv')
    return p


def compute_wr(df: pd.DataFrame, lookback: int) -> pd.Series:
    hh = df['high'].rolling(lookback, min_periods=lookback).max()
    ll = df['low'].rolling(lookback, min_periods=lookback).min()
    rng = hh - ll
    wr = -100.0 * (hh - df['close']) / rng
    return wr.where(rng != 0)


def load_time_filters(config_file: Path) -> dict[str, dict[str, set[int]]]:
    payload = json.loads(config_file.read_text(encoding='utf-8'))
    raw = payload.get('time_filters', {})
    out: dict[str, dict[str, set[int]]] = {'asia': {}, 'ny': {}}
    for sess in ('asia', 'ny'):
        m = raw.get(sess, {})
        if not isinstance(m, dict):
            continue
        for day, hours in m.items():
            if isinstance(hours, list):
                out[sess][str(day)] = {int(h) for h in hours if isinstance(h, (int, float))}
    return out


def apply_session_filter(df: pd.DataFrame, mode: str, config_file: Path) -> pd.Series:
    if mode == 'off':
        return pd.Series(True, index=df.index)
    if mode == 'ny_broad':
        return df['ny_hour'].between(6, 17, inclusive='both')
    tf = load_time_filters(config_file)
    ny = tf.get('ny', {})
    return pd.Series([hour in ny.get(day, set()) for day, hour in zip(df['ny_day_name'], df['ny_hour'])], index=df.index)


def main() -> int:
    args = build_parser().parse_args()

    raw = DataLoader().load_data('gold_prices', start_date=args.start_date, end_date=args.end_date)
    df = pd.DataFrame({
        'ts': pd.to_datetime(raw['timestamp'], unit='ms', utc=True),
        'open': raw['openPrice'].astype(float),
        'high': raw['highPrice'].astype(float),
        'low': raw['lowPrice'].astype(float),
        'close': raw['closePrice'].astype(float),
    }).sort_values('ts').reset_index(drop=True)

    df['wr'] = compute_wr(df, args.wr_lookback)
    df['change_1m'] = df['close'] - df['close'].shift(1)
    df['change_3bar_total'] = df['change_1m'] + df['change_1m'].shift(1) + df['change_1m'].shift(2)
    df['down'] = df['close'] < df['open']
    df['three_consecutive_down'] = df['down'] & df['down'].shift(1) & df['down'].shift(2)
    df['threebar_total_change_condition'] = df['change_3bar_total'] < args.min_total_change
    df['wr_between_range'] = df['wr'].between(args.wr_min, args.wr_max, inclusive='both')
    df['signal'] = df['three_consecutive_down'] & df['threebar_total_change_condition'] & df['wr_between_range']

    ts_ny = pd.to_datetime(df['ts'], utc=True).dt.tz_convert('America/New_York')
    df['ny_day_name'] = ts_ny.dt.day_name()
    df['ny_hour'] = ts_ny.dt.hour
    df['eligible'] = apply_session_filter(df, args.session_filter, ROOT / args.config_file)

    trades = []
    i = 0
    n = len(df)
    while i < n - 1:
        if not (bool(df.at[i, 'signal']) and bool(df.at[i, 'eligible'])):
            i += 1
            continue

        entry_i = i
        entry = float(df.at[i, 'close'])
        stop = entry + args.stop_usd
        tp = entry - args.tp_usd if args.tp_usd > 0 else None

        max_i = min(n - 1, entry_i + args.hold_bars)
        exit_i = max_i
        exit_price = float(df.at[max_i, 'close'])
        reason = 'timeout'

        for j in range(entry_i + 1, max_i + 1):
            h = float(df.at[j, 'high'])
            l = float(df.at[j, 'low'])
            if h >= stop:
                exit_i = j
                exit_price = stop
                reason = 'stop'
                break
            if tp is not None and l <= tp:
                exit_i = j
                exit_price = tp
                reason = 'tp'
                break

        pnl = entry - exit_price
        trades.append({
            'entry_i': entry_i,
            'exit_i': exit_i,
            'entry_time': df.at[entry_i, 'ts'],
            'exit_time': df.at[exit_i, 'ts'],
            'entry_price': entry,
            'exit_price': exit_price,
            'pnl_usd': pnl,
            'reason': reason,
        })

        i = exit_i + 1

    out = pd.DataFrame(trades)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    if out.empty:
        print('No trades generated.')
        print(f'Saved: {out_path}')
        return 0

    wins = out.loc[out['pnl_usd'] > 0, 'pnl_usd'].sum()
    losses = -out.loc[out['pnl_usd'] < 0, 'pnl_usd'].sum()
    pf = wins / losses if losses > 0 else float('inf')

    print('BACKTEST SUMMARY')
    print(f"Params: WR({args.wr_lookback}) in [{args.wr_min:.1f}, {args.wr_max:.1f}], min_total_change<{args.min_total_change:.1f}, hold={args.hold_bars}, stop={args.stop_usd}, tp={args.tp_usd}, session={args.session_filter}")
    print(f"Trades: {len(out)}")
    print(f"Win rate: {(out['pnl_usd'] > 0).mean() * 100:.2f}%")
    print(f"Total profit (USD, 1 unit): ${out['pnl_usd'].sum():.2f}")
    print(f"Average PnL per trade: ${out['pnl_usd'].mean():.4f}")
    print(f"Profit factor: {pf:.3f}")
    print('Reason counts:')
    print(out['reason'].value_counts().to_string())
    print(f'Saved: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

