#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path('/Users/alpha/Desktop/python/AlphaGold')
TRADES_PATH = ROOT / 'training/backtest_trades_best_base_corrected.csv'
NY_TZ = ZoneInfo('America/New_York')
HK_TZ = ZoneInfo('Asia/Hong_Kong')
CUTOFF_HOUR = 17


def load_trades() -> pd.DataFrame:
    df = pd.read_csv(TRADES_PATH)
    for col in ('ts', 'entry_time', 'exit_time', 'last_target_time'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
    return df


def session_name(row: pd.Series) -> str:
    h_hkt = int(row['hour_hkt'])
    h_ny = int(row['hour_ny'])
    if 7 <= h_hkt <= 15:
        return 'Asia'
    if 8 <= h_ny <= 16:
        return 'NY'
    return 'Other'


def summarize(df: pd.DataFrame, col: str) -> list[dict[str, object]]:
    g = df.groupby(col)['pnl'].agg(
        trades='size',
        total_pnl='sum',
        avg_trade='mean',
        median_trade='median',
        win_rate_pct=lambda s: float((s > 0).mean() * 100.0),
        gross_profit=lambda s: float(s[s > 0].sum()),
        gross_loss=lambda s: float(s[s < 0].sum()),
    ).reset_index()
    g['profit_factor'] = g.apply(lambda r: (r['gross_profit'] / abs(r['gross_loss'])) if r['gross_loss'] < 0 else None, axis=1)
    rows: list[dict[str, object]] = []
    for _, row in g.iterrows():
        key = row[col]
        if pd.isna(key):
            value: object = None
        elif isinstance(key, (int, float)) and str(col).startswith('hour_'):
            value = int(key)
        else:
            value = str(key)
        rows.append(
            {
                col: value,
                'trades': int(row['trades']),
                'total_pnl': float(row['total_pnl']),
                'avg_trade': float(row['avg_trade']),
                'median_trade': float(row['median_trade']),
                'win_rate_pct': float(row['win_rate_pct']),
                'profit_factor': float(row['profit_factor']) if row['profit_factor'] is not None else None,
            }
        )
    return rows


def main() -> None:
    x = load_trades()
    x['trading_day_ny'] = (x['entry_time'].dt.tz_convert(NY_TZ) - pd.Timedelta(hours=CUTOFF_HOUR)).dt.floor('D')
    x['month_ny'] = x['trading_day_ny'].dt.strftime('%Y-%m')
    x['weekday_hkt'] = x['entry_time'].dt.tz_convert(HK_TZ).dt.day_name()
    x['hour_hkt'] = x['entry_time'].dt.tz_convert(HK_TZ).dt.hour
    x['hour_ny'] = x['entry_time'].dt.tz_convert(NY_TZ).dt.hour
    x['session'] = x[['hour_hkt', 'hour_ny']].apply(session_name, axis=1)

    payload = {
        'coverage': {
            'trades': int(len(x)),
            'entry_start_utc': x['entry_time'].min().isoformat(),
            'entry_end_utc': x['entry_time'].max().isoformat(),
        },
        'monthly_ny_trading_day': summarize(x, 'month_ny'),
        'weekday_hkt': summarize(x, 'weekday_hkt'),
        'hour_hkt': summarize(x, 'hour_hkt'),
        'hour_ny': summarize(x, 'hour_ny'),
        'session': summarize(x, 'session'),
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()

