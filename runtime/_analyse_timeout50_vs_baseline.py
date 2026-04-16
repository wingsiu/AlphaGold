#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path('/Users/alpha/Desktop/python/AlphaGold')
BASE_REPORT_PATH = ROOT / 'training/backtest_report_best_base_corrected.json'
NEW_REPORT_PATH = ROOT / 'training/backtest_report_best_base_corrected_timeout50_noretrain.json'
BASE_TRADES_PATH = ROOT / 'training/backtest_trades_best_base_corrected.csv'
NEW_TRADES_PATH = ROOT / 'training/backtest_trades_best_base_corrected_timeout50_noretrain.csv'
NY_TZ = ZoneInfo('America/New_York')
HK_TZ = ZoneInfo('Asia/Hong_Kong')
CUTOFF_HOUR = 17


def _load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ('ts', 'entry_time', 'exit_time', 'last_target_time'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
    return df


def _bucket_stats(df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    x = df.copy()
    x['entry_time'] = pd.to_datetime(x['entry_time'], utc=True)
    x['hour_hkt'] = x['entry_time'].dt.tz_convert(HK_TZ).dt.hour
    x['hour_ny'] = x['entry_time'].dt.tz_convert(NY_TZ).dt.hour
    x['weekday_hkt'] = x['entry_time'].dt.tz_convert(HK_TZ).dt.day_name()

    def _session_name(row: pd.Series) -> str:
        h_hkt = int(row['hour_hkt'])
        h_ny = int(row['hour_ny'])
        if 7 <= h_hkt <= 15:
            return 'Asia'
        if 8 <= h_ny <= 16:
            return 'NY'
        return 'Other'

    x['session'] = x[['hour_hkt', 'hour_ny']].apply(_session_name, axis=1)

    def _summarize(col: str) -> list[dict[str, object]]:
        g = x.groupby(col)['pnl'].agg(
            trades='size',
            total_pnl='sum',
            avg_trade='mean',
            win_rate_pct=lambda s: float((s > 0).mean() * 100.0),
        ).reset_index()
        g = g.sort_values(['total_pnl', 'avg_trade', 'trades'], ascending=[False, False, False])
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
                    'win_rate_pct': float(row['win_rate_pct']),
                }
            )
        return rows

    return {
        'by_weekday_hkt': _summarize('weekday_hkt'),
        'by_hour_hkt': _summarize('hour_hkt'),
        'by_hour_ny': _summarize('hour_ny'),
        'by_session': _summarize('session'),
    }


def main() -> None:
    base_report = json.loads(BASE_REPORT_PATH.read_text())
    new_report = json.loads(NEW_REPORT_PATH.read_text())
    base_trades = _load_trades(BASE_TRADES_PATH)
    new_trades = _load_trades(NEW_TRADES_PATH)

    base_pnl = base_report['directional_pnl']
    new_pnl = new_report['directional_pnl']
    base_all = base_pnl['all']
    new_all = new_pnl['all']
    base_long = base_pnl['long_up']
    new_long = new_pnl['long_up']
    base_short = base_pnl['short_down']
    new_short = new_pnl['short_down']

    base_time = _bucket_stats(base_trades)
    new_time = new_all['time_distribution']
    new_long_time = new_long['time_distribution']
    new_short_time = new_short['time_distribution']

    payload = {
        'baseline': {
            'trades': int(base_pnl['trades']),
            'total_pnl': float(base_pnl['total_pnl']),
            'avg_trade': float(base_pnl['avg_trade']),
            'avg_day': float(base_pnl['avg_day']),
            'positive_days_pct': float(base_pnl['positive_days_pct']),
            'profit_factor': float(base_all['profit_factor']),
            'trade_max_drawdown': float(base_all['trade_max_drawdown']),
            'daily_max_drawdown': float(base_all['daily_max_drawdown']),
            'avg_duration_min': float(base_all['avg_duration_min']),
            'median_duration_min': float(base_all['median_duration_min']),
            'exit_reason_counts': base_all['exit_reason_counts'],
            'timeout_stats': base_all['timeout_stats'],
            'long': {
                'trades': int(base_long['trades']),
                'total_pnl': float(base_long['total_pnl']),
                'win_rate_pct': float(base_long['win_rate_pct']),
                'profit_factor': float(base_long['profit_factor']),
                'avg_duration_min': float(base_long['avg_duration_min']),
            },
            'short': {
                'trades': int(base_short['trades']),
                'total_pnl': float(base_short['total_pnl']),
                'win_rate_pct': float(base_short['win_rate_pct']),
                'profit_factor': float(base_short['profit_factor']),
                'avg_duration_min': float(base_short['avg_duration_min']),
            },
        },
        'timeout50': {
            'trades': int(new_pnl['trades']),
            'total_pnl': float(new_pnl['total_pnl']),
            'avg_trade': float(new_pnl['avg_trade']),
            'avg_day': float(new_pnl['avg_day']),
            'positive_days_pct': float(new_pnl['positive_days_pct']),
            'profit_factor': float(new_all['profit_factor']),
            'trade_max_drawdown': float(new_all['trade_max_drawdown']),
            'daily_max_drawdown': float(new_all['daily_max_drawdown']),
            'avg_duration_min': float(new_all['avg_duration_min']),
            'median_duration_min': float(new_all['median_duration_min']),
            'exit_reason_counts': new_all['exit_reason_counts'],
            'timeout_stats': new_all['timeout_stats'],
            'long': {
                'trades': int(new_long['trades']),
                'total_pnl': float(new_long['total_pnl']),
                'win_rate_pct': float(new_long['win_rate_pct']),
                'profit_factor': float(new_long['profit_factor']),
                'avg_duration_min': float(new_long['avg_duration_min']),
            },
            'short': {
                'trades': int(new_short['trades']),
                'total_pnl': float(new_short['total_pnl']),
                'win_rate_pct': float(new_short['win_rate_pct']),
                'profit_factor': float(new_short['profit_factor']),
                'avg_duration_min': float(new_short['avg_duration_min']),
            },
        },
        'delta_timeout50_minus_baseline': {
            'trades': int(new_pnl['trades'] - base_pnl['trades']),
            'total_pnl': round(float(new_pnl['total_pnl'] - base_pnl['total_pnl']), 2),
            'avg_trade': round(float(new_pnl['avg_trade'] - base_pnl['avg_trade']), 4),
            'avg_day': round(float(new_pnl['avg_day'] - base_pnl['avg_day']), 4),
            'positive_days_pct': round(float(new_pnl['positive_days_pct'] - base_pnl['positive_days_pct']), 4),
            'profit_factor': round(float(new_all['profit_factor'] - base_all['profit_factor']), 6),
            'trade_max_drawdown': round(float(new_all['trade_max_drawdown'] - base_all['trade_max_drawdown']), 2),
            'daily_max_drawdown': round(float(new_all['daily_max_drawdown'] - base_all['daily_max_drawdown']), 2),
            'avg_duration_min': round(float(new_all['avg_duration_min'] - base_all['avg_duration_min']), 4),
            'median_duration_min': round(float(new_all['median_duration_min'] - base_all['median_duration_min']), 4),
            'long_total_pnl': round(float(new_long['total_pnl'] - base_long['total_pnl']), 2),
            'short_total_pnl': round(float(new_short['total_pnl'] - base_short['total_pnl']), 2),
            'long_trades': int(new_long['trades'] - base_long['trades']),
            'short_trades': int(new_short['trades'] - base_short['trades']),
            'timeout_trades': int(new_all['timeout_stats']['trades'] - base_all['timeout_stats']['trades']),
            'timeout_avg_pnl': round(float((new_all['timeout_stats']['avg_pnl'] or 0.0) - (base_all['timeout_stats']['avg_pnl'] or 0.0)), 4),
        },
        'time_distribution': {
            'timeout50_all_best_weekday_hkt_by_total_pnl': new_time['by_weekday_hkt'][:3],
            'timeout50_all_best_hour_hkt_by_total_pnl': new_time['by_hour_hkt'][:5],
            'timeout50_all_best_hour_ny_by_total_pnl': new_time['by_hour_ny'][:5],
            'timeout50_all_sessions': new_time['by_session'],
            'timeout50_long_best_hour_hkt': new_long_time['by_hour_hkt'][:5],
            'timeout50_short_best_hour_hkt': new_short_time['by_hour_hkt'][:5],
            'baseline_all_best_weekday_hkt_by_total_pnl': base_time['by_weekday_hkt'][:3],
            'baseline_all_best_hour_hkt_by_total_pnl': base_time['by_hour_hkt'][:5],
            'baseline_all_best_hour_ny_by_total_pnl': base_time['by_hour_ny'][:5],
            'baseline_all_sessions': base_time['by_session'],
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()

