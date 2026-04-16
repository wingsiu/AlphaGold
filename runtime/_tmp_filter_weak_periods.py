#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path('/Users/alpha/Desktop/python/AlphaGold')
TRADES = ROOT / 'training/backtest_trades_best_base_corrected.csv'
BASE_REPORT = ROOT / 'training/backtest_report_best_base_corrected.json'
WEAK_JSON = ROOT / 'runtime/_tmp_weak_zone_gate_probs_v2.json'
OUT_CSV = ROOT / 'runtime/_tmp_backtest_trades_best_base_without_weak_periods.csv'
OUT_SUMMARY = ROOT / 'runtime/_tmp_backtest_best_base_without_weak_periods_summary.json'

SESSION_SPECS = {
    'hkt': {'timezone': 'Asia/Hong_Kong', 'start_hour': 8, 'start_minute': 0, 'end_hour': 16, 'end_minute': 0},
    'london': {'timezone': 'Europe/London', 'start_hour': 8, 'start_minute': 0, 'end_hour': 16, 'end_minute': 30},
    'ny': {'timezone': 'America/New_York', 'start_hour': 9, 'start_minute': 30, 'end_hour': 16, 'end_minute': 0},
}


def in_session(local_ts: pd.Series, spec: dict[str, int | str]) -> pd.Series:
    minute_of_day = local_ts.dt.hour * 60 + local_ts.dt.minute
    start_min = int(spec['start_hour']) * 60 + int(spec['start_minute'])
    end_min = int(spec['end_hour']) * 60 + int(spec['end_minute'])
    return (minute_of_day >= start_min) & (minute_of_day < end_min)


def session_hour_label(session_key: str, hour: int) -> str:
    if session_key == 'ny' and hour == 9:
        return '09:30'
    return f'{hour:02d}:00'


def cell_mask(df: pd.DataFrame, session: str, day: str, hour_label: str) -> pd.Series:
    spec = SESSION_SPECS[session]
    local_ts = df['entry_time'].dt.tz_convert(ZoneInfo(str(spec['timezone'])))
    return in_session(local_ts, spec) & (local_ts.dt.day_name() == day) & (local_ts.dt.hour.map(lambda h: session_hour_label(session, int(h))) == hour_label)


def main() -> None:
    trades = pd.read_csv(TRADES)
    trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
    trades['row_id'] = range(len(trades))
    weak_payload = json.loads(WEAK_JSON.read_text())
    base_report = json.loads(BASE_REPORT.read_text())

    weak_ids: set[int] = set()
    for cell in weak_payload['weak_cells']:
        mask = cell_mask(trades, str(cell['session']), str(cell['day']), str(cell['hour']))
        weak_ids.update(trades.loc[mask, 'row_id'].astype(int).tolist())

    kept = trades[~trades['row_id'].isin(sorted(weak_ids))].copy()
    removed = trades[trades['row_id'].isin(sorted(weak_ids))].copy()
    kept.drop(columns=['row_id'], inplace=True)
    removed.drop(columns=['row_id'], inplace=True)
    OUT_CSV.write_text(kept.to_csv(index=False), encoding='utf-8')

    baseline = base_report['directional_pnl']
    summary = {
        'weak_cell_definition': weak_payload['weak_cell_definition'],
        'weak_cells_count': len(weak_payload['weak_cells']),
        'baseline': {
            'trades': baseline['trades'],
            'total_pnl': baseline['total_pnl'],
            'avg_trade': baseline['avg_trade'],
            'avg_day': baseline['avg_day'],
            'positive_days_pct': baseline['positive_days_pct'],
            'max_drawdown': baseline['max_drawdown'],
        },
        'removed_union': {
            'trades': int(len(removed)),
            'total_pnl': float(removed['pnl'].astype(float).sum()) if len(removed) else 0.0,
            'avg_trade': float(removed['pnl'].astype(float).mean()) if len(removed) else None,
            'win_rate_pct': float((removed['pnl'].astype(float) > 0).mean() * 100.0) if len(removed) else None,
        },
        'kept_direct_filter_only': {
            'trades': int(len(kept)),
            'total_pnl': float(kept['pnl'].astype(float).sum()) if len(kept) else 0.0,
            'avg_trade': float(kept['pnl'].astype(float).mean()) if len(kept) else None,
            'win_rate_pct': float((kept['pnl'].astype(float) > 0).mean() * 100.0) if len(kept) else None,
        },
        'delta_kept_minus_baseline': {
            'trades': int(len(kept) - int(baseline['trades'])),
            'total_pnl': float(kept['pnl'].astype(float).sum()) - float(baseline['total_pnl']),
            'avg_trade': (float(kept['pnl'].astype(float).mean()) if len(kept) else None) - float(baseline['avg_trade']) if len(kept) else None,
        },
        'filtered_csv': str(OUT_CSV),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

