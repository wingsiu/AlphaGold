#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data import DataLoader


THRESHOLDS = [5.0, 10.0]


def load_time_filters(config_file: Path) -> dict[str, dict[str, set[int]]]:
    payload = json.loads(config_file.read_text(encoding='utf-8'))
    raw_filters = payload.get('time_filters', {})
    normalized: dict[str, dict[str, set[int]]] = {'asia': {}, 'ny': {}}
    for session in ('asia', 'ny'):
        session_map = raw_filters.get(session, {})
        if not isinstance(session_map, dict):
            continue
        for day_name, hours in session_map.items():
            if isinstance(hours, list):
                normalized[session][str(day_name)] = {int(h) for h in hours if isinstance(h, (int, float))}
    return normalized


def load_1m_gold(start_date: str, end_date: str) -> pd.DataFrame:
    raw = DataLoader().load_data('gold_prices', start_date=start_date, end_date=end_date)
    return pd.DataFrame(
        {
            'ts': pd.to_datetime(raw['timestamp'], unit='ms', utc=True),
            'open': raw['openPrice'].astype(float),
            'high': raw['highPrice'].astype(float),
            'low': raw['lowPrice'].astype(float),
            'close': raw['closePrice'].astype(float),
        }
    ).sort_values('ts').reset_index(drop=True)


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['slot'] = df['ts'].dt.minute % 15
    df['block_start'] = df['ts'].dt.floor('15min')

    minute0 = df.loc[df['slot'] == 0, ['ts', 'block_start', 'open', 'low', 'close']].copy()
    minute0 = minute0.rename(columns={'ts': 'minute0_ts', 'open': 'm0_open', 'low': 'm0_low', 'close': 'm0_close'})
    minute0['m0_open_minus_low'] = minute0['m0_open'] - minute0['m0_low']
    minute0['m0_open_minus_close'] = minute0['m0_open'] - minute0['m0_close']

    blocks = df.groupby('block_start').agg(
        block_open=('open', 'first'),
        block_high=('high', 'max'),
        block_low=('low', 'min'),
        block_close=('close', 'last'),
        rows=('close', 'size'),
    ).reset_index()
    blocks['block_change'] = blocks['block_close'] - blocks['block_open']
    blocks['block_down'] = blocks['block_change'] < 0
    blocks['block_up'] = blocks['block_change'] > 0

    out = minute0.merge(blocks, on='block_start', how='inner')
    out['continue_down_after_m0'] = out['block_close'] < out['m0_close']
    out['recover_above_m0_close'] = out['block_close'] > out['m0_close']
    out['recover_above_m0_open'] = out['block_close'] > out['m0_open']

    ts_ny = pd.to_datetime(out['minute0_ts'], utc=True).dt.tz_convert('America/New_York')
    out['ny_day_name'] = ts_ny.dt.day_name()
    out['ny_hour'] = ts_ny.dt.hour
    out['is_ny_session_broad'] = out['ny_hour'].between(6, 17, inclusive='both')

    filters = load_time_filters(ROOT / 'ml_config.json')
    out['is_ny_session_config'] = [
        bool(hour in filters.get('ny', {}).get(day, set()))
        for day, hour in zip(out['ny_day_name'], out['ny_hour'])
    ]

    return out


def summarize_threshold(scope_name: str, data: pd.DataFrame, threshold: float) -> dict[str, float | int | str]:
    sub = data.loc[data['m0_open_minus_low'] > threshold].copy()
    if sub.empty:
        return {
            'scope': scope_name,
            'threshold_open_minus_low_gt': threshold,
            'rows': 0,
            'pct_block_down': float('nan'),
            'pct_continue_down_after_m0': float('nan'),
            'pct_recover_above_m0_close': float('nan'),
            'pct_recover_above_m0_open': float('nan'),
            'avg_block_change': float('nan'),
            'median_block_change': float('nan'),
            'avg_m0_open_minus_low': float('nan'),
        }
    return {
        'scope': scope_name,
        'threshold_open_minus_low_gt': threshold,
        'rows': int(len(sub)),
        'pct_block_down': float(sub['block_down'].mean() * 100),
        'pct_continue_down_after_m0': float(sub['continue_down_after_m0'].mean() * 100),
        'pct_recover_above_m0_close': float(sub['recover_above_m0_close'].mean() * 100),
        'pct_recover_above_m0_open': float(sub['recover_above_m0_open'].mean() * 100),
        'avg_block_change': float(sub['block_change'].mean()),
        'median_block_change': float(sub['block_change'].median()),
        'avg_m0_open_minus_low': float(sub['m0_open_minus_low'].mean()),
    }


def summarize_baseline(scope_name: str, data: pd.DataFrame) -> dict[str, float | int | str]:
    return {
        'scope': scope_name,
        'threshold_open_minus_low_gt': 0.0,
        'rows': int(len(data)),
        'pct_block_down': float(data['block_down'].mean() * 100),
        'pct_continue_down_after_m0': float(data['continue_down_after_m0'].mean() * 100),
        'pct_recover_above_m0_close': float(data['recover_above_m0_close'].mean() * 100),
        'pct_recover_above_m0_open': float(data['recover_above_m0_open'].mean() * 100),
        'avg_block_change': float(data['block_change'].mean()),
        'median_block_change': float(data['block_change'].median()),
        'avg_m0_open_minus_low': float(data['m0_open_minus_low'].mean()),
    }


def main() -> int:
    out_dir = ROOT / 'training' / 'restart_20260325' / 'reports'
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_1m_gold(start_date='2025-05-20', end_date='2026-02-04')
    data = build_dataset(df)

    scopes = {
        'all_data': data,
        'ny_session_broad': data.loc[data['is_ny_session_broad']].copy(),
        'ny_session_config': data.loc[data['is_ny_session_config']].copy(),
    }

    rows: list[dict[str, float | int | str]] = []
    for scope_name, scoped in scopes.items():
        rows.append(summarize_baseline(scope_name, scoped))
        for threshold in THRESHOLDS:
            rows.append(summarize_threshold(scope_name, scoped, threshold))

    summary = pd.DataFrame(rows)
    summary_csv = out_dir / 'min0_drop_thresholds_summary.csv'
    detail_csv = out_dir / 'min0_drop_thresholds_detail.csv'
    text_out = out_dir / 'min0_drop_thresholds_summary.txt'
    summary.to_csv(summary_csv, index=False)
    data.to_csv(detail_csv, index=False)

    def pick(scope: str, thr: float) -> pd.Series:
        return summary.loc[(summary['scope'] == scope) & (summary['threshold_open_minus_low_gt'] == thr)].iloc[0]

    lines = [
        'Minute-0 drop threshold analysis',
        '===============================',
        '',
        'Definition:',
        '  minute-0 drop = minute0 open - minute0 low',
        '  full 15m drop = block close < block open',
        '',
    ]

    for scope in ('all_data', 'ny_session_broad', 'ny_session_config'):
        base = pick(scope, 0.0)
        lines.append(scope)
        lines.append(f"  baseline blocks                  : {int(base['rows'])}")
        lines.append(f"  baseline pct full-15m down       : {base['pct_block_down']:.2f}%")
        lines.append(f"  baseline avg minute0 open-low    : {base['avg_m0_open_minus_low']:.4f}")
        for thr in THRESHOLDS:
            row = pick(scope, thr)
            lines.append(f"  minute0 open-low > {thr:.0f} rows          : {int(row['rows'])}")
            lines.append(f"  minute0 open-low > {thr:.0f} pct full-15m down : {row['pct_block_down']:.2f}%")
            lines.append(f"  minute0 open-low > {thr:.0f} pct continue down : {row['pct_continue_down_after_m0']:.2f}%")
            lines.append(f"  minute0 open-low > {thr:.0f} avg block change : {row['avg_block_change']:.4f}")
        lines.append('')

    text_out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'saved {summary_csv}')
    print(f'saved {detail_csv}')
    print(f'saved {text_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

