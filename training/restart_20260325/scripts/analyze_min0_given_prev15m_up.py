#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data import DataLoader


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
    df = pd.DataFrame(
        {
            'ts': pd.to_datetime(raw['timestamp'], unit='ms', utc=True),
            'open': raw['openPrice'].astype(float),
            'high': raw['highPrice'].astype(float),
            'low': raw['lowPrice'].astype(float),
            'close': raw['closePrice'].astype(float),
        }
    ).sort_values('ts').reset_index(drop=True)
    return df


def build_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df['slot'] = df['ts'].dt.minute % 15
    df['block_start'] = df['ts'].dt.floor('15min')
    df['prev_close_1m'] = df['close'].shift(1)

    minute0 = df.loc[df['slot'] == 0, ['ts', 'block_start', 'open', 'high', 'low', 'close', 'prev_close_1m']].copy()
    minute0 = minute0.rename(
        columns={
            'ts': 'minute0_ts',
            'open': 'm0_open',
            'high': 'm0_high',
            'low': 'm0_low',
            'close': 'm0_close',
        }
    )

    blocks = df.groupby('block_start').agg(
        block_open=('open', 'first'),
        block_close=('close', 'last'),
        block_high=('high', 'max'),
        block_low=('low', 'min'),
        rows=('close', 'size'),
    ).reset_index()
    blocks['block_change'] = blocks['block_close'] - blocks['block_open']
    blocks['block_up'] = blocks['block_change'] > 0
    blocks['block_down'] = blocks['block_change'] < 0
    blocks['prev_block_change'] = blocks['block_change'].shift(1)
    blocks['prev_block_up'] = blocks['block_up'].shift(1)
    blocks['prev_block_down'] = blocks['block_down'].shift(1)

    analysis = minute0.merge(
        blocks[['block_start', 'prev_block_change', 'prev_block_up', 'prev_block_down']],
        on='block_start',
        how='left',
    )

    analysis['m0_down_close_open'] = analysis['m0_close'] < analysis['m0_open']
    analysis['m0_down_prev_close'] = analysis['m0_close'] < analysis['prev_close_1m']
    analysis['m0_low_below_open'] = analysis['m0_low'] < analysis['m0_open']
    analysis['m0_low_below_prev_close'] = analysis['m0_low'] < analysis['prev_close_1m']
    analysis['m0_change'] = analysis['m0_close'] - analysis['m0_open']
    analysis['m0_open_minus_low'] = analysis['m0_open'] - analysis['m0_low']

    minute0_ts = pd.to_datetime(analysis['minute0_ts'], utc=True)
    minute0_ny = minute0_ts.dt.tz_convert('America/New_York')
    analysis['ny_day_name'] = minute0_ny.dt.day_name()
    analysis['ny_hour'] = minute0_ny.dt.hour
    analysis['is_ny_session_broad'] = analysis['ny_hour'].between(6, 17, inclusive='both')

    filters = load_time_filters(ROOT / 'ml_config.json')
    analysis['is_ny_session_config'] = [
        bool(hour in filters.get('ny', {}).get(day, set()))
        for day, hour in zip(analysis['ny_day_name'], analysis['ny_hour'])
    ]

    valid = analysis.dropna(subset=['prev_block_up']).copy()
    return valid, blocks


def summarize_group(name: str, sub: pd.DataFrame) -> dict[str, float | int | str]:
    return {
        'group': name,
        'rows': int(len(sub)),
        'pct_m0_down_close_open': float(sub['m0_down_close_open'].mean() * 100),
        'pct_m0_down_prev_close': float(sub['m0_down_prev_close'].mean() * 100),
        'pct_m0_low_below_open': float(sub['m0_low_below_open'].mean() * 100),
        'pct_m0_low_below_prev_close': float(sub['m0_low_below_prev_close'].mean() * 100),
        'avg_m0_change': float(sub['m0_change'].mean()),
        'median_m0_change': float(sub['m0_change'].median()),
        'avg_m0_open_minus_low': float(sub['m0_open_minus_low'].mean()),
        'median_m0_open_minus_low': float(sub['m0_open_minus_low'].median()),
        'avg_prev_block_change': float(sub['prev_block_change'].mean()),
    }


def main() -> int:
    out_dir = ROOT / 'training' / 'restart_20260325' / 'reports'
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_1m_gold(start_date='2025-05-20', end_date='2026-02-04')
    analysis, _ = build_analysis(df)

    def build_scope(scope_name: str, scope_mask: pd.Series) -> list[dict[str, float | int | str]]:
        scoped = analysis.loc[scope_mask].copy()
        prev_up_mask = scoped['prev_block_up'].eq(True)
        prev_down_mask = scoped['prev_block_down'].eq(True)
        prev_not_up_mask = ~prev_up_mask
        return [
            summarize_group(f'{scope_name}__all', scoped),
            summarize_group(f'{scope_name}__prev_15m_up', scoped.loc[prev_up_mask]),
            summarize_group(f'{scope_name}__prev_15m_not_up', scoped.loc[prev_not_up_mask]),
            summarize_group(f'{scope_name}__prev_15m_down', scoped.loc[prev_down_mask]),
        ]

    summary_rows = []
    summary_rows.extend(build_scope('all_data', analysis['prev_block_up'].notna()))
    summary_rows.extend(build_scope('ny_session_broad', analysis['is_ny_session_broad']))
    summary_rows.extend(build_scope('ny_session_config', analysis['is_ny_session_config']))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / 'min0_given_prev15m_up_summary.csv'
    detail_csv = out_dir / 'min0_given_prev15m_up_detail.csv'
    text_out = out_dir / 'min0_given_prev15m_up_summary.txt'

    summary_df.to_csv(summary_csv, index=False)
    analysis.to_csv(detail_csv, index=False)

    def row(group: str) -> pd.Series:
        return summary_df.loc[summary_df['group'] == group].iloc[0]

    overall = row('all_data__all')
    prev_up = row('all_data__prev_15m_up')
    prev_not_up = row('all_data__prev_15m_not_up')
    broad_overall = row('ny_session_broad__all')
    broad_prev_up = row('ny_session_broad__prev_15m_up')
    config_overall = row('ny_session_config__all')
    config_prev_up = row('ny_session_config__prev_15m_up')

    uplift_close_open = prev_up['pct_m0_down_close_open'] - overall['pct_m0_down_close_open']
    uplift_prev_close = prev_up['pct_m0_down_prev_close'] - overall['pct_m0_down_prev_close']

    lines = [
        'Conditional minute-0 analysis: previous 15-minute candle up',
        '===========================================================',
        '',
        'Definition of previous 15m candle up:',
        '  previous block close - previous block open > 0',
        '',
        f"All blocks with previous block available        : {overall['rows']}",
        f"Blocks where previous 15m candle was up         : {prev_up['rows']}",
        f"Blocks where previous 15m candle was not up     : {prev_not_up['rows']}",
        '',
        'Metric 1: minute 0 closes below its own open',
        f"  overall                                       : {overall['pct_m0_down_close_open']:.2f}%",
        f"  if previous 15m candle was up                 : {prev_up['pct_m0_down_close_open']:.2f}%",
        f"  uplift vs overall                             : {uplift_close_open:+.2f} pct-pts",
        '',
        'Metric 2: minute 0 closes below previous minute close',
        f"  overall                                       : {overall['pct_m0_down_prev_close']:.2f}%",
        f"  if previous 15m candle was up                 : {prev_up['pct_m0_down_prev_close']:.2f}%",
        f"  uplift vs overall                             : {uplift_prev_close:+.2f} pct-pts",
        '',
        'Metric 3: minute 0 low breaks below previous minute close',
        f"  overall                                       : {overall['pct_m0_low_below_prev_close']:.2f}%",
        f"  if previous 15m candle was up                 : {prev_up['pct_m0_low_below_prev_close']:.2f}%",
        '',
        'Average minute-0 change (close - open)',
        f"  overall                                       : {overall['avg_m0_change']:.4f}",
        f"  if previous 15m candle was up                 : {prev_up['avg_m0_change']:.4f}",
        '',
        'Average minute-0 open - low',
        f"  overall                                       : {overall['avg_m0_open_minus_low']:.4f}",
        f"  if previous 15m candle was up                 : {prev_up['avg_m0_open_minus_low']:.4f}",
        '',
        'NY session (broad: 06:00-17:00 America/New_York)',
        f"  rows in NY session                            : {int(broad_overall['rows'])}",
        f"  avg minute-0 open-low overall                 : {broad_overall['avg_m0_open_minus_low']:.4f}",
        f"  avg minute-0 open-low if previous 15m up      : {broad_prev_up['avg_m0_open_minus_low']:.4f}",
        f"  pct minute-0 close<open overall               : {broad_overall['pct_m0_down_close_open']:.2f}%",
        f"  pct minute-0 close<open if previous 15m up    : {broad_prev_up['pct_m0_down_close_open']:.2f}%",
        '',
        'NY session (config-filtered hours from ml_config.json)',
        f"  rows in config NY session                     : {int(config_overall['rows'])}",
        f"  avg minute-0 open-low overall                 : {config_overall['avg_m0_open_minus_low']:.4f}",
        f"  avg minute-0 open-low if previous 15m up      : {config_prev_up['avg_m0_open_minus_low']:.4f}",
        f"  pct minute-0 close<open overall               : {config_overall['pct_m0_down_close_open']:.2f}%",
        f"  pct minute-0 close<open if previous 15m up    : {config_prev_up['pct_m0_down_close_open']:.2f}%",
        '',
    ]

    if uplift_close_open > 0:
        lines.append('Conclusion: yes, minute 0 is more likely to be down after an up 15m candle by the close<open definition.')
    else:
        lines.append('Conclusion: no, minute 0 is not more likely to be down after an up 15m candle by the close<open definition.')

    text_out.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('\n'.join(lines))
    print(f'saved {summary_csv}')
    print(f'saved {detail_csv}')
    print(f'saved {text_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

