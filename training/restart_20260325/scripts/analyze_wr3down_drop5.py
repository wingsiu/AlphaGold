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


LOOKBACK_WR = 90
DEFAULT_TARGET_VALUES = [3.0, 5.0, 7.5, 10.0]
DEFAULT_HOLD_BARS = 15


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='WR + 3-down drop analysis with optional time filters and target optimization')
    p.add_argument('--start-date', default='2025-05-20')
    p.add_argument('--end-date', default='2026-02-04')
    p.add_argument('--wr-lookback', type=int, default=LOOKBACK_WR)
    p.add_argument('--hold-bars', type=int, default=DEFAULT_HOLD_BARS)
    p.add_argument(
        '--target-values',
        default=','.join(str(x) for x in DEFAULT_TARGET_VALUES),
        help='Comma-separated drop targets in points, e.g. 3,5,7.5,10',
    )
    p.add_argument(
        '--session-filter',
        choices=['off', 'ny_broad', 'ny_config'],
        default='off',
        help='Entry-time filter: off, NY broad hours, or NY config hours from ml_config.json',
    )
    p.add_argument('--config-file', default='ml_config.json', help='Config file for ny_config session filter')
    return p


def parse_target_values(raw: str) -> list[float]:
    vals: list[float] = []
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError('No valid target values parsed from --target-values')
    return vals


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


def compute_wr(df: pd.DataFrame, lookback: int = LOOKBACK_WR) -> pd.Series:
    hh = df['high'].rolling(lookback, min_periods=lookback).max()
    ll = df['low'].rolling(lookback, min_periods=lookback).min()
    rng = hh - ll
    wr = -100.0 * (hh - df['close']) / rng
    wr = wr.where(rng != 0)
    return wr


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


def build_signals(df: pd.DataFrame, wr_lookback: int) -> pd.DataFrame:
    out = df.copy()
    out['wr'] = compute_wr(out, wr_lookback)
    out['change_1m'] = out['close'] - out['close'].shift(1)
    out['change_3bar_total'] = out['change_1m'] + out['change_1m'].shift(1) + out['change_1m'].shift(2)
    out['down'] = out['close'] < out['open']
    out['three_consecutive_down'] = out['down'] & out['down'].shift(1) & out['down'].shift(2)
    out['threebar_total_change_lt_minus5'] = out['change_3bar_total'] < -5.0
    out['wr_between_minus20_minus80'] = out['wr'].between(-80.0, -20.0, inclusive='both')
    out['signal'] = (
        out['three_consecutive_down']
        & out['threebar_total_change_lt_minus5']
        & out['wr_between_minus20_minus80']
    )
    ts_ny = pd.to_datetime(out['ts'], utc=True).dt.tz_convert('America/New_York')
    out['ny_day_name'] = ts_ny.dt.day_name()
    out['ny_hour'] = ts_ny.dt.hour
    out['is_ny_broad'] = out['ny_hour'].between(6, 17, inclusive='both')
    return out


def apply_session_filter(df: pd.DataFrame, session_filter: str, config_file: Path) -> pd.Series:
    if session_filter == 'off':
        return pd.Series(True, index=df.index)
    if session_filter == 'ny_broad':
        return df['is_ny_broad'].fillna(False)
    if session_filter == 'ny_config':
        filters = load_time_filters(config_file)
        allowed_ny = filters.get('ny', {})
        return pd.Series(
            [
                bool(hour in allowed_ny.get(day, set()))
                for day, hour in zip(df['ny_day_name'], df['ny_hour'])
            ],
            index=df.index,
        )
    raise ValueError(f'Unknown session filter: {session_filter}')


def evaluate(df: pd.DataFrame, hold_bars: int, target_values: list[float], eligible_mask: pd.Series) -> pd.DataFrame:
    rows = []
    signal_count = int((df['signal'] & eligible_mask).sum())

    future_cols = [df['low'].shift(-i) for i in range(1, hold_bars + 1)]
    future_min = pd.concat(future_cols, axis=1).min(axis=1)
    valid = future_cols[-1].notna()

    sig_valid = df['signal'] & eligible_mask & valid
    base_valid = eligible_mask & valid

    sig_n = int(sig_valid.sum())
    base_n = int(base_valid.sum())

    max_drop = df['close'] - future_min

    for target in target_values:
        hit = future_min <= (df['close'] - target)

        sig_hit_rate = float(hit[sig_valid].mean() * 100.0) if sig_n else float('nan')
        base_hit_rate = float(hit[base_valid].mean() * 100.0) if base_n else float('nan')
        sig_avg_max_drop = float(max_drop[sig_valid].mean()) if sig_n else float('nan')
        base_avg_max_drop = float(max_drop[base_valid].mean()) if base_n else float('nan')

        rows.append(
            {
                'hold_bars': hold_bars,
                'target_pts': float(target),
                'signals': sig_n,
                'signal_hit_rate_pct': sig_hit_rate,
                'baseline_hit_rate_pct': base_hit_rate,
                'edge_vs_baseline_pct_pts': sig_hit_rate - base_hit_rate,
                'signal_avg_max_drop': sig_avg_max_drop,
                'baseline_avg_max_drop': base_avg_max_drop,
                'total_signals_in_sample': signal_count,
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    args = build_parser().parse_args()

    out_dir = ROOT / 'training' / 'restart_20260325' / 'reports'
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = parse_target_values(args.target_values)
    df = load_1m_gold(start_date=args.start_date, end_date=args.end_date)
    sig = build_signals(df, wr_lookback=args.wr_lookback)
    eligible_mask = apply_session_filter(sig, args.session_filter, ROOT / args.config_file)
    res = evaluate(sig, hold_bars=args.hold_bars, target_values=targets, eligible_mask=eligible_mask)

    detail_path = out_dir / 'wr3down_signals_detail.csv'
    summary_path = out_dir / 'wr3down_drop5_summary.csv'
    text_path = out_dir / 'wr3down_drop5_summary.txt'

    sig[
        [
            'ts',
            'open',
            'high',
            'low',
            'close',
            'wr',
            'change_1m',
            'change_3bar_total',
            'three_consecutive_down',
            'threebar_total_change_lt_minus5',
            'wr_between_minus20_minus80',
            'signal',
            'ny_day_name',
            'ny_hour',
            'is_ny_broad',
        ]
    ].to_csv(detail_path, index=False)
    res.to_csv(summary_path, index=False)

    total = int((sig['signal'] & eligible_mask).sum())
    best = res.sort_values('edge_vs_baseline_pct_pts', ascending=False).iloc[0]
    lines = [
        'WR + 3-down pattern analysis',
        '============================',
        '',
        f"Condition: Williams %R({args.wr_lookback}) between -80 and -20, and 3 consecutive down 1m candles (close < open) AND total close-to-close change across those 3 bars < -5.",
        f"Session filter: {args.session_filter}",
        f"Hold bars: {args.hold_bars}",
        f"Targets tested (pts): {targets}",
        f'Total signals (eligible): {total}',
        '',
        f"Best target by edge: {best['target_pts']:.2f} pts  edge={best['edge_vs_baseline_pct_pts']:.2f} pct-pts  signal_hit={best['signal_hit_rate_pct']:.2f}%  baseline_hit={best['baseline_hit_rate_pct']:.2f}%",
        '',
        'hold_bars | target_pts | signals | signal_hit_rate% | baseline_hit_rate% | edge_pct_pts | signal_avg_max_drop | baseline_avg_max_drop',
    ]

    for _, r in res.iterrows():
        lines.append(
            f"{int(r['hold_bars']):>9} | {r['target_pts']:>10.2f} | {int(r['signals']):>7} | {r['signal_hit_rate_pct']:>16.2f} | "
            f"{r['baseline_hit_rate_pct']:>18.2f} | {r['edge_vs_baseline_pct_pts']:>12.2f} | "
            f"{r['signal_avg_max_drop']:>19.4f} | {r['baseline_avg_max_drop']:>21.4f}"
        )

    text_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('\n'.join(lines))
    print(f'saved {summary_path}')
    print(f'saved {detail_path}')
    print(f'saved {text_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

