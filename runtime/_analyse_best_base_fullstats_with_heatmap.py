#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path('/Users/alpha/Desktop/python/AlphaGold')
STATS_PATH = ROOT / 'runtime/backtest_best_base_corrected_directional_pnl_fullstats.json'
TRADES_PATH = ROOT / 'training/backtest_trades_best_base_corrected.csv'

DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
SESSION_SPECS: dict[str, dict[str, Any]] = {
    'hkt': {
        'label': 'HKT',
        'timezone': 'Asia/Hong_Kong',
        'start_hour': 8,
        'start_minute': 0,
        'end_hour': 16,
        'end_minute': 0,
        'note': 'HKT session window 08:00 <= local time < 16:00',
    },
    'london': {
        'label': 'London',
        'timezone': 'Europe/London',
        'start_hour': 8,
        'start_minute': 0,
        'end_hour': 16,
        'end_minute': 30,
        'note': 'London session window 08:00 <= local time < 16:30 (DST-aware)',
    },
    'ny': {
        'label': 'NY',
        'timezone': 'America/New_York',
        'start_hour': 9,
        'start_minute': 30,
        'end_hour': 16,
        'end_minute': 0,
        'note': 'NY session window 09:30 <= local time < 16:00 (DST-aware)',
    },
}


def _load_trades() -> pd.DataFrame:
    df = pd.read_csv(TRADES_PATH)
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    return df


def _hour_values_for_session(spec: dict[str, Any]) -> list[int]:
    start_hour = int(spec['start_hour'])
    end_hour = int(spec['end_hour'])
    end_minute = int(spec['end_minute'])
    last_hour = end_hour if end_minute > 0 else end_hour - 1
    return list(range(start_hour, last_hour + 1))


def _hour_label(session_key: str, hour: int) -> str:
    if session_key == 'ny' and hour == 9:
        return '09:30'
    return f'{hour:02d}:00'


def _in_session(local_ts: pd.Series, spec: dict[str, Any]) -> pd.Series:
    minute_of_day = local_ts.dt.hour * 60 + local_ts.dt.minute
    start_min = int(spec['start_hour']) * 60 + int(spec['start_minute'])
    end_min = int(spec['end_hour']) * 60 + int(spec['end_minute'])
    return (minute_of_day >= start_min) & (minute_of_day < end_min)


def _safe_pct_wins(pnl: pd.Series) -> float | None:
    if len(pnl) == 0:
        return None
    return float((pnl > 0).mean() * 100.0)


def _render_table(title: str, day_rows: dict[str, dict[str, Any]], hour_labels: list[str], value_format) -> str:
    col_width = max(8, max(len(h) for h in hour_labels) + 1)
    lines = [title]
    header = f"{'Day':<12}" + ''.join(h.rjust(col_width) for h in hour_labels)
    lines.append(header)
    lines.append('-' * len(header))
    for day in DAY_ORDER:
        vals = day_rows[day]
        line = f'{day:<12}'
        for hour in hour_labels:
            line += value_format(vals[hour]).rjust(col_width)
        lines.append(line)
    return '\n'.join(lines)


def _build_session_heatmap(df: pd.DataFrame, session_key: str, spec: dict[str, Any]) -> dict[str, Any]:
    tz = ZoneInfo(str(spec['timezone']))
    local_ts = df['entry_time'].dt.tz_convert(tz)
    sdf = df.loc[_in_session(local_ts, spec)].copy()
    local_ts = local_ts.loc[sdf.index]
    sdf['local_time'] = local_ts
    sdf['local_day'] = local_ts.dt.day_name()
    sdf['local_hour'] = local_ts.dt.hour

    hour_values = _hour_values_for_session(spec)
    hour_labels = [_hour_label(session_key, hour) for hour in hour_values]
    hour_label_map = {hour: _hour_label(session_key, hour) for hour in hour_values}

    cell_stats: dict[str, dict[str, Any]] = {day: {label: None for label in hour_labels} for day in DAY_ORDER}
    trade_count_heatmap: dict[str, dict[str, int | None]] = {day: {label: None for label in hour_labels} for day in DAY_ORDER}
    win_rate_heatmap: dict[str, dict[str, float | None]] = {day: {label: None for label in hour_labels} for day in DAY_ORDER}
    avg_trade_heatmap: dict[str, dict[str, float | None]] = {day: {label: None for label in hour_labels} for day in DAY_ORDER}
    total_pnl_heatmap: dict[str, dict[str, float | None]] = {day: {label: None for label in hour_labels} for day in DAY_ORDER}

    for day in DAY_ORDER:
        day_df = sdf[sdf['local_day'] == day]
        for hour in hour_values:
            label = hour_label_map[hour]
            hour_df = day_df[day_df['local_hour'] == hour]
            if hour_df.empty:
                continue
            pnl = hour_df['pnl'].astype(float)
            trades = int(len(hour_df))
            total_pnl = float(pnl.sum())
            avg_trade = float(pnl.mean())
            win_rate_pct = _safe_pct_wins(pnl)
            stats = {
                'trades': trades,
                'total_pnl': total_pnl,
                'avg_trade': avg_trade,
                'win_rate_pct': win_rate_pct,
            }
            cell_stats[day][label] = stats
            trade_count_heatmap[day][label] = trades
            win_rate_heatmap[day][label] = win_rate_pct
            avg_trade_heatmap[day][label] = avg_trade
            total_pnl_heatmap[day][label] = total_pnl

    session_pnl = sdf['pnl'].astype(float) if len(sdf) else pd.Series(dtype='float64')
    rendered_tables = {
        'trade_count': _render_table(
            f"{spec['label']} SESSION HEATMAP — TRADE COUNT",
            trade_count_heatmap,
            hour_labels,
            lambda v: '--' if v is None else f'{int(v)}',
        ),
        'win_rate_pct': _render_table(
            f"{spec['label']} SESSION HEATMAP — WIN RATE %",
            win_rate_heatmap,
            hour_labels,
            lambda v: '--' if v is None else f'{float(v):.1f}% ',
        ),
        'avg_trade': _render_table(
            f"{spec['label']} SESSION HEATMAP — AVG TRADE PNL",
            avg_trade_heatmap,
            hour_labels,
            lambda v: '--' if v is None else f'{float(v):.2f}',
        ),
        'total_pnl': _render_table(
            f"{spec['label']} SESSION HEATMAP — TOTAL PNL",
            total_pnl_heatmap,
            hour_labels,
            lambda v: '--' if v is None else f'{float(v):.2f}',
        ),
    }

    return {
        'label': spec['label'],
        'timezone': spec['timezone'],
        'session_window': {
            'start': f"{int(spec['start_hour']):02d}:{int(spec['start_minute']):02d}",
            'end_exclusive': f"{int(spec['end_hour']):02d}:{int(spec['end_minute']):02d}",
            'note': spec['note'],
        },
        'trades': int(len(sdf)),
        'total_pnl': float(session_pnl.sum()) if len(sdf) else 0.0,
        'avg_trade': float(session_pnl.mean()) if len(sdf) else None,
        'win_rate_pct': _safe_pct_wins(session_pnl),
        'hour_labels': hour_labels,
        'day_labels': DAY_ORDER,
        'cell_stats': cell_stats,
        'trade_count_heatmap': trade_count_heatmap,
        'win_rate_pct_heatmap': win_rate_heatmap,
        'avg_trade_heatmap': avg_trade_heatmap,
        'total_pnl_heatmap': total_pnl_heatmap,
        'rendered_tables': rendered_tables,
    }


def main() -> None:
    payload = json.loads(STATS_PATH.read_text())
    all_stats = payload['all']
    long_stats = payload['long_up']
    short_stats = payload['short_down']
    time_dist = all_stats['time_distribution']
    trades_df = _load_trades()
    session_heatmaps = {key: _build_session_heatmap(trades_df, key, spec) for key, spec in SESSION_SPECS.items()}

    out = {
        'top_level': {
            'trades': payload['trades'],
            'total_pnl': payload['total_pnl'],
            'avg_trade': payload['avg_trade'],
            'n_days': payload['n_days'],
            'avg_trades_per_day': payload['avg_trades_per_day'],
            'avg_day': payload['avg_day'],
            'positive_days_pct': payload['positive_days_pct'],
            'max_drawdown': payload['max_drawdown'],
            'long': payload['long'],
            'short': payload['short'],
            'streaks': payload['streaks'],
            'reverse_signal': payload['reverse_signal'],
            'target_hit': payload['target_hit'],
            'timeout': payload['timeout'],
        },
        'all_full': {
            'trades': all_stats['trades'],
            'total_pnl': all_stats['total_pnl'],
            'avg_trade': all_stats['avg_trade'],
            'median_trade': all_stats['median_trade'],
            'win_rate_pct': all_stats['win_rate_pct'],
            'gross_profit': all_stats['gross_profit'],
            'gross_loss': all_stats['gross_loss'],
            'profit_factor': all_stats['profit_factor'],
            'best_trade': all_stats['best_trade'],
            'worst_trade': all_stats['worst_trade'],
            'avg_win': all_stats['avg_win'],
            'avg_loss': all_stats['avg_loss'],
            'avg_duration_min': all_stats['avg_duration_min'],
            'median_duration_min': all_stats['median_duration_min'],
            'min_duration_min': all_stats['min_duration_min'],
            'max_duration_min': all_stats['max_duration_min'],
            'median_day': all_stats['median_day'],
            'best_day': all_stats['best_day'],
            'worst_day': all_stats['worst_day'],
            'trade_max_drawdown': all_stats['trade_max_drawdown'],
            'daily_max_drawdown': all_stats['daily_max_drawdown'],
            'exit_reason_counts': all_stats['exit_reason_counts'],
            'reverse_signal_stats': all_stats['reverse_signal_stats'],
            'target_hit_stats': all_stats['target_hit_stats'],
            'timeout_stats': all_stats['timeout_stats'],
            'target_updates_mean': all_stats['target_updates_mean'],
            'target_updates_median': all_stats['target_updates_median'],
            'target_updates_max': all_stats['target_updates_max'],
        },
        'long_up_full': {
            'trades': long_stats['trades'],
            'total_pnl': long_stats['total_pnl'],
            'avg_trade': long_stats['avg_trade'],
            'median_trade': long_stats['median_trade'],
            'win_rate_pct': long_stats['win_rate_pct'],
            'gross_profit': long_stats['gross_profit'],
            'gross_loss': long_stats['gross_loss'],
            'profit_factor': long_stats['profit_factor'],
            'avg_duration_min': long_stats['avg_duration_min'],
            'median_duration_min': long_stats['median_duration_min'],
            'exit_reason_counts': long_stats['exit_reason_counts'],
            'timeout_stats': long_stats['timeout_stats'],
        },
        'short_down_full': {
            'trades': short_stats['trades'],
            'total_pnl': short_stats['total_pnl'],
            'avg_trade': short_stats['avg_trade'],
            'median_trade': short_stats['median_trade'],
            'win_rate_pct': short_stats['win_rate_pct'],
            'gross_profit': short_stats['gross_profit'],
            'gross_loss': short_stats['gross_loss'],
            'profit_factor': short_stats['profit_factor'],
            'avg_duration_min': short_stats['avg_duration_min'],
            'median_duration_min': short_stats['median_duration_min'],
            'exit_reason_counts': short_stats['exit_reason_counts'],
            'timeout_stats': short_stats['timeout_stats'],
        },
        'time_distribution': {
            'by_weekday_hkt': time_dist['by_weekday_hkt'],
            'by_hour_hkt': time_dist['by_hour_hkt'],
            'by_hour_ny': time_dist['by_hour_ny'],
            'by_session': time_dist['by_session'],
            'legacy_weekday_hour_hkt_heatmap': time_dist.get('weekday_hour_hkt_heatmap', {}),
        },
        'session_heatmaps': session_heatmaps,
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

