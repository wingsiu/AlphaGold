#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path('/Users/alpha/Desktop/python/AlphaGold')
TRADES = ROOT / 'training/backtest_trades_best_base_corrected.csv'
FULLSTATS = ROOT / 'runtime/backtest_best_base_corrected_directional_pnl_fullstats.json'
REPORT = ROOT / 'training/backtest_report_best_base_corrected.json'

DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
SESSION_SPECS = {
    'hkt': {'timezone': 'Asia/Hong_Kong', 'start_hour': 8, 'start_minute': 0, 'end_hour': 16, 'end_minute': 0},
    'london': {'timezone': 'Europe/London', 'start_hour': 8, 'start_minute': 0, 'end_hour': 16, 'end_minute': 30},
    'ny': {'timezone': 'America/New_York', 'start_hour': 9, 'start_minute': 30, 'end_hour': 16, 'end_minute': 0},
}
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


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


def summarize(df: pd.DataFrame) -> dict[str, object]:
    pnl = df['pnl'].astype(float)
    winners = df[pnl > 0]
    losers = df[pnl <= 0]
    return {
        'trades': int(len(df)),
        'total_pnl': float(pnl.sum()),
        'avg_trade': float(pnl.mean()) if len(df) else None,
        'win_rate_pct': float((pnl > 0).mean() * 100.0) if len(df) else None,
        'entry_prob_mean': float(df['entry_signal_prob'].astype(float).mean()) if len(df) else None,
        'entry_prob_median': float(df['entry_signal_prob'].astype(float).median()) if len(df) else None,
        'last_prob_mean': float(df['last_signal_prob'].astype(float).mean()) if len(df) else None,
        'last_prob_median': float(df['last_signal_prob'].astype(float).median()) if len(df) else None,
        'winner_entry_prob_mean': float(winners['entry_signal_prob'].astype(float).mean()) if len(winners) else None,
        'loser_entry_prob_mean': float(losers['entry_signal_prob'].astype(float).mean()) if len(losers) else None,
        'winner_last_prob_mean': float(winners['last_signal_prob'].astype(float).mean()) if len(winners) else None,
        'loser_last_prob_mean': float(losers['last_signal_prob'].astype(float).mean()) if len(losers) else None,
    }


def sweep_thresholds(df: pd.DataFrame, column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_pnl = float(df['pnl'].astype(float).sum())
    base_trades = int(len(df))
    for thr in THRESHOLDS:
        kept = df[df[column].astype(float) >= thr]
        removed = df[df[column].astype(float) < thr]
        kept_pnl = float(kept['pnl'].astype(float).sum()) if len(kept) else 0.0
        removed_pnl = float(removed['pnl'].astype(float).sum()) if len(removed) else 0.0
        rows.append({
            'threshold': thr,
            'kept_trades': int(len(kept)),
            'kept_total_pnl': kept_pnl,
            'kept_avg_trade': float(kept['pnl'].astype(float).mean()) if len(kept) else None,
            'kept_win_rate_pct': float((kept['pnl'].astype(float) > 0).mean() * 100.0) if len(kept) else None,
            'removed_trades': int(len(removed)),
            'removed_total_pnl': removed_pnl,
            'removed_loss_trades': int((removed['pnl'].astype(float) <= 0).sum()) if len(removed) else 0,
            'removed_win_trades': int((removed['pnl'].astype(float) > 0).sum()) if len(removed) else 0,
            'delta_total_pnl_vs_base': kept_pnl - base_pnl,
            'trade_reduction_pct': (1.0 - (len(kept) / base_trades)) * 100.0 if base_trades else 0.0,
        })
    return rows


def main() -> None:
    report = json.loads(REPORT.read_text())
    stats = json.loads(FULLSTATS.read_text())
    trades = pd.read_csv(TRADES)
    trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
    trades['row_id'] = range(len(trades))

    weak_cells: list[dict[str, object]] = []
    all_sessions = stats['all']['time_distribution']['session_heatmaps']
    for session_name, sess in all_sessions.items():
        for day in DAY_ORDER:
            for hour, cell in sess['cell_stats'][day].items():
                if not cell:
                    continue
                if int(cell['trades']) < 5:
                    continue
                if float(cell['total_pnl']) < 0:
                    weak_cells.append({
                        'session': session_name,
                        'day': day,
                        'hour': hour,
                        'trades': int(cell['trades']),
                        'total_pnl': float(cell['total_pnl']),
                        'avg_trade': float(cell['avg_trade']),
                        'win_rate_pct': float(cell['win_rate_pct']),
                    })
    weak_cells = sorted(weak_cells, key=lambda r: (r['total_pnl'], r['avg_trade']))

    per_cell: list[dict[str, object]] = []
    weak_row_ids: set[int] = set()
    for cell in weak_cells:
        mask = cell_mask(trades, str(cell['session']), str(cell['day']), str(cell['hour']))
        cell_df = trades[mask].copy()
        weak_row_ids.update(cell_df['row_id'].astype(int).tolist())
        per_cell.append({
            'cell': cell,
            'summary': summarize(cell_df),
            'entry_signal_prob_sweep': sweep_thresholds(cell_df, 'entry_signal_prob'),
            'last_signal_prob_sweep': sweep_thresholds(cell_df, 'last_signal_prob'),
        })

    weak_union_df = trades[trades['row_id'].isin(sorted(weak_row_ids))].copy()
    non_weak_df = trades[~trades['row_id'].isin(sorted(weak_row_ids))].copy()
    payload = {
        'baseline_gates': {
            'stage1_min_prob': report['config'].get('stage1_min_prob'),
            'stage2_min_prob': report['config'].get('stage2_min_prob'),
        },
        'weak_cell_definition': 'cells with at least 5 trades and negative total_pnl in all.time_distribution.session_heatmaps; aggregate uses union of trades that fall into any weak cell',
        'weak_cells': weak_cells,
        'aggregate_union': {
            'weak_cells_union': summarize(weak_union_df),
            'non_weak_union': summarize(non_weak_df),
            'weak_entry_signal_prob_sweep': sweep_thresholds(weak_union_df, 'entry_signal_prob'),
            'weak_last_signal_prob_sweep': sweep_thresholds(weak_union_df, 'last_signal_prob'),
        },
        'per_cell': per_cell,
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()

