#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

P = Path('/Users/alpha/Desktop/python/AlphaGold/runtime/backtest_best_base_corrected_directional_pnl_fullstats.json')
obj = json.loads(P.read_text())


def flatten_cells(session_obj: dict[str, object], min_trades: int = 1) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for session_name, sess in session_obj.items():
        cell_stats = sess['cell_stats']
        for day, hours in cell_stats.items():
            for hour, stats in hours.items():
                if not stats:
                    continue
                if int(stats['trades']) < min_trades:
                    continue
                out.append(
                    {
                        'session': session_name,
                        'day': day,
                        'hour': hour,
                        'trades': int(stats['trades']),
                        'total_pnl': float(stats['total_pnl']),
                        'avg_trade': float(stats['avg_trade']),
                        'win_rate_pct': float(stats['win_rate_pct']),
                    }
                )
    return out


print('BEST_CELLS_MIN5')
all_sessions = obj['all']['time_distribution']['session_heatmaps']
all_cells_5 = flatten_cells(all_sessions, min_trades=5)
all_best = sorted(all_cells_5, key=lambda r: (r['total_pnl'], r['avg_trade'], r['win_rate_pct'], r['trades']), reverse=True)[:12]
for r in all_best:
    print(f"{r['session']:>6} | {r['day']:<9} | {r['hour']} | trades={r['trades']:>2} | total_pnl={r['total_pnl']:>7.2f} | avg_trade={r['avg_trade']:>6.2f} | win_rate={r['win_rate_pct']:>5.1f}%")

print('\nWORST_CELLS_MIN5')
all_worst = sorted(all_cells_5, key=lambda r: (r['total_pnl'], r['avg_trade'], -r['trades']))[:12]
for r in all_worst:
    print(f"{r['session']:>6} | {r['day']:<9} | {r['hour']} | trades={r['trades']:>2} | total_pnl={r['total_pnl']:>7.2f} | avg_trade={r['avg_trade']:>6.2f} | win_rate={r['win_rate_pct']:>5.1f}%")

print('\nSESSION_SUMMARY_BY_BUCKET')
for bucket in ['all', 'long_up', 'short_down']:
    print(f'\nBUCKET {bucket}')
    sessions = obj[bucket]['time_distribution']['session_heatmaps']
    for name in ['hkt', 'london', 'ny']:
        s = sessions[name]
        avg_trade = 'None' if s['avg_trade'] is None else f"{s['avg_trade']:.4f}"
        wr = 'None' if s['win_rate_pct'] is None else f"{s['win_rate_pct']:.2f}"
        print(f"  {name:>6}: trades={s['trades']:>3} total_pnl={s['total_pnl']:>8.2f} avg_trade={avg_trade:>7} win_rate={wr}")

print('\nBEST_CELLS_BY_BUCKET_MIN5')
for bucket in ['long_up', 'short_down']:
    cells = flatten_cells(obj[bucket]['time_distribution']['session_heatmaps'], min_trades=5)
    best = sorted(cells, key=lambda r: (r['total_pnl'], r['avg_trade'], r['win_rate_pct'], r['trades']), reverse=True)[:8]
    print(f'\nBUCKET {bucket}')
    for r in best:
        print(f"{r['session']:>6} | {r['day']:<9} | {r['hour']} | trades={r['trades']:>2} | total_pnl={r['total_pnl']:>7.2f} | avg_trade={r['avg_trade']:>6.2f} | win_rate={r['win_rate_pct']:>5.1f}%")

