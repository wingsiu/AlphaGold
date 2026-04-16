#!/usr/bin/env python3
import json
import pathlib
import subprocess

root = pathlib.Path('/Users/alpha/Desktop/python/AlphaGold')
cmd = [
    'python3', 'training/image_trend_ml.py',
    '--start-date', '2025-05-20',
    '--end-date', '2026-04-10',
    '--timeframe', '1min',
    '--eval-mode', 'single_split',
    '--test-start-date', '2025-11-25T17:02:00+00:00',
    '--test-size', '0.4',
    '--disable-time-filter',
    '--window', '150',
    '--window-15m', '0',
    '--min-window-range', '40',
    '--min-15m-drop', '15',
    '--min-15m-rise', '0',
    '--horizon', '25',
    '--trend-threshold', '0.008',
    '--adverse-limit', '15',
    '--long-target-threshold', '0.006',
    '--short-target-threshold', '0.008',
    '--long-adverse-limit', '12',
    '--short-adverse-limit', '18',
    '--classifier', 'gradient_boosting',
    '--use-state-features',
    '--pred-history-len', '150',
    '--max-flat-ratio', '2.5',
    '--stage1-min-prob', '0.48',
    '--stage2-min-prob', '0.50',
    '--prep-cache-dir', 'training/_prep_cache_single_split_aligned',
    '--model-out', 'training/backtest_model_best_base_corrected.joblib',
    '--report-out', 'training/backtest_report_best_base_corrected.json',
    '--trades-out', 'training/backtest_trades_best_base_corrected.csv',
]
proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
(root / 'runtime' / 'best_base_corrected.stdout.txt').write_text(proc.stdout, encoding='utf-8')
(root / 'runtime' / 'best_base_corrected.stderr.txt').write_text(proc.stderr, encoding='utf-8')
(root / 'runtime' / 'best_base_corrected.exit.txt').write_text(str(proc.returncode), encoding='utf-8')
print(json.dumps({
    'returncode': proc.returncode,
    'stdout_tail': proc.stdout[-4000:],
    'stderr_tail': proc.stderr[-4000:],
}, indent=2))

