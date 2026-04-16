import json
from collections import Counter
from pathlib import Path

base = Path('/Users/alpha/Desktop/python/AlphaGold/training')
files = {
    'new_r0_broad': base / '_wfC_r0_flat300_500_s150_60_u50_65_d50_65.json',
    'regen_C_native': base / 'backtest_report_wf_state_sweep_C_regen.json',
    'r0_nearC': base / '_wfC_r0_nearC_flat345_s150_60_u60_64_d62_66.json',
    'r0_flat345': base / '_wfC_r0_flat345_s148_55_u56_62_d62_66.json',
    'hist_C': base / 'backtest_report_wf_state_sweep_C.json',
}

for name, path in files.items():
    obj = json.loads(path.read_text())
    pnl = obj.get('directional_pnl') or {}
    all_ = pnl.get('all', pnl)
    print(name)
    print({k: all_.get(k) for k in ['trades', 'total_pnl', 'profit_factor', 'avg_day', 'positive_days_pct', 'trade_max_drawdown', 'daily_max_drawdown']})
    cycles = obj.get('walkforward_cycles', [])
    stage1 = Counter()
    flat = Counter()
    up = Counter()
    down = Counter()
    reason = Counter()
    for c in cycles:
        s = c.get('selected_config', {})
        stage1[s.get('stage1_min_prob')] += 1
        flat[s.get('max_flat_ratio')] += 1
        up[s.get('stage2_min_prob_up', s.get('stage2_min_prob'))] += 1
        down[s.get('stage2_min_prob_down', s.get('stage2_min_prob'))] += 1
        reason[s.get('reason')] += 1
    print('stage1', dict(stage1))
    print('flat', dict(flat))
    print('up', dict(up))
    print('down', dict(down))
    print('reason', dict(reason))
    print('-' * 60)

