import json, glob, os

files = sorted(glob.glob('training/_tmp_image_trend_sweep/*.json'))
print(f'Total sweep artifacts: {len(files)}\n')

rows = []
for f in files:
    d = json.loads(open(f).read())
    pnl = d.get('directional_pnl', {})
    cfg = d.get('config', {})
    rows.append({
        'file':      os.path.basename(f),
        'horizon':   cfg.get('horizon'),
        'threshold': cfg.get('trend_threshold'),
        'drop':      cfg.get('min_15m_drop'),
        's1p':       d.get('stage1_min_prob'),
        's2p':       d.get('stage2_min_prob'),
        'trades':    pnl.get('trades', 0),
        'total_pnl': round(pnl.get('total_pnl', 0), 2),
        'avg_trade': round(pnl.get('avg_trade', 0), 2),
        'pos_days':  pnl.get('positive_days_pct'),
        'avg_day':   round(pnl.get('avg_day') or 0, 2),
        's2_bacc':   round(d.get('stage2', {}).get('balanced_accuracy', 0), 4),
    })

rows.sort(key=lambda x: -(x['total_pnl'] or 0))
hdr = f"  {'file':<45} {'h':>4} {'thr':>6} {'d':>4} {'s1p':>5} {'s2p':>5} {'trades':>7} {'total_pnl':>10} {'avg_tr':>7} {'pos%':>6} {'s2bacc':>7}"
print(hdr)
print('-' * len(hdr))
for r in rows[:15]:
    pd = f"{r['pos_days']:.1f}" if r['pos_days'] is not None else 'N/A'
    print(f"  {r['file']:<45} {str(r['horizon']):>4} {str(r['threshold']):>6} {str(r['drop']):>4} "
          f"{str(r['s1p']):>5} {str(r['s2p']):>5} {r['trades']:>7} {r['total_pnl']:>10.2f} "
          f"{r['avg_trade']:>7.2f} {pd:>6} {r['s2_bacc']:>7.4f}")

print()
print("--- Today's baseline run (window=150, no 15m image, optimize-prob) ---")
b = json.loads(open('training/image_trend_report_w150_noimgb.json').read())
bcfg = b['config']
bpnl = b['directional_pnl']
print(f"  horizon={bcfg['horizon']}  threshold={bcfg['trend_threshold']}  drop={bcfg['min_15m_drop']}")
print(f"  s1p={b['stage1_min_prob']}  s2p={b['stage2_min_prob']}")
print(f"  trades={bpnl['trades']}  total_pnl={bpnl['total_pnl']:.2f}  avg_trade={bpnl['avg_trade']:.2f}")
print(f"  pos_days={bpnl.get('positive_days_pct')}  avg_day={bpnl.get('avg_day')}")
print(f"  s2_bacc={b['stage2']['balanced_accuracy']:.4f}")

