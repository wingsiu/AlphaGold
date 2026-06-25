#!/usr/bin/env python3
"""v8 results detail: per-month PnL breakdown for best configs."""
import sys
sys.path.insert(0, '/Users/alpha/AlphaGold')

import numpy as np
import pandas as pd
from v15.research.v8_lowvol_regime_research import (
    load_askbid_data, compute_indicators, define_regime,
    ARCHETYPES, _build_forward_arrays, _signal_indices_and_prices,
    vectorized_long_exit, vectorized_short_exit, MAX_BARS,
)

BEST_CONFIGS = {
    'micro_momentum': {'tp': 5, 'sl': 10},
    'range_fade': {'tp': 15, 'sl': 20},
}

MONTHS = pd.date_range('2025-09-01', '2026-07-01', freq='MS', tz='UTC')

def main():
    print("Loading & computing...")
    df = load_askbid_data()
    df = compute_indicators(df)
    df = define_regime(df)
    df = df.dropna(subset=['atr14', 'ema50_slope', 'rsi_5', 'bb_pct', 'pos_in_range'])
    fwd = _build_forward_arrays(df, MAX_BARS)

    month_masks = {}
    for m_start in MONTHS:
        m_end = m_start + pd.offsets.MonthEnd(1)
        month_masks[m_start.strftime('%Y-%m')] = (df.index >= m_start) & (df.index <= m_end)
    month_masks['2026-06-p'] = (df.index >= '2026-06-01') & (df.index <= '2026-06-09')

    for arch_name, cfg in BEST_CONFIGS.items():
        tp, sl = cfg['tp'], cfg['sl']
        sig_fn = ARCHETYPES[arch_name]
        signals = sig_fn(df)

        print(f"\n{'='*85}")
        print(f"  {arch_name}  TP={tp}  SL={sl}")
        print(f"{'='*85}")
        print(f"{'Period':<12s} {'Bars':>6s} {'Regime':>6s} {'Longs':>6s} {'Shorts':>6s} {'Trades':>6s} {'PnL':>10s} {'WR':>7s} {'Avg':>7s} {'Cum':>10s}")

        cumulative = 0.0
        all_pnls_list = []
        for pname in list(month_masks.keys()):
            li, lp, si, sp = _signal_indices_and_prices(signals, df, month_masks[pname])
            pnls = []
            if len(li) > 0:
                lp_vec = vectorized_long_exit(li, lp, fwd, tp, sl)
                pnls.extend(lp_vec[~np.isnan(lp_vec)].tolist())
            if len(si) > 0:
                sp_vec = vectorized_short_exit(si, sp, fwd, tp, sl)
                pnls.extend(sp_vec[~np.isnan(sp_vec)].tolist())

            n_bars = month_masks[pname].sum()
            n_reg = df.loc[month_masks[pname], 'in_regime'].sum()

            if len(pnls) > 0:
                all_pnls_list.extend(pnls)
                a = np.array(pnls)
                wr = (a > 0).mean() * 100
                avg = a.mean()
                total = a.sum()
                cumulative += total
                print(f"  {pname:<12s} {n_bars:>6d} {n_reg:>6d} {len(li):>6d} {len(si):>6d} {len(a):>6d} {total:>+10.1f} {wr:>6.1f}% {avg:>+7.2f} {cumulative:>+10.1f}")
            elif len(li) + len(si) > 0:
                print(f"  {pname:<12s} {n_bars:>6d} {n_reg:>6d} {len(li):>6d} {len(si):>6d} {len(li)+len(si):>6d} {'N/A':>10s}")

        if all_pnls_list:
            a = np.array(all_pnls_list)
            wr = (a > 0).mean() * 100
            pos = a[a > 0].sum()
            neg = abs(a[a < 0].sum())
            pf = pos / neg if neg > 0 else 99
            cum = a.cumsum()
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak).min()
            print(f"  {'─'*75}")
            print(f"  {'TOTAL':<12s} {'':>6s} {'':>6s} {'':>6s} {'':>6s} {len(a):>6d} {a.sum():>+10.1f} {wr:>6.1f}% {a.mean():>+7.2f}")
            print(f"  PF={pf:.2f}  MaxDD={dd:.1f}  Sharpe≈{a.mean()/a.std()*np.sqrt(252*390):.2f} (ann)")

if __name__ == '__main__':
    main()
