#!/usr/bin/env python3
"""Diagnose why range_fade loses in Sep-Apr but wins in May-June.

Tests additional gates beyond ATR/trend:
  1. Range stability: is the 50-bar range actually stable (width < X%)?
  2. Mean reversion strength: do past fade trades actually revert?
  3. Vol contraction: is vol declining vs 200-bar average?
  4. Range position entropy: is price bouncing between extremes or drifting?
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from v15.research.v9_regime_ensemble import (
    load_askbid_data, compute_indicators, define_regime,
    _build_forward_arrays, vectorized_long_exit, vectorized_short_exit,
)

TP, SL = 15, 20
MAX_BARS = 30

def main():
    print("Loading...")
    df = load_askbid_data()
    df = compute_indicators(df)
    df = define_regime(df)
    df = df.dropna(subset=['atr14', 'ema50_slope', 'rsi_5', 'pos_in_range'])

    # --- Additional gate features ---
    # 1. Range stability: std of range width over last 50 bars / mean range width
    df['range_width'] = df['range_high'] - df['range_low']
    df['range_stability'] = df['range_width'].rolling(50).std() / (df['range_width'].rolling(50).mean() + 0.01)
    # Lower = more stable range

    # 2. ATR contraction: current ATR14 / ATR200
    df['atr200'] = df['atr14'].rolling(200).mean()
    df['atr_contraction'] = df['atr14'] / (df['atr200'] + 0.01)
    # < 1.0 = vol contracting (good for range fade)

    # 3. ADX-like trend strength: |ema50_slope| / ATR
    df['trend_strength'] = df['ema50_slope'].abs() / (df['atr14'] + 0.01)

    # 4. Range position oscillation: how many times did pos_in_range cross 0.5 recently?
    cross_mid = ((df['pos_in_range'] > 0.5) != (df['pos_in_range'].shift(1) > 0.5)).astype(int)
    df['range_crosses'] = cross_mid.rolling(50).sum()
    # Higher = more oscillation (good for range fade)

    # 5. Close-to-EMA50 distance: if price hugs EMA50, fading extremes works
    df['dist_ema50_abs'] = abs(df['close'] - df['ema50']) / (df['atr14'] + 0.01)

    # --- Signals ---
    in_regime = df['regime_lowvol']
    long_sig = (df['pos_in_range'] < 0.20) & in_regime
    short_sig = (df['pos_in_range'] > 0.80) & in_regime

    # --- Monthly PnL split by gate conditions ---
    months = pd.date_range('2025-09-01', '2026-07-01', freq='MS', tz='UTC')
    fwd = _build_forward_arrays(df, MAX_BARS)

    # Test additional gates
    print(f"\n{'='*90}")
    print("RANGE_FADE DIAGNOSIS: Sep-Apr vs May-June")
    print(f"{'='*90}")

    # Overall stats for regime bars
    regime_bars = df[df['regime_lowvol']]
    print(f"\nRegime characteristics (mean ± std):")
    for col in ['atr14', 'range_stability', 'atr_contraction', 'trend_strength',
                'range_crosses', 'dist_ema50_abs', 'pos_in_range']:
        vals = regime_bars[col].dropna()
        print(f"  {col:<22s}: {vals.mean():.4f} ± {vals.std():.4f}")

    # Compare winning vs losing months
    print(f"\n{'='*90}")
    print(f"{'Period':<20s} {'Trades':>7s} {'PnL':>10s} {'WR':>7s} {'ATR':>7s} "
          f"{'Stab':>7s} {'Contr':>7s} {'Trend':>7s} {'Cross':>7s} {'|EMA50|':>8s}")
    print("-" * 90)

    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(1)
        mask = (df.index >= m_start) & (df.index <= m_end)
        sub = df.loc[mask]

        # Get signals for this month
        li, lp = [], []
        si, sp = [], []
        long_idx = sub.index[long_sig.loc[sub.index]]
        short_idx = sub.index[short_sig.loc[sub.index]]

        for idx in long_idx:
            li.append(df.index.get_loc(idx))
            lp.append(df['close_ask'].loc[idx])
        for idx in short_idx:
            si.append(df.index.get_loc(idx))
            sp.append(df['close_bid'].loc[idx])

        pnls = []
        if li:
            lpnls = vectorized_long_exit(np.array(li), np.array(lp), fwd, TP, SL)
            pnls.extend(lpnls[~np.isnan(lpnls)])
        if si:
            spnls = vectorized_short_exit(np.array(si), np.array(sp), fwd, TP, SL)
            pnls.extend(spnls[~np.isnan(spnls)])

        if len(pnls) < 5:
            continue

        a = np.array(pnls)
        reg = sub[sub['regime_lowvol']]
        atr_m = reg['atr14'].mean() if len(reg) else 0
        stab_m = reg['range_stability'].mean() if len(reg) else 0
        contr_m = reg['atr_contraction'].mean() if len(reg) else 0
        trend_m = reg['trend_strength'].mean() if len(reg) else 0
        cross_m = reg['range_crosses'].mean() if len(reg) else 0
        dist_m = reg['dist_ema50_abs'].mean() if len(reg) else 0

        label = m_start.strftime('%Y-%m')
        print(f"  {label:<20s} {len(a):>7d} {a.sum():>+10.1f} {100*(a>0).mean():>6.1f}% "
              f"{atr_m:>7.2f} {stab_m:>7.3f} {contr_m:>7.3f} {trend_m:>7.3f} {cross_m:>7.1f} {dist_m:>8.2f}")

    # --- Test gates ---
    print(f"\n{'='*90}")
    print("GATE SWEEP: which additional condition filters losing months?")
    print(f"{'='*90}")

    gates = {
        'no_gate': pd.Series(True, index=df.index),
        'stable_range': df['range_stability'] < 0.5,
        'vol_contracting': df['atr_contraction'] < 0.9,
        'weak_trend_strict': df['trend_strength'] < 0.02,
        'oscillating': df['range_crosses'] > 15,
        'near_ema': df['dist_ema50_abs'] < 3.0,
        'stable+contracting': (df['range_stability'] < 0.5) & (df['atr_contraction'] < 0.9),
        'stable+oscillate': (df['range_stability'] < 0.5) & (df['range_crosses'] > 15),
        'contract+oscillate': (df['atr_contraction'] < 0.9) & (df['range_crosses'] > 15),
        'all_three': (df['range_stability'] < 0.5) & (df['atr_contraction'] < 0.9) & (df['range_crosses'] > 15),
    }

    sep_apr_mask = (df.index >= '2025-09-01') & (df.index < '2026-05-01')
    mj_mask = (df.index >= '2026-05-01') & (df.index <= '2026-06-09')

    for gate_name, gate_cond in gates.items():
        combined_gate = in_regime & gate_cond
        lg = long_sig & gate_cond
        sg = short_sig & gate_cond

        # Sep-Apr trades
        sa_long = lg & sep_apr_mask
        sa_short = sg & sep_apr_mask
        sa_pnls = []
        for idx in df.index[sa_long]:
            ei = df.index.get_loc(idx)
            p = vectorized_long_exit(np.array([ei]), np.array([df.iloc[ei]['close_ask']]), fwd, TP, SL)
            if not np.isnan(p[0]): sa_pnls.append(p[0])
        for idx in df.index[sa_short]:
            ei = df.index.get_loc(idx)
            p = vectorized_short_exit(np.array([ei]), np.array([df.iloc[ei]['close_bid']]), fwd, TP, SL)
            if not np.isnan(p[0]): sa_pnls.append(p[0])

        # May-June trades
        mj_long = lg & mj_mask
        mj_short = sg & mj_mask
        mj_pnls = []
        for idx in df.index[mj_long]:
            ei = df.index.get_loc(idx)
            p = vectorized_long_exit(np.array([ei]), np.array([df.iloc[ei]['close_ask']]), fwd, TP, SL)
            if not np.isnan(p[0]): mj_pnls.append(p[0])
        for idx in df.index[mj_short]:
            ei = df.index.get_loc(idx)
            p = vectorized_short_exit(np.array([ei]), np.array([df.iloc[ei]['close_bid']]), fwd, TP, SL)
            if not np.isnan(p[0]): mj_pnls.append(p[0])

        sa_a = np.array(sa_pnls) if sa_pnls else np.array([])
        mj_a = np.array(mj_pnls) if mj_pnls else np.array([])

        n_reg_sa = (combined_gate & sep_apr_mask).sum()
        n_reg_mj = (combined_gate & mj_mask).sum()

        print(f"  {gate_name:<25s}: Sep-Apr {len(sa_a):>5d}t {sa_a.sum():>+8.1f}pt ({n_reg_sa:>5,} bars) | "
              f"MJ {len(mj_a):>4d}t {mj_a.sum():>+7.1f}pt ({n_reg_mj:>4,} bars)")

    # --- Find best gate that kills Sep-Apr losses while preserving MJ profits ---
    print(f"\n{'='*90}")
    print("BEST GATE CANDIDATES (maximize MJ PnL while keeping Sep-Apr ≥ 0)")
    print(f"{'='*90}")

    candidates = []
    for gate_name, gate_cond in gates.items():
        if gate_name == 'no_gate':
            continue
        combined_gate = in_regime & gate_cond
        lg = long_sig & gate_cond
        sg = short_sig & gate_cond

        sa_pnls, mj_pnls = [], []
        for idx in df.index[lg & sep_apr_mask]:
            ei = df.index.get_loc(idx)
            p = vectorized_long_exit(np.array([ei]), np.array([df.iloc[ei]['close_ask']]), fwd, TP, SL)
            if not np.isnan(p[0]): sa_pnls.append(p[0])
        for idx in df.index[sg & sep_apr_mask]:
            ei = df.index.get_loc(idx)
            p = vectorized_short_exit(np.array([ei]), np.array([df.iloc[ei]['close_bid']]), fwd, TP, SL)
            if not np.isnan(p[0]): sa_pnls.append(p[0])
        for idx in df.index[lg & mj_mask]:
            ei = df.index.get_loc(idx)
            p = vectorized_long_exit(np.array([ei]), np.array([df.iloc[ei]['close_ask']]), fwd, TP, SL)
            if not np.isnan(p[0]): mj_pnls.append(p[0])
        for idx in df.index[sg & mj_mask]:
            ei = df.index.get_loc(idx)
            p = vectorized_short_exit(np.array([ei]), np.array([df.iloc[ei]['close_bid']]), fwd, TP, SL)
            if not np.isnan(p[0]): mj_pnls.append(p[0])

        sa_pnl = sum(sa_pnls) if sa_pnls else 0
        mj_pnl = sum(mj_pnls) if mj_pnls else 0
        total_pnl = sa_pnl + mj_pnl

        candidates.append({
            'gate': gate_name,
            'sa_trades': len(sa_pnls), 'sa_pnl': sa_pnl,
            'mj_trades': len(mj_pnls), 'mj_pnl': mj_pnl,
            'total_pnl': total_pnl,
            'total_trades': len(sa_pnls) + len(mj_pnls),
        })

    cand_df = pd.DataFrame(candidates)
    cand_df = cand_df.sort_values('total_pnl', ascending=False)
    for _, row in cand_df.head(10).iterrows():
        print(f"  {row['gate']:<25s}: SA {row['sa_trades']:>5.0f}t {row['sa_pnl']:>+8.1f}pt | "
              f"MJ {row['mj_trades']:>4.0f}t {row['mj_pnl']:>+7.1f}pt | "
              f"TOTAL {row['total_trades']:>5.0f}t {row['total_pnl']:>+8.1f}pt")

    print(f"\nDONE.")


if __name__ == '__main__':
    main()
