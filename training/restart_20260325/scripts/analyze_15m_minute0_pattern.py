#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data import DataLoader


def load_1m_gold(start_date: str, end_date: str) -> pd.DataFrame:
    raw = DataLoader().load_data('gold_prices', start_date=start_date, end_date=end_date)
    df = pd.DataFrame(
        {
            'ts': pd.to_datetime(raw['timestamp'], unit='ms', utc=True),
            'open': raw['openPrice'].astype(float),
            'high': raw['highPrice'].astype(float),
            'low': raw['lowPrice'].astype(float),
            'close': raw['closePrice'].astype(float),
            'volume': raw['lastTradedVolume'].fillna(0.0).astype(float),
        }
    ).sort_values('ts').reset_index(drop=True)
    return df


def build_slot_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df['slot'] = df['ts'].dt.minute % 15
    df['block_start'] = df['ts'].dt.floor('15min')
    df['ret_co'] = df['close'] - df['open']
    df['ret_prev_close'] = df['close'] - df['close'].shift(1)
    df['down_co'] = df['ret_co'] < 0
    df['down_prev_close'] = df['ret_prev_close'] < 0

    block_low = df.groupby('block_start')['low'].transform('min')
    df['is_block_low'] = df['low'].eq(block_low)
    low_ties = df.groupby('block_start')['is_block_low'].transform('sum')
    df['block_low_credit'] = df['is_block_low'] / low_ties

    block_high = df.groupby('block_start')['high'].transform('max')
    df['is_block_high'] = df['high'].eq(block_high)
    high_ties = df.groupby('block_start')['is_block_high'].transform('sum')
    df['block_high_credit'] = df['is_block_high'] / high_ties

    blocks = int(df['block_start'].nunique())

    slot = df.groupby('slot').agg(
        rows=('slot', 'size'),
        pct_down_close_open=('down_co', 'mean'),
        pct_down_prev_close=('down_prev_close', 'mean'),
        avg_ret_close_open=('ret_co', 'mean'),
        median_ret_close_open=('ret_co', 'median'),
        avg_ret_prev_close=('ret_prev_close', 'mean'),
        block_low_hits=('block_low_credit', 'sum'),
        block_high_hits=('block_high_credit', 'sum'),
    )
    slot['pct_block_low'] = slot['block_low_hits'] / blocks
    slot['pct_block_high'] = slot['block_high_hits'] / blocks
    slot = slot.reset_index()

    first = df[df['slot'] == 0][['block_start', 'open', 'close', 'low']].rename(
        columns={'open': 'm0_open', 'close': 'm0_close', 'low': 'm0_low'}
    )
    block_last = df.groupby('block_start').tail(1)[['block_start', 'close']].rename(columns={'close': 'block_close'})
    reversal = first.merge(block_last, on='block_start', how='inner')
    reversal['m0_down'] = reversal['m0_close'] < reversal['m0_open']
    reversal['recover_above_m0_close'] = reversal['block_close'] > reversal['m0_close']
    reversal['recover_above_m0_open'] = reversal['block_close'] > reversal['m0_open']
    reversal['m0_drop_size'] = reversal['m0_close'] - reversal['m0_open']

    return slot, reversal


def build_summary(slot: pd.DataFrame, reversal: pd.DataFrame) -> str:
    slot0 = slot.loc[slot['slot'] == 0].iloc[0]
    worst_down = slot.sort_values('pct_down_close_open', ascending=False).iloc[0]
    most_block_low = slot.sort_values('pct_block_low', ascending=False).iloc[0]
    m0_down = reversal[reversal['m0_down']].copy()

    lines = [
        '15-minute slot pattern analysis',
        '================================',
        '',
        f"slot0 pct_down_close_open      : {slot0['pct_down_close_open'] * 100:.2f}%",
        f"slot0 pct_down_prev_close      : {slot0['pct_down_prev_close'] * 100:.2f}%",
        f"slot0 pct_block_low            : {slot0['pct_block_low'] * 100:.2f}%",
        f"slot0 avg_ret_close_open       : {slot0['avg_ret_close_open']:.4f}",
        '',
        f"most bearish slot by close-open down frequency : slot {int(worst_down['slot'])} ({worst_down['pct_down_close_open'] * 100:.2f}%)",
        f"most common slot for 15m block low             : slot {int(most_block_low['slot'])} ({most_block_low['pct_block_low'] * 100:.2f}%)",
        '',
        f"blocks with minute-0 down move                 : {len(m0_down)}",
        f"recover above minute-0 close by block end      : {m0_down['recover_above_m0_close'].mean() * 100:.2f}%",
        f"recover above minute-0 open by block end       : {m0_down['recover_above_m0_open'].mean() * 100:.2f}%",
    ]
    return '\n'.join(lines) + '\n'


def main() -> int:
    out_dir = ROOT / 'training' / 'restart_20260325' / 'reports'
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_1m_gold(start_date='2025-05-20', end_date='2026-02-04')
    slot, reversal = build_slot_stats(df)

    slot_out = out_dir / 'minute0_15m_slot_stats.csv'
    summary_out = out_dir / 'minute0_15m_summary.txt'

    slot.to_csv(slot_out, index=False)
    summary_text = build_summary(slot, reversal)
    summary_out.write_text(summary_text, encoding='utf-8')

    print(summary_text)
    print(f'saved {slot_out}')
    print(f'saved {summary_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

