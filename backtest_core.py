"""
AlphaGold v13 — Reusable Backtest Simulation Core
=================================================
Standalone module exposing `simulate_v13_core` so it can be imported by
`daily_reconciliation.py`, parameter sweeps, etc. without executing the
full backtest script in `backtest.py` (which parses sys.argv at import).
"""
import pandas as pd

from config.v13_config import EXECUTION_CONFIG


def simulate_v13_core(df_test, tp, sl, horizon_minutes):
    """
    Extracted trade simulation loop for parameter sweeping.
    Expects df_test to already have side_signal, s1_prob, s2_prob columns.
    Uses real bid/ask columns when available, else synthetic spread.
    """
    _spread = EXECUTION_CONFIG["spread_default"]
    if 'open_ask' not in df_test.columns:
        if 'openPrice_ask' in df_test.columns:
            df_test = df_test.copy()
            df_test['open_ask']  = df_test['openPrice_ask']
            df_test['open_bid']  = df_test['openPrice_bid']
            df_test['close_ask'] = df_test['closePrice_ask']
            df_test['close_bid'] = df_test['closePrice_bid']
            df_test['high_ask']  = df_test['highPrice_ask']
            df_test['low_bid']   = df_test['lowPrice_bid']
        else:
            df_test = df_test.copy()
            df_test['open_ask']  = df_test['open']  + _spread
            df_test['open_bid']  = df_test['open']  - _spread
            df_test['close_ask'] = df_test['close'] + _spread
            df_test['close_bid'] = df_test['close'] - _spread
            df_test['high_ask']  = df_test['high']  + _spread
            df_test['low_bid']   = df_test['low']   - _spread

    all_trades, active_pos = [], None
    for i in range(len(df_test) - 1):
        row      = df_test.iloc[i]
        next_row = df_test.iloc[i + 1]
        now_ts   = row.name
        sig      = int(row['side_signal'])

        if active_pos:
            s = active_pos['side']
            exit_info = None
            if s == 1:
                if   row['low_bid']  <= active_pos['stop']:    exit_info = (active_pos['stop'],   'stop_loss')
                elif row['high_ask'] >= active_pos['target']:  exit_info = (active_pos['target'], 'target_hit')
                elif now_ts          >= active_pos['timeout']: exit_info = (row['close_bid'],     'timeout')
            else:
                if   row['high_ask'] >= active_pos['stop']:    exit_info = (active_pos['stop'],   'stop_loss')
                elif row['low_bid']  <= active_pos['target']:  exit_info = (active_pos['target'], 'target_hit')
                elif now_ts          >= active_pos['timeout']: exit_info = (row['close_ask'],     'timeout')
            if exit_info:
                px, reason = exit_info
                pnl = (px - active_pos['entry_price']) * s
                all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': px,
                                    'exit_reason': reason, 'pnl': pnl})
                active_pos = None

        if active_pos:
            s = active_pos['side']
            if sig != 0 and sig == -s:
                px  = row['close_bid'] if s == 1 else row['close_ask']
                pnl = (px - active_pos['entry_price']) * s
                all_trades.append({**active_pos, 'exit_time': now_ts, 'exit_price': px,
                                    'exit_reason': 'reverse_signal', 'pnl': pnl})
                active_pos = None
            elif sig == s:
                active_pos['timeout'] = now_ts + pd.Timedelta(minutes=horizon_minutes)
                active_pos['target_updates'] += 1
                new_t = row['close'] + (tp if s == 1 else -tp)
                if (s == 1 and new_t > active_pos['target']) or (s == -1 and new_t < active_pos['target']):
                    active_pos['target'] = new_t

        if active_pos is None and sig != 0:
            ep = next_row['open_ask'] if sig == 1 else next_row['open_bid']
            active_pos = {
                'side': sig,
                'entry_time': next_row.name,
                'entry_price': ep,
                'stop':    ep - sl  if sig == 1 else ep + sl,
                'target':  ep + tp  if sig == 1 else ep - tp,
                'timeout': next_row.name + pd.Timedelta(minutes=horizon_minutes),
                'target_updates': 0,
                's1_prob': row['s1_prob'],
                's2_prob': row['s2_prob'],
            }

    return all_trades
