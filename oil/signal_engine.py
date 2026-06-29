"""Oil live signal engine — shared by oil_live_bot and parity_check.

Single code path for WR90 / retrace / SI detection, ML scoring, and cascade windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from oil.wf_ml import score_wf_model

# --- Config (must match v15/backtest/backtest_oil.py) ---
NY_S, NY_E, NY_FC_H, NY_FC_M = 3, 12, 14, 28
LONG_MAX_B = 60
LONG_ENTRY = -80
LONG_CV = 15000
LONG_EP_MIN = 3
LONG_TP, LONG_SL = 80, 30
LONG_WR_ML_TH = 0.55

SI_CHANGE_MAX, SI_VOL_MIN = -14.0, 800
SI_TP, SI_SL, SI_MAX_B = 120, 80, 90
SI_PROB = 0.55

RET_DLOW, RET_RNG = 20, 30
RET_CHG, RET_WICK = -10, 16
RET_TP, RET_SL = 30, 15
RET_ML_TH = 0.60

WR_FEATS = ['wr', 'volume', 'range', 'avg_r3', 'cad', 'ret_1b', 'ret_3b',
            'vol_r', 'h_dlow', 'l_dlow', 'body', 'up', 'up_p1']
SI_FEATS = ['prev_change', 'prev2_change', 'prev_lower_wick', 'prev_volume',
            'prev_range', 'prev_spread', 'ATR', 'ATR_ratio',
            'ret_1m', 'ret_3m', 'ret_5m', 'vol_ratio_20',
            'up_count3_15min', 'ret_3_15m', 'ret_5_15m', 'dist_day_high']
RET_FEATS = ['cad', 'avg_r3', 'bc', 'wb', 'range', 'ret_1b', 'ret_3b', 'ret_5b',
             'vol_r', 'h_dlow', 'l_dlow', 'body', 'up', 'up_p1', 'up_p2', 'body_p1', 'range_p1']


@dataclass
class OilSignalState:
    """In-memory state for live / replay signal evaluation."""
    open_deal_id: Optional[str] = None
    wr90_in_cluster: bool = False
    wr90_cv: float = 0.0
    wr90_bc: int = 0
    wr90_active_until_15m: Optional[str] = None
    ret_active_until_15m: Optional[str] = None
    si_active_until_1m: Optional[str] = None
    last_evaluated_15m_bar: Optional[str] = None
    last_submitted_bar_ts: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def load_dict(self, data: dict) -> None:
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)


@dataclass
class SignalDecision:
    leg: str
    entry_ts: pd.Timestamp
    prob: Optional[float]
    would_enter: bool
    reason: str
    detail: str = ""


def min15_completed_series(index: pd.DatetimeIndex) -> pd.Series:
    """Slot-start of the last *completed* 15×1m window (:00/:15/:30/:45).

    A slot needs minutes [start … start+14]. Example: 1m @ 12:34 → min15=12:15
    because the 12:30 slot (through 12:44) is still forming.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    slot_start = idx.floor('15min')
    slot_last = slot_start + pd.Timedelta(minutes=14)
    prev = slot_start - pd.Timedelta(minutes=15)
    return pd.Series(np.where(idx >= slot_last, slot_start, prev), index=index)


def min15_forming_slot(ts: pd.Timestamp) -> pd.Timestamp:
    """Slot-start of the 15m bucket containing ``ts`` (may still be forming)."""
    t = pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize('UTC')
    return t.floor('15min')


def latest_completed_15m(latest_1m: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Slot-start of the latest fully closed 15m window given the newest 1m bar."""
    s = min15_completed_series(pd.DatetimeIndex([latest_1m]))
    return pd.Timestamp(s.iloc[0])


def attach_min15(df1m: pd.DataFrame) -> pd.DataFrame:
    """Add ``min15`` column: last completed 15m slot-start for each 1m row."""
    out = df1m.copy()
    out['min15'] = min15_completed_series(pd.DatetimeIndex(out.index))
    return out


def d15_through_completed(cached: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """``build_15m`` through the latest completed slot (excludes forming bucket)."""
    if len(cached) == 0:
        return build_15m(cached), None
    completed = latest_completed_15m(cached.index[-1])
    d15 = build_15m(cached)
    if completed is None or completed not in d15.index:
        return d15.iloc[:0], None
    loc = d15.index.get_loc(completed)
    return d15.iloc[: loc + 1], completed


def build_15m(df1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1m → 15m by slot-start (:00/:15/:30/:45). Index = slot start."""
    if df1m.empty:
        return df1m.copy()
    df = df1m.copy()
    slots = pd.DatetimeIndex(df.index).floor('15min')
    if slots.tz is None:
        slots = slots.tz_localize('UTC')
    df['_slot'] = slots
    counts = df.groupby('_slot', sort=True).size()
    d = df.groupby('_slot', sort=True).agg(
        {'open': 'first', 'high': 'max', 'low': 'min',
         'close_ask': 'last', 'close_bid': 'last', 'volume': 'sum'})
    full = counts[counts >= 15].index
    d = d.loc[d.index.isin(full)]
    d.index = pd.DatetimeIndex(d.index)
    if d.index.tz is None:
        d.index = d.index.tz_localize('UTC')
    n = 14
    hh = d['high'].rolling(n).max()
    ll = d['low'].rolling(n).min()
    d['wr'] = ((hh - d['close_ask']) / (hh - ll + 0.01)) * -100
    ny = d.index.tz_convert('America/New_York')
    d['Dlow'] = d.groupby(ny.date)['low'].cummin()
    d['range'] = d['high'] - d['low']
    d['avg_r3'] = d['range'].rolling(3, 3).mean()
    d['wb'] = np.minimum(d['open'], d['close_ask']) - d['low']
    d['bc'] = d['close_ask'] - d['open']
    d['cad'] = d['close_ask'] - d['Dlow']
    d['ny_h'] = ny.hour
    d['ny_m'] = ny.minute
    d['ins'] = (d['ny_h'] >= NY_S) & (d['ny_h'] <= NY_E)
    d['ret_1b'] = d['close_ask'].pct_change(1)
    d['ret_3b'] = d['close_ask'].pct_change(3)
    d['ret_5b'] = d['close_ask'].pct_change(5)
    d['vol_r'] = d['volume'] / (d['volume'].rolling(20).mean() + 0.01)
    d['h_dlow'] = d['high'] - d['Dlow']
    d['l_dlow'] = d['low'] - d['Dlow']
    d['body'] = abs(d['close_ask'] - d['open'])
    d['up'] = (d['close_ask'] > d['open']).astype(int)
    d['up_p1'] = d['up'].shift(1)
    d['up_p2'] = d['up'].shift(2)
    d['body_p1'] = d['body'].shift(1)
    d['range_p1'] = d['range'].shift(1)
    return d


def compute_si_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['change'] = df['close_ask'] - df['open']
    df['prev_change'] = df['change'].shift(1)
    df['prev2_change'] = df['change'].shift(2)
    df['prev_lower_wick'] = df['close_ask'].shift(1) - df['low'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_range'] = df['high'].shift(1) - df['low'].shift(1)
    df['prev_spread'] = df['close_ask'].shift(1) - df['close_bid'].shift(1)
    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close_ask'].shift()),
                    abs(df['low'] - df['close_ask'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['ATR_ratio'] = df['prev_range'] / (df['ATR'] + 0.01)
    df['ret_1m'] = df['close_ask'].pct_change()
    df['ret_3m'] = df['ret_1m'].rolling(3, 1).sum()
    df['ret_5m'] = df['ret_1m'].rolling(5, 1).sum()
    df['vol_ma_20'] = df['volume'].rolling(20, 5).mean()
    df['vol_ratio_20'] = df['prev_volume'] / (df['vol_ma_20'] + 0.01)
    df['ny_hour'] = df.index.tz_convert('America/New_York').hour.isin(list(range(3, 13)))

    df = attach_min15(df)
    slots = pd.DatetimeIndex(df.index).floor('15min')
    if slots.tz is None:
        slots = slots.tz_localize('UTC')
    df['_slot'] = slots
    counts = df.groupby('_slot', sort=True).size()
    g = df.groupby('_slot', sort=True).agg({'open': 'first', 'close_ask': 'last'})
    g = g.loc[g.index.isin(counts[counts >= 15].index)]
    g['up'] = np.where(g['close_ask'] > g['open'], 1,
                       np.where(g['close_ask'] < g['open'], -1, 0))
    g['up_count3'] = g['up'].rolling(3, 1).sum()
    g['ret'] = g['close_ask'].pct_change()
    g['ret_3_15m'] = g['ret'].rolling(3, 1).sum()
    g['ret_5_15m'] = g['ret'].rolling(5, 1).sum()
    f15 = g[['up_count3', 'ret_3_15m', 'ret_5_15m']].copy()
    f15.index.name = 'min15'
    f15 = f15.reset_index()
    left = df.reset_index(names='timestamp').sort_values('timestamp')
    m15 = pd.merge_asof(
        left,
        f15.sort_values('min15'),
        on='min15',
        direction='backward',
    )
    m15.index = m15['timestamp']
    df['up_count3_15min'] = m15['up_count3']
    df['ret_3_15m'] = m15['ret_3_15m']
    df['ret_5_15m'] = m15['ret_5_15m']
    daily_high = df['high'].resample('D').max().reindex(df.index, method='ffill')
    df['dist_day_high'] = daily_high - df['close_ask']
    return df


def sim_si_fixed(ei: int, ep: float, df: pd.DataFrame) -> tuple[float, int, str]:
    stop = ep + SI_SL
    target = ep - SI_TP
    hz = min(SI_MAX_B, len(df) - ei - 1)
    nyz = df.index.tz_convert('America/New_York')
    for i in range(1, hz + 1):
        b = df.iloc[ei + i]
        bh = nyz[ei + i]
        if bh.hour > NY_FC_H or (bh.hour == NY_FC_H and bh.minute >= NY_FC_M):
            return b['close_ask'], i, 'ny_close'
        if b['high'] >= stop:
            return stop, i, 'sl'
        if b['low'] <= target:
            return target, i, 'tp'
    px = df.iloc[ei + hz]['close_ask']
    pnl = ep - px
    return (ep + SI_SL, hz, 'timeout') if pnl < -SI_SL else (px, hz, 'timeout')


def _in_cascade(bar_ts: pd.Timestamp, deadline_iso: Optional[str]) -> bool:
    if not deadline_iso:
        return False
    return pd.Timestamp(bar_ts) <= pd.Timestamp(deadline_iso)


def init_wr90_cluster_state(d15: pd.DataFrame, st: OilSignalState) -> None:
    """Replay completed 15m bars to restore cluster CV/Ep (no entries).

    Pass only *completed* 15m bars (exclude the currently forming bar on live).
    """
    st.wr90_in_cluster = False
    st.wr90_cv = 0.0
    st.wr90_bc = 0
    if len(d15) < 1:
        return
    for i in range(len(d15)):
        bar = d15.iloc[i]
        is_oversold = bool(bar['wr'] < LONG_ENTRY and bar['ins'])
        if is_oversold:
            if not st.wr90_in_cluster:
                st.wr90_cv = 0.0
                st.wr90_bc = 0
            st.wr90_in_cluster = True
            st.wr90_cv += float(bar['volume'])
            st.wr90_bc += 1
        elif st.wr90_in_cluster:
            st.wr90_in_cluster = False
            st.wr90_cv = 0.0
            st.wr90_bc = 0


def detect_wr90_cluster(d15: pd.DataFrame, st: OilSignalState) -> Optional[tuple[int, float, int]]:
    if len(d15) < 15:
        return None
    bar = d15.iloc[-1]
    is_oversold = bool(bar['wr'] < LONG_ENTRY and bar['ins'])
    if is_oversold:
        if not st.wr90_in_cluster:
            st.wr90_cv = 0.0
            st.wr90_bc = 0
        st.wr90_in_cluster = True
        st.wr90_cv += float(bar['volume'])
        st.wr90_bc += 1
        return None
    if st.wr90_in_cluster:
        cv, bc = st.wr90_cv, st.wr90_bc
        st.wr90_in_cluster = False
        st.wr90_cv = 0.0
        st.wr90_bc = 0
        if bool(bar['ins']) and cv >= LONG_CV and bc >= LONG_EP_MIN:
            return len(d15) - 1, cv, bc
    return None


def should_evaluate_15m_bar(
    completed_15m_bar: pd.Timestamp,
    st: OilSignalState,
) -> bool:
    """True when this completed 15m close has not been scored yet."""
    if st.last_evaluated_15m_bar is None:
        return True
    return pd.Timestamp(completed_15m_bar) > pd.Timestamp(st.last_evaluated_15m_bar)


def mark_15m_evaluated(latest_15m_bar: pd.Timestamp, st: OilSignalState) -> None:
    st.last_evaluated_15m_bar = pd.Timestamp(latest_15m_bar).isoformat()


def _set_cascade(entry_ts: pd.Timestamp, minutes: int, field: str, st: OilSignalState) -> None:
    setattr(st, field, (pd.Timestamp(entry_ts) + pd.Timedelta(minutes=minutes)).isoformat())


def evaluate_wr90(d15: pd.DataFrame, st: OilSignalState, flat: bool) -> SignalDecision:
    latest_15m = d15.index[-1]
    wr = detect_wr90_cluster(d15, st)
    if not flat:
        return SignalDecision('wr90', latest_15m, None, False, 'in_trade')
    if _in_cascade(latest_15m, st.wr90_active_until_15m):
        return SignalDecision('wr90', latest_15m, None, False, 'cascade_window')
    if not wr:
        return SignalDecision('wr90', latest_15m, None, False, 'no_cluster')
    idx, cv, ep = wr
    entry_ts = d15.index[idx]
    prob = score_wf_model('wr90', entry_ts, d15.iloc[idx], WR_FEATS)
    if prob is None:
        return SignalDecision('wr90', entry_ts, None, False, 'no_model')
    would = prob >= LONG_WR_ML_TH
    return SignalDecision(
        'wr90', entry_ts, prob, would, 'pass' if would else 'below_threshold',
        f'CV={cv:.0f} Ep={ep}',
    )


def evaluate_retrace(d15: pd.DataFrame, st: OilSignalState, flat: bool) -> SignalDecision:
    latest_15m = d15.index[-1]
    if not flat:
        return SignalDecision('ret', latest_15m, None, False, 'in_trade')
    if _in_cascade(latest_15m, st.ret_active_until_15m):
        return SignalDecision('ret', latest_15m, None, False, 'cascade_window')
    mask = ((d15['cad'] > RET_DLOW) & (d15['avg_r3'] > RET_RNG) &
            (d15['bc'] < RET_CHG) & (d15['wb'] < RET_WICK) & d15['ins'])
    if not bool(mask.iloc[-1]):
        return SignalDecision('ret', latest_15m, None, False, 'no_signal')
    prob = score_wf_model('ret', latest_15m, d15.iloc[-1], RET_FEATS)
    if prob is None:
        return SignalDecision('ret', latest_15m, None, False, 'no_model')
    would = prob >= RET_ML_TH
    return SignalDecision(
        'ret', latest_15m, prob, would, 'pass' if would else 'below_threshold',
    )


def evaluate_si(d1m_feats: pd.DataFrame, latest_bar: pd.Timestamp, st: OilSignalState, flat: bool) -> SignalDecision:
    if not flat:
        return SignalDecision('si', latest_bar, None, False, 'in_trade')
    if _in_cascade(latest_bar, st.si_active_until_1m):
        return SignalDecision('si', latest_bar, None, False, 'cascade_window')
    if len(d1m_feats) < 30:
        return SignalDecision('si', latest_bar, None, False, 'warmup')
    recent = d1m_feats.iloc[-1]
    prev_chg = float(recent.get('prev_change', 0))
    prev_vol = float(recent.get('prev_volume', 0))
    if not (prev_chg < SI_CHANGE_MAX and prev_vol > SI_VOL_MIN and recent.get('ny_hour', False)):
        return SignalDecision('si', latest_bar, None, False, 'no_trigger')
    p2 = float(recent.get('prev2_change', 0))
    wick = float(recent.get('prev_lower_wick', 999))
    uc3 = float(recent.get('up_count3_15min', 0))
    ddh = float(recent.get('dist_day_high', 999))
    if not (p2 < 10.0 and p2 > -14.0 and wick < 35.0 and uc3 != -3 and ddh < 180.0):
        return SignalDecision('si', latest_bar, None, False, 'filtered')
    si_mask = ((d1m_feats['prev_change'] < SI_CHANGE_MAX) &
               (d1m_feats['prev2_change'] < 10.0) & (d1m_feats['prev2_change'] > -14.0) &
               (d1m_feats['prev_lower_wick'] < 35.0) & (d1m_feats['prev_volume'] > SI_VOL_MIN) &
               d1m_feats['ny_hour'] & (d1m_feats['up_count3_15min'] != -3) &
               (d1m_feats['dist_day_high'] < 180.0))
    si_sigs = sorted(d1m_feats.index[si_mask].tolist())
    in_si, si_ex = False, -1
    on_bar = False
    for sig in si_sigs:
        ei = d1m_feats.index.get_loc(sig)
        if ei + SI_MAX_B >= len(d1m_feats):
            continue
        if in_si and ei <= si_ex:
            continue
        ep = d1m_feats.iloc[ei]['close_bid']
        _, bars, _ = sim_si_fixed(ei, ep, d1m_feats)
        in_si, si_ex = True, ei + bars
        if sig == latest_bar:
            on_bar = True
    if not on_bar:
        return SignalDecision('si', latest_bar, None, False, 'not_on_bar')
    prob = score_wf_model('si', latest_bar, d1m_feats.loc[latest_bar], SI_FEATS)
    if prob is None:
        return SignalDecision('si', latest_bar, None, False, 'no_model')
    would = prob >= SI_PROB
    return SignalDecision('si', latest_bar, prob, would, 'pass' if would else 'below_threshold')


def apply_entry(decision: SignalDecision, st: OilSignalState) -> None:
    """Update cascade windows after a simulated/live entry."""
    if not decision.would_enter:
        return
    if decision.leg == 'wr90':
        _set_cascade(decision.entry_ts, 15 * LONG_MAX_B, 'wr90_active_until_15m', st)
    elif decision.leg == 'ret':
        _set_cascade(decision.entry_ts, 15 * LONG_MAX_B, 'ret_active_until_15m', st)
    elif decision.leg == 'si':
        _set_cascade(decision.entry_ts, SI_MAX_B, 'si_active_until_1m', st)
    st.open_deal_id = 'sim'
    st.last_submitted_bar_ts = pd.Timestamp(decision.entry_ts).isoformat()


def clear_position(st: OilSignalState) -> None:
    st.open_deal_id = None


def clear_replay_slot(st: OilSignalState) -> None:
    """Release single-slot hold without cascade windows (replay/backtest parity)."""
    st.open_deal_id = None
    st.wr90_active_until_15m = None
    st.ret_active_until_15m = None
    st.si_active_until_1m = None


def sim_exit_ts(decision: SignalDecision, d1m: pd.DataFrame, d15: pd.DataFrame) -> pd.Timestamp:
    """When a simulated trade would exit (matches backtest sim_full / sim_si_fixed)."""
    entry = pd.Timestamp(decision.entry_ts)
    if decision.leg == 'si':
        feats = compute_si_features(d1m)
        ei = feats.index.get_loc(entry)
        ep = float(feats.iloc[ei]['close_bid'])
        _, bars, _ = sim_si_fixed(ei, ep, feats)
        return pd.Timestamp(feats.index[ei + bars])
    idx = d15.index.get_loc(entry)
    tp = LONG_TP if decision.leg == 'wr90' else RET_TP
    sl = LONG_SL if decision.leg == 'wr90' else RET_SL
    ep = float(d15.iloc[idx]['close_ask'])
    ct, cs = ep + tp, ep - sl
    last = min(idx + LONG_MAX_B, len(d15) - 1)
    for j in range(idx + 1, last + 1):
        b = d15.iloc[j]
        if (b['ny_h'] > NY_FC_H) or (b['ny_h'] == NY_FC_H and b['ny_m'] >= NY_FC_M):
            return pd.Timestamp(d15.index[j])
        if b['high'] >= ct:
            return pd.Timestamp(d15.index[j])
        if b['low'] <= cs:
            return pd.Timestamp(d15.index[j])
    return pd.Timestamp(d15.index[last])


def evaluate_minute(
    cached: pd.DataFrame,
    st: OilSignalState,
    *,
    submit: bool = False,
) -> dict[str, Any]:
    """Evaluate one minute — same path as live bot after cache update."""
    d15, completed_15m = d15_through_completed(cached)
    d1m_feats = compute_si_features(cached)
    latest_bar = cached.index[-1]
    flat = st.open_deal_id is None
    out: dict[str, Any] = {
        'latest_1m': str(latest_bar),
        'latest_15m': str(completed_15m) if completed_15m is not None else None,
        'min15': str(latest_completed_15m(latest_bar)),
        'flat': flat,
        'wr90_cluster': {'in_cluster': st.wr90_in_cluster, 'cv': st.wr90_cv, 'ep': st.wr90_bc},
    }
    si = evaluate_si(d1m_feats, latest_bar, st, flat)
    out['si'] = si
    if si.would_enter and submit:
        apply_entry(si, st)

    wr = ret = None
    if completed_15m is not None and len(d15) > 0:
        if should_evaluate_15m_bar(completed_15m, st):
            wr = evaluate_wr90(d15, st, flat and st.open_deal_id is None)
            flat_now = st.open_deal_id is None
            ret = evaluate_retrace(d15, st, flat_now)
            mark_15m_evaluated(completed_15m, st)
            if wr and wr.would_enter and submit and st.open_deal_id is None:
                apply_entry(wr, st)
            flat_now = st.open_deal_id is None
            if ret and ret.would_enter and submit and flat_now:
                apply_entry(ret, st)
    out['wr90'] = wr
    out['ret'] = ret
    return out


def _completed_15m(d1m: pd.DataFrame) -> pd.DataFrame:
    """15m bars excluding the currently forming bucket."""
    d15, _ = d15_through_completed(d1m)
    return d15


def replay_leg_entries(
    d1m: pd.DataFrame,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    leg: str,
    warmup_days: int = 90,
) -> list[SignalDecision]:
    """Replay one leg independently (matches backtest per-leg simulation)."""
    warmup_start = day_start - pd.Timedelta(days=warmup_days)
    window = d1m[(d1m.index >= warmup_start) & (d1m.index <= day_end)]
    st = OilSignalState()
    d15_w = _completed_15m(window[window.index < day_start])
    if leg == 'wr90' and len(d15_w) > 0:
        init_wr90_cluster_state(d15_w, st)
    entries: list[SignalDecision] = []
    busy_until: Optional[pd.Timestamp] = None
    period = window[window.index >= day_start]
    if leg in ('wr90', 'ret'):
        d15_day = build_15m(window)
        eval_15m = d15_day[(d15_day.index >= day_start) & (d15_day.index <= day_end)].index
        iter_times = sorted({
            period.index[period.index.searchsorted(t, side='right') - 1]
            for t in eval_15m
            if period.index.searchsorted(t, side='right') > 0
        })
    else:
        iter_times = list(period.index)
    for ts in iter_times:
        if busy_until is not None and ts >= busy_until:
            clear_replay_slot(st)
            busy_until = None
        sub = window[window.index <= ts]
        ev = evaluate_minute(sub, st, submit=False)
        if busy_until is not None and ts < busy_until:
            continue
        dec = ev.get(leg)
        if dec and dec.would_enter and st.open_deal_id is None:
            st.open_deal_id = 'sim'
            entries.append(dec)
            d15_sub, _ = d15_through_completed(sub)
            busy_until = sim_exit_ts(dec, sub, d15_sub)
    return entries


def replay_entries(
    d1m: pd.DataFrame,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    warmup_days: int = 90,
) -> list[SignalDecision]:
    """Minute replay of live logic with cascade + single-position hold."""
    warmup_start = day_start - pd.Timedelta(days=warmup_days)
    window = d1m[(d1m.index >= warmup_start) & (d1m.index <= day_end)]
    st = OilSignalState()
    d15_w = _completed_15m(window[window.index < day_start])
    if len(d15_w) > 0:
        init_wr90_cluster_state(d15_w, st)
    entries: list[SignalDecision] = []
    busy_until: Optional[pd.Timestamp] = None
    period = window[window.index >= day_start]
    for i in range(len(period)):
        ts = period.index[i]
        if busy_until is not None and ts >= busy_until:
            clear_replay_slot(st)
            busy_until = None
        sub = window[window.index <= ts]
        ev = evaluate_minute(sub, st, submit=False)
        if busy_until is not None and ts < busy_until:
            continue
        for key in ('si', 'wr90', 'ret'):
            dec = ev.get(key)
            if dec and dec.would_enter and st.open_deal_id is None:
                st.open_deal_id = 'sim'
                entries.append(dec)
                d15_sub, _ = d15_through_completed(sub)
                busy_until = sim_exit_ts(dec, sub, d15_sub)
                break
    return entries


def load_mysql_bars(warmup_days: int = 90, end: Optional[datetime] = None, table: str = "prices") -> pd.DataFrame:
    """Load 1m oil bars from MySQL (default `prices` — same as backtest_oil)."""
    from data.data_loader import DataLoader
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=warmup_days)
    loader = DataLoader()
    raw = loader.load_data(table, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    raw.index = pd.to_datetime(raw['timestamp'], unit='ms')
    df = pd.DataFrame(index=raw.index)
    for c, src in [('open', 'openPrice_ask'), ('high', 'highPrice_ask'),
                   ('low', 'lowPrice_ask'), ('close_ask', 'closePrice_ask'),
                   ('close_bid', 'closePrice_bid'), ('volume', 'lastTradedVolume')]:
        df[c] = raw.get(src, pd.Series(np.nan, index=raw.index)).astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df[~df.index.duplicated(keep='last')].sort_index()


def merge_ig_overlay(mysql_df: pd.DataFrame, ig_tail: pd.DataFrame) -> pd.DataFrame:
    """MySQL base + IG bars newer than MySQL latest (or all IG if fresher)."""
    if ig_tail.empty:
        return mysql_df
    out = mysql_df.copy()
    for idx, row in ig_tail.iterrows():
        out.loc[idx] = row
    return out[~out.index.duplicated(keep='last')].sort_index()
