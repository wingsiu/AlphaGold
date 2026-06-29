#!/usr/bin/env python3
"""Oil v16 Live Trading Bot — WR90 + Ret + RetShort + LongRet + SI
=================================================================
Five-leg v16 portfolio (single-slot merge). Polls every second.

Usage: python3 trading_bot_oil_v16.py
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from oil.signal_engine import (
    OilSignalState,
    SignalDecision,
    build_15m,
    compute_si_features,
    d15_through_completed,
    init_wr90_cluster_state as se_init_wr90_cluster,
    load_mysql_bars,
    SI_FEATS,
    LONG_ENTRY,
    LONG_CV,
    LONG_EP_MIN,
    LONG_MAX_B,
)
from v16.oil.signal_engine import evaluate_minute_v16
from v16.config.oil_config import (
    OIL_MODEL_DIR,
    RETRACE,
    RET_SHORT,
    LONG_RETRACE_15M,
    WR90,
    SHORT_IMPULSE as SI_CFG,
)

LONG_TP, LONG_SL = float(WR90["tp"]), float(WR90["sl"])
SI_TP, SI_SL, SI_MAX_B = float(SI_CFG["tp"]), float(SI_CFG["sl"]), int(SI_CFG["max_bars"])
RET_TP, RET_SL = float(RETRACE["tp"]), float(RETRACE["sl"])
RET_SHORT_TP, RET_SHORT_SL = float(RET_SHORT["tp"]), float(RET_SHORT["sl"])
LONG_RET_TP, LONG_RET_SL = float(LONG_RETRACE_15M["tp"]), float(LONG_RETRACE_15M["sl"])
LONG_WR_ML_TH = 0.55
RET_ML_TH = 0.55
SI_PROB = 0.50

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v16.oil.journal import OilSignalJournal

# ======================= LOGGING ========================
os.makedirs("runtime", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("runtime/oil_live_bot_v16.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("oil_bot_v16")

# ======================= CONFIG (aligned with oil_backtest.py) ========================
MINUTE_OIL_TRADE_START_SEC = 8   # :08–:10 fetch + score / orders
MINUTE_OIL_SYNC_START_SEC = 14   # :14–:15 position sync
FEATURE_WARMUP_DAYS = 90
STATE_PATH = PROJECT_ROOT / "runtime" / "oil_live_bot_v16_state.json"
HEALTH_PATH = PROJECT_ROOT / "runtime" / "oil_bot_v16_health.json"
CLOSE_BRIDGE_PATH = PROJECT_ROOT / "runtime" / "live_oil_v16_trade_closes.json"
OIL_CACHE_PATH = PROJECT_ROOT / "runtime" / "oil_1m_v16_cache.pkl"
# Same table as v15/backtest/backtest_oil.py
OIL_TABLE = "prices"

# Session
NY_S, NY_E, NY_FC_H, NY_FC_M = 3, 12, 14, 28

# Short impulse — v16 config (LONG_TP etc. set above from v16.config.oil_config)
SI_CHANGE_MAX, SI_VOL_MIN = float(SI_CFG["change_max"]), float(SI_CFG["vol_min"])

# Retrace thresholds from v16 config
RET_DLOW, RET_RNG = RETRACE["dlow"], RETRACE["rng"]
RET_CHG, RET_WICK = RETRACE["chg"], RETRACE["wick"]

# Feature lists (exact match with backtest)
WR_FEATS = ['wr', 'volume', 'range', 'avg_r3', 'cad', 'ret_1b', 'ret_3b',
            'vol_r', 'h_dlow', 'l_dlow', 'body', 'up', 'up_p1']

SI_FEATS = ['prev_change', 'prev2_change', 'prev_lower_wick', 'prev_volume',
            'prev_range', 'prev_spread', 'ATR', 'ATR_ratio',
            'ret_1m', 'ret_3m', 'ret_5m', 'vol_ratio_20',
            'up_count3_15min', 'ret_3_15m', 'ret_5_15m', 'dist_day_high']

RET_FEATS = ['cad', 'avg_r3', 'bc', 'wb', 'range', 'ret_1b', 'ret_3b', 'ret_5b',
             'vol_r', 'h_dlow', 'l_dlow', 'body', 'up', 'up_p1', 'up_p2', 'body_p1', 'range_p1']


# ======================= IG SERVICE ========================
from brokers.ig_live import IGLiveBrokerAdapter
from brokers.base import OrderRequest
from ig_scripts.ig_data_api import (
    IGService, Price, API_CONFIG,
    fetch_open_positions, fetch_prices as ig_fetch_prices,
)

ig = IGService(
    api_key=API_CONFIG["api_key"],
    username=API_CONFIG["username"],
    password=API_CONFIG["password"],
    base_url=API_CONFIG["base_url"],
)
log.info("IG service initialized (oil)")


# ======================= DATA ========================
def load_historical_from_db():
    """One-time warmup from MySQL (same table as backtest)."""
    log.info(
        f"Loading oil data ({FEATURE_WARMUP_DAYS}d warmup) from MySQL table '{OIL_TABLE}'..."
    )
    return load_mysql_bars(warmup_days=FEATURE_WARMUP_DAYS, table=OIL_TABLE)


def merge_bars_into_cache(cached: pd.DataFrame, new_bars: pd.DataFrame) -> pd.DataFrame:
    if new_bars.empty:
        return cached
    for idx, row in new_bars.iterrows():
        if idx not in cached.index:
            cached.loc[idx] = row
    cached = cached[~cached.index.duplicated(keep="last")]
    cached.sort_index(inplace=True)
    return cached


def trim_oil_cache(cached: pd.DataFrame) -> pd.DataFrame:
    """Keep rolling warmup window (gold-style cache trim)."""
    if cached.empty:
        return cached
    max_ts = cached.index.max()
    cutoff = max_ts - pd.Timedelta(days=FEATURE_WARMUP_DAYS)
    return cached[cached.index >= cutoff]


def save_oil_cache(cached: pd.DataFrame) -> None:
    if cached.empty:
        return
    try:
        OIL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cached.to_pickle(OIL_CACHE_PATH)
    except Exception as e:
        log.error(f"Oil cache save failed: {e}")


def load_oil_cache_into(cached: pd.DataFrame) -> pd.DataFrame:
    if not OIL_CACHE_PATH.exists():
        return cached
    try:
        disk_df = pd.read_pickle(OIL_CACHE_PATH)
        if disk_df.empty:
            return cached
        if disk_df.index.tz is None:
            disk_df.index = disk_df.index.tz_localize("UTC")
        else:
            disk_df.index = disk_df.index.tz_convert("UTC")
        before = len(cached)
        cached = merge_bars_into_cache(cached, disk_df)
        added = len(cached) - before
        if added > 0:
            log.info(f"Restored {added} oil bar(s) from disk cache (latest={cached.index[-1]})")
    except Exception as e:
        log.warning(f"Oil disk cache restore failed: {e}")
    return cached


def fetch_ig_bars_since(last_bar_ts: pd.Timestamp | None):
    """Fetch bars from last cached bar to now, matching gold bot's format."""
    try:
        now = datetime.now(timezone.utc)
        if last_bar_ts is not None:
            start = last_bar_ts.to_pydatetime() - timedelta(minutes=1)
        else:
            start = now - timedelta(minutes=120)
        result = ig_fetch_prices(ig, Price.Oil, start_time=start, end_time=now)
        if result:
            df = pd.DataFrame(result)
            # Match gold bot's timestamp parsing: millisecond epoch in 'timestamp' column
            if "timestamp" in df.columns:
                df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            else:
                df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
            # Rename columns to match internal naming (same rename as gold bot)
            df = df.rename(columns={
                "openPrice_ask": "open",
                "highPrice_ask": "high",
                "lowPrice_ask": "low",
                "closePrice_ask": "close_ask",
                "closePrice_bid": "close_bid",
                "lastTradedVolume": "volume",
            })
            # Keep only needed columns
            df = df[["open", "high", "low", "close_ask", "close_bid", "volume"]]
            return df
    except Exception as e:
        log.error(f"IG fetch error: {e}")
    return pd.DataFrame()


# build_15m imported from oil.signal_engine


# ======================= WR90 LONG — legacy wrappers (signal_engine is canonical) ========================
def init_wr90_cluster_state(d15: pd.DataFrame) -> None:
    """Replay 15m history so cluster CV/Ep match backtest after restart."""
    state._wr90_in_cluster = False
    state._wr90_cv = 0.0
    state._wr90_bc = 0
    if len(d15) < 2:
        return
    # Replay all completed bars except the latest (evaluated live on new 15m close).
    for i in range(len(d15) - 1):
        bar = d15.iloc[i]
        is_oversold = bool(bar['wr'] < LONG_ENTRY and bar['ins'])
        if is_oversold:
            if not state._wr90_in_cluster:
                state._wr90_cv = 0.0
                state._wr90_bc = 0
            state._wr90_in_cluster = True
            state._wr90_cv += float(bar['volume'])
            state._wr90_bc += 1
        elif state._wr90_in_cluster:
            state._wr90_in_cluster = False
            state._wr90_cv = 0.0
            state._wr90_bc = 0


def _ts_after_deadline(bar_ts: pd.Timestamp, deadline_iso: Optional[str]) -> bool:
    if not deadline_iso:
        return False
    return pd.Timestamp(bar_ts) <= pd.Timestamp(deadline_iso)


def _set_wr90_cascade_window(entry_bar_ts: pd.Timestamp) -> None:
    state._wr90_active_until_15m = (
        pd.Timestamp(entry_bar_ts) + pd.Timedelta(minutes=15 * LONG_MAX_B)
    ).isoformat()


def _set_ret_cascade_window(entry_bar_ts: pd.Timestamp) -> None:
    state._ret_active_until_15m = (
        pd.Timestamp(entry_bar_ts) + pd.Timedelta(minutes=15 * LONG_MAX_B)
    ).isoformat()


def _set_si_cascade_window(entry_bar_ts: pd.Timestamp, bars: int = SI_MAX_B) -> None:
    state._si_active_until_1m = (
        pd.Timestamp(entry_bar_ts) + pd.Timedelta(minutes=bars)
    ).isoformat()


def detect_wr90_cluster(d15):
    """Check if a WR90 oversold cluster just ended (matching backtest logic).
    
    Uses/writes state._wr90_in_cluster, _wr90_cv, _wr90_bc to accumulate
    across bars. Only signals on the bar AFTER the cluster ends (WR >= -75),
    not during the oversold cluster.
    
    Returns (entry_idx, cv, ep_count) if a valid cluster just ended, or None.
    """
    if len(d15) < 15:
        return None
    bar = d15.iloc[-1]
    is_oversold = bool(bar['wr'] < LONG_ENTRY and bar['ins'])
    
    if is_oversold:
        if not state._wr90_in_cluster:
            state._wr90_cv = 0.0
            state._wr90_bc = 0
        state._wr90_in_cluster = True
        state._wr90_cv += float(bar['volume'])
        state._wr90_bc += 1
        return None
    elif state._wr90_in_cluster:
        cv = state._wr90_cv
        bc = state._wr90_bc
        state._wr90_in_cluster = False
        state._wr90_cv = 0.0
        state._wr90_bc = 0
        
        current_ins = bool(bar['ins'])
        if current_ins and cv >= LONG_CV and bc >= LONG_EP_MIN:
            return (len(d15) - 1, cv, bc)
        return None
    else:
        return None


def should_evaluate_15m_bar(latest_15m_bar: pd.Timestamp) -> bool:
    """True when this 15m close has not been scored yet."""
    if state._last_evaluated_15m_bar is None:
        return True
    return pd.Timestamp(latest_15m_bar) > pd.Timestamp(state._last_evaluated_15m_bar)


def mark_15m_bar_evaluated(latest_15m_bar: pd.Timestamp) -> None:
    state._last_evaluated_15m_bar = pd.Timestamp(latest_15m_bar).isoformat()
    save_oil_state()


def process_15m_signals(
    d15: pd.DataFrame,
    latest_15m_bar: pd.Timestamp,
    latest_bar: pd.Timestamp,
    flat: bool,
    journal: OilSignalJournal,
) -> tuple[str, str]:
    """WR90 + retrace entry logic (once per new 15m bar)."""
    wr_result_str = ""
    ret_result_str = ""
    if not flat:
        journal.record_score(
            str(latest_bar), pattern_name='short_impulse',
            pattern_side=0, pattern_prob=None, action='in_trade',
        )
        return wr_result_str, ret_result_str

    # --- WR90 LONG ---
    if _ts_after_deadline(latest_15m_bar, state._wr90_active_until_15m):
        wr_result_str = "WR90: cascade window"
    else:
        wr = detect_wr90_cluster(d15)
        if wr:
            entry_idx, cv, ep_count = wr
            prob = compute_wr90_ml_prob(entry_idx, d15)
            if prob is not None:
                wr_result_str = f"WR90 prob={prob:.3f} CV={cv:.0f} Ep={ep_count}"
                if prob >= LONG_WR_ML_TH:
                    bar = d15.iloc[entry_idx]
                    entry_ts = d15.index[entry_idx]
                    wr_signal = {
                        'type': 'wr90_long', 'side': 1,
                        'entry_price': float(bar['close_ask']),
                        'tp': LONG_TP, 'sl': LONG_SL,
                        'prob': float(prob),
                    }
                    submitted = submit_oil_trade(wr_signal, entry_ts, journal)
                    if submitted:
                        _set_wr90_cascade_window(entry_ts)
                    journal.record_score(
                        str(latest_bar), pattern_name='wr90_long',
                        pattern_side=1, pattern_prob=float(prob),
                        action='entry' if submitted else 'blocked',
                    )
            else:
                wr_result_str = "WR90: no WF model for month"
        else:
            wr_result_str = "WR90: no cluster"

    # --- OIL RETRACE ---
    if state.open_deal_id:
        ret_result_str = "Retrace: in_trade"
    elif _ts_after_deadline(latest_15m_bar, state._ret_active_until_15m):
        ret_result_str = "Retrace: cascade window"
    else:
        ret_sigs = detect_retrace_signals(d15)
        sig_idx = ret_sigs[-1]['idx'] if ret_sigs else None
        if sig_idx is not None and d15.index[sig_idx] == latest_15m_bar:
            bar = d15.iloc[sig_idx]
            lp = score_wf_model('ret', d15.index[sig_idx], bar, RET_FEATS)
            if lp is not None:
                ret_result_str = f"Retrace prob={lp:.3f}"
                if lp >= RET_ML_TH:
                    ret_signal = {
                        'type': 'oil_retrace', 'side': 1,
                        'entry_price': float(bar['close_ask']),
                        'tp': RET_TP, 'sl': RET_SL,
                        'prob': float(lp),
                    }
                    submitted = submit_oil_trade(ret_signal, latest_15m_bar, journal)
                    if submitted:
                        _set_ret_cascade_window(latest_15m_bar)
                    journal.record_score(
                        str(latest_bar), pattern_name='oil_retrace',
                        pattern_side=1, pattern_prob=float(lp),
                        action='entry' if submitted else 'blocked',
                    )
            else:
                ret_result_str = "Retrace: no WF model for month"
        else:
            ret_result_str = "Retrace: no signal"

    return wr_result_str, ret_result_str


def compute_wr90_ml_prob(entry_idx, d15):
    """Score WR90 entry bar with monthly WF .joblib (same as backtest)."""
    ts = d15.index[entry_idx]
    return score_wf_model('wr90', ts, d15.iloc[entry_idx], WR_FEATS)


# ======================= SHORT IMPULSE (1m — aligned with backtest) ========================
def compute_si_features(df):
    """Compute SI features matching backtest's compute_si_features exactly."""
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

    # 15-min context
    d15 = df.resample('15min', label='right', closed='right').agg(
        {'open': 'first', 'close_ask': 'last'}).dropna()
    d15['up'] = np.where(d15['close_ask'] > d15['open'], 1,
                         np.where(d15['close_ask'] < d15['open'], -1, 0))
    d15['up_count3'] = d15['up'].rolling(3, 1).sum()
    d15['ret'] = d15['close_ask'].pct_change()
    d15['ret_3_15m'] = d15['ret'].rolling(3, 1).sum()
    d15['ret_5_15m'] = d15['ret'].rolling(5, 1).sum()
    f15 = d15[['up_count3', 'ret_3_15m', 'ret_5_15m']].reset_index()
    m15 = pd.merge_asof(df.reset_index().sort_values('timestamp'),
                         f15.rename(columns={'timestamp': 't15'}),
                         left_on='timestamp', right_on='t15',
                         direction='backward', tolerance=pd.Timedelta(minutes=15))
    m15.index = m15['timestamp']
    df['up_count3_15min'] = m15['up_count3']
    df['ret_3_15m'] = m15['ret_3_15m']
    df['ret_5_15m'] = m15['ret_5_15m']

    daily_high = df['high'].resample('D').max().reindex(df.index, method='ffill')
    df['dist_day_high'] = daily_high - df['close_ask']
    return df


def sim_si_fixed(ei, ep, df):
    """Simulate one short impulse trade. Returns (exit_price, bars, reason)."""
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


# ======================= OIL RETRACE (15m — aligned with backtest) ========================
def detect_retrace_signals(d15):
    """Find retrace signal indices (same as backtest)."""
    mask = ((d15['cad'] > RET_DLOW) & (d15['avg_r3'] > RET_RNG) &
            (d15['bc'] < RET_CHG) & (d15['wb'] < RET_WICK) & d15['ins'])
    return [{'idx': i} for i in range(len(d15)) if mask.iloc[i]]


# ======================= POSITION STATE ========================
class OilState:
    def __init__(self):
        self.open_deal_id: Optional[str] = None
        self.open_side: int = 0
        self.open_source: Optional[str] = None
        self.open_entry_price: Optional[float] = None
        self.open_tp: Optional[float] = None
        self.open_sl: Optional[float] = None
        self.closed_first_seen_at: Optional[str] = None
        self.last_pnl: float = 0.0
        self._last_submitted_bar_ts: Optional[pd.Timestamp] = None
        self._startup_ts: Optional[pd.Timestamp] = None
        self._wr90_in_cluster: bool = False
        self._wr90_cv: float = 0.0
        self._wr90_bc: int = 0
        # Backtest sim_no_cascade: no new WR90/retrace entries while a leg trade is active.
        self._wr90_active_until_15m: Optional[str] = None
        self._ret_active_until_15m: Optional[str] = None
        self._si_active_until_1m: Optional[str] = None
        self._last_evaluated_15m_bar: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "open_deal_id": self.open_deal_id,
            "open_side": self.open_side,
            "open_source": self.open_source,
            "open_entry_price": self.open_entry_price,
            "open_tp": self.open_tp,
            "open_sl": self.open_sl,
            "closed_first_seen_at": self.closed_first_seen_at,
            "last_pnl": self.last_pnl,
            "_wr90_in_cluster": self._wr90_in_cluster,
            "_wr90_cv": self._wr90_cv,
            "_wr90_bc": self._wr90_bc,
            "_wr90_active_until_15m": self._wr90_active_until_15m,
            "_ret_active_until_15m": self._ret_active_until_15m,
            "_si_active_until_1m": self._si_active_until_1m,
            "_last_evaluated_15m_bar": self._last_evaluated_15m_bar,
        }

    def load_from_dict(self, data: dict) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


state = OilState()
signal_st = OilSignalState()


def _sync_signal_from_oil() -> None:
    signal_st.open_deal_id = state.open_deal_id
    signal_st.wr90_in_cluster = state._wr90_in_cluster
    signal_st.wr90_cv = state._wr90_cv
    signal_st.wr90_bc = state._wr90_bc
    signal_st.wr90_active_until_15m = state._wr90_active_until_15m
    signal_st.ret_active_until_15m = state._ret_active_until_15m
    signal_st.si_active_until_1m = state._si_active_until_1m
    signal_st.last_evaluated_15m_bar = state._last_evaluated_15m_bar
    if state._last_submitted_bar_ts is not None:
        signal_st.last_submitted_bar_ts = pd.Timestamp(state._last_submitted_bar_ts).isoformat()


def _sync_oil_from_signal() -> None:
    state._wr90_in_cluster = signal_st.wr90_in_cluster
    state._wr90_cv = signal_st.wr90_cv
    state._wr90_bc = signal_st.wr90_bc
    state._wr90_active_until_15m = signal_st.wr90_active_until_15m
    state._ret_active_until_15m = signal_st.ret_active_until_15m
    state._si_active_until_1m = signal_st.si_active_until_1m
    state._last_evaluated_15m_bar = signal_st.last_evaluated_15m_bar


def _decision_log_line(dec: Optional[SignalDecision], prefix: str) -> str:
    if dec is None:
        return ""
    if dec.would_enter:
        prob = f" prob={dec.prob:.3f}" if dec.prob is not None else ""
        return f"{prefix}{prob} ENTRY"
    if dec.prob is not None:
        return f"{prefix} prob={dec.prob:.3f} ({dec.reason})"
    return f"{prefix}: {dec.reason}"


def _submit_from_decision(dec: SignalDecision, bar_df: pd.DataFrame, journal: OilSignalJournal) -> bool:
    if dec.leg == 'wr90':
        bar = dec.entry_ts
        d15_tmp = build_15m(bar_df)
        if bar not in d15_tmp.index:
            return False
        sig = {
            'type': 'wr90_long', 'side': 1,
            'entry_price': float(d15_tmp.loc[bar, 'close_ask']),
            'tp': LONG_TP, 'sl': LONG_SL, 'prob': float(dec.prob or 0),
        }
        submitted = submit_oil_trade(sig, bar, journal)
        journal.record_score(
            str(bar_df.index[-1]), pattern_name='wr90_long',
            pattern_side=1, pattern_prob=float(dec.prob or 0),
            action='entry' if submitted else 'blocked',
        )
        if submitted:
            signal_st.wr90_active_until_15m = (
                pd.Timestamp(bar) + pd.Timedelta(minutes=15 * LONG_MAX_B)
            ).isoformat()
            _sync_oil_from_signal()
        return submitted
    if dec.leg == 'ret':
        d15_tmp = build_15m(bar_df)
        bar = dec.entry_ts
        if bar not in d15_tmp.index:
            return False
        sig = {
            'type': 'oil_retrace', 'side': 1,
            'entry_price': float(d15_tmp.loc[bar, 'close_ask']),
            'tp': RET_TP, 'sl': RET_SL, 'prob': float(dec.prob or 0),
        }
        submitted = submit_oil_trade(sig, bar, journal)
        journal.record_score(
            str(bar_df.index[-1]), pattern_name='oil_retrace',
            pattern_side=1, pattern_prob=float(dec.prob or 0),
            action='entry' if submitted else 'blocked',
        )
        if submitted:
            signal_st.ret_active_until_15m = (
                pd.Timestamp(bar) + pd.Timedelta(minutes=15 * LONG_MAX_B)
            ).isoformat()
            _sync_oil_from_signal()
        return submitted
    if dec.leg == 'si':
        row = compute_si_features(bar_df).loc[dec.entry_ts]
        sig = {
            'type': 'short_impulse', 'side': -1,
            'entry_price': float(row['close_bid']),
            'tp': SI_TP, 'sl': SI_SL, 'prob': float(dec.prob or 0),
        }
        submitted = submit_oil_trade(sig, dec.entry_ts, journal)
        journal.record_score(
            str(dec.entry_ts), pattern_name='short_impulse',
            pattern_side=-1, pattern_prob=float(dec.prob or 0),
            action='entry' if submitted else 'score',
        )
        if submitted:
            signal_st.si_active_until_1m = (
                pd.Timestamp(dec.entry_ts) + pd.Timedelta(minutes=SI_MAX_B)
            ).isoformat()
            _sync_oil_from_signal()
        return submitted
    if dec.leg == 'ret_short':
        from v16.oil.long_retrace import enrich_d15_wicks
        d15_tmp = enrich_d15_wicks(build_15m(bar_df))
        bar = dec.entry_ts
        if bar not in d15_tmp.index:
            return False
        sig = {
            'type': 'oil_retrace_short', 'side': -1,
            'entry_price': float(d15_tmp.loc[bar, 'close_bid']),
            'tp': RET_SHORT_TP, 'sl': RET_SHORT_SL, 'prob': float(dec.prob or 0),
        }
        submitted = submit_oil_trade(sig, bar, journal)
        if submitted:
            signal_st.ret_active_until_15m = (
                pd.Timestamp(bar) + pd.Timedelta(minutes=15 * LONG_MAX_B)
            ).isoformat()
            _sync_oil_from_signal()
        return submitted
    if dec.leg == 'long_ret':
        from v16.oil.long_retrace import enrich_d15_long_retrace
        d15_tmp = enrich_d15_long_retrace(build_15m(bar_df))
        bar = dec.entry_ts
        if bar not in d15_tmp.index:
            return False
        sig = {
            'type': 'long_retrace', 'side': 1,
            'entry_price': float(d15_tmp.loc[bar, 'close_ask']),
            'tp': LONG_RET_TP, 'sl': LONG_RET_SL, 'prob': float(dec.prob or 0),
        }
        submitted = submit_oil_trade(sig, bar, journal)
        if submitted:
            signal_st.ret_active_until_15m = (
                pd.Timestamp(bar) + pd.Timedelta(minutes=15 * LONG_MAX_B)
            ).isoformat()
            _sync_oil_from_signal()
        return submitted
    return False


def write_health_json(ev: Optional[dict] = None, *, note: str = "") -> None:
    try:
        payload = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "status": "ok",
            "note": note,
            "cache_bars": len(cached) if cached is not None and len(cached) else 0,
            "cache_latest_1m": str(cached.index[-1]) if cached is not None and len(cached) else None,
            "open_deal_id": state.open_deal_id,
            "open_source": state.open_source,
            "wr90_cluster": {
                "in_cluster": signal_st.wr90_in_cluster,
                "cv": signal_st.wr90_cv,
                "ep": signal_st.wr90_bc,
            },
            "cascade": {
                "wr90_until_15m": signal_st.wr90_active_until_15m,
                "ret_until_15m": signal_st.ret_active_until_15m,
                "si_until_1m": signal_st.si_active_until_1m,
            },
            "last_evaluated_15m": signal_st.last_evaluated_15m_bar,
            "parity_check": "python3 _check_oil_parity.py",
            "log_tail": "tail -30 runtime/oil_live_bot.log",
        }
        if ev:
            for key in ("si", "wr90", "ret"):
                dec = ev.get(key)
                if dec and isinstance(dec, SignalDecision):
                    payload[f"last_{key}"] = {
                        "reason": dec.reason,
                        "prob": dec.prob,
                        "would_enter": dec.would_enter,
                        "detail": dec.detail,
                    }
        HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        HEALTH_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"Health write failed: {e}")


# Module-level cache ref for health/submit helpers
cached: pd.DataFrame = pd.DataFrame()


def load_oil_state() -> None:
    if not STATE_PATH.exists():
        return
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state.load_from_dict(data)
        _sync_signal_from_oil()
        log.info(f"Loaded oil state open_deal_id={state.open_deal_id}")
    except Exception as e:
        log.warning(f"Oil state load failed: {e}")


def save_oil_state() -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    except Exception as e:
        log.error(f"Oil state save failed: {e}")


def clear_oil_position_state() -> None:
    state.open_deal_id = None
    state.open_side = 0
    state.open_source = None
    state.open_entry_price = None
    state.open_tp = None
    state.open_sl = None
    state.closed_first_seen_at = None
    save_oil_state()


def append_close_bridge(
    deal_id: str,
    *,
    reason: str,
    exit_time: str | None = None,
    exit_price: float | None = None,
    pnl: float | None = None,
) -> None:
    rows: list[dict] = []
    if CLOSE_BRIDGE_PATH.exists():
        try:
            loaded = json.loads(CLOSE_BRIDGE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                rows = loaded
        except Exception:
            rows = []
    rows.append({
        "deal_id": deal_id,
        "exit_reason": reason,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "pnl": pnl,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    CLOSE_BRIDGE_PATH.write_text(json.dumps(rows[-50:], indent=2), encoding="utf-8")


# ======================= POSITION SYNC ========================
def sync_ig_trade_results(journal: OilSignalJournal | None = None) -> None:
    """Match gold bot: confirm closes at IG and update oil journal."""
    if not state.open_deal_id:
        return
    deal_id = state.open_deal_id
    try:
        positions = fetch_open_positions(ig)
        still_open = any(
            p.get("position", {}).get("dealId") == deal_id for p in positions
        )
        if still_open:
            if state.closed_first_seen_at:
                state.closed_first_seen_at = None
                save_oil_state()
            return

        from ig_scripts.ig_data_api import get_closed_trade_by_deal_id

        closed = get_closed_trade_by_deal_id(ig, deal_id)
        if closed and closed.get("pnl") is not None:
            pnl = float(closed["pnl"])
            log.info(f"Trade {deal_id} closed PnL={pnl:.2f} source={state.open_source}")
            if journal is not None:
                try:
                    journal.close_trade(
                        deal_id,
                        exit_time=closed.get("exit_time"),
                        exit_price=closed.get("exit_price"),
                        pnl=pnl,
                        exit_reason=str(closed.get("reason") or "broker_close"),
                    )
                except Exception as je:
                    log.error(f"Oil journal close failed: {je}")
            append_close_bridge(
                deal_id,
                reason=str(closed.get("reason") or "broker_close"),
                exit_time=closed.get("exit_time"),
                exit_price=closed.get("exit_price"),
                pnl=pnl,
            )
            state.last_pnl = pnl
            clear_oil_position_state()
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        if not state.closed_first_seen_at:
            state.closed_first_seen_at = now_iso
            save_oil_state()
        elif (
            datetime.now(timezone.utc) - datetime.fromisoformat(state.closed_first_seen_at)
        ).total_seconds() > 300:
            log.warning(f"Force-clearing stale oil open_deal_id {deal_id}")
            if journal is not None:
                try:
                    journal.close_trade(
                        deal_id,
                        exit_time=now_iso,
                        exit_reason="stale_sync",
                    )
                except Exception as je:
                    log.error(f"Oil journal stale close failed: {je}")
            clear_oil_position_state()
    except Exception as e:
        log.error(f"Oil trade sync failed: {e}")


def sync_ig_position(journal: OilSignalJournal | None = None):
    """Legacy wrapper — use full trade sync."""
    sync_ig_trade_results(journal)


# ======================= IG BROKER ========================
broker = IGLiveBrokerAdapter(
    ig, instrument=Price.Oil,
    stop_loss_pct=SI_SL,
    take_profit_pct=SI_TP,
)


# ======================= TRADE SUBMISSION ========================
def submit_oil_trade(signal, bar_ts, journal=None):
    if state._last_submitted_bar_ts and bar_ts <= state._last_submitted_bar_ts:
        log.warning(f"Blocked duplicate submit on bar {bar_ts}")
        return False
    if state.open_deal_id:
        return False
    try:
        open_pos = fetch_open_positions(ig)
        if open_pos:
            # Only block if bot's own tracked position is still open
            our_open = any(
                p.get("position", {}).get("dealId") == state.open_deal_id
                for p in open_pos
            )
            if our_open:
                log.error(f"BLOCKED — bot position {state.open_deal_id} still open")
                return False
            # Manual trades don't block
            log.info(f"Manual trade(s) open — not blocking oil bot entry")
    except Exception as e:
        log.error(f"fetch_open_positions failed: {e} — BLOCKING entry")
        return False

    side = signal.get('side', 0)
    tp = signal.get('tp', SI_TP)
    sl = signal.get('sl', SI_SL)
    entry_price = signal.get('entry_price', 0)

    request = OrderRequest(
        symbol="oil",
        side="buy" if side == 1 else "sell",
        size=1.0,
        signal_time_utc=pd.Timestamp(bar_ts).tz_convert("UTC").isoformat(),
        entry_time_utc=pd.Timestamp(bar_ts).tz_convert("UTC").isoformat(),
        entry_price=float(entry_price),
        probability=float(signal.get('prob', 1.0)),
        signal_model_family="oil_combined",
        metadata={"stop_distance": sl, "limit_distance": tp, "source": signal['type']},
    )
    result = broker.submit_order(request)
    if result.submitted:
        state._last_submitted_bar_ts = bar_ts
        deal_id = result.deal_id
        real_entry_price = entry_price
        try:
            os.environ["IG_REQUEST_CONSUMER"] = "bot_oil"
            pos_wrap = broker.get_position_by_deal_id(deal_id)
            if pos_wrap:
                level = pos_wrap.get("position", {}).get("level")
                if level is not None:
                    real_entry_price = float(level)
        except Exception:
            pass
        state.open_deal_id = deal_id
        state.open_side = side
        state.open_source = signal['type']
        state.open_entry_price = real_entry_price
        state.open_tp = tp
        state.open_sl = sl
        state.closed_first_seen_at = None
        save_oil_state()
        if journal is not None:
            journal.open_trade({
                'deal_id': deal_id,
                'source': signal['type'],
                'side': side,
                'entry_time': str(pd.Timestamp(bar_ts).tz_convert('UTC')),
                'entry_price': entry_price,
                'backtest_entry_price': entry_price,
                'real_entry_price': real_entry_price,
                'tp': tp,
                'sl': sl,
                'horizon': None,
                'probability': float(signal.get('prob', 1.0)),
            })
        log.info(f"✓ ORDER SUBMITTED: {signal['type']} side={side} @ {real_entry_price:.1f} "
                 f"TP={tp} SL={sl} deal_id={deal_id}")
        return True
    else:
        log.error(f"Order failed: {result.reason if hasattr(result, 'reason') else result}")
        return False


# ======================= PID LOCK ========================
PID_FILE = PROJECT_ROOT / "runtime" / "oil_live_bot_v16.pid"

def acquire_pid_lock():
    if PID_FILE.exists():
        old_pid = PID_FILE.read_text().strip()
        try:
            os.kill(int(old_pid), 0)
            log.error(f"Oil bot already running with PID {old_pid} — exiting")
            sys.exit(1)
        except (OSError, ValueError):
            PID_FILE.unlink(missing_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    atexit.register(lambda: PID_FILE.unlink(missing_ok=True))


# ======================= MAIN LOOP ========================
def sync_open_position_on_startup():
    """On startup, detect any existing open IG positions and track oil ones."""
    try:
        positions = fetch_open_positions(ig)
        if positions:
            for pos_wrap in positions:
                pos = pos_wrap.get("position", {})
                epic = pos.get("epic") or pos.get("instrumentName") or ""
                if (
                    "OIL" in epic.upper()
                    or "CL.BMU" in epic.upper()
                    or "CC.D.CL" in epic.upper()
                ):
                    deal_id = pos.get("dealId", "")
                    direction = str(pos.get("direction", "")).upper()
                    level = float(pos.get("level", 0))
                    log.info(f"Startup: Found existing oil position {deal_id} "
                             f"dir={direction} level={level} — tracking")
                    state.open_deal_id = deal_id
                    state.open_side = 1 if direction == "BUY" else -1
                    state.open_source = "wr90_long"
                    state.open_entry_price = level
                    state.open_tp = LONG_TP
                    state.open_sl = LONG_SL
                    save_oil_state()
                    return
    except Exception as e:
        log.warning(f"Startup position sync failed: {e}")


def main():
    global cached
    acquire_pid_lock()
    os.environ["IG_REQUEST_CONSUMER"] = "bot_oil"
    load_oil_state()
    sync_open_position_on_startup()
    log.info("=" * 60)
    log.info("  OIL LIVE BOT v16 — WR90 + Ret + RetShort + LongRet + SI")
    log.info(f"  WR90:      WR<{LONG_ENTRY} CV>={LONG_CV} Ep>={LONG_EP_MIN} "
             f"TP={LONG_TP}/SL={LONG_SL}")
    log.info(f"  Retrace:   Dlow>{RET_DLOW} Rng>{RET_RNG} Chg<{RET_CHG} "
             f"Wick<{RET_WICK} TP={RET_TP}/SL={RET_SL}")
    log.info(f"  RetShort:  TP={RET_SHORT_TP}/SL={RET_SHORT_SL}")
    log.info(f"  LongRet:   TP={LONG_RET_TP}/SL={LONG_RET_SL}")
    log.info(f"  SI:        prev_chg<{SI_CHANGE_MAX} Vol>{SI_VOL_MIN} "
             f"TP={SI_TP}/SL={SI_SL}")
    log.info(f"  WF models: {OIL_MODEL_DIR}")
    log.info(f"  Oil IG epic: {Price.Oil.epic}")
    log.info("=" * 60)

    cached = load_historical_from_db()
    cached = load_oil_cache_into(cached)
    log.info(f"  Loaded {len(cached):,} bars ({cached.index[0]} → {cached.index[-1]})")

    journal = OilSignalJournal()
    journal.resolve_trades_view(bot_state=state.to_dict(), allow_ig=False)
    last_trade_minute = None
    last_sync_minute = None

    _sync_signal_from_oil()
    d15_init, _ = d15_through_completed(cached)
    se_init_wr90_cluster(d15_init, signal_st)
    _sync_oil_from_signal()
    log.info(
        f"  WR90 cluster state replayed: in_cluster={signal_st.wr90_in_cluster} "
        f"cv={signal_st.wr90_cv:.0f} ep={signal_st.wr90_bc}"
    )
    write_health_json(note="startup")

    while True:
        try:
            now = datetime.now(timezone.utc)
            sec = now.second
            minute_key = now.strftime('%Y-%m-%dT%H:%M')

            if sec >= MINUTE_OIL_SYNC_START_SEC and minute_key != last_sync_minute:
                last_sync_minute = minute_key
                os.environ["IG_REQUEST_CONSUMER"] = "bot_oil_sync"
                sync_ig_trade_results(journal)
                journal.resolve_trades_view(bot_state=state.to_dict(), allow_ig=False)

            if sec >= MINUTE_OIL_TRADE_START_SEC and minute_key != last_trade_minute:
                last_trade_minute = minute_key
                os.environ["IG_REQUEST_CONSUMER"] = "bot_oil"
                last_bar_ts = cached.index[-1] if len(cached) > 0 else None
                new_bars = fetch_ig_bars_since(last_bar_ts)
                if new_bars.empty:
                    # Heartbeat every 5 min when no new bars (outside trading hours)
                    if not hasattr(main, '_last_heartbeat') or (now - main._last_heartbeat).total_seconds() > 300:
                        log.info(f"Heartbeat: alive, {len(cached)} bars in cache, latest={cached.index[-1] if cached is not None and len(cached) > 0 else 'none'}")
                        main._last_heartbeat = now
                    continue

                for idx, row in new_bars.iterrows():
                    if idx not in cached.index:
                        cached.loc[idx] = row
                cached = cached[~cached.index.duplicated(keep='last')]
                cached.sort_index(inplace=True)
                cached = trim_oil_cache(cached)
                save_oil_cache(cached)
                log.info(
                    f"Cache fetch: +{len(new_bars)} bar(s) from IG, "
                    f"total={len(cached)}, latest={cached.index[-1]}"
                )

                # Build feature frames every minute
                d1m_feats = compute_si_features(cached)
                latest_bar = cached.index[-1]

                # --- Save 1m bar features to journal ---
                try:
                    latest_row = d1m_feats.loc[latest_bar]
                    feat_dict = {f: float(latest_row.get(f, 0)) for f in SI_FEATS}
                    feat_dict['open'] = float(latest_row.get('open', 0))
                    feat_dict['high'] = float(latest_row.get('high', 0))
                    feat_dict['low'] = float(latest_row.get('low', 0))
                    feat_dict['close'] = float(latest_row.get('close_ask', 0))
                    journal.record_bar_feature(
                        str(latest_bar),
                        json.dumps(feat_dict, default=str),
                    )
                except Exception:
                    pass

                _sync_signal_from_oil()
                ev = evaluate_minute_v16(cached, signal_st, submit=False)
                _sync_oil_from_signal()
                save_oil_state()

                si_result = _decision_log_line(ev.get('si'), 'SI')
                wr_result_str = _decision_log_line(ev.get('wr90'), 'WR90')
                ret_result_str = _decision_log_line(ev.get('ret'), 'Ret')
                rs_result = _decision_log_line(ev.get('ret_short'), 'RetSh')
                lr_result = _decision_log_line(ev.get('long_ret'), 'LngRet')

                flat = state.open_deal_id is None
                winner = ev.get('winner')
                if flat and winner and winner.would_enter:
                    submitted = _submit_from_decision(winner, cached, journal)
                    if submitted:
                        _sync_signal_from_oil()
                        tag = f"{winner.leg} prob={winner.prob:.3f} ENTRY"
                        if winner.leg == 'si':
                            si_result = tag
                        elif winner.leg == 'wr90':
                            wr_result_str = tag
                        elif winner.leg == 'ret':
                            ret_result_str = tag
                        elif winner.leg == 'ret_short':
                            rs_result = tag
                        elif winner.leg == 'long_ret':
                            lr_result = tag

                write_health_json(ev)
                parts = [f"[{latest_bar}]"]
                for bit in (si_result, wr_result_str, ret_result_str, rs_result, lr_result):
                    if bit:
                        parts.append(bit)
                if len(parts) == 1:
                    parts.append("no signals")
                log.info(" | ".join(parts))

            time.sleep(1)

        except KeyboardInterrupt:
            log.info("Shutting down (Ctrl+C)...")
            break
        except Exception as e:
            log.error(f"Loop error: {e}", exc_info=True)
            time.sleep(5)


if __name__ == '__main__':
    main()
