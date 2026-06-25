#!/usr/bin/env python3
"""Oil Live Trading Bot — WR90 Long + Short Impulse + Oil Retrace
=================================================================
Three-leg bot aligned with oil_backtest.py (v29 combined).

Polls every second; fetches IG prices once per minute at :06 (6th second),
after the 1-min bar closes. Initial cache loads from MySQL.

Config (from v29 backtest):
  WR90 Long      : WR<-75, CV>=5K, Ep>=2, TP=60/SL=20, WF-XGBoost>=0.65
  Short Impulse  : prev_change<-14, vol>800, +4 raw filters, TP=120/SL=80, WF-XGB>=0.55
  Oil Retrace    : cad>20, avgR3>30, bc<-10, wb<16, TP=30/SL=15, WF-XGB>=0.60

Trades on IX.D.WTICOUS.IFS.IP.
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

from oil.wf_ml import OIL_MODEL_DIR, score_wf_model

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import DataLoader
from v15.research.v29_oil_journal import OilSignalJournal

# ======================= LOGGING ========================
os.makedirs("runtime", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("runtime/oil_live_bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("oil_bot")

# ======================= CONFIG (aligned with oil_backtest.py) ========================
MINUTE_FETCH_SECOND = 6
FEATURE_WARMUP_DAYS = 90
# OIL_EPIC comes from Price.Oil.epic = "CC.D.CL.BMU.IP"
OIL_TABLE = "oil_prices"

# Session
NY_S, NY_E, NY_FC_H, NY_FC_M = 3, 12, 14, 28

# WR90 Long (15m bars)
LONG_MAX_B = 60
LONG_ENTRY = -75
LONG_CV = 5000
LONG_EP_MIN = 2
LONG_TP, LONG_SL = 60, 20
LONG_WR_ML_TH = 0.65

# Short Impulse (1m bars)
SI_CHANGE_MAX, SI_VOL_MIN = -14.0, 800
SI_TP, SI_SL, SI_MAX_B = 120, 80, 90
SI_PROB = 0.55

# Oil Retrace (15m bars)
RET_DLOW, RET_RNG = 20, 30
RET_CHG, RET_WICK = -10, 16
RET_TP, RET_SL = 30, 15
RET_ML_TH = 0.60

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
    """Load warmup data from MySQL."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=FEATURE_WARMUP_DAYS)
    log.info(f"Loading oil data {start.date()} → {end.date()} from MySQL...")
    loader = DataLoader()
    try:
        raw = loader.load_data(
            table_name=OIL_TABLE,
            start_date=start.strftime('%Y-%m-%d'),
            end_date=end.strftime('%Y-%m-%d'),
        )
    except Exception:
        log.warning("MySQL oil table not found, trying prices table...")
        raw = loader.load_data(
            table_name='prices',
            start_date=start.strftime('%Y-%m-%d'),
            end_date=end.strftime('%Y-%m-%d'),
        )
    raw.index = pd.to_datetime(raw['timestamp'], unit='ms')
    df = pd.DataFrame(index=raw.index)
    for c, src in [('open', 'openPrice_ask'), ('high', 'highPrice_ask'),
                   ('low', 'lowPrice_ask'), ('close_ask', 'closePrice_ask'),
                   ('close_bid', 'closePrice_bid'), ('volume', 'lastTradedVolume')]:
        df[c] = raw.get(src, pd.Series(np.nan, index=raw.index)).astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


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


# ======================= 15M FRAME (aligned with backtest) ========================
def build_15m(df1m):
    """Build full 15-min OHLCV frame with all features needed for WR90 + Retrace."""
    d = df1m.resample('15min', label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min',
         'close_ask': 'last', 'close_bid': 'last', 'volume': 'sum'}).dropna()
    n = 14
    hh = d['high'].rolling(n).max()
    ll = d['low'].rolling(n).min()
    d['wr'] = ((hh - d['close_ask']) / (hh - ll + 0.01)) * -100
    ny = d.index.tz_convert('America/New_York')
    d['Dlow'] = d['low'].groupby(ny.date).transform('min')
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


# ======================= WR90 LONG — Cluster-End Detection + Walk-Forward ML ========================
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
        # Inside cluster - accumulate volume/bar count
        state._wr90_in_cluster = True
        state._wr90_cv += float(bar['volume'])
        state._wr90_bc += 1
        return None
    elif state._wr90_in_cluster:
        # Cluster just ended - snapshot, reset state, evaluate
        cv = state._wr90_cv
        bc = state._wr90_bc
        state._wr90_in_cluster = False
        state._wr90_cv = 0.0
        state._wr90_bc = 0
        
        # Current bar must be in session (matches backtest in_s.iloc[ebi] check)
        current_ins = bool(bar['ins'])
        if current_ins and cv >= LONG_CV and bc >= LONG_EP_MIN:
            return (len(d15) - 1, cv, bc)
        return None
    else:
        return None


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
        self._last_submitted_bar_ts: Optional[pd.Timestamp] = None
        self._startup_ts: Optional[pd.Timestamp] = None  # Block entries until fresh bar arrives
        # WR90 cluster tracking (matches backtest's cluster-end detection)
        self._wr90_in_cluster: bool = False
        self._wr90_cv: float = 0.0
        self._wr90_bc: int = 0


state = OilState()


# ======================= POSITION SYNC ========================
def sync_ig_position():
    """Check if our tracked oil position is still open on IG."""
    try:
        positions = fetch_open_positions(ig)
    except Exception:
        return
    # Filter to oil positions only
    oil_positions = [
        p for p in positions
        if p.get("position", {}).get("dealId") == state.open_deal_id
    ]
    if state.open_deal_id and not oil_positions:
        log.info(f"Position {state.open_deal_id} closed — clearing state")
        state.open_deal_id = None
        state.open_side = 0
        state.open_source = None


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
        state.open_deal_id = result.deal_id
        state.open_side = side
        state.open_source = signal['type']
        state.open_entry_price = entry_price
        state.open_tp = tp
        state.open_sl = sl
        # Record to journal database so trades appear in alphagold.db
        if journal is not None:
            journal.open_trade({
                'deal_id': result.deal_id,
                'source': signal['type'],
                'side': side,
                'entry_time': str(pd.Timestamp(bar_ts).tz_convert('UTC')),
                'entry_price': entry_price,
                'tp': tp,
                'sl': sl,
                'horizon': None,
                'probability': float(signal.get('prob', 1.0)),
            })
        log.info(f"✓ ORDER SUBMITTED: {signal['type']} side={side} @ {entry_price:.1f} "
                 f"TP={tp} SL={sl} deal_id={result.deal_id}")
        return True
    else:
        log.error(f"Order failed: {result.reason if hasattr(result, 'reason') else result}")
        return False


# ======================= PID LOCK ========================
PID_FILE = PROJECT_ROOT / "runtime" / "oil_live_bot.pid"

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
                if "OIL" in epic.upper() or "WTICO" in epic.upper() or "IX.D.WTICOUS" in pos.get("instrumentName", ""):
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
                    return
    except Exception as e:
        log.warning(f"Startup position sync failed: {e}")


def main():
    acquire_pid_lock()
    sync_open_position_on_startup()
    log.info("=" * 60)
    log.info("  OIL LIVE BOT v29 — WR90 + Short Impulse + Oil Retrace")
    log.info(f"  WR90:   WR<{LONG_ENTRY} CV>={LONG_CV} Ep>={LONG_EP_MIN} "
             f"TP={LONG_TP}/SL={LONG_SL} ML>={LONG_WR_ML_TH}")
    log.info(f"  Retrace: Dlow>{RET_DLOW} Rng>{RET_RNG} Chg<{RET_CHG} "
             f"Wick<{RET_WICK} TP={RET_TP}/SL={RET_SL} ML>={RET_ML_TH}")
    log.info(f"  SI:     prev_chg<{SI_CHANGE_MAX} Vol>{SI_VOL_MIN} "
             f"TP={SI_TP}/SL={SI_SL} ML>={SI_PROB}")
    log.info(f"  WF models: {OIL_MODEL_DIR}")
    log.info("=" * 60)

    cached = load_historical_from_db()
    log.info(f"  Loaded {len(cached):,} bars ({cached.index[0]} → {cached.index[-1]})")

    journal = OilSignalJournal()
    last_minute = None

    # Pre-mark latest cached bars as "already processed" to prevent
    # replaying old signals on restart. Only bars arriving fresh from IG
    # after this point will trigger new entries.
    state._startup_ts = pd.Timestamp.now(tz="UTC")
    d15_init = build_15m(cached)
    last_processed_15m_bar = d15_init.index[-1] if len(d15_init) > 0 else None
    if last_processed_15m_bar is not None:
        log.info(f"  Startup lockout: entries blocked until bar > {last_processed_15m_bar}")
    d15_init = None  # free memory

    while True:
        try:
            now = datetime.now(timezone.utc)
            sec = now.second
            minute_key = now.strftime('%Y-%m-%dT%H:%M')

            if sec == MINUTE_FETCH_SECOND and minute_key != last_minute:
                last_minute = minute_key
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

                # Sync position
                sync_ig_position()

                # Build feature frames every minute
                d15 = build_15m(cached)
                d1m_feats = compute_si_features(cached)
                latest_bar = cached.index[-1]
                flat = not state.open_deal_id

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

                # ======== Run 1m predictions EVERY minute (SI) ========
                si_result = ""
                if flat and len(d1m_feats) >= 30:
                    recent = d1m_feats.iloc[-1]
                    prev_chg = float(recent.get('prev_change', 0))
                    prev_vol = float(recent.get('prev_volume', 0))
                    ny_hour = recent.get('ny_hour', False)
                    if prev_chg < SI_CHANGE_MAX and prev_vol > SI_VOL_MIN and ny_hour:
                        p2 = float(recent.get('prev2_change', 0))
                        wick = float(recent.get('prev_lower_wick', 999))
                        uc3 = float(recent.get('up_count3_15min', 0))
                        ddh = float(recent.get('dist_day_high', 999))
                        if (p2 < 10.0 and p2 > -14.0
                                and wick < 35.0 and uc3 != -3 and ddh < 180.0):
                            si_mask = ((d1m_feats['prev_change'] < SI_CHANGE_MAX) &
                                       (d1m_feats['prev2_change'] < 10.0) &
                                       (d1m_feats['prev2_change'] > -14.0) &
                                       (d1m_feats['prev_lower_wick'] < 35.0) &
                                       (d1m_feats['prev_volume'] > SI_VOL_MIN) &
                                       d1m_feats['ny_hour'] &
                                       (d1m_feats['up_count3_15min'] != -3) &
                                       (d1m_feats['dist_day_high'] < 180.0))
                            si_sigs = sorted(d1m_feats.index[si_mask].tolist())
                            si_recs = []
                            in_si = False
                            si_ex = -1
                            for sig in si_sigs:
                                ei = d1m_feats.index.get_loc(sig)
                                if ei + SI_MAX_B >= len(d1m_feats):
                                    continue
                                if in_si and ei <= si_ex:
                                    continue
                                ep = d1m_feats.iloc[ei]['close_bid']
                                ex_price, bars, reason = sim_si_fixed(ei, ep, d1m_feats)
                                si_recs.append({
                                    'entry_idx': sig, 'pnl': ep - ex_price,
                                    'reason': reason,
                                    'exit_ts': d1m_feats.index[ei + bars],
                                })
                                in_si = True
                                si_ex = ei + bars
                            if si_recs and si_recs[-1]['entry_idx'] == latest_bar:
                                last_si_prob = score_wf_model(
                                    'si', latest_bar, d1m_feats.loc[latest_bar], SI_FEATS,
                                )
                            else:
                                last_si_prob = None
                            if last_si_prob is not None:
                                si_result = f"SI prob={last_si_prob:.3f}"
                                if last_si_prob >= SI_PROB and si_recs[-1]['entry_idx'] == latest_bar:
                                    si_signal = {
                                        'type': 'short_impulse', 'side': -1,
                                        'entry_price': float(recent['close_bid']),
                                        'tp': SI_TP, 'sl': SI_SL,
                                        'prob': float(last_si_prob),
                                    }
                                    submitted = submit_oil_trade(si_signal, latest_bar, journal)
                                    log.info(f"[SI] SHORT @ {latest_bar} entry={si_signal['entry_price']:.1f} prob={last_si_prob:.3f}")
                                    journal.record_score(
                                        str(latest_bar), pattern_name='short_impulse',
                                        pattern_side=-1, pattern_prob=float(last_si_prob),
                                        action='entry' if submitted else 'score',
                                    )
                                else:
                                    journal.record_score(
                                        str(latest_bar), pattern_name='short_impulse',
                                        pattern_side=0, pattern_prob=float(last_si_prob),
                                        action='score',
                                    )
                            elif si_recs:
                                si_result = "SI: signal not on current bar"
                        else:
                            si_result = f"SI filtered: p2={p2:.1f} wick={wick:.1f} uc3={uc3} ddh={ddh:.0f}"
                    else:
                        si_result = f"SI no trigger: chg={prev_chg:.1f} vol={prev_vol:.0f} ny={ny_hour}"
                elif state.open_deal_id:
                    si_result = "SI in_trade"

                # ======== 15m predictions (only on new 15m bar) ========
                wr_result_str = ""
                ret_result_str = ""
                if len(d15) > 0:
                    latest_15m_bar = d15.index[-1]
                    if latest_15m_bar != last_processed_15m_bar:
                        last_processed_15m_bar = latest_15m_bar
                        if flat:
                            # --- WR90 LONG ---
                            wr = detect_wr90_cluster(d15)
                            if wr:
                                entry_idx, cv, ep_count = wr
                                prob = compute_wr90_ml_prob(entry_idx, d15)
                                if prob is not None:
                                    wr_result_str = f"WR90 prob={prob:.3f} CV={cv:.0f} Ep={ep_count}"
                                    if prob >= LONG_WR_ML_TH:
                                        bar = d15.iloc[entry_idx]
                                        wr_signal = {
                                            'type': 'wr90_long', 'side': 1,
                                            'entry_price': float(bar['close_ask']),
                                            'tp': LONG_TP, 'sl': LONG_SL,
                                            'prob': float(prob),
                                        }
                                        submitted = submit_oil_trade(wr_signal, d15.index[entry_idx], journal)
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
                                        journal.record_score(
                                            str(latest_bar), pattern_name='oil_retrace',
                                            pattern_side=1, pattern_prob=float(lp),
                                            action='entry' if submitted else 'blocked',
                                        )
                                else:
                                    ret_result_str = "Retrace: no WF model for month"
                            else:
                                ret_result_str = "Retrace: no signal"
                        else:
                            journal.record_score(
                                str(latest_bar), pattern_name='short_impulse',
                                pattern_side=0, pattern_prob=None, action='in_trade',
                            )

                # ==== Consolidated prediction log every minute ====
                parts = [f"[{latest_bar}]"]
                if si_result:
                    parts.append(si_result)
                if wr_result_str:
                    parts.append(wr_result_str)
                if ret_result_str:
                    parts.append(ret_result_str)
                if not si_result and not wr_result_str and not ret_result_str:
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
