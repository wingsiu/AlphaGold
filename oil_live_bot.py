#!/usr/bin/env python3
"""Oil Live Trading Bot — WR90 Long + Short Impulse with XGBoost.
=================================================================
Polls every second; fetches IG prices once per minute at :06 (6th second),
after the 1-min bar closes. Initial cache loads from MySQL (gold bot's DB).

Trades WR90 Long (15m bars) + Short Impulse (1m bars) on IX.D.WTICOUS.IFS.IP.
"""
import sys; from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np; import pandas as pd; import xgboost as xgb
from data.data_loader import DataLoader
from datetime import datetime, timezone, timedelta
import os; import atexit; import time; import logging; import warnings
warnings.filterwarnings('ignore')

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

# ======================= CONFIG ========================
MINUTE_FETCH_SECOND = 6
FEATURE_WARMUP_DAYS = 90
OIL_EPIC = "IX.D.WTICOUS.IFS.IP"
OIL_TABLE = "oil_prices"

# WR90 Long (15m bars)
NY_S, NY_E = 3, 12
LONG_EP_MIN, LONG_CV = 3, 15000
LONG_ENTRY = -80
LONG_TP, LONG_SL = 80, 30

# Short Impulse (1m bars)
SI_CHANGE_MAX, SI_VOL_MIN = -14.0, 800
SI_UK_HOURS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SI_TP, SI_SL, SI_MAX_B = 90, 60, 60
SI_PROB = 0.55

SI_XGB_FEATURES = [
    'prev_change', 'prev2_change', 'prev_lower_wick', 'prev_upper_wick',
    'prev_volume', 'prev_range', 'prev_spread', 'ATR', 'ATR_ratio',
    'ret_1m', 'ret_3m', 'ret_5m', 'vol_ratio_20',
    'up_count3_15min', 'ret_3_15m', 'ret_5_15m', 'dist_day_high'
]

# ======================= IG SERVICE ========================
from brokers.ig_live import IGLiveBrokerAdapter
from brokers.base import OrderRequest
from ig_scripts.ig_data_api import IGService, Price, API_CONFIG, fetch_open_positions, fetch_prices as ig_fetch_prices

ig = IGService(
    api_key=API_CONFIG["api_key"],
    username=API_CONFIG["username"],
    password=API_CONFIG["password"],
    base_url=API_CONFIG["base_url"],
)
log.info("IG service initialized (oil)")

# ======================= DATA ========================
def load_historical_from_db():
    """Load warmup data from MySQL (same source as gold bot)."""
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
    cols = [
        ('open', 'openPrice_ask'), ('high', 'highPrice_ask'),
        ('low', 'lowPrice_ask'), ('close_ask', 'closePrice_ask'),
        ('close_bid', 'closePrice_bid'), ('volume', 'lastTradedVolume'),
    ]
    for c, src in cols:
        df[c] = raw.get(src, pd.Series(np.nan, index=raw.index)).astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def fetch_ig_minute_bar():
    """Fetch latest 1-min bar from IG (at :06 of each minute)."""
    from ig_scripts.ig_data_api import fetch_and_store_prices_from_latest
    try:
        fetch_and_store_prices_from_latest(ig, Price.Oil)
        # Read back from IG's stored data
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=2)
        result = ig_fetch_prices(ig,
            OIL_EPIC,
            resolution="MINUTE",
            start_time=start,
            end_time=now,
        )
        if result:
            df = pd.DataFrame(result)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            return df
    except Exception as e:
        log.error(f"IG fetch error: {e}")
    return pd.DataFrame()

# ======================= WR90 SIGNAL ========================
def build_15m(df1m):
    d = df1m.resample('15min', label='right', closed='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min',
         'close_ask': 'last', 'close_bid': 'last', 'volume': 'sum'}).dropna()
    n = 14; hh = d['high'].rolling(n).max(); ll = d['low'].rolling(n).min()
    d['wr'] = ((hh - d['close_ask']) / (hh - ll + 0.01)) * -100
    ny = d.index.tz_convert('America/New_York')
    d['ny_h'], d['ny_m'] = ny.hour, ny.minute
    d['in_sess'] = (d['ny_h'] >= NY_S) & (d['ny_h'] <= NY_E)
    return d

def detect_wr90_signal(d15):
    if len(d15) < 15: return None
    in_s = d15['in_sess']; o = (d15['wr'] < LONG_ENTRY) & in_s
    cv, bc = 0.0, 0; last_ep_start = None
    for i in range(len(d15)):
        if o.iloc[i] and d15['in_sess'].iloc[i]:
            if last_ep_start is None: last_ep_start = i
            cv += d15['volume'].iloc[i]; bc += 1
        elif last_ep_start is not None:
            ebi = i
            if ebi < len(d15) and d15['in_sess'].iloc[min(ebi, len(d15)-1)] \
               and cv >= LONG_CV and bc >= LONG_EP_MIN and ebi > last_ep_start:
                bar = d15.iloc[ebi]
                return {
                    'type': 'WR90_LONG', 'entry_price': float(bar['close_ask']),
                    'bar_time': str(d15.index[ebi]),
                    'cum_vol': float(cv), 'bars': bc,
                    'wr': float(bar['wr']), 'ny_hour': int(bar['ny_h']),
                    'tp': LONG_TP, 'sl': LONG_SL
                }
            cv, bc = 0.0, 0; last_ep_start = None
    return None

# ======================= SHORT IMPULSE ========================
def compute_si_features(df):
    df = df.copy()
    df['change'] = df['close_ask'] - df['open']
    df['prev_change'] = df['change'].shift(1)
    df['prev2_change'] = df['change'].shift(2)
    df['prev_lower_wick'] = df['close_ask'].shift(1) - df['low'].shift(1)
    df['prev_upper_wick'] = df['high'].shift(1) - df['close_ask'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_range'] = df['high'].shift(1) - df['low'].shift(1)
    df['prev_spread'] = df['close_ask'].shift(1) - df['close_bid'].shift(1)
    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close_ask'].shift()),
                    abs(df['low'] - df['close_ask'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['ATR_ratio'] = df['prev_range'] / (df['ATR'] + 0.01)
    df['vol_ma_20'] = df['volume'].rolling(20, min_periods=5).mean()
    df['vol_ratio_20'] = df['prev_volume'] / (df['vol_ma_20'] + 1e-6)
    df['ret_1m'] = df['close_ask'].pct_change()
    df['ret_3m'] = df['ret_1m'].rolling(3, min_periods=1).sum()
    df['ret_5m'] = df['ret_1m'].rolling(5, min_periods=1).sum()
    df['uk_hour'] = df.index.hour.isin(SI_UK_HOURS)
    df['hour'] = df.index.hour.astype(float)

    # 15-min context
    try:
        d15 = df.resample('15min', label='right', closed='right').agg(
            {'open': 'first', 'close_ask': 'last'}).dropna()
        d15['up'] = np.where(d15['close_ask'] > d15['open'], 1,
                             np.where(d15['close_ask'] < d15['open'], -1, 0))
        d15['up_count3'] = d15['up'].rolling(3, min_periods=1).sum()
        d15['ret'] = d15['close_ask'].pct_change()
        d15['ret_3_15m'] = d15['ret'].rolling(3, min_periods=1).sum()
        d15['ret_5_15m'] = d15['ret'].rolling(5, min_periods=1).sum()
        f15 = d15[['up_count3', 'ret_3_15m', 'ret_5_15m']].reset_index()
        m15 = pd.merge_asof(df.reset_index().sort_values('timestamp'),
                             f15.rename(columns={'timestamp': 't15'}),
                             left_on='timestamp', right_on='t15',
                             direction='backward', tolerance=pd.Timedelta(minutes=15))
        m15.index = m15['timestamp']
        for c in ['up_count3', 'ret_3_15m', 'ret_5_15m']:
            df[c + '_15min'] = m15.get(c)
    except Exception:
        for c in ['up_count3_15min', 'ret_3_15m', 'ret_5_15m']:
            if c not in df.columns:
                df[c] = np.nan

    daily_high = df['high'].resample('D').max().rename('day_high').reset_index()
    dh = pd.merge_asof(df.reset_index().sort_values('timestamp'),
                        daily_high.rename(columns={'timestamp': 'day_ts'}),
                        left_on='timestamp', right_on='day_ts', direction='backward')
    dh.index = dh['timestamp']
    df['dist_day_high'] = dh['day_high'] - df['close_ask']
    return df


def train_si_xgb(df_si):
    """Train XGBoost on recent history for SI probability."""
    try:
        future = df_si['close_bid'].shift(-SI_MAX_B) - df_si['close_bid']
        df_si['target_down'] = (future < -SI_TP).astype(int)
        valid = df_si.dropna(subset=SI_XGB_FEATURES + ['target_down'])
        if len(valid) < 200: return None
        X = valid[SI_XGB_FEATURES].values.astype(np.float32)
        y = valid['target_down'].values
        if y.sum() < 10: return None
        model = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05,
                                   random_state=42, verbosity=0, use_label_encoder=False,
                                   eval_metric='logloss')
        model.fit(X, y)
        return model
    except Exception:
        return None


def check_si_signal(df_si):
    if len(df_si) < 30: return None, None
    recent = df_si.iloc[-1]
    required = SI_XGB_FEATURES + ['uk_hour']
    if any(pd.isna(recent.get(c)) for c in required): return None, None
    if not (float(recent['prev_change']) < SI_CHANGE_MAX
            and float(recent['prev_volume']) > SI_VOL_MIN
            and recent['uk_hour']):
        return None, None
    feat = [float(recent[c]) for c in SI_XGB_FEATURES]
    if any(np.isnan(feat)): return None, None
    model = train_si_xgb(df_si)
    if model is None: return None, None
    X = np.array([feat])
    prob = float(model.predict_proba(X)[0, 1])
    if prob >= SI_PROB:
        return {
            'type': 'SHORT_IMPULSE',
            'entry_price': float(recent['close_bid']),
            'bar_time': str(df_si.index[-1]),
            'tp': SI_TP, 'sl': SI_SL, 'prob': prob
        }, prob
    return None, prob

# ======================= POSITION STATE ========================
class OilState:
    def __init__(self):
        self.open_deal_id = None
        self.open_side = 0
        self.open_source = None
        self.open_entry_price = None
        self.open_tp = None
        self.open_sl = None
        self._last_submitted_bar_ts = None

state = OilState()

# ======================= POSITION SYNC ========================
def sync_ig_position():
    """Check if our tracked position is still open."""
    try:
        from ig_scripts.ig_data_api import fetch_positions
        positions = fetch_positions(ig, epic=OIL_EPIC)
    except Exception:
        return
    active_deals = [p.get('dealId', p.get('deal_id', '')) for p in positions if p]
    if state.open_deal_id and state.open_deal_id not in active_deals:
        log.info(f"Position {state.open_deal_id} closed — clearing state")
        state.open_deal_id = None; state.open_side = 0; state.open_source = None

# ======================= IG BROKER ========================
broker = IGLiveBrokerAdapter(
    ig, instrument=Price.Oil,
    stop_loss_pct=SI_SL,
    take_profit_pct=SI_TP,
)

# ======================= TRADE ========================
def submit_oil_trade(signal, bar_ts):
    if state._last_submitted_bar_ts and bar_ts <= state._last_submitted_bar_ts:
        log.warning(f"Blocked duplicate submit on bar {bar_ts}")
        return False
    if state.open_deal_id:
        return False
    try:
        open_pos = fetch_open_positions(ig)
        if open_pos:
            deal_ids = [p.get("position", {}).get("dealId", "?") for p in open_pos]
            log.error(f"IG has {len(open_pos)} open position(s) {deal_ids} — BLOCKING entry")
            return False
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
        signal_model_family="oil_retrace",
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
def main():
    acquire_pid_lock()
    log.info("=" * 50)
    log.info("  OIL LIVE BOT — Polling (:06 sec each minute)")
    log.info("=" * 50)

    # Load initial cache from MySQL
    cached = load_historical_from_db()
    log.info(f"  Loaded {len(cached):,} bars ({cached.index[0]} → {cached.index[-1]})")

    last_minute = None

    while True:
        try:
            now = datetime.now(timezone.utc)
            sec = now.second
            minute_key = now.strftime('%Y-%m-%dT%H:%M')

            if sec == MINUTE_FETCH_SECOND and minute_key != last_minute:
                last_minute = minute_key
                log.debug(f"Fetching IG data at :{sec:02d}s for {minute_key}")
                new_bars = fetch_ig_minute_bar()
                if not new_bars.empty:
                    for idx, row in new_bars.iterrows():
                        if idx not in cached.index:
                            cached.loc[idx] = row
                    cached = cached[~cached.index.duplicated(keep='last')]
                    cached.sort_index(inplace=True)
                    log.debug(f"  Cached: {len(cached)} bars, latest: {cached.index[-1]}")

                    # Sync position
                    sync_ig_position()

                    # Check signals
                    d15 = build_15m(cached)
                    wr90 = detect_wr90_signal(d15)
                    if wr90:
                        wr90['side'] = 1
                        log.info(f"[WR90] LONG signal @ {wr90['bar_time']} — "
                                 f"entry={wr90['entry_price']:.1f} wr={wr90['wr']:.0f}")
                        submit_oil_trade(wr90, cached.index[-1])

                    if not state.open_deal_id:
                        d1m = compute_si_features(cached)
                        si, prob = check_si_signal(d1m)
                        if si:
                            si['side'] = -1
                            log.info(f"[SI] SHORT signal @ {si['bar_time']} — "
                                     f"entry={si['entry_price']:.1f} prob={prob:.3f}")
                            submit_oil_trade(si, cached.index[-1])
                        elif prob is not None:
                            log.debug(f"[SI] No signal — prob={prob:.3f} < {SI_PROB}")

            time.sleep(1)

        except KeyboardInterrupt:
            log.info("Shutting down (Ctrl+C)...")
            break
        except Exception as e:
            log.error(f"Loop error: {e}", exc_info=True)
            time.sleep(5)

if __name__ == '__main__':
    main()
