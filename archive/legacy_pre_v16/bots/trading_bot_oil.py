#!/usr/bin/env python3
"""
AlphaGold Oil Trading Bot — WR90 Long + Short Impulse + Oil Retrace
====================================================================
Three-leg hybrid bot modeled after v14 gold hybrid.

Sources: v14 hybrid bot structure + v29 research ML params.

Signal priority: WR90 Long > Short Impulse > Oil Retrace (only when flat).

Usage:
  python3 trading_bot_oil.py
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brokers.ig_live import IGLiveBrokerAdapter
from brokers.base import OrderRequest
from ig_scripts.ig_data_api import (
    API_CONFIG, IGService, Price,
    fetch_and_store_prices_from_latest,
    fetch_open_positions, fetch_prices as ig_fetch_prices,
)
from data.data_loader import DataLoader
from v15.research.v29_oil_journal import OilSignalJournal

UTC = timezone.utc

# ======================== CONFIG ========================
NY_S, NY_E, NY_FC_H, NY_FC_M = 3, 12, 14, 28
LONG_MAX_B = 60
LONG_ENTRY = -75
LONG_CV, LONG_EP_MIN = 5000, 2
LONG_TP, LONG_SL = 60, 20
LONG_WR_ML_TH = 0.65

SI_CHANGE_MAX, SI_VOL_MIN = -14.0, 800
SI_TP, SI_SL, SI_MAX_B = 120, 80, 90
SI_PROB = 0.55

RET_TP, RET_SL = 30, 15
RET_ML_TH = 0.60

FEATURE_WARMUP_DAYS = 90
OIL_TABLE = "prices"

WR_FEATS = ['wr', 'volume', 'range', 'avg_r3', 'cad', 'ret_1b', 'ret_3b',
            'vol_r', 'h_dlow', 'l_dlow', 'body', 'up', 'up_p1']
SI_FEATS = ['prev_change', 'prev2_change', 'prev_lower_wick', 'prev_volume',
            'prev_range', 'prev_spread', 'ATR', 'ATR_ratio',
            'ret_1m', 'ret_3m', 'ret_5m', 'vol_ratio_20',
            'up_count3_15min', 'ret_3_15m', 'ret_5_15m', 'dist_day_high']
RET_FEATS = ['cad', 'avg_r3', 'bc', 'wb', 'range', 'ret_1b', 'ret_3b', 'ret_5b',
             'vol_r', 'h_dlow', 'l_dlow', 'body', 'up', 'up_p1', 'up_p2', 'body_p1', 'range_p1']


@dataclass
class BotState:
    open_deal_id: Optional[str] = None
    open_side: int = 0
    open_source: Optional[str] = None
    open_entry_time: Optional[str] = None
    open_entry_price: Optional[float] = None
    open_tp: Optional[float] = None
    open_sl: Optional[float] = None
    open_horizon: int = 60
    open_horizon_deadline: Optional[str] = None
    last_predicted_bar_ts: Optional[str] = None
    _last_submitted_bar_ts: Optional[str] = None

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}

    @classmethod
    def from_dict(cls, data: dict) -> "BotState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TradingBotOil:
    """Oil trading bot — WR90 Long, Short Impulse, Oil Retrace."""

    def __init__(self):
        self.log = self._init_logger()
        self.log.info("--- Oil Trading Bot Startup ---")

        self.state_path = PROJECT_ROOT / "runtime" / "trading_bot_oil_state.json"
        self.state = self._load_state()

        self.service = IGService(
            api_key=API_CONFIG["api_key"],
            username=API_CONFIG["username"],
            password=API_CONFIG["password"],
            base_url=API_CONFIG["base_url"],
        )
        self.broker = IGLiveBrokerAdapter(
            self.service, instrument=Price.Oil,
            stop_loss_pct=SI_SL, take_profit_pct=SI_TP,
        )

        self.cached = pd.DataFrame()
        self.journal = OilSignalJournal()
        self._init_cache()

    def _init_logger(self):
        log = logging.getLogger("OilBot")
        if log.handlers:
            return log
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
        (PROJECT_ROOT / "runtime").mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(PROJECT_ROOT / "runtime" / "oil_bot.log")
        fh.setFormatter(fmt)
        log.addHandler(fh)
        return log

    def _load_state(self) -> BotState:
        if self.state_path.exists():
            try:
                return BotState.from_dict(json.loads(self.state_path.read_text()))
            except Exception as e:
                self.log.error(f"State load failed: {e}")
        return BotState()

    def _save_state(self):
        try:
            self.state_path.write_text(json.dumps(self.state.to_dict(), indent=2))
        except Exception as e:
            self.log.error(f"State save failed: {e}")

    def _clear_position(self):
        self.state.open_deal_id = None
        self.state.open_side = 0
        self.state.open_source = None
        self.state.open_entry_time = None
        self.state.open_entry_price = None
        self.state.open_tp = None
        self.state.open_sl = None
        self.state.open_horizon_deadline = None
        self._save_state()

    def _init_cache(self):
        """Load warmup data from MySQL."""
        end = datetime.now(UTC)
        start = end - pd.Timedelta(days=FEATURE_WARMUP_DAYS)
        self.log.info(f"Loading oil data {start.date()} → {end.date()} from MySQL...")
        loader = DataLoader()
        try:
            raw = loader.load_data(table_name=OIL_TABLE,
                                   start_date=start.strftime('%Y-%m-%d'),
                                   end_date=end.strftime('%Y-%m-%d'))
        except Exception:
            self.log.warning("MySQL oil table not found, trying prices table...")
            raw = loader.load_data(table_name='prices',
                                   start_date=start.strftime('%Y-%m-%d'),
                                   end_date=end.strftime('%Y-%m-%d'))
        raw.index = pd.to_datetime(raw['timestamp'], unit='ms')
        df = pd.DataFrame(index=raw.index)
        for c, src in [('open', 'openPrice_ask'), ('high', 'highPrice_ask'),
                       ('low', 'lowPrice_ask'), ('close_ask', 'closePrice_ask'),
                       ('close_bid', 'closePrice_bid'), ('volume', 'lastTradedVolume')]:
            df[c] = raw.get(src, pd.Series(np.nan, index=raw.index)).astype(float)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        self.cached = df
        self.log.info(f"  Loaded {len(self.cached):,} bars ({self.cached.index[0]} → {self.cached.index[-1]})")

    # ======================== FEATURE / SIGNAL FUNCTIONS ========================

    def _build_15m(self):
        d = self.cached.resample('15min', label='right', closed='right').agg(
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
        d['ny_h'] = ny.hour; d['ny_m'] = ny.minute
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

    def _detect_wr90(self, d15):
        if len(d15) < 15: return None
        in_s = d15['ins']
        o = (d15['wr'] < LONG_ENTRY) & in_s
        cv, bc = 0.0, 0
        last_start = None
        for i in range(len(d15)):
            if o.iloc[i]:
                if last_start is None: last_start = i
                cv += d15['volume'].iloc[i]; bc += 1
            elif last_start is not None:
                ebi = i
                if ebi < len(d15) and d15['ins'].iloc[min(ebi, len(d15) - 1)] \
                   and cv >= LONG_CV and bc >= LONG_EP_MIN:
                    bar = d15.iloc[ebi]
                    return {
                        'type': 'wr90', 'side': 1,
                        'entry_price': float(bar['close_ask']),
                        'bar_idx': ebi, 'bar_time': d15.index[ebi],
                        'tp': LONG_TP, 'sl': LONG_SL,
                    }
                cv, bc = 0.0, 0; last_start = None
        return None

    def _compute_si_features(self):
        df = self.cached.copy()
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
        df['ret_3_15m'] = m15['ret_3_15m']; df['ret_5_15m'] = m15['ret_5_15m']
        daily_high = df['high'].resample('D').max().reindex(df.index, method='ffill')
        df['dist_day_high'] = daily_high - df['close_ask']
        return df

    def _detect_si(self, d1m_feats):
        if len(d1m_feats) < 30:
            return None, None
        recent = d1m_feats.iloc[-1]
        if not (recent['prev_change'] < SI_CHANGE_MAX and recent['prev_volume'] > SI_VOL_MIN and recent['ny_hour']):
            return None, None
        feat = np.array([float(recent.get(c, 0)) for c in SI_FEATS])
        if np.isnan(feat).any():
            return None, None
        # Simple prob: train on-the-fly (simplified for live)
        model = self._train_si_xgb(d1m_feats)
        if model is None: return None, None
        prob = float(model.predict_proba(feat.reshape(1, -1))[0, 1])
        return prob, recent

    def _train_si_xgb(self, df_si):
        try:
            future = df_si['close_bid'].shift(-SI_MAX_B) - df_si['close_bid']
            df_si['target_down'] = (future < -SI_TP).astype(int)
            valid = df_si.dropna(subset=SI_FEATS + ['target_down'])
            if len(valid) < 200: return None
            X = valid[SI_FEATS].values.astype(np.float32)
            y = valid['target_down'].values
            if y.sum() < 10: return None
            model = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05,
                                       random_state=42, verbosity=0, use_label_encoder=False,
                                       eval_metric='logloss')
            model.fit(X, y)
            return model
        except Exception:
            return None

    # ======================== POSITION MANAGEMENT ========================

    def _sync_position(self):
        if not self.state.open_deal_id:
            return
        try:
            positions = fetch_open_positions(self.service)
            active = [p.get('position', {}).get('dealId', '') for p in positions]
            if self.state.open_deal_id not in active:
                self.log.info(f"Position {self.state.open_deal_id} closed")
                self._clear_position()
        except Exception as e:
            self.log.error(f"Position sync failed: {e}")

    def _check_horizon(self):
        if not self.state.open_deal_id or not self.state.open_horizon_deadline:
            return
        deadline = pd.Timestamp(self.state.open_horizon_deadline)
        if deadline.tzinfo is None:
            deadline = deadline.tz_localize('UTC')
        if datetime.now(UTC) < deadline:
            return
        self.log.info(f"Horizon timeout — closing {self.state.open_deal_id}")
        try:
            pos = self.broker.get_position_by_deal_id(self.state.open_deal_id)
            if pos:
                direction = pos.get('position', {}).get('direction', 'BUY')
                size = pos.get('position', {}).get('size', 1.0)
                close_dir = 'SELL' if direction == 'BUY' else 'BUY'
                self.broker.close_position(deal_id=self.state.open_deal_id,
                                           direction=close_dir, size=float(size))
            self._clear_position()
        except Exception as e:
            self.log.error(f"Horizon close error: {e}")

    def _submit_signal(self, sig: dict, bar_ts: pd.Timestamp):
        if self.state._last_submitted_bar_ts and bar_ts <= pd.Timestamp(self.state._last_submitted_bar_ts):
            self.log.warning(f"Duplicate bar {bar_ts} — skipping")
            return False
        if self.state.open_deal_id:
            return False

        # Verify IG is flat
        try:
            if fetch_open_positions(self.service):
                self.log.error("IG has open positions — BLOCKING entry")
                return False
        except Exception:
            return False

        side_str = 'buy' if sig['side'] == 1 else 'sell'
        request = OrderRequest(
            symbol='oil', side=side_str, size=1.0,
            signal_time_utc=bar_ts.isoformat(),
            entry_time_utc=bar_ts.isoformat(),
            entry_price=float(sig['entry_price']),
            probability=float(sig.get('prob', 1.0)),
            signal_model_family='oil_combined',
            metadata={
                'stop_distance': sig['sl'], 'limit_distance': sig['tp'],
                'source': sig['type'],
            },
        )
        result = self.broker.submit_order(request)
        if result.submitted:
            self.state._last_submitted_bar_ts = bar_ts.isoformat()
            self.state.open_deal_id = result.deal_id
            self.state.open_side = sig['side']
            self.state.open_source = sig['type']
            self.state.open_entry_time = bar_ts.isoformat()
            self.state.open_entry_price = float(sig['entry_price'])
            self.state.open_tp = sig['tp']
            self.state.open_sl = sig['sl']
            self.state.open_horizon = 60 if sig['type'] == 'wr90' else 90
            deadline = bar_ts + pd.Timedelta(minutes=self.state.open_horizon)
            self.state.open_horizon_deadline = deadline.isoformat()
            self._save_state()
            self.log.info(f"✓ ORDER SUBMITTED: {sig['type']} side={sig['side']} "
                          f"@ {sig['entry_price']:.1f} TP={sig['tp']} SL={sig['sl']} "
                          f"deal_id={result.deal_id}")
            return True
        else:
            self.log.error(f"Order failed: {result.reason}")
            return False

    # ======================== MAIN ROUTINE ========================

    def poll(self):
        """Called every ~5 seconds; fetches data once per minute at :06s."""
        now = datetime.now(UTC)
        sec = now.second

        # Sync & horizon check
        self._sync_position()
        self._check_horizon()

        # Fetch IG data at :06s
        if sec != 6:
            return
        try:
            fetch_and_store_prices_from_latest(self.service, Price.Oil)
            result = ig_fetch_prices(self.service, Price.Oil, resolution='MINUTE',
                                     start_time=now - pd.Timedelta(minutes=2), end_time=now)
            if not result:
                return
            new_df = pd.DataFrame(result)
            new_df.index = pd.to_datetime(new_df.index)
            if new_df.index.tz is None:
                new_df.index = new_df.index.tz_localize('UTC')
            for idx, row in new_df.iterrows():
                if idx not in self.cached.index:
                    self.cached.loc[idx] = row
            self.cached = self.cached[~self.cached.index.duplicated(keep='last')].sort_index()
        except Exception as e:
            self.log.error(f"Fetch error: {e}")
            return

        latest_bar = self.cached.index[-1]
        if self.state.last_predicted_bar_ts and latest_bar <= pd.Timestamp(self.state.last_predicted_bar_ts):
            return
        self.state.last_predicted_bar_ts = latest_bar.isoformat()

        # Save features
        try:
            feat_df = self._compute_si_features()
            row = feat_df.loc[latest_bar]
            feat_dict = {f: float(row.get(f, 0)) for f in SI_FEATS}
            feat_dict['open'] = float(row.get('open', 0))
            feat_dict['high'] = float(row.get('high', 0))
            feat_dict['low'] = float(row.get('low', 0))
            feat_dict['close'] = float(row.get('close_ask', 0))
            self.journal.record_bar_feature(str(latest_bar), json.dumps(feat_dict, default=str))
        except Exception:
            pass

        # Check WR90 (only if flat)
        if not self.state.open_deal_id:
            d15 = self._build_15m()
            wr90 = self._detect_wr90(d15)
            if wr90:
                self.log.info(f"[WR90] LONG @ {wr90['bar_time']} entry={wr90['entry_price']:.1f}")
                submitted = self._submit_signal(wr90, latest_bar)
                self.journal.record_score(str(latest_bar), pattern_name='wr90_long', pattern_side=1,
                                           pattern_prob=1.0, action='entry' if submitted else 'blocked')

        # Check Short Impulse (only if flat)
        if not self.state.open_deal_id:
            try:
                si_feats = self._compute_si_features()
                prob, recent = self._detect_si(si_feats)
                if prob is not None and prob >= SI_PROB:
                    sig = {
                        'type': 'short_impulse', 'side': -1,
                        'entry_price': float(recent['close_bid']),
                        'tp': SI_TP, 'sl': SI_SL, 'prob': prob,
                    }
                    self.log.info(f"[SI] SHORT prob={prob:.3f} @ {sig['entry_price']:.1f}")
                    submitted = self._submit_signal(sig, latest_bar)
                    self.journal.record_score(str(latest_bar), pattern_name='short_impulse',
                                               pattern_side=-1, pattern_prob=prob,
                                               action='entry' if submitted else 'score')
                elif prob is not None:
                    self.journal.record_score(str(latest_bar), pattern_name='short_impulse',
                                               pattern_side=0, pattern_prob=prob, action='score')
            except Exception as e:
                self.log.error(f"SI error: {e}")

        if self.state.open_deal_id:
            self.journal.record_score(str(latest_bar), action='in_trade')

    def run(self):
        self.log.info("=" * 50)
        self.log.info("  OIL TRADING BOT — Running")
        self.log.info("=" * 50)

        while True:
            try:
                self.poll()
                time.sleep(1.5)
            except KeyboardInterrupt:
                self.log.info("Shutdown requested")
                break
            except Exception as e:
                self.log.error(f"Loop error: {e}", exc_info=True)
                time.sleep(5)


if __name__ == '__main__':
    pid_file = PROJECT_ROOT / "runtime" / "trading_bot_oil.pid"
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            os.kill(old_pid, 0)
            print(f"ERROR: Oil bot already running (PID {old_pid})", file=sys.stderr)
            sys.exit(1)
        except (OSError, ValueError):
            pid_file.unlink(missing_ok=True)
    pid_file.write_text(str(os.getpid()))
    atexit.register(lambda: pid_file.unlink() if pid_file.exists() else None)

    TradingBotOil().run()
