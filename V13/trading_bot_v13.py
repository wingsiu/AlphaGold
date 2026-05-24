#!/usr/bin/env python3
"""AlphaGold v13 Trading Bot.
Uses Stage 1 (Filter) + Stage 2 (Directional) XGBoost models on a 1-minute timeframe.
"""

import sys
import os
import time
import json
import logging
import joblib
import ta
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import json
from zoneinfo import ZoneInfo
from typing import Optional, Any
from dataclasses import dataclass, asdict

# Setup project paths
from V13._paths import PROJECT_ROOT

# Imports from your application
from ig_scripts.ig_data_api import Price, IGService, fetch_prices, fetch_market_snapshot, fetch_open_positions, API_CONFIG
from brokers.ig_live import IGLiveBrokerAdapter
from execution.engine import ExecutionEngine
from config.v13_config import FILTER_CONFIG, TARGET_CONFIG, MODEL_CONFIG, EXECUTION_CONFIG, WF_CONFIG
from xgboost_filter_model.train_filter_1min import prepare_features as prepare_base_features
from xgboost_filter_model.train_filter_v10 import add_liquidity_indicators

# --- Constants ---
UTC = timezone.utc
NY_TZ = ZoneInfo("America/New_York")
HK_TZ = ZoneInfo("Asia/Hong_Kong")
LONDON_TZ = ZoneInfo("Europe/London")
TRADING_DAY_CUTOFF_HOUR_NY = 17

# --- Helper Functions for Features ---

def _ts_to_ny(ts) -> pd.Timestamp:
    """Normalize a date/datetime string or Timestamp to America/New_York."""
    t = pd.to_datetime(ts)
    if t.tzinfo is None:
        return t.tz_localize(NY_TZ)
    return t.tz_convert(NY_TZ)

def _session_info(ts, timezone, start_h, start_m, end_h, end_m):
    local_ts = ts.tz_convert(timezone)
    minute_of_day = local_ts.hour * 60 + local_ts.minute
    s = start_h * 60 + start_m
    e = end_h * 60 + end_m
    if s <= minute_of_day < e:
        return 1.0, (minute_of_day - s) / (e - s)
    return 0.0, 0.0

def add_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for m in [10, 30, 90]:
        df[f'ma_{m}'] = df['close'].rolling(window=m).mean()
        df[f'price_vs_ma_{m}'] = (df['close'] - df[f'ma_{m}']) / (df[f'ma_{m}'] + 1e-9)
    df['ma_10_vs_30'] = (df['ma_10'] - df['ma_30']) / (df['ma_30'] + 1e-9)
    df['ma_30_vs_90'] = (df['ma_30'] - df['ma_90']) / (df['ma_90'] + 1e-9)
    return df

def add_directional_features(df: pd.DataFrame) -> pd.DataFrame:
    # MUST mirror xgboost_filter_model/train_directional_model_v2.py exactly,
    # otherwise train/serve skew on directional_change_* and wick_ratio_*
    # → biased S2 probabilities. See investigation 2026-05-18.
    df = df.copy()
    for w in [15, 30, 90]:
        rolling_high  = df['high'].rolling(window=w).max()
        rolling_low   = df['low'].rolling(window=w).min()
        rolling_open  = df['open'].shift(w - 1)
        rolling_close = df['close']
        rolling_range = (rolling_high - rolling_low).replace(0, np.nan)
        df[f'directional_change_{w}'] = (rolling_close - rolling_open) / rolling_range

        upper_wicks = (df['high'] - df[['open', 'close']].max(axis=1)).rolling(window=w).sum()
        lower_wicks = (df[['open', 'close']].min(axis=1) - df['low']).rolling(window=w).sum()
        df[f'wick_ratio_{w}'] = upper_wicks / (lower_wicks + 1e-6)
    return df

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in [14, 30]:
        df[f'rsi_{w}'] = ta.momentum.RSIIndicator(df['close'], window=w).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    for w in [15, 30, 60]:
        df[f'roc_{w}'] = ta.momentum.ROCIndicator(df['close'], window=w).roc()
    return df

def prepare_v13_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Base Feature Engineering (re-uses exactly what the model was trained on)
    df = prepare_base_features(df, move_threshold=10, er_threshold=0.3, future_window=45, for_live_inference=True)

    # 2. Add Liquidity Zone Indicators
    df = add_liquidity_indicators(df)

    # 3. Add v13 explicit features missing from base
    # Session Progress
    def _get_sessions(ts):
        asia_f, asia_p = _session_info(ts, HK_TZ, 8, 0, 16, 0)
        lon_f, lon_p = _session_info(ts, LONDON_TZ, 8, 0, 16, 30)
        ny_f, ny_p = _session_info(ts, NY_TZ, 9, 30, 16, 0)
        return pd.Series([asia_f, asia_p, lon_f, lon_p, ny_f, ny_p])

    # Avoid SettingWithCopyWarning if possible by updating immediately
    sessions_df = df.index.to_series().apply(_get_sessions)
    sessions_df.columns = ['is_asia', 'asia_progress', 'is_london', 'london_progress', 'is_ny', 'ny_progress']
    for c in sessions_df.columns:
        df[c] = sessions_df[c]

    # Day rolling features (UTC+2)
    day_start_offset = pd.Timedelta(hours=2)
    df["day_utc2"] = (df.index + day_start_offset).floor("D")
    df["day_open"] = df.groupby("day_utc2")["open"].transform("first")
    df["day_high_rolling"] = df.groupby("day_utc2")["high"].cummax()
    df["day_low_rolling"] = df.groupby("day_utc2")["low"].cummin()
    df["Dchange_utc2_rel"] = (df["close"] - df["day_open"]) / (df["day_open"] + 1e-9)

    max_oc = df[["day_open", "close"]].max(axis=1)
    min_oc = df[["day_open", "close"]].min(axis=1)
    df["Dupper_wick_utc2_rel"] = (df["day_high_rolling"] - max_oc) / (df["day_open"] + 1e-9)
    df["Dlower_wick_utc2_rel"] = (min_oc - df["day_low_rolling"]) / (df["day_open"] + 1e-9)


    # ── Energetic-bar filter (MUST mirror prepare_data_v13 ordering) ────────
    # The model was trained on a dataframe that was filtered to energetic bars
    # (move > min_bar_move AND volume > min_volume) BEFORE the directional/MA/
    # momentum features were computed. Those features therefore roll over
    # consecutive energetic bars, not raw 1-min bars. We must replicate that
    # here at serving time, otherwise MA/RSI/MACD/ROC/directional_change/etc.
    # are computed on a different population than at training time
    # (train/serve skew → biased S1/S2 probabilities, e.g. "always DOWN").
    df["bar_move"] = (df["close"] - df["open"]).abs()
    df = df[(df["bar_move"] > FILTER_CONFIG["min_bar_move"]) &
            (df["volume"]    > FILTER_CONFIG["min_volume"])].copy()

    # Final TA feature groups
    df = add_directional_features(df)
    df = add_ma_features(df)
    df = add_momentum_features(df)

    return df

@dataclass
class BotState:
    consecutive_losses: int = 0
    last_pnl: float = 0.0
    open_deal_id: Optional[str] = None
    open_entry_time: Optional[str] = None  # ISO-8601 UTC; used to enforce EXECUTION_CONFIG['horizon'] timeout
    closed_first_seen_at: Optional[str] = None  # ISO-8601 UTC; first poll where IG reported the open_deal_id no longer open
    pending_level_resync_bar: Optional[str] = None  # ISO-8601 UTC; bar AFTER signal bar whose open we use to set SL/TP (mirrors backtest next_row entry)
    last_retrain_date: Optional[str] = None # ISO format (YYYY-MM-DD)
    last_reconciliation_day: Optional[str] = None # YYYY-MM-DD

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BotState":
        # Filter for only known fields to avoid errors with legacy state formats
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

def extract_image_payload(df: pd.DataFrame, i: int, window: int = 150):
    w = df.iloc[i - window + 1 : i + 1]
    c0 = float(w["close"].iloc[0]) or 1.0
    open_rel  = w["open"].to_numpy()  / c0 - 1.0
    high_rel  = w["high"].to_numpy()  / c0 - 1.0
    low_rel   = w["low"].to_numpy()   / c0 - 1.0
    close_rel = w["close"].to_numpy() / c0 - 1.0
    body_rel  = (w["close"].to_numpy() - w["open"].to_numpy()) / c0
    range_rel = (w["high"].to_numpy()  - w["low"].to_numpy())  / c0
    vol = w["volume"].to_numpy(dtype=float)
    vol_mean, vol_std = np.mean(vol), np.std(vol)
    vol_z = np.zeros_like(vol) if vol_std < 1e-9 else (vol - vol_mean) / vol_std
    v0 = float(vol[0])
    vol_rel = np.zeros_like(vol) if abs(v0) < 1e-9 else vol / v0 - 1.0
    vd = np.diff(vol, prepend=vol[0])
    vd_std = np.std(vd)
    vol_diff_norm = np.zeros_like(vd) if vd_std < 1e-9 else vd / vd_std

    return np.stack([open_rel, high_rel, low_rel, close_rel, body_rel, range_rel, vol_z, vol_rel, vol_diff_norm], axis=0).flatten()

class AlphaGoldV13Bot:
    def __init__(self):
        self.logger = self._init_logger()
        self.logger.info("--- AlphaGold v13 Bot Startup ---")

        # Paths
        self.state_path = PROJECT_ROOT / "runtime" / "trading_bot_state.json"
        self.state = self._load_state()

        self.logger.info(f"Execution Config: {json.dumps(EXECUTION_CONFIG, indent=2)}")
        self.logger.info(f"Target Config: {json.dumps(TARGET_CONFIG, indent=2)}")
        self.logger.info(f"Filter Config: {json.dumps(FILTER_CONFIG, indent=2)}")

        self.service = IGService(
            api_key=API_CONFIG["api_key"],
            username=API_CONFIG["username"],
            password=API_CONFIG["password"],
            base_url=API_CONFIG["base_url"],
        )
        self.broker = IGLiveBrokerAdapter(
            self.service,
            instrument=Price.Gold,
            stop_loss_pct=EXECUTION_CONFIG["sl"],
            # Use absolute-point TP (40) to match backtest, NOT take_profit_pct (0.8%).
            take_profit_pct=EXECUTION_CONFIG["tp"],
        )
        self.engine = ExecutionEngine(self.broker)

        # Load Artifacts from runtime/bot_assets
        assets_dir = PROJECT_ROOT / "V13" / "runtime" / "bot_assets"
        s1_file = assets_dir / "filter_model_v13_wf_image.joblib"
        s2_file = assets_dir / "directional_model_v13_wf.joblib"
        img_file = assets_dir / "image_trend_model.joblib"

        self.s1_model = joblib.load(s1_file)
        self.s2_model = joblib.load(s2_file)
        img_bundle = joblib.load(img_file)
        self.img_s1_model = img_bundle["stage1"]
        self.img_win = img_bundle["config"].get("window", 150)

        # Feature Lists
        self.v13_s1_cols = [
            'returns', 'adx', 'adx_slope', 'volatility', 'er_15', 'er_30', 'er_90', 'fractal_dimension',
            'wr_15', 'wr_30', 'wr_90', 'change_15', 'upper_wick_15', 'lower_wick_15', 'change_30',
            'upper_wick_30', 'lower_wick_30', 'change_90', 'upper_wick_90', 'lower_wick_90',
            'down_efficiency_ratio', 'up_efficiency_ratio', 'volume_price_corr', 'volume_trend',
            'volume_osc', 'change', 'upper_wick', 'lower_wick', 'bar_change', 'bar_upper_wick',
            'bar_lower_wick', 'day_progress', 'is_asia', 'asia_progress', 'is_london',
            'london_progress', 'is_ny', 'ny_progress', 'is_eq_high', 'is_eq_low', 'near_high_zone',
            'near_low_zone', 'recovery_long', 'recovery_short', 'image_s1_prob'
        ]
        self.v13_s2_cols = self.v13_s1_cols + [
            'directional_change_15', 'wick_ratio_15', 
            'directional_change_30', 'wick_ratio_30', 
            'directional_change_90', 'wick_ratio_90',
            'price_vs_ma_10', 'price_vs_ma_30', 'price_vs_ma_90', 'ma_10_vs_30', 'ma_30_vs_90',
            'rsi_14', 'rsi_30', 'macd', 'macd_signal', 'macd_diff', 'roc_15', 'roc_30', 'roc_60'
        ]

        # Use the same warm-up condition as backtest for consistency.
        self.feature_warmup_days = int(WF_CONFIG.get("feature_warmup_days", 120))
        self.min_prediction_bars = max(self.img_win + 100, 400)

        self.last_ts = None
        # Track the last bar timestamp we've already scored, so the same minute
        # bar is never predicted twice (minute-level alignment with backtest).
        self.last_predicted_bar_ts = None
        self.cached_df = pd.DataFrame()
        self._warmup_cache()

    def _warmup_cache(self):
        """Initial fetch to populate history at startup."""
        self.logger.info(
            f"Performing cache warmup (syncing latest from IG and loading ~{self.feature_warmup_days} days from DB)..."
        )
        try:
            from data.data_loader import DataLoader
            from ig_scripts.ig_data_api import fetch_and_store_prices_from_latest
            
            # 1. Sync missing data from IG to the local database
            fetch_and_store_prices_from_latest(self.service, Price.Gold)
            
            # 2. Load history from local database and keep last N days for feature warm-up
            df = DataLoader().load_data(Price.Gold.db_name)
            if not df.empty:
                df.index = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.rename(columns={'openPrice': 'open', 'highPrice': 'high', 'lowPrice': 'low', 'closePrice': 'close', 'lastTradedVolume': 'volume'}).sort_index()
                warmup_cutoff = df.index.max() - pd.Timedelta(days=self.feature_warmup_days)
                df = df[df.index >= warmup_cutoff].copy()
                self.cached_df = df[~df.index.duplicated(keep='last')]
                self.logger.info(f"Cache warmed up with {len(self.cached_df)} rows from database.")
            else:
                self.logger.warning("Database returned empty dataframe.")
        except Exception as e:
            self.logger.error(f"Cache warmup failed: {e}")

    def _load_state(self) -> BotState:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    return BotState.from_dict(json.load(f))
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
        return BotState()

    def _save_state(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def _sync_trade_results(self):
        """Check if open position has closed and update consecutive_losses."""
        if not self.state.open_deal_id:
            return

        try:
            # Check if our deal_id is still in open positions
            open_pos = fetch_open_positions(self.service)
            is_still_open = False
            for p in open_pos:
                if p.get('position', {}).get('dealId') == self.state.open_deal_id:
                    is_still_open = True
                    break

            if not is_still_open:
                self.logger.info(f"Position {self.state.open_deal_id} is no longer open. Fetching closed trade result...")
                
                # Use the broker adapter to get closed trade details
                closed_trade = self.broker.get_closed_trade_by_deal_id(self.state.open_deal_id)
                
                if closed_trade:
                    pnl = closed_trade.get("pnl", 0.0) or 0.0
                    self.logger.info(f"Trade {self.state.open_deal_id} closed. PnL: {pnl:.2f}. Reason: {closed_trade.get('reason')}")
                    
                    self.state.last_pnl = pnl
                    if pnl < 0:
                        self.state.consecutive_losses += 1
                    else:
                        self.state.consecutive_losses = 0
                    
                    self.state.open_deal_id = None
                    self.state.open_entry_time = None
                    self.state.closed_first_seen_at = None
                    self._save_state()
                    self.logger.info(f"State updated: consecutive_losses={self.state.consecutive_losses}")
                else:
                    # IG sometimes takes a moment to move the trade to history.
                    # Track when we first saw the position-not-open and force-clear
                    # after a grace period so a missing history entry doesn't
                    # permanently block all future entries.
                    now_iso = datetime.now(timezone.utc).isoformat()
                    grace_minutes = 5
                    if not self.state.closed_first_seen_at:
                        self.state.closed_first_seen_at = now_iso
                        self._save_state()
                        self.logger.warning(f"Position {self.state.open_deal_id} closed but not found in history yet. Will retry (grace={grace_minutes}m)...")
                    else:
                        first_seen = pd.Timestamp(self.state.closed_first_seen_at)
                        elapsed_min = (pd.Timestamp(now_iso) - first_seen).total_seconds() / 60.0
                        if elapsed_min >= grace_minutes:
                            self.logger.warning(
                                f"Position {self.state.open_deal_id} not in history after {elapsed_min:.1f}m "
                                f"(grace {grace_minutes}m exceeded) \u2014 force-clearing state to unblock new entries."
                            )
                            self.state.open_deal_id = None
                            self.state.open_entry_time = None
                            self.state.closed_first_seen_at = None
                            self._save_state()
                        else:
                            self.logger.warning(
                                f"Position {self.state.open_deal_id} closed but not in history yet ({elapsed_min:.1f}/{grace_minutes}m). Will retry..."
                            )
            else:
                # Position is open again — clear any stale closed-detect marker.
                if self.state.closed_first_seen_at:
                    self.state.closed_first_seen_at = None
                    self._save_state()

        except Exception as e:
            self.logger.error(f"Error syncing trade results: {e}")

    def _check_horizon_timeout(self):
        """Force-close the open position if it has been held longer than
        EXECUTION_CONFIG['horizon'] minutes (mirrors backtest 'timeout' exit).
        """
        if not self.state.open_deal_id or not self.state.open_entry_time:
            return
        try:
            entry_ts = pd.Timestamp(self.state.open_entry_time)
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize('UTC')
            now_utc = pd.Timestamp(datetime.now(timezone.utc))
            elapsed_min = (now_utc - entry_ts).total_seconds() / 60.0
            horizon_min = float(EXECUTION_CONFIG.get("horizon", 45))
            if elapsed_min < horizon_min:
                return

            # Look up direction & size from IG so we can submit the close order.
            pos_wrap = self.broker.get_position_by_deal_id(self.state.open_deal_id)
            if not pos_wrap:
                # IG no longer reports it as open — let _sync_trade_results clean it up.
                self.logger.info(
                    f"Horizon elapsed ({elapsed_min:.1f}m >= {horizon_min}m) but position "
                    f"{self.state.open_deal_id} not found on IG; skipping close."
                )
                return
            pos = pos_wrap.get('position', {})
            direction = str(pos.get('direction', '')).upper()  # 'BUY' or 'SELL'
            size = float(pos.get('size') or EXECUTION_CONFIG.get("size", 1.0))
            close_direction = "SELL" if direction == "BUY" else "BUY"

            self.logger.info(
                f"HORIZON TIMEOUT deal_id={self.state.open_deal_id} elapsed={elapsed_min:.1f}m "
                f"horizon={horizon_min}m — closing position ({direction} {size} -> {close_direction})"
            )
            close_res = self.broker.close_position(
                deal_id=self.state.open_deal_id,
                direction=close_direction,
                size=size,
            )
            self.logger.info(f"Horizon close result: {close_res}")
            # _sync_trade_results on the next poll will reconcile PnL and clear state.
        except Exception as e:
            self.logger.error(f"Error in horizon timeout check: {e}")

    def _resync_levels_to_backtest(self, df: pd.DataFrame) -> None:
        """Once the bar AFTER the signal bar is available, amend SL/TP so they
        match backtest convention exactly:
            ep = next_row['open'] + spread (LONG) or - spread (SHORT)
            stop  = ep - sl (LONG) or ep + sl (SHORT)
            target= ep + tp (LONG) or ep - tp (SHORT)
        Backtest enters at next_row's open price; live IG fill is a few seconds
        later at market — this aligns the SL/TP anchors so live trade outcomes
        track the backtest.
        """
        pending = self.state.pending_level_resync_bar
        if not pending or not self.state.open_deal_id:
            return
        try:
            target_ts = pd.Timestamp(pending)
            if target_ts.tzinfo is None:
                target_ts = target_ts.tz_localize('UTC')
        except Exception:
            self.state.pending_level_resync_bar = None
            self._save_state()
            return

        # Give up if the target bar is more than 10 min in the past (cache may have
        # rolled it off, or position already closed).
        now_utc = pd.Timestamp(datetime.now(timezone.utc))
        if (now_utc - target_ts).total_seconds() > 600:
            self.logger.warning(
                f"Level resync target bar {target_ts} too old (>10m); abandoning."
            )
            self.state.pending_level_resync_bar = None
            self._save_state()
            return

        if target_ts not in df.index:
            return  # bar not finalized yet — try again on next poll

        # Look up direction from IG (we don't trust local state for direction)
        try:
            pos_wrap = self.broker.get_position_by_deal_id(self.state.open_deal_id)
        except Exception as e:
            self.logger.warning(f"Level resync: get_position failed: {e}")
            return
        if not pos_wrap:
            self.logger.info(
                f"Level resync: position {self.state.open_deal_id} no longer open; clearing pending."
            )
            self.state.pending_level_resync_bar = None
            self._save_state()
            return
        direction = str(pos_wrap.get('position', {}).get('direction', '')).upper()
        side = 1 if direction == 'BUY' else -1

        bar_open_mid = float(df.loc[target_ts, 'open'])
        # Prefer real bid/ask columns from IG (carried through from fetch_prices),
        # fall back to mid + spread_default only if missing.
        ask_col = df.loc[target_ts].get('openPrice_ask') if hasattr(df.loc[target_ts], 'get') else None
        bid_col = df.loc[target_ts].get('openPrice_bid') if hasattr(df.loc[target_ts], 'get') else None
        try:
            bar_open_ask = float(ask_col) if ask_col is not None and not pd.isna(ask_col) else None
            bar_open_bid = float(bid_col) if bid_col is not None and not pd.isna(bid_col) else None
        except (TypeError, ValueError):
            bar_open_ask = bar_open_bid = None

        if bar_open_ask is None or bar_open_bid is None:
            spread = float(EXECUTION_CONFIG.get('spread_default', 0.0))
            bar_open_ask = bar_open_mid + spread
            bar_open_bid = bar_open_mid - spread

        tp = float(EXECUTION_CONFIG['tp'])
        sl = float(EXECUTION_CONFIG['sl'])

        ep = bar_open_ask if side == 1 else bar_open_bid
        stop_level = ep - sl if side == 1 else ep + sl
        limit_level = ep + tp if side == 1 else ep - tp

        try:
            self.broker.amend_position_levels(
                deal_id=self.state.open_deal_id,
                stop_level=round(stop_level, 2),
                limit_level=round(limit_level, 2),
            )
            self.logger.info(
                f"Level resync OK deal_id={self.state.open_deal_id} side={'BUY' if side==1 else 'SELL'} "
                f"bar={target_ts} open_mid={bar_open_mid:.2f} ask={bar_open_ask:.2f} bid={bar_open_bid:.2f} "
                f"ep={ep:.2f} -> stop={stop_level:.2f} limit={limit_level:.2f}"
            )
            self.state.pending_level_resync_bar = None
            self._save_state()
        except Exception as e:
            self.logger.error(f"Level resync amend failed: {e}")

    def _init_logger(self):
        l = logging.getLogger("AlphaGoldV13")
        # Avoid adding duplicate handlers if re-instantiated
        if l.handlers:
            return l
        l.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        # Console
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        l.addHandler(sh)
        # File — writes to runtime/trading_bot_v13.log
        log_path = PROJECT_ROOT / "V13" / "runtime" / "logs" / "trading_bot_v13.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        l.addHandler(fh)
        return l

    def poll(self, store_to_db: bool = False):
        now = datetime.now(timezone.utc)
        now_pd = pd.Timestamp(now)
        now_ny = now_pd.tz_convert(NY_TZ)

        # --- Daily Reconciliation Trigger ---
        if not store_to_db:
            current_trading_day = (now_ny - pd.Timedelta(hours=TRADING_DAY_CUTOFF_HOUR_NY)).floor("D")
            last_recon_str = self.state.last_reconciliation_day
            last_recon_day = _ts_to_ny(last_recon_str).floor("D") if last_recon_str else None

            if last_recon_day is not None and current_trading_day > last_recon_day:
                self.logger.info("New trading day started (NY 17:00). Triggering daily reconciliation for previous day...")
                import subprocess
                prev_day_str = last_recon_day.strftime('%Y-%m-%d')

                # Determine additional reports
                recon_cmd = [sys.executable, str(PROJECT_ROOT / "v14" / "tools" / "daily_reconciliation.py"), prev_day_str]

                # 1. Weekly: If yesterday was Friday (weekday 4)
                if last_recon_day.weekday() == 4:
                    recon_cmd.append("--weekly")

                # 2. Cycle: If today is a Retrain Day (cycle boundary)
                from config.v13_config import WF_CONFIG
                wf_start = _ts_to_ny(WF_CONFIG["wf_start"])
                days_since_start = (current_trading_day - wf_start.floor("D")).days
                retrain_days = WF_CONFIG.get("retrain_days", 14)
                if days_since_start % retrain_days == 0:
                    self.logger.info(f"Cycle boundary detected (day {days_since_start}). Adding cycle report...")
                    recon_cmd.append("--cycle")

                try:
                    subprocess.Popen(recon_cmd)
                except Exception as e:
                    self.logger.error("Failed to trigger daily reconciliation: %s", e)

            # Update the state to the current trading day
            new_recon_day_str = current_trading_day.strftime('%Y-%m-%d')
            if self.state.last_reconciliation_day != new_recon_day_str:
                self.state.last_reconciliation_day = new_recon_day_str
                self._save_state()
        # ------------------------------------

        # 0. Sync trade results before predicting
        if not store_to_db:
            self._sync_trade_results()
            # Enforce horizon (45-min) timeout to mirror backtest exit logic.
            self._check_horizon_timeout()

        # 0. Always show account status first if storing to DB
        # because account info is high-priority and independent of price data.
        if store_to_db:
            try:
                from ig_scripts.ig_data_api import fetch_primary_account_summary
                acc_summary = fetch_primary_account_summary(self.service)
                print(f"\n--- Account Status Update [{now.strftime('%Y-%m-%d %H:%M:%S')}] ---", flush=True)
                print(f"Account:   {acc_summary.get('account_name')} ({acc_summary.get('account_id')})", flush=True)
                print(f"Balance:   {acc_summary.get('balance')} {acc_summary.get('currency')}", flush=True)
                print(f"Equity:    {acc_summary.get('equity')}", flush=True)
                print(f"Available: {acc_summary.get('available')}", flush=True)
                print(f"P&L:       {acc_summary.get('profit_loss')}", flush=True)
                print(f"--------------------------------------------------\n", flush=True)
                self.logger.info(f"Account Update: Equity={acc_summary.get('equity')}, P&L={acc_summary.get('profit_loss')}")
            except Exception as e:
                self.logger.error(f"Failed to fetch account status: {e}")

        self.logger.info(f"Polling IG (stream update)...")

        # 1. Incremental fetch: Just get the last few minutes to ensure we bridge the gap
        # or capture the most recent closed candle.
        try:
            prices = fetch_prices(self.service, Price.Gold, start_time=now - pd.Timedelta(minutes=5), end_time=now)
            if not prices:
                self.logger.warning("No price data returned from IG.")
                return

            new_df = pd.DataFrame(prices)
            new_df.index = pd.to_datetime(new_df['timestamp'], unit='ms', utc=True)
            new_df = new_df.rename(columns={'openPrice': 'open', 'highPrice': 'high', 'lowPrice': 'low', 'closePrice': 'close', 'lastTradedVolume': 'volume'}).sort_index()

            # Merge with cache and keep unique timestamps (preferring new data)
            self.cached_df = pd.concat([self.cached_df, new_df])
            self.cached_df = self.cached_df[~self.cached_df.index.duplicated(keep='last')].sort_index()

            # Keep last N days in memory to match backtest warm-up behavior.
            max_ts = self.cached_df.index.max()
            cutoff = max_ts - pd.Timedelta(days=self.feature_warmup_days)
            self.cached_df = self.cached_df[self.cached_df.index >= cutoff]

            if store_to_db:
                # Store only the newest prices from this specific poll in MySQL
                from ig_scripts.ig_data_api import insert_prices
                insert_prices(prices, Price.Gold)
                # store-only cycle: do NOT run prediction or trading on this tick
                return

        except Exception as e:
            self.logger.error(f"Fetch failed: {e}")
            return

        df = self.cached_df.copy()

        # Drop the in-progress (not-yet-closed) current-minute bar.
        # Backtest only ever sees finalized OHLCV bars, so live must do the same:
        # at wall-clock 12:24:05 we want bar [12:23] (just closed at 12:24:00),
        # NOT bar [12:24] which is only 5 seconds into its minute.
        current_minute_floor = pd.Timestamp(now).floor('1min')
        df = df[df.index < current_minute_floor]
        if df.empty:
            self.logger.warning("No finalized bar available yet — waiting for next minute.")
            return
        raw_latest_ts = df.index[-1]   # raw latest FINALIZED bar (before feature dropna)

        # If we just opened a position last poll, try to align SL/TP to backtest
        # (uses the OPEN of the bar AFTER the signal bar). Safe to call every poll —
        # no-ops when nothing pending.
        if not store_to_db:
            self._resync_levels_to_backtest(df)

        if len(df) < self.min_prediction_bars:
            self.logger.warning(
                f"Cache size too small to run predictions: length={len(df)}, "
                f"need at least {self.min_prediction_bars} bars."
            )
            return

        # Minute-level dedupe: skip if we've already scored this bar in a
        # previous poll within the same minute (e.g. xx:05 already ran).
        if self.last_predicted_bar_ts is not None and raw_latest_ts == self.last_predicted_bar_ts:
            self.logger.info(
                f"[{raw_latest_ts}] Bar already scored — skipping duplicate prediction."
            )
            return
        # Mark this bar as scored regardless of whether the filter/model produces
        # a signal, so it won't be re-evaluated by any later poll in the same min.
        self.last_predicted_bar_ts = raw_latest_ts

        # 2. Bar-quality filter — model was trained ONLY on energetic bars.
        # Skip prediction on quiet bars to avoid out-of-distribution false signals.
        latest_bar = df.iloc[-1]
        bar_move = abs(latest_bar['close'] - latest_bar['open'])
        bar_vol   = latest_bar['volume']
        if bar_move <= FILTER_CONFIG["min_bar_move"] or bar_vol <= FILTER_CONFIG["min_volume"]:
            self.logger.info(
                f"[{raw_latest_ts}] Bar filtered (O={latest_bar['open']:.2f} C={latest_bar['close']:.2f} move={bar_move:.2f}, vol={bar_vol}) — "
                f"thresholds move>{FILTER_CONFIG['min_bar_move']}, vol>{FILTER_CONFIG['min_volume']}"
            )
            return

        # 3a. Keep raw 1-min frame for image-model window (must mirror backtest:
        # add_image_model_predictions in train_filter_v13_wf_image.py iterates over
        # the PRE-FILTER 1-min df with df.iloc[i-150+1:i+1]). Using the post-filter
        # (energetic-only) df here was a bug: the image model was trained on 150
        # CONSECUTIVE 1-min bars, not 150 non-consecutive energetic bars, which
        # produced an out-of-distribution image_s1_prob and divergent S1 scores
        # vs backtest.
        raw_df = df

        # 3b. Features — for_live_inference=True ensures the latest bar is NOT dropped
        df = prepare_v13_features(df)

        # latest_ts is now read AFTER prepare_v13_features so it is the real latest
        # bar with fully valid features (warm-up NaNs have been trimmed from the start,
        # but the most recent bar is preserved).
        latest_ts = df.index[-1]
        self.last_ts = latest_ts

        if latest_ts != raw_latest_ts:
            self.logger.warning(
                f"Feature df ends at {latest_ts}, raw cache ends at {raw_latest_ts}. "
                f"Gap = {int((raw_latest_ts - latest_ts).total_seconds() // 60)} min — "
                f"scoring most recent available bar."
            )

        # Image Prob — computed from the RAW 1-min df at latest_ts (parity with backtest).
        if latest_ts not in raw_df.index:
            self.logger.error(
                f"latest_ts {latest_ts} missing from raw 1-min df — cannot compute image_s1_prob."
            )
            return
        raw_idx = raw_df.index.get_loc(latest_ts)
        if raw_idx < self.img_win - 1:
            self.logger.warning(
                f"Raw 1-min cache too small for image window ({raw_idx + 1} < {self.img_win}) — skipping bar."
            )
            return
        img_vec = extract_image_payload(raw_df, raw_idx, self.img_win)
        ts = latest_ts
        asia_f, asia_p = _session_info(ts, HK_TZ, 8, 0, 16, 0)
        lon_f, lon_p = _session_info(ts, LONDON_TZ, 8, 0, 16, 30)
        ny_f, ny_p = _session_info(ts, NY_TZ, 9, 30, 16, 0)
        extra = [df.loc[latest_ts, "Dchange_utc2_rel"], df.loc[latest_ts, "Dupper_wick_utc2_rel"], df.loc[latest_ts, "Dlower_wick_utc2_rel"],
                 asia_f, asia_p, lon_f, lon_p, ny_f, ny_p]
        df.loc[latest_ts, "image_s1_prob"] = self.img_s1_model.predict_proba(np.concatenate([img_vec, extra]).reshape(1, -1))[0][1]

        # 4. Model Logic
        s1_p = self.s1_model.predict_proba(df.loc[[latest_ts], self.v13_s1_cols])[0][1]

        # Dynamic S2: raise entry bar after consecutive losses, capped at s2_max_threshold.
        # Uses base threshold so position-management signals (roll/reverse) stay at 0.55 level.
        s2_base      = EXECUTION_CONFIG["s2_threshold"]
        s2_increment = EXECUTION_CONFIG["s2_loss_increment"]
        s2_max       = EXECUTION_CONFIG["s2_max_threshold"]
        dynamic_s2   = min(s2_max, s2_base + self.state.consecutive_losses * s2_increment)

        side = 0
        s2_p = None
        if s1_p >= EXECUTION_CONFIG["s1_threshold"]:
            s2_p = self.s2_model.predict_proba(df.loc[[latest_ts], self.v13_s2_cols])[0][1]
            if s2_p >= dynamic_s2:
                side = 1
            elif s2_p <= (1.0 - dynamic_s2):
                side = -1

        self.logger.info(f"[{latest_ts}] O={latest_bar['open']:.2f} H={latest_bar['high']:.2f} L={latest_bar['low']:.2f} C={latest_bar['close']:.2f} V={latest_bar['volume']} | S1={s1_p:.4f} S2={s2_p} Thresh={dynamic_s2:.3f} (base={s2_base}, losses={self.state.consecutive_losses}) Signal={side}")

        # Trigger Execution if signal exists and we don't have an open position
        if side != 0 and store_to_db == False:
            if self.state.open_deal_id:
                self.logger.info(f"Signal {side} ignored because position {self.state.open_deal_id} is already open.")
                return

            try:
                self.logger.info(f"Triggering execution for side {side}...")
                exec_res = self.engine.handle_signal(
                    mode="live",
                    signal_model_family="v13_hybrid",
                    signal={
                        "side": "buy" if side == 1 else "sell",
                        "probability": s2_p if side == 1 else (1.0 - s2_p),
                        "tradable": True
                    },
                    entry_time=latest_ts,
                    entry_price=df["close"].iloc[-1],
                    size=EXECUTION_CONFIG.get("size", 1.0)
                )
                
                if exec_res.get("submitted"):
                    deal_id = exec_res.get("deal_id")
                    self.logger.info(f"Order successfully submitted. Deal ID: {deal_id}")
                    self.state.open_deal_id = deal_id
                    # Save entry time so the horizon timeout (EXECUTION_CONFIG['horizon'])
                    # can force-close the position if neither TP nor SL hit in time.
                    self.state.open_entry_time = pd.Timestamp(latest_ts).tz_convert('UTC').isoformat()
                    # Schedule SL/TP resync to the bar AFTER the signal bar (mirrors
                    # backtest's next_row entry-price convention). _resync_levels_to_backtest
                    # will fire on the next poll once that bar is finalized.
                    next_bar_ts = pd.Timestamp(latest_ts) + pd.Timedelta(minutes=1)
                    if next_bar_ts.tzinfo is None:
                        next_bar_ts = next_bar_ts.tz_localize('UTC')
                    else:
                        next_bar_ts = next_bar_ts.tz_convert('UTC')
                    self.state.pending_level_resync_bar = next_bar_ts.isoformat()
                    self._save_state()
                else:
                    self.logger.error(f"Order submission failed: {exec_res.get('reason')}")
                    
            except Exception as e:
                self.logger.error(f"Execution failed: {e}")

    def run(self):
        self.logger.info("Bot execution loop started.")
        while True:
            try:
                now = datetime.now(timezone.utc)
                sec = now.second

                # Check for Weekend Retraining (Every two weeks on Saturday morning)
                # target: Saturday (weekday 5) at 01:00 UTC
                if now.weekday() == 5 and now.hour == 1:
                    should_retrain = False
                    if self.state.last_retrain_date is None:
                        should_retrain = True
                    else:
                        last_date = datetime.fromisoformat(self.state.last_retrain_date).date()
                        days_since = (now.date() - last_date).days
                        if days_since >= 14:
                            should_retrain = True

                    if should_retrain:
                        self.logger.info("Starting scheduled bi-weekly weekend retraining...")
                        try:
                            import subprocess
                            # Persist full stdout/stderr of both training stages so we can
                            # inspect metrics (accuracy, classification reports) afterward.
                            retrain_log_dir = PROJECT_ROOT / "runtime" / "retrain_logs"
                            retrain_log_dir.mkdir(parents=True, exist_ok=True)
                            day_tag = now.date().isoformat()

                            # 1. Retrain Stage 1
                            s1_script = PROJECT_ROOT / "V13" / "xgboost" / "train_filter_v13_wf_image.py"
                            self.logger.info("Running Stage 1 retraining...")
                            res1 = subprocess.run([sys.executable, str(s1_script)], capture_output=True, text=True)
                            (retrain_log_dir / f"stage1_{day_tag}.log").write_text(
                                f"=== STDOUT ===\n{res1.stdout}\n=== STDERR ===\n{res1.stderr}\n",
                                encoding="utf-8",
                            )

                            if res1.returncode == 0:
                                # 2. Retrain Stage 2
                                s2_script = PROJECT_ROOT / "V13" / "xgboost" / "train_stage2_v13_directional.py"
                                self.logger.info("Running Stage 2 retraining...")
                                res2 = subprocess.run([sys.executable, str(s2_script)], capture_output=True, text=True)
                                (retrain_log_dir / f"stage2_{day_tag}.log").write_text(
                                    f"=== STDOUT ===\n{res2.stdout}\n=== STDERR ===\n{res2.stderr}\n",
                                    encoding="utf-8",
                                )

                                if res2.returncode == 0:
                                    self.logger.info("All retraining successful. Hot-reloading models...")
                                    
                                    # Copy newly trained models to runtime/bot_assets (if scripts don't do it)
                                    import shutil
                                    assets_dir = PROJECT_ROOT / "V13" / "runtime" / "bot_assets"
                                    shutil.copy(PROJECT_ROOT / "V13" / "xgboost" / "filter_model_v13_wf_image.joblib", assets_dir / "filter_model_v13_wf_image.joblib")
                                    shutil.copy(PROJECT_ROOT / "V13" / "xgboost" / "directional_model_v13_wf.joblib", assets_dir / "directional_model_v13_wf.joblib")

                                    # Reload models
                                    self.s1_model = joblib.load(assets_dir / "filter_model_v13_wf_image.joblib")
                                    self.s2_model = joblib.load(assets_dir / "directional_model_v13_wf.joblib")
                                    
                                    self.state.last_retrain_date = now.date().isoformat()
                                    self._save_state()
                                    self.logger.info(f"Models reloaded. Next retraining scheduled for {now.date() + pd.Timedelta(days=14)}")
                                else:
                                    self.logger.error(f"Stage 2 retraining failed: {res2.stderr}")
                            else:
                                self.logger.error(f"Stage 1 retraining failed: {res1.stderr}")
                        except Exception as re:
                            self.logger.error(f"Error during automatic retraining: {re}")

                # 1. On Every 5th second of each minute: Predict & Trade only
                #    (uses the most recently closed minute bar from IG; does NOT write to MySQL)
                if sec == 5:
                    self.poll(store_to_db=False)
                    # We sleep slightly more than 1s to avoid double-triggering within the same second
                    time.sleep(1.2)
                
                # 2. On Every 30th second of each minute: Fetch & Store to MySQL only
                #    (no prediction, no trading — keeps MySQL fresh + shows account status)
                elif sec == 30:
                    self.poll(store_to_db=True)
                    time.sleep(1.2)
                
                else:
                    # Check every 0.5s to catch the precise second triggers
                    time.sleep(0.5)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    AlphaGoldV13Bot().run()

