"""
Gold Trading Bot - Optimized Data Management
✅ Fetch bulk data once at 5:05 PM NY
✅ Update incrementally during trading
✅ Dynamic position sizing: floor(Capital/80) × 0.1
✅ Max 10 contracts absolute cap
✅ Shows position status and running P&L

RUN THIS FILE IN PYCHARM!
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.blocking import BlockingScheduler
import pickle
import os

from trade_logger import TradeLogger
from config import STRATEGY_CONFIG, IG_CONFIG, DB_CONFIG
from ig_api import (
    get_accounts,
    get_positions,
    place_order,
    delete_position,
    fetch_prices_without_insert,
    Price,
    get_activity_history,
    get_closed_position_by_deal_id  # ✅ NEW: Get closed position details
)
from daily_backtest import DailyBacktest


class DataManager:
    """Manages price data efficiently"""

    def __init__(self, logger, cache_file=None):
        self.logger = logger
        self.cache_file = cache_file or DB_CONFIG['CACHE_FILE']
        self.df = None
        self.last_update = None
        self.indicators_calculated = False

    def _fetch_from_ig_api(self, days_back=7):
        """
        Fetch data from IG API with proper date range
        Returns list of price data or None
        """
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days_back)

            self.logger.info(f"   Fetching from IG: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            prices_data = fetch_prices_without_insert(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                instrument=Price.Gold
            )

            if prices_data and len(prices_data) > 0:
                self.logger.info(f"   ✅ Received {len(prices_data)} bars from IG")
                return prices_data
            else:
                self.logger.warning(f"   ⚠️ No data for {days_back} days range")
                return None

        except Exception as e:
            self.logger.error(f"   ❌ IG API error: {e}")
            return None

    def load_or_fetch_bulk_data(self, force_refresh=False):
        """Load from cache or fetch bulk data"""
        try:
            # 1. Try loading from cache first
            if not force_refresh and os.path.exists(self.cache_file):
                self.logger.info(f"📂 Loading cached data...")
                try:
                    with open(self.cache_file, 'rb') as f:
                        cache = pickle.load(f)
                        self.df = cache['df']
                        self.last_update = cache['last_update']
                        self.indicators_calculated = cache.get('indicators_calculated', False)

                    # Ensure dt column is timezone-aware
                    if self.df['dt'].dt.tz is None:
                        self.df['dt'] = self.df['dt'].dt.tz_localize('UTC')

                    # Ensure last_update is timezone-aware
                    if hasattr(self.last_update, 'tzinfo'):
                        if self.last_update.tzinfo is None:
                            self.last_update = pytz.UTC.localize(self.last_update)
                    else:
                        self.last_update = pd.Timestamp(self.last_update).to_pydatetime()
                        if self.last_update.tzinfo is None:
                            self.last_update = pytz.UTC.localize(self.last_update)

                    hours_old = (datetime.now(pytz.UTC) - self.last_update).total_seconds() / 3600

                    if hours_old < 72:  # Cache less than 3 days old
                        self.logger.info(f"✅ Loaded {len(self.df)} bars from cache ({hours_old:.1f}h old)")
                        return True
                    else:
                        self.logger.info(f"⚠️ Cache is {hours_old:.1f} hours old, refreshing...")
                except Exception as cache_error:
                    self.logger.warning(f"⚠️ Cache load failed: {cache_error}")

            # 2. Fetch from IG API
            self.logger.info("📡 Fetching bulk data from IG...")

            prices_data = None

            # Try different date ranges until we get data
            for days in [7, 10, 14]:
                self.logger.info(f"   Trying {days} days back...")
                prices_data = self._fetch_from_ig_api(days_back=days)
                if prices_data and len(prices_data) > 0:
                    break

            # 3. If IG fails, try CSV fallback
            if not prices_data or len(prices_data) == 0:
                self.logger.warning("⚠️ IG API returned no data, trying CSV fallback...")
                csv_path = 'ig_scripts/gold_prices.csv'
                if os.path.exists(csv_path):
                    self.logger.info(f"📂 Loading from CSV: {csv_path}")
                    self.df = pd.read_csv(csv_path)

                    if 'timestamp' in self.df.columns:
                        self.df['dt'] = pd.to_datetime(self.df['timestamp'], unit='ms', utc=True)
                    elif 'dt' in self.df.columns:
                        self.df['dt'] = pd.to_datetime(self.df['dt'], utc=True)

                    self.df = self.df.sort_values('dt').tail(3500).reset_index(drop=True)
                    self.last_update = pd.Timestamp(self.df['dt'].iloc[-1]).to_pydatetime()
                    self.indicators_calculated = False

                    self.logger.info(f"✅ Loaded {len(self.df)} bars from CSV")
                    self.logger.info(f"   Range: {self.df['dt'].min()} to {self.df['dt'].max()}")
                    self._save_cache()
                    return True
                else:
                    self.logger.error("❌ No data source available")
                    return False

            # 4. Process IG data
            self.df = pd.DataFrame(prices_data)

            required_cols = ['timestamp', 'openPrice', 'highPrice', 'lowPrice', 'closePrice']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                self.logger.error(f"❌ Missing columns: {missing_cols}")
                return False

            # ✅ Force UTC timezone
            self.df['dt'] = pd.to_datetime(self.df['timestamp'], unit='ms', utc=True)
            self.df = self.df.sort_values('dt').reset_index(drop=True)

            if len(self.df) > 3500:
                self.df = self.df.tail(3500).reset_index(drop=True)

            self.last_update = pd.Timestamp(self.df['dt'].iloc[-1]).to_pydatetime()
            self.indicators_calculated = False

            self.logger.info(f"✅ Processed {len(self.df)} bars")
            self.logger.info(f"   Range: {self.df['dt'].min()} to {self.df['dt'].max()}")

            self._save_cache()
            return True

        except Exception as e:
            self.logger.error(f"❌ Error loading data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def update_with_latest_bar(self):
        """Fetch only latest bars from IG"""
        try:
            # Ensure last_update is timezone-aware datetime
            if hasattr(self.last_update, 'tzinfo'):
                if self.last_update.tzinfo is None:
                    last_update_utc = pytz.UTC.localize(self.last_update)
                else:
                    last_update_utc = self.last_update
            else:
                last_update_utc = pd.Timestamp(self.last_update).to_pydatetime()
                if last_update_utc.tzinfo is None:
                    last_update_utc = pytz.UTC.localize(last_update_utc)

            hours_since_update = (datetime.now(pytz.UTC) - last_update_utc).total_seconds() / 3600

            # Determine how many days to fetch
            if hours_since_update > 48:
                days_back = min(7, int(hours_since_update / 24) + 1)
                self.logger.info(f"📡 Data is {hours_since_update:.1f}h old, fetching {days_back} days")
            else:
                days_back = 2

            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days_back)

            prices_data = fetch_prices_without_insert(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                instrument=Price.Gold
            )

            if not prices_data or len(prices_data) == 0:
                self.logger.info("⏸️ No new data available (market closed)")
                return False

            new_df = pd.DataFrame(prices_data)

            if 'timestamp' not in new_df.columns:
                return False

            # ✅ Force UTC timezone
            new_df['dt'] = pd.to_datetime(new_df['timestamp'], unit='ms', utc=True)
            new_df = new_df.sort_values('dt')

            # ✅ Ensure self.df['dt'] is also tz-aware
            if self.df['dt'].dt.tz is None:
                self.df['dt'] = self.df['dt'].dt.tz_localize('UTC')

            # Get only bars newer than our last bar
            last_bar_time = self.df['dt'].iloc[-1]
            new_bars = new_df[new_df['dt'] > last_bar_time]

            if len(new_bars) == 0:
                self.logger.info("⏸️ No new bars since last update")
                return False

            # Append new bars
            self.df = pd.concat([self.df, new_bars], ignore_index=True)

            # Remove duplicates
            self.df = self.df.drop_duplicates(subset=['timestamp'], keep='last')
            self.df = self.df.sort_values('dt').reset_index(drop=True)

            old_last = self.last_update
            self.last_update = pd.Timestamp(self.df['dt'].iloc[-1]).to_pydatetime()
            self.indicators_calculated = False

            if len(self.df) > 3500:
                self.df = self.df.tail(3500).reset_index(drop=True)

            self.logger.info(f"✅ Added {len(new_bars)} new bar(s)")
            self.logger.info(f"   Old: {old_last} → New: {self.last_update}")
            self.logger.info(f"   Latest price: ${new_bars.iloc[-1]['closePrice']:.2f}")
            self._save_cache()
            return True

        except Exception as e:
            self.logger.error(f"❌ Error updating bar: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def calculate_indicators(self):
        """Calculate WR(90), daily range, etc."""
        try:
            self.logger.info("📊 Calculating indicators...")

            utc_tz = pytz.UTC
            ny_tz = pytz.timezone('America/New_York')

            # Ensure timezone awareness
            if self.df['dt'].dt.tz is None:
                self.df['dt_utc'] = self.df['dt'].dt.tz_localize(utc_tz)
            else:
                self.df['dt_utc'] = self.df['dt']

            self.df['dt_ny'] = self.df['dt_utc'].dt.tz_convert(ny_tz)
            self.df['ny_hour'] = self.df['dt_ny'].dt.hour
            self.df['ny_date'] = self.df['dt_ny'].dt.date

            # Daily range (full day)
            daily_stats = self.df.groupby('ny_date').agg({
                'highPrice': 'max',
                'lowPrice': 'min'
            }).reset_index()
            daily_stats['daily_range'] = daily_stats['highPrice'] - daily_stats['lowPrice']

            daily_range_map = dict(zip(daily_stats['ny_date'], daily_stats['daily_range']))
            self.df['daily_range'] = self.df['ny_date'].map(daily_range_map)

            # ✅ Intraday range (no lookahead) for live regime selection
            self.df['intraday_high'] = self.df.groupby('ny_date')['highPrice'].cummax()
            self.df['intraday_low'] = self.df.groupby('ny_date')['lowPrice'].cummin()
            self.df['daily_range_rt'] = self.df['intraday_high'] - self.df['intraday_low']

            # Previous day range
            unique_dates = sorted(self.df['ny_date'].unique())
            prev_day_range_map = {}
            for i in range(1, len(unique_dates)):
                prev_day_range_map[unique_dates[i]] = daily_range_map.get(unique_dates[i-1], np.nan)

            self.df['prev_day_range'] = self.df['ny_date'].map(prev_day_range_map)

            if self.df['prev_day_range'].isna().any():
                avg_range = self.df['daily_range'].mean()
                self.df['prev_day_range'] = self.df['prev_day_range'].fillna(avg_range)

            # WR(90)
            period = 90
            if len(self.df) >= period:
                highest_high = self.df['highPrice'].rolling(period).max()
                lowest_low = self.df['lowPrice'].rolling(period).min()
                range_90 = highest_high - lowest_low

                self.df['wr_90'] = -100 * (highest_high - self.df['closePrice']) / range_90
                self.df['wr_90'] = self.df['wr_90'].replace([np.inf, -np.inf], np.nan)
                self.df['prev_wr_90'] = self.df['wr_90'].shift(1)
                self.df['high_90'] = highest_high
                self.df['low_90'] = lowest_low
                self.df['range_90'] = range_90
            else:
                self.logger.warning(f"⚠️ Only {len(self.df)} bars, need {period} for WR(90)")

            self.indicators_calculated = True
            self.logger.info("✅ Indicators ready")

        except Exception as e:
            self.logger.error(f"❌ Indicator error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _save_cache(self):
        """Save to pickle file"""
        try:
            cache = {
                'df': self.df,
                'last_update': self.last_update,
                'indicators_calculated': self.indicators_calculated
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache, f)
            self.logger.info(f"💾 Cache saved: {len(self.df)} bars")
        except Exception as e:
            self.logger.error(f"❌ Cache save error: {e}")

    def get_latest_bar(self):
        """Get most recent bar"""
        if self.df is None or len(self.df) == 0:
            return None
        return self.df.iloc[-1]

    def should_refresh_bulk_data(self):
        """Check if need bulk refresh - only at scheduled time"""
        now_ny = datetime.now(pytz.timezone('America/New_York'))

        # ONLY refresh at 5:05 PM NY (weekdays)
        if (now_ny.hour == STRATEGY_CONFIG['REFRESH_HOUR_NY'] and
            STRATEGY_CONFIG['REFRESH_MINUTE_NY'] <= now_ny.minute < STRATEGY_CONFIG['REFRESH_MINUTE_NY'] + 5):
            if now_ny.weekday() < 5:
                self.logger.info("⏰ Scheduled refresh time (5:05 PM NY)")
                return True

        # Emergency refresh if cache >3 days old
        if self.last_update:
            if hasattr(self.last_update, 'tzinfo'):
                if self.last_update.tzinfo is None:
                    last_update_utc = pytz.UTC.localize(self.last_update)
                else:
                    last_update_utc = self.last_update
            else:
                last_update_utc = pd.Timestamp(self.last_update).to_pydatetime()
                if last_update_utc.tzinfo is None:
                    last_update_utc = pytz.UTC.localize(last_update_utc)

            hours_old = (datetime.now(pytz.UTC) - last_update_utc).total_seconds() / 3600
            if hours_old > 72:
                self.logger.info(f"⚠️ Emergency refresh: cache is {hours_old:.1f}h old (>3 days)")
                return True

        return False


class GoldTrader:
    """Main trading bot"""

    def __init__(self):
        self.logger = TradeLogger(
            log_dir=DB_CONFIG['LOG_DIR'],
            db_path=DB_CONFIG['DB_PATH']
        )
        # ✅ ADD THIS LINE:
        self.ny_tz = pytz.timezone('America/New_York')

        self.data_manager = DataManager(self.logger)
        self.daily_backtest = DailyBacktest(self.logger)
        self.logger.info("="*80)
        self.logger.info("🚀 GOLD TRADING BOT - OPTIMIZED")
        self.logger.info("="*80)

        if not self.data_manager.load_or_fetch_bulk_data():
            raise Exception("Failed to load data")

        self.data_manager.calculate_indicators()

        self.peak_balance = None
        self.current_position_size = None
        self._initialize_account()

        # ✅ ADD THESE LINES:
        self._last_position_count = 0
        self._tracked_deal_ids = set()
        self._open_positions = {}  # Track entry info: {deal_id: {'timestamp': entry_timestamp, 'price': entry_price, 'direction': direction}}


        self.logger.info(f"Balance: ${self.peak_balance:.2f}")
        self.logger.info(f"Position size: {self.current_position_size:.2f} contracts")
        self.logger.info("="*80)

    def check_for_closed_positions(self):
        """
        Check IG for positions that were closed by stop/target and log them
        This catches trades that were closed at broker level
        """
        try:
            from datetime import timedelta

            # Get current positions
            current_positions = get_positions()
            gold_positions = [p for p in current_positions if p['market']['epic'] == IG_CONFIG['GOLD_EPIC']]

            # Check if a position was closed since last cycle
            if self._last_position_count > 0 and len(gold_positions) == 0:
                self.logger.warning("⚠️ Position was closed externally (stop/target hit at broker)")

                # Get recent activity to find the closing deal
                try:
                    now_ny = datetime.now(self.ny_tz)
                    from_time = now_ny - timedelta(hours=2)  # ✅ Extended to 2 hours to ensure we catch the entry

                    self.logger.info(f"🔍 Fetching activity history from {from_time.strftime('%Y-%m-%d %H:%M:%S')}...")
                    activities = get_activity_history(from_date=from_time)

                    if not activities:
                        self.logger.warning("⚠️ No activity history returned")
                        return

                    self.logger.info(f"✅ Got {len(activities)} activities, searching for closed gold position...")

                    # Find most recent CLOSE activity for gold that we haven't logged yet
                    closed_activity_found = False
                    for activity in activities:
                        # Check if this is a position close
                        activity_type = activity.get('type', '')
                        if 'POSITION' not in activity_type.upper():
                            continue

                        deal_id = activity.get('dealId')

                        self.logger.info(f"   📋 Found POSITION activity - Deal ID: {deal_id}")

                        # Skip if we've already logged this deal
                        if deal_id in self._tracked_deal_ids:
                            self.logger.info(f"   ⏭️ Already logged deal {deal_id}, skipping")
                            continue

                        # Check if it's for gold
                        epic = activity.get('epic')
                        if epic != IG_CONFIG['GOLD_EPIC']:
                            self.logger.info(f"   ⏭️ Not gold (epic: {epic}), skipping")
                            continue

                        self.logger.info(f"   ✅ Gold position found!")

                        # Check if it's a close
                        status = activity.get('status', '')
                        action_status = activity.get('actionStatus', '')
                        self.logger.info(f"   📌 Status: {status}, ActionStatus: {action_status}")

                        if 'CLOSED' not in status.upper() and 'CLOSED' not in action_status.upper():
                            self.logger.info(f"   ⏭️ Not a closed position, skipping")
                            continue

                        self.logger.info(f"   ✅ CLOSED position confirmed!")
                        closed_activity_found = True

                        # Extract trade details
                        direction = activity.get('direction', '')  # BUY or SELL
                        size = float(activity.get('size', 0.5))

                        # Get prices
                        level = float(activity.get('level', 0))  # Close price

                        # ✅ Initialize timestamps and entry info
                        entry_timestamp = None
                        exit_timestamp = None
                        tracked_entry_price = None

                        if deal_id in self._open_positions:
                            pos_info = self._open_positions[deal_id]
                            if isinstance(pos_info, dict):
                                entry_timestamp = pos_info.get('timestamp')
                                tracked_entry_price = pos_info.get('price')
                            else:
                                entry_timestamp = pos_info

                        # ✅ Try to get entry price from activity details
                        details = activity.get('details', {})
                        opening_level = details.get('openLevel')
                        if opening_level:
                            try:
                                opening_level = float(opening_level)
                            except (ValueError, TypeError):
                                opening_level = None

                        if not opening_level and tracked_entry_price:
                            opening_level = tracked_entry_price
                            self.logger.info(f"📌 Using tracked entry price: ${opening_level:.2f}")

                        # ✅ Prefer transaction data for timestamps/prices when available
                        tx_info = get_closed_position_by_deal_id(deal_id)
                        if tx_info:
                            if tx_info.get('entry_price'):
                                opening_level = tx_info['entry_price']
                            if tx_info.get('exit_price'):
                                level = tx_info['exit_price']
                            if tx_info.get('open_timestamp'):
                                entry_timestamp = tx_info['open_timestamp']
                            if tx_info.get('close_timestamp'):
                                exit_timestamp = tx_info['close_timestamp']

                        # ✅ IMPROVED: Better fallback logic for entry price
                        if not opening_level:
                            # Try to find matching open position activity
                            found_entry = False
                            for other_activity in activities:
                                other_deal_id = other_activity.get('dealId')
                                if other_deal_id != deal_id:
                                    continue

                                other_type = other_activity.get('type', '')
                                if 'POSITION' not in other_type.upper():
                                    continue

                                other_status = other_activity.get('status', '')
                                # Look for OPEN status (not CLOSED)
                                if 'CLOSED' in other_status.upper() or 'OPEN' not in other_status.upper():
                                    continue

                                other_level = other_activity.get('level')
                                if other_level:
                                    try:
                                        opening_level = float(other_level)
                                        found_entry = True
                                        break
                                    except (ValueError, TypeError):
                                        pass

                            # Last resort: check tracked positions or current price
                            if not opening_level:
                                if deal_id in self._open_positions:
                                    self.logger.warning(f"⚠️ Entry price not in activity, using approximation")
                                    # Estimate based on direction and current position
                                    opening_level = level * 1.001 if direction == 'BUY' else level * 0.999
                                else:
                                    self.logger.error(f"❌ Cannot determine entry price for deal {deal_id}")
                                    # Skip this trade instead of guessing
                                    continue

                        # Calculate P&L
                        if direction == 'BUY':
                            gross_pnl = (level - opening_level) * size
                        else:
                            gross_pnl = (opening_level - level) * size

                        net_pnl = gross_pnl - IG_CONFIG['TRADE_COST']

                        # Determine exit reason
                        if 'STOP' in status.upper() or 'STOP' in action_status.upper():
                            exit_reason = 'stop'
                        elif 'LIMIT' in status.upper() or 'LIMIT' in action_status.upper():
                            exit_reason = 'target'
                        else:
                            exit_reason = 'manual'

                        # Get regime
                        bar = self.data_manager.get_latest_bar()
                        if bar is not None:
                            daily_range = bar.get('daily_range', 50)
                            regime, _ = self._get_regime(daily_range)
                        else:
                            regime = 'UNKNOWN'

                        # Calculate stop distance
                        stop_distance = abs(opening_level - level) if exit_reason == 'stop' else 0

                        # Continue with entry timestamp search if we don't have it already
                        if not entry_timestamp:
                            # Estimate from activity date using dateutil parser
                            activity_date = activity.get('date')
                            if activity_date:
                                try:
                                    from dateutil import parser
                                    close_dt = parser.parse(activity_date)

                                    # ✅ IMPROVED: Search for matching open activity for accurate entry time
                                    for other_activity in activities:
                                        if other_activity.get('dealId') != deal_id:
                                            continue
                                        other_status = other_activity.get('status', '')
                                        if 'OPEN' in other_status.upper() and 'CLOSED' not in other_status.upper():
                                            entry_date_str = other_activity.get('date')
                                            if entry_date_str:
                                                try:
                                                    entry_dt = parser.parse(entry_date_str)
                                                    entry_timestamp = int(entry_dt.timestamp() * 1000)
                                                    self.logger.info(f"📌 Found entry time from activity: {entry_dt.strftime('%H:%M:%S')}")
                                                    break
                                                except:
                                                    pass

                                    # Fallback if no open activity found
                                    if not entry_timestamp:
                                        self.logger.warning(f"⚠️ Estimating entry time (30 min before close)")
                                        entry_dt = close_dt - timedelta(minutes=30)
                                        entry_timestamp = int(entry_dt.timestamp() * 1000)
                                except Exception as parse_err:
                                    self.logger.error(f"⚠️ Error parsing date: {parse_err}")
                                    entry_timestamp = int((datetime.now(self.ny_tz).timestamp() - 1800) * 1000)
                            else:
                                entry_timestamp = int((datetime.now(self.ny_tz).timestamp() - 1800) * 1000)


                        # ✅ Parse exit timestamp from activity (fallback)
                        if not exit_timestamp:
                            activity_date = activity.get('date', '')
                            if activity_date:
                                try:
                                    if 'T' in activity_date:
                                        exit_dt = datetime.strptime(activity_date.split('.')[0], '%Y-%m-%dT%H:%M:%S')
                                        exit_dt = pytz.UTC.localize(exit_dt)
                                        exit_timestamp = int(exit_dt.timestamp() * 1000)
                                        self.logger.info(f"📌 Parsed exit timestamp: {exit_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                                except Exception as parse_err:
                                    self.logger.warning(f"⚠️ Could not parse exit date '{activity_date}': {parse_err}")

                        if not exit_timestamp:
                            exit_timestamp = int(datetime.now(self.ny_tz).timestamp() * 1000)
                            self.logger.warning("⚠️ Using current time for exit timestamp (parsing failed)")

                        # Log the trade
                        self.logger.info(f"")
                        self.logger.info(f"🚪 DETECTED EXTERNALLY CLOSED POSITION:")
                        self.logger.info(f"   Direction: {direction}")
                        self.logger.info(f"   Entry: ${opening_level:.2f}")
                        self.logger.info(f"   Exit: ${level:.2f}")
                        self.logger.info(f"   Size: {size}")
                        self.logger.info(f"   P&L: ${net_pnl:.2f}")
                        self.logger.info(f"   Reason: {exit_reason}")
                        self.logger.info(f"   Deal ID: {deal_id}")

                        # ✅ Log to database with actual timestamps from IG
                        self.logger.log_trade(
                            entry_timestamp=entry_timestamp,
                            exit_timestamp=exit_timestamp,
                            direction=direction,
                            entry_price=opening_level,
                            exit_price=level,
                            position_size=size,
                            pnl=net_pnl,
                            exit_reason=exit_reason,
                            regime=regime,
                            stop_distance=stop_distance,
                            bars_held=0  # Unknown for externally closed positions
                        )

                        # Mark this deal as logged
                        self._tracked_deal_ids.add(deal_id)

                        # Remove from open positions tracking
                        if deal_id in self._open_positions:
                            del self._open_positions[deal_id]

                        # Only log the first closed position found
                        break

                    if not closed_activity_found:
                        self.logger.warning(f"⚠️ No closed gold position found in {len(activities)} activities")
                        self.logger.info(f"   Possible reasons:")
                        self.logger.info(f"   - Position closed too long ago (>2 hours)")
                        self.logger.info(f"   - Activity not yet available in IG API")
                        self.logger.info(f"   - Already logged (deal in tracked set)")

                except Exception as e:
                    self.logger.error(f"❌ Error fetching activity history: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())

            # Update position count for next cycle
            self._last_position_count = len(gold_positions)

        except Exception as e:
            self.logger.error(f"❌ Error checking closed positions: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _initialize_account(self):
        """Get account from IG"""
        try:
            accounts = get_accounts()
            if accounts:
                account = accounts[0]
                balance = account['balance']['balance']
                equity = balance + account['balance'].get('profitLoss', 0)
                self.peak_balance = equity
                self.current_position_size = self._calculate_position_size(equity)
                self.logger.info(f"✅ Account: ${balance:.2f}")
            else:
                raise Exception("No account data")
        except Exception as e:
            self.logger.error(f"❌ Account init failed: {e}")
            self.peak_balance = STRATEGY_CONFIG.get('STARTING_CAPITAL', 1000)
            self.current_position_size = self._calculate_position_size(self.peak_balance)

    def _calculate_position_size(self, capital):
        """Formula: floor(capital/80) × 0.1, max 10"""
        size = np.floor(capital / STRATEGY_CONFIG['CAPITAL_PER_CONTRACT']) * STRATEGY_CONFIG['CONTRACT_MULTIPLIER']
        size = max(STRATEGY_CONFIG['MIN_CONTRACTS'], size)
        size = min(size, STRATEGY_CONFIG['MAX_CONTRACTS'])
        return size

    def _get_regime(self, daily_range):
        """Get volatility regime"""
        if daily_range < STRATEGY_CONFIG['LOW_VOL_THRESHOLD']:
            return 'LOW', STRATEGY_CONFIG['LOW_VOL_HOURS']
        elif daily_range <= STRATEGY_CONFIG['HIGH_VOL_THRESHOLD']:
            return 'MEDIUM', STRATEGY_CONFIG['MED_VOL_HOURS']
        else:
            return 'HIGH', STRATEGY_CONFIG['HIGH_VOL_HOURS']

    def _get_next_trading_hour(self, current_hour, trading_hours):
        """Calculate next trading hour"""
        try:
            sorted_hours = sorted(trading_hours)

            # Find next hour after current
            for hour in sorted_hours:
                if hour > current_hour:
                    return hour

            # If no hour found after current, return first hour of next day
            if len(sorted_hours) > 0:
                return sorted_hours[0]

            return None
        except Exception as e:
            return None

    def _log_position_status(self, positions, current_price):
        """Log detailed position status"""
        if len(positions) == 0:
            return

        self.logger.info("="*60)
        self.logger.info("📍 OPEN POSITIONS:")

        for i, position in enumerate(positions, 1):
            try:
                pos = position['position']
                market = position['market']

                direction = pos['direction']
                size = pos['size']
                entry_price = pos['level']
                stop_level = pos.get('stopLevel', 'None')
                limit_level = pos.get('limitLevel', 'None')

                # Calculate running P&L
                if direction == 'BUY':
                    pnl = (current_price - entry_price) * size
                    pnl_emoji = "🟢" if pnl > 0 else "🔴"
                else:
                    pnl = (entry_price - current_price) * size
                    pnl_emoji = "🟢" if pnl > 0 else "🔴"

                # Calculate distances
                if stop_level != 'None':
                    stop_dist = abs(current_price - stop_level)
                else:
                    stop_dist = 0

                if limit_level != 'None':
                    limit_dist = abs(limit_level - current_price)
                else:
                    limit_dist = 0

                self.logger.info(f"  Position #{i}: {direction} {size} @ ${entry_price:.2f}")
                self.logger.info(f"    Current: ${current_price:.2f}")
                self.logger.info(f"    {pnl_emoji} P&L: ${pnl:.2f} ({(pnl/entry_price/size*100):.2f}%)")
                if stop_level != 'None':
                    self.logger.info(f"    Stop: ${stop_level:.2f} (${stop_dist:.2f} away)")
                if limit_level != 'None':
                    self.logger.info(f"    Target: ${limit_level:.2f} (${limit_dist:.2f} away)")

            except Exception as e:
                self.logger.error(f"❌ Error logging position: {e}")

        self.logger.info("="*60)

    def _check_signals(self, current_bar):
        """Check for entry signals with detailed logging"""
        if pd.isna(current_bar.get('wr_90')) or pd.isna(current_bar.get('prev_wr_90')):
            return None, "Indicators not ready"

        if pd.isna(current_bar.get('range_90')) or pd.isna(current_bar.get('prev_day_range')):
            return None, "Range data not ready"

        # ✅ ENHANCED: Cleaner range filter message
        min_range = current_bar['prev_day_range'] * STRATEGY_CONFIG['RANGE_PCT']
        if current_bar['range_90'] <= min_range:
            return None, f"Range filter: R90=${current_bar['range_90']:.2f} need >${min_range:.2f} (40% of prev day ${current_bar['prev_day_range']:.2f})"

        long_signal = (current_bar['prev_wr_90'] <= STRATEGY_CONFIG['LONG_ENTRY']) and \
                      (current_bar['wr_90'] > STRATEGY_CONFIG['LONG_ENTRY'])
        short_signal = (current_bar['prev_wr_90'] >= STRATEGY_CONFIG['SHORT_ENTRY']) and \
                       (current_bar['wr_90'] < STRATEGY_CONFIG['SHORT_ENTRY'])

        if long_signal:
            return 'LONG', None
        elif short_signal:
            return 'SHORT', None
        else:
            if current_bar['wr_90'] <= STRATEGY_CONFIG['LONG_ENTRY']:
                return None, f"Waiting LONG (WR: {current_bar['wr_90']:.1f} ≤ {STRATEGY_CONFIG['LONG_ENTRY']}, need cross above)"
            elif current_bar['wr_90'] >= STRATEGY_CONFIG['SHORT_ENTRY']:
                return None, f"Waiting SHORT (WR: {current_bar['wr_90']:.1f} ≥ {STRATEGY_CONFIG['SHORT_ENTRY']}, need cross below)"
            else:
                return None, f"No signal (WR: {current_bar['wr_90']:.1f} in neutral zone [{STRATEGY_CONFIG['LONG_ENTRY']}, {STRATEGY_CONFIG['SHORT_ENTRY']}])"

    def _check_exits(self, positions, current_bar):
        """Check exit conditions and pull actual close data from IG"""
        for position in positions:
            try:
                direction = position['position']['direction']
                entry_price = position['position']['level']
                position_size = position['position']['size']
                deal_id = position['position'].get('dealId')
                current_price = current_bar['closePrice']

                exit_triggered = False
                exit_reason = None
                exit_price = current_price
                exit_details = ""

                # Check WR crossover exits
                if direction == 'BUY':
                    if (current_bar.get('prev_wr_90', 0) >= STRATEGY_CONFIG['LONG_EXIT'] and
                            current_bar.get('wr_90', 0) < STRATEGY_CONFIG['LONG_EXIT']):
                        exit_triggered = True
                        exit_reason = 'wr_cross'
                        exit_details = f"WR crossed below {STRATEGY_CONFIG['LONG_EXIT']} (was {current_bar.get('prev_wr_90', 0):.1f}, now {current_bar.get('wr_90', 0):.1f})"

                elif direction == 'SELL':
                    if (current_bar.get('prev_wr_90', 0) <= STRATEGY_CONFIG['SHORT_EXIT'] and
                            current_bar.get('wr_90', 0) > STRATEGY_CONFIG['SHORT_EXIT']):
                        exit_triggered = True
                        exit_reason = 'wr_cross'
                        exit_details = f"WR crossed above {STRATEGY_CONFIG['SHORT_EXIT']} (was {current_bar.get('prev_wr_90', 0):.1f}, now {current_bar.get('wr_90', 0):.1f})"

                # ✅ FIX: For WR cross, close position first, then pull actual close price from IG
                if exit_triggered and exit_reason == 'wr_cross':
                    self.logger.info(f"🚪 Closing position by {exit_reason}: {exit_details}")

                    # Close the position at IG
                    delete_position(position['position'])

                    # ✅ CRITICAL FIX: Pull actual close price AND timestamp from IG after closing
                    import time
                    time.sleep(2)  # Wait for IG to process the closure

                    # ✅ Get actual exit timestamp (default to now if not available)
                    exit_timestamp = int(datetime.now(self.ny_tz).timestamp() * 1000)
                    entry_timestamp = None

                    closed_pos = get_closed_position_by_deal_id(deal_id)
                    if closed_pos:
                        # ✅ CRITICAL: Get actual ENTRY price from IG (not just exit price)
                        if closed_pos.get('entry_price'):
                            entry_price = closed_pos['entry_price']
                            self.logger.info(f"✅ Got actual entry price from IG: ${entry_price:.2f}")

                        if closed_pos.get('exit_price'):
                            exit_price = closed_pos['exit_price']
                            self.logger.info(f"✅ Got actual close price from IG: ${exit_price:.2f}")

                        # ✅ NEW: Get actual exit timestamp from IG
                        if closed_pos.get('close_timestamp'):
                            exit_timestamp = closed_pos['close_timestamp']
                            exit_dt = datetime.fromtimestamp(exit_timestamp / 1000, tz=self.ny_tz)
                            self.logger.info(f"✅ Got actual close time from IG: {exit_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                        else:
                            self.logger.warning("⚠️ No close timestamp from IG, using current time")

                        if closed_pos.get('open_timestamp'):
                            entry_timestamp = closed_pos['open_timestamp']
                            entry_dt = datetime.fromtimestamp(entry_timestamp / 1000, tz=self.ny_tz)
                            self.logger.info(f"✅ Got actual open time from IG: {entry_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    else:
                        self.logger.warning("⚠️ Could not get IG data, using current price and time")

                    # Calculate P&L with actual close price
                    if direction == 'BUY':
                        gross_pnl = (exit_price - entry_price) * position_size
                    else:
                        gross_pnl = (entry_price - exit_price) * position_size

                    net_pnl = gross_pnl - IG_CONFIG['TRADE_COST']
                    pnl_emoji = "🟢" if net_pnl > 0 else "🔴"

                    # Enhanced exit logging
                    self.logger.info("")
                    self.logger.info("=" * 60)
                    self.logger.info(f"🚪 POSITION CLOSED: {exit_reason.upper()}")
                    self.logger.info("=" * 60)
                    self.logger.info(f"📍 Position: {direction} {position_size} @ ${entry_price:.2f}")
                    self.logger.info(f"📉 Exit: ${exit_price:.2f} (from IG)")
                    self.logger.info(f"🎯 Reason: {exit_details}")
                    self.logger.info(f"📊 Price Move: ${abs(exit_price - entry_price):.2f} ({abs(exit_price - entry_price) / entry_price * 100:.2f}%)")
                    self.logger.info(f"{pnl_emoji} Gross P&L: ${gross_pnl:.2f}")
                    self.logger.info(f"💰 Net P&L: ${net_pnl:.2f} (after ${IG_CONFIG['TRADE_COST']:.2f} costs)")
                    self.logger.info(f"📈 Return: {(net_pnl / entry_price / position_size * 100):.2f}%")
                    self.logger.info("=" * 60)

                    # ✅ Only fall back to tracked entry time if not already set from IG
                    if entry_timestamp is None:
                        if deal_id in self._open_positions:
                            pos_info = self._open_positions[deal_id]
                            if isinstance(pos_info, dict):
                                entry_timestamp = pos_info.get('timestamp')
                            else:
                                entry_timestamp = pos_info

                    if entry_timestamp is None:
                        estimated_duration_ms = 30 * 60 * 1000
                        entry_timestamp = exit_timestamp - estimated_duration_ms
                        self.logger.warning("⚠️ Entry timestamp not tracked, estimating from exit time")

                    # Get current regime
                    bar = self.data_manager.get_latest_bar()
                    if bar is not None:
                        daily_range = bar.get('daily_range', 50)
                        regime, _ = self._get_regime(daily_range)
                    else:
                        regime = 'UNKNOWN'

                    # ✅ Log trade with actual IG timestamps
                    self.logger.log_trade(
                        entry_timestamp=entry_timestamp,
                        exit_timestamp=exit_timestamp,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_size=position_size,
                        pnl=net_pnl,
                        exit_reason=exit_reason,
                        regime=regime,
                        stop_distance=0,
                        bars_held=0
                    )

                    # Remove from tracking
                    if deal_id in self._open_positions:
                        del self._open_positions[deal_id]
                    self._tracked_deal_ids.add(deal_id)

                    self.logger.info("✅ Position closed and logged with IG data")

            except Exception as e:
                self.logger.error(f"❌ Exit error: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

    def refresh_bulk_data(self):
        """Scheduled bulk refresh"""
        self.logger.info("🔄 Bulk data refresh...")
        if self.data_manager.load_or_fetch_bulk_data(force_refresh=True):
            self.data_manager.calculate_indicators()
            self.logger.info("✅ Refresh complete")

    def run_cycle(self):
        """Main cycle - runs every minute"""
        try:
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("🔄 Cycle start...")

            # ✅ CHECK FOR EXTERNALLY CLOSED POSITIONS FIRST
            self.check_for_closed_positions()

            if self.data_manager.should_refresh_bulk_data():
                self.refresh_bulk_data()
                return

            updated = self.data_manager.update_with_latest_bar()

            if updated:
                self.data_manager.calculate_indicators()

            current_bar = self.data_manager.get_latest_bar()
            if current_bar is None:
                self.logger.warning("⚠️ No data available")
                return

            if pd.isna(current_bar.get('wr_90')):
                self.logger.warning("⚠️ Indicators not ready")
                return

            now_ny = datetime.now(pytz.timezone('America/New_York'))
            current_hour = now_ny.hour
            # ✅ Use intraday range (no lookahead) if available
            daily_range = current_bar.get('daily_range_rt', current_bar['daily_range'])
            regime, regime_hours = self._get_regime(daily_range)
            is_trading_hour = current_hour in regime_hours

            # Log market status
            self.logger.log_market_status(
                daily_range, regime, current_hour, is_trading_hour,
                current_bar['closePrice'], current_bar['wr_90'], current_bar['range_90']
            )

            # Show WR(90) calculation details
            self.logger.info(
                f"📊 WR(90): High: ${current_bar['high_90']:.2f} | Low: ${current_bar['low_90']:.2f} | Range: ${current_bar['range_90']:.2f}")

            # ✅ ALWAYS show range filter status
            min_range_required = current_bar['prev_day_range'] * STRATEGY_CONFIG['RANGE_PCT']
            range_status = "✅ PASS" if current_bar['range_90'] > min_range_required else "❌ BLOCKED"
            self.logger.info(
                f"📏 Range Filter: R90=${current_bar['range_90']:.2f} vs Required=${min_range_required:.2f} (40% of prev day ${current_bar['prev_day_range']:.2f}) [{range_status}]")

            accounts = get_accounts()
            if not accounts:
                self.logger.warning("⚠️ Failed to get account info")
                return

            account = accounts[0]
            balance = account['balance']['balance']
            equity = balance + account['balance'].get('profitLoss', 0)

            if equity > self.peak_balance:
                self.peak_balance = equity
                new_size = self._calculate_position_size(equity)
                if new_size > self.current_position_size:
                    self.logger.info(f"📈 Size: {self.current_position_size:.2f} → {new_size:.2f}")
                    self.current_position_size = new_size

            positions = get_positions()
            gold_positions = [p for p in positions if p['market']['epic'] == IG_CONFIG['GOLD_EPIC']]
            num_positions = len(gold_positions)

            self.logger.log_account_status(
                balance, equity, self.peak_balance,
                self.current_position_size, num_positions,
                account['balance'].get('profitLoss', 0)
            )

            if num_positions > 0:
                self._log_position_status(gold_positions, current_bar['closePrice'])

            if not is_trading_hour:
                next_hour = self._get_next_trading_hour(current_hour, regime_hours)
                if next_hour is not None:
                    hours_until = next_hour - current_hour
                    if hours_until < 0:
                        hours_until += 24
                    self.logger.info(f"⏸️ Outside trading hours ({regime} regime, hour {current_hour})")
                    self.logger.info(
                        f"⏰ Next trading hour: {next_hour}:00 NY (in {hours_until}h) | Hours: {sorted(regime_hours)}")
                else:
                    self.logger.info(f"⏸️ Outside trading hours ({regime} regime, hour {current_hour})")

                if num_positions > 0:
                    self._check_exits(gold_positions, current_bar)
                return

            if num_positions > 0:
                self._check_exits(gold_positions, current_bar)
                positions = get_positions()
                gold_positions = [p for p in positions if p['market']['epic'] == IG_CONFIG['GOLD_EPIC']]
                num_positions = len(gold_positions)

            if num_positions == 0:
                signal, skip_reason = self._check_signals(current_bar)

                if signal:
                    if signal == 'LONG':
                        stop_dist = current_bar['closePrice'] - current_bar['low_90']
                    else:
                        stop_dist = current_bar['high_90'] - current_bar['closePrice']

                    if stop_dist > STRATEGY_CONFIG['MAX_STOP_DISTANCE']:
                        self.logger.log_signal(signal, current_bar['closePrice'], regime,
                                               current_bar['wr_90'], False,
                                               f"Stop ${stop_dist:.2f} > ${STRATEGY_CONFIG['MAX_STOP_DISTANCE']}")
                        self.logger.info(
                            f"⏭️ Stop too wide: ${stop_dist:.2f} > ${STRATEGY_CONFIG['MAX_STOP_DISTANCE']}")
                        return

                    self._execute_trade(signal, current_bar, regime, stop_dist)
                elif skip_reason:
                    self.logger.info(f"⏭️ {skip_reason}")

            self.logger.info("✅ Cycle complete")

        except Exception as e:
            self.logger.error(f"❌ Cycle error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _execute_trade(self, direction, current_bar, regime, stop_distance):
        """Execute trade"""
        try:
            if direction == 'LONG':
                stop_price = current_bar['low_90']
                target_price = current_bar['closePrice'] + current_bar['range_90'] * STRATEGY_CONFIG['LONG_TARGET_PCT']
                ig_direction = 1
            else:
                stop_price = current_bar['high_90']
                target_price = current_bar['closePrice'] - current_bar['range_90'] * STRATEGY_CONFIG['SHORT_TARGET_PCT']
                ig_direction = -1

            limit_distance = abs(target_price - current_bar['closePrice'])
            stop_distance_order = abs(stop_price - current_bar['closePrice'])

            self.logger.info(f"📤 {direction} order:")
            self.logger.info(f"   Size: {self.current_position_size:.2f}")
            self.logger.info(f"   Entry: ${current_bar['closePrice']:.2f}")
            self.logger.info(f"   Stop: ${stop_price:.2f} (-${stop_distance_order:.2f})")
            self.logger.info(f"   Target: ${target_price:.2f} (+${limit_distance:.2f})")

            # ✅ Call with correct parameters (up, size, limit, stop)
            result = place_order(
                up=ig_direction,
                size=self.current_position_size,
                limit=limit_distance,
                stop=stop_distance_order
            )

            if result:
                # ✅ Track comprehensive entry information for external close detection
                entry_timestamp = int(datetime.now(self.ny_tz).timestamp() * 1000)
                deal_id = result.get('dealReference')

                if deal_id:
                    # Store entry details for later reference
                    self._open_positions[deal_id] = {
                        'timestamp': entry_timestamp,
                        'price': current_bar['closePrice'],
                        'direction': direction,
                        'size': self.current_position_size
                    }
                    self.logger.info(f"📌 Tracking position: Deal ID {deal_id} @ ${current_bar['closePrice']:.2f}")

                self.logger.log_signal(direction, current_bar['closePrice'], regime,
                                       current_bar['wr_90'], True, None)
                self.logger.info("✅ Order executed!")
            else:
                self.logger.error("❌ Order failed")
                self.logger.log_signal(direction, current_bar['closePrice'], regime,
                                     current_bar['wr_90'], False, "IG API failed")

        except Exception as e:
            self.logger.error(f"❌ Execute error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def print_status_dashboard(self):
        """Print performance dashboard"""
        try:
            perf = self.logger.get_performance_summary()
            if not perf:
                return

            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info("📊 PERFORMANCE DASHBOARD")
            self.logger.info("="*80)

            today = perf['today']
            all_time = perf['all_time']

            today_wr = (today['winners']/today['trades']*100) if today['trades'] > 0 else 0
            all_wr = (all_time['winners']/all_time['trades']*100) if all_time['trades'] > 0 else 0

            self.logger.info(f"TODAY:")
            self.logger.info(f"  Trades: {today['trades']} | Winners: {today['winners']} | Win Rate: {today_wr:.1f}%")
            self.logger.info(f"  Total P&L: ${today['total_pnl']:.2f} | Avg: ${today['avg_pnl']:.2f}")
            self.logger.info(f"  Best: ${today['best_trade']:.2f} | Worst: ${today['worst_trade']:.2f}")

            self.logger.info(f"ALL TIME:")
            self.logger.info(f"  Trades: {all_time['trades']} | Winners: {all_time['winners']} | Win Rate: {all_wr:.1f}%")
            self.logger.info(f"  Total P&L: ${all_time['total_pnl']:.2f}")

            self.logger.info("="*80)

        except Exception as e:
            self.logger.error(f"❌ Dashboard error: {e}")

    def run_daily_backtest(self):
        """Run daily backtest comparison"""
        try:
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("🔍 RUNNING DAILY BACKTEST")
            self.logger.info("=" * 80)

            self.daily_backtest.run_daily_analysis(self.data_manager)

        except Exception as e:
            self.logger.error(f"❌ Daily backtest error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


def main():
    """Main entry point"""
    import sys
    import fcntl

    # ✅ PREVENT MULTIPLE INSTANCES
    lock_file_path = '/tmp/gold_trader.lock'

    try:
        lock_file = open(lock_file_path, 'w')
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        print(f"✅ Lock acquired (PID: {os.getpid()})")
    except IOError:
        print("=" * 80)
        print("❌ ERROR: Another bot instance is already running!")
        print("=" * 80)

        # Try to find the running instance
        try:
            with open(lock_file_path, 'r') as f:
                existing_pid = f.read().strip()
            print(f"Existing bot PID: {existing_pid}")
            print(f"\nTo stop it, run:")
            print(f"  kill {existing_pid}")
        except:
            pass

        print(f"\nOr remove the lock file manually:")
        print(f"  rm {lock_file_path}")
        print("=" * 80)
        sys.exit(1)

    try:
        print("=" * 80)
        print("🚀 GOLD TRADING BOT - READY TO START")
        print("=" * 80)

        print("\n📡 Initializing trader...")
        trader = GoldTrader()

        print("\n⏰ Setting up scheduler...")
        scheduler = BlockingScheduler(timezone=pytz.timezone('America/New_York'))

        # Trading cycle every minute at :05 seconds (UPDATED!)
        scheduler.add_job(
            trader.run_cycle,
            'cron',
            second=5,  # ✅ Changed from 1 to 5
            id='trading_cycle'
        )

        # Bulk refresh at 5:05 PM NY (Mon-Fri)
        scheduler.add_job(
            trader.refresh_bulk_data,
            'cron',
            hour=17,
            minute=5,
            day_of_week='mon-fri',
            id='bulk_refresh'
        )

        # Daily backtest at 5:30 PM NY (Mon-Fri)
        scheduler.add_job(
            trader.run_daily_backtest,
            'cron',
            hour=17,
            minute=30,
            day_of_week='mon-fri',
            id='daily_backtest'
        )

        # Dashboard every 15 min
        scheduler.add_job(
            trader.print_status_dashboard,
            'cron',
            minute='*/15',
            id='dashboard'
        )

        print("\n📅 SCHEDULE:")
        print("  🔄 Trading: Every minute at :05 sec")
        print("  📡 Refresh: 5:05 PM NY (Mon-Fri)")
        print("  🔍 Backtest: 5:30 PM NY (Mon-Fri)")
        print("  📊 Dashboard: Every 15 minutes")
        print(f"\n🔒 Lock file: {lock_file_path}")
        print(f"🆔 Process ID: {os.getpid()}")
        print("\n✅ Bot running! Press Ctrl+C to stop")
        print("=" * 80 + "\n")

        scheduler.start()

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("👋 Shutting down (user interrupt)...")
        print("=" * 80)
        if 'trader' in locals():
            trader.logger.info("🛑 Bot stopped by user")
    except SystemExit:
        print("\n" + "=" * 80)
        print("👋 Shutting down (system exit)...")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ FATAL ERROR: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        if 'trader' in locals():
            trader.logger.error(f"❌ Fatal error: {e}")
            trader.logger.error(traceback.format_exc())
    finally:
        # ✅ CLEAN UP LOCK FILE
        try:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            os.remove(lock_file_path)
            print(f"✅ Lock file removed: {lock_file_path}")
        except:
            pass

if __name__ == "__main__":
    main()

