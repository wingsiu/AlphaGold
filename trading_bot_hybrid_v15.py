#!/usr/bin/env python3
"""
AlphaGold v15 Hybrid Trading Bot
==================================
v14 pattern-first architecture + v15 deterministic energetic gate (no HMM).

Key changes from v14 hybrid bot:
  - Scorer swapped to V15HybridLiveScorer (deterministic energetic gate)
  - Feature building includes v15 en_* columns
  - Signal parity with v15 backtest (both use bar_move > 3 & vol > 200 gate)

Usage:
  python3 trading_bot_hybrid_v15.py
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brokers.ig_live import IGLiveBrokerAdapter
from config.hybrid_config import (
    ENERGETIC_EXECUTION_CONFIG,
    EXECUTION_CONFIG,
    HYBRID_CONFIG,
    WF_CONFIG,
)
from brokers.base import OrderRequest
from ig_scripts.ig_data_api import (
    API_CONFIG,
    IGService,
    Price,
    fetch_and_store_prices_from_latest,
    fetch_open_positions,
    fetch_prices,
)
from trading_bot_base import AlphaGoldBaseBot, BotState
from v15.hybrid_live import V15HybridLiveScorer, LiveSignal
from xgboost_filter_model.time_slot_filter import is_blocked_entry, load_weak_filter, resolve_v14_time_filter_path
from mobile_api.journal import SignalJournal, trading_day_label, trading_day_start_utc

UTC = timezone.utc


@dataclass
class HybridBotState(BotState):
    last_trained_wf_cycle: Optional[int] = None
    open_position_source: Optional[str] = None  # pattern | energetic
    open_pattern_name: Optional[str] = None
    open_tp: Optional[float] = None
    open_sl: Optional[float] = None
    open_horizon: Optional[int] = None
    open_side: Optional[int] = None
    entry_tp: Optional[float] = None
    entry_horizon: Optional[int] = None
    open_horizon_deadline: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "HybridBotState":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class AlphaGoldHybridV15Bot(AlphaGoldBaseBot):
    """v15 Pattern + energetic fallback live bot — deterministic gate, no HMM."""

    def __init__(self):
        self.logger = self._init_logger()
        self.logger.info("--- AlphaGold v15 HYBRID Bot Startup (no HMM) ---")

        self.state_path = PROJECT_ROOT / "runtime" / "trading_bot_hybrid_v15_state.json"
        self.state = self._load_hybrid_state()

        self.logger.info(f"Pattern exec defaults: {json.dumps(EXECUTION_CONFIG, indent=2)}")
        self.logger.info(f"Energetic exec: {json.dumps(ENERGETIC_EXECUTION_CONFIG, indent=2)}")
        self.logger.info(
            f"Hybrid: pattern reverse={HYBRID_CONFIG['pattern_close_on_reverse']}  "
            f"refresh={HYBRID_CONFIG['pattern_same_dir_refresh']}  "
            f"energetic reverse={HYBRID_CONFIG['energetic_close_on_reverse']}  "
            f"refresh={HYBRID_CONFIG['energetic_same_dir_refresh']}"
        )
        self.logger.info("Energetic gate: v15 deterministic (bar_move > 3 & vol > 200, NO HMM)")

        self.service = IGService(
            api_key=API_CONFIG["api_key"],
            username=API_CONFIG["username"],
            password=API_CONFIG["password"],
            base_url=API_CONFIG["base_url"],
        )
        self.broker = IGLiveBrokerAdapter(
            self.service,
            instrument=Price.Gold,
            stop_loss_pct=ENERGETIC_EXECUTION_CONFIG["sl"],
            take_profit_pct=ENERGETIC_EXECUTION_CONFIG["tp"],
        )

        self.weak_period_cells: list[dict] = []
        wf_path = resolve_v14_time_filter_path(PROJECT_ROOT)
        if wf_path and Path(wf_path).exists():
            self.weak_period_cells = load_weak_filter(wf_path)
            self.logger.info(
                f"Time slot filter ON: {len(self.weak_period_cells)} blocked cells"
            )

        self.scorer = V15HybridLiveScorer(self.logger)
        self._load_models()
        self.feature_warmup_days = int(WF_CONFIG.get("feature_warmup_days", 120))
        self.min_prediction_bars = 400
        self.last_ts = None
        self.last_predicted_bar_ts = None
        self._last_submitted_bar_ts: Optional[pd.Timestamp] = None
        self.cached_df = pd.DataFrame()
        self._warmup_cache()
        self._feature_df: pd.DataFrame | None = None
        self._feature_df_end: pd.Timestamp | None = None
        self.journal = SignalJournal()

    def _init_logger(self):
        log = logging.getLogger("AlphaGoldHybridV15")
        if log.handlers:
            return log
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
        log_path = PROJECT_ROOT / "runtime" / "trading_bot_hybrid_v15.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        log.addHandler(fh)
        return log

    def _load_models(self):
        self.scorer.reload()
        self._feature_df = None
        self._feature_df_end = None

    def _maybe_weekend_retrain(self) -> None:
        now = datetime.now(UTC)
        if now.hour != 1:
            return

        from xgboost_filter_model.pattern_training import wf_incremental_train_target

        pending = wf_incremental_train_target(pd.Timestamp(now))
        if pending is None:
            return
        cycle_num, cycle_start = pending
        if self.state.last_trained_wf_cycle == cycle_num:
            return

        self.logger.info(
            f"WF cycle closed — training cycle_{cycle_num} "
            f"(start {cycle_start.date()}, data before {cycle_start.date()})…"
        )
        retrain_log_dir = PROJECT_ROOT / "runtime" / "retrain_logs"
        retrain_log_dir.mkdir(parents=True, exist_ok=True)
        day_tag = now.date().isoformat()

        try:
            retrain_env = {
                **os.environ,
                "V14_WF_TRAIN_MODE": "incremental",
                "PYTHONPATH": str(PROJECT_ROOT),
            }
            wf_end = date.today().strftime("%Y-%m-%d")
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "retrain_hybrid_wf.py"),
                "2025-06-01",
                wf_end,
            ]
            self.logger.info("Running incremental hybrid retrain + weak-filter rebuild…")
            res = subprocess.run(cmd, capture_output=True, text=True, env=retrain_env)
            (retrain_log_dir / f"retrain_hybrid_v15_{day_tag}.log").write_text(
                f"=== STDOUT ===\n{res.stdout}\n=== STDERR ===\n{res.stderr}\n",
                encoding="utf-8",
            )
            if res.returncode != 0:
                self.logger.error(f"retrain_hybrid_wf failed: {res.stderr[-2000:]}")
                return
            self.logger.info("Hybrid retraining complete — hot-reloading scorer…")
            self._load_models()
            self.state.last_retrain_date = now.date().isoformat()
            self.state.last_trained_wf_cycle = cycle_num
            self._save_state()
        except Exception as e:
            self.logger.error(f"Weekend retrain error: {e}")

    def _load_hybrid_state(self) -> HybridBotState:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    return HybridBotState.from_dict(json.load(f))
            except Exception as e:
                self.logger.error(f"Failed to load hybrid state: {e}")
        return HybridBotState()

    def _save_state(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def _clear_position_state(self):
        self.state.open_deal_id = None
        self.state.open_entry_time = None
        self.state.closed_first_seen_at = None
        self.state.pending_level_resync_bar = None
        self.state.open_position_source = None
        self.state.open_pattern_name = None
        self.state.open_tp = None
        self.state.open_sl = None
        self.state.open_horizon = None
        self.state.open_side = None
        self.state.entry_tp = None
        self.state.entry_horizon = None
        self.state.open_horizon_deadline = None

    # =====================================================================
    # (All position management, signal submission, and polling logic is
    #  identical to v14 — inherited from AlphaGoldBaseBot / copied from
    #  trading_bot_hybrid_v14.py.  The only v15 change is the scorer.)
    # =====================================================================

    def _horizon_deadline_ts(self) -> Optional[pd.Timestamp]:
        if self.state.open_horizon_deadline:
            ts = pd.Timestamp(self.state.open_horizon_deadline)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return ts
        if not self.state.open_entry_time:
            return None
        entry_ts = pd.Timestamp(self.state.open_entry_time)
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        horizon_min = float(
            self.state.open_horizon or ENERGETIC_EXECUTION_CONFIG.get("horizon", 30)
        )
        return entry_ts + pd.Timedelta(minutes=horizon_min)

    def _set_horizon_deadline(self, from_ts: pd.Timestamp, horizon_minutes: int) -> None:
        ts = pd.Timestamp(from_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        self.state.open_horizon_deadline = (ts + pd.Timedelta(minutes=horizon_minutes)).isoformat()

    def _refresh_mode_for_source(self, source: str) -> str:
        if source == "pattern":
            return str(HYBRID_CONFIG.get("pattern_same_dir_refresh", "entry"))
        return str(HYBRID_CONFIG.get("energetic_same_dir_refresh", "global"))

    def _apply_same_dir_refresh(
        self, latest_ts: pd.Timestamp, close_price: float, refresh_mode: str,
    ) -> bool:
        if refresh_mode == "none" or not self.state.open_deal_id:
            return False
        side = int(self.state.open_side or 0)
        if side == 0:
            return False
        source = self.state.open_position_source or "energetic"
        if refresh_mode == "entry":
            tp_dist = float(self.state.entry_tp or self.state.open_tp or EXECUTION_CONFIG["tp"])
            horizon = int(self.state.entry_horizon or self.state.open_horizon or EXECUTION_CONFIG["horizon"])
        elif refresh_mode == "global":
            tp_dist = float(ENERGETIC_EXECUTION_CONFIG["tp"])
            horizon = int(ENERGETIC_EXECUTION_CONFIG["horizon"])
        else:
            self.logger.warning(f"Unsupported live refresh mode: {refresh_mode}")
            return False
        self._set_horizon_deadline(latest_ts, horizon)
        pos_wrap = self.broker.get_position_by_deal_id(self.state.open_deal_id)
        if not pos_wrap:
            self._save_state()
            return True
        pos = pos_wrap.get("position", {})
        stop_level = pos.get("stopLevel")
        current_limit = pos.get("limitLevel")
        new_limit = float(close_price) + tp_dist if side == 1 else float(close_price) - tp_dist
        should_amend = False
        if current_limit is None:
            should_amend = True
        else:
            cur = float(current_limit)
            if side == 1 and new_limit > cur:
                should_amend = True
            elif side == -1 and new_limit < cur:
                should_amend = True
        if should_amend and stop_level is not None:
            try:
                self.broker.amend_position_levels(
                    deal_id=self.state.open_deal_id,
                    stop_level=round(float(stop_level), 2),
                    limit_level=round(new_limit, 2),
                )
                self.logger.info(
                    f"Same-dir refresh ({refresh_mode}) source={source} side={side} "
                    f"limit={new_limit:.2f} horizon_reset={horizon}m"
                )
            except Exception as e:
                self.logger.error(f"Same-dir refresh amend failed: {e}")
        else:
            self.logger.info(
                f"Same-dir refresh ({refresh_mode}) source={source} "
                f"horizon_reset={horizon}m limit_unchanged={current_limit}"
            )
        self._save_state()
        return True

    def _manage_open_position(
        self, latest_ts: pd.Timestamp, close_price: float,
        pat_sig: Optional[LiveSignal], en_sig: Optional[LiveSignal],
    ) -> bool:
        if not self.state.open_deal_id:
            return False
        open_side = int(self.state.open_side or 0)
        source = self.state.open_position_source or "pattern"
        pat_side = pat_sig.side if pat_sig else 0
        en_side = en_sig.side if en_sig else 0
        sig_side = pat_side if source == "pattern" else en_side
        if source == "energetic" and pat_sig:
            closed_ok = self._close_position_market("pattern_priority")
            if closed_ok:
                self.logger.info(
                    "Energetic position closed for pattern_priority — "
                    "pattern entry will fire on next poll (5s)"
                )
            else:
                self.logger.error(
                    "CRITICAL: Failed to close energetic position for "
                    "pattern_priority — BLOCKING pattern entry to prevent double trade"
                )
            return True
        close_on_reverse = (
            HYBRID_CONFIG.get("pattern_close_on_reverse", False)
            if source == "pattern"
            else HYBRID_CONFIG.get("energetic_close_on_reverse", True)
        )
        if close_on_reverse and sig_side != 0 and sig_side == -open_side:
            self._close_position_market("reverse_signal")
            return True
        refresh_mode = self._refresh_mode_for_source(source)
        if sig_side != 0 and sig_side == open_side and refresh_mode != "none":
            self._apply_same_dir_refresh(latest_ts, close_price, refresh_mode)
            return True
        return True

    def _sync_trade_results(self):
        if not self.state.open_deal_id:
            try:
                open_pos = fetch_open_positions(self.service)
                if open_pos:
                    ids = [p.get("position", {}).get("dealId") for p in open_pos]
                    self.logger.warning(
                        f"Startup reconciliation: state has no open_deal_id "
                        f"but IG reports {len(open_pos)} position(s) deal_ids={ids}."
                    )
            except Exception:
                pass
            return
        try:
            open_pos = fetch_open_positions(self.service)
            is_still_open = any(
                p.get("position", {}).get("dealId") == self.state.open_deal_id for p in open_pos
            )
            if not is_still_open:
                closed = self.broker.get_closed_trade_by_deal_id(self.state.open_deal_id)
                if closed:
                    pnl = closed.get("pnl", 0.0) or 0.0
                    self.logger.info(
                        f"Trade {self.state.open_deal_id} closed PnL={pnl:.2f} "
                        f"source={self.state.open_position_source}"
                    )
                    try:
                        self.journal.close_trade(
                            self.state.open_deal_id,
                            exit_time=closed.get("exit_time"),
                            exit_price=closed.get("exit_price"),
                            pnl=float(pnl),
                            exit_reason=closed.get("reason") or "broker_close",
                        )
                    except Exception as je:
                        self.logger.error(f"Journal close failed: {je}")
                    self.state.last_pnl = pnl
                    self.state.consecutive_losses = (
                        self.state.consecutive_losses + 1 if pnl < 0 else 0
                    )
                    self._clear_position_state()
                    self._save_state()
                else:
                    now_iso = datetime.now(UTC).isoformat()
                    if not self.state.closed_first_seen_at:
                        self.state.closed_first_seen_at = now_iso
                        self._save_state()
                    elif (
                        datetime.now(UTC) - datetime.fromisoformat(self.state.closed_first_seen_at)
                    ).total_seconds() > 300:
                        self.logger.warning("Force-clearing stale open_deal_id")
                        try:
                            self.journal.close_trade(
                                self.state.open_deal_id,
                                exit_time=datetime.now(UTC).isoformat(),
                                exit_reason="stale_sync",
                            )
                        except Exception as je:
                            self.logger.error(f"Journal stale close failed: {je}")
                        self._clear_position_state()
                        self._save_state()
            else:
                if self.state.closed_first_seen_at:
                    self.state.closed_first_seen_at = None
                    self._save_state()
        except Exception as e:
            self.logger.error(f"Error syncing trade results: {e}")

    def _check_horizon_timeout(self):
        if not self.state.open_deal_id:
            return
        deadline = self._horizon_deadline_ts()
        if deadline is None:
            return
        try:
            now_ts = pd.Timestamp(datetime.now(UTC))
            if now_ts < deadline:
                return
            pos_wrap = self.broker.get_position_by_deal_id(self.state.open_deal_id)
            if not pos_wrap:
                self._sync_trade_results()
                return
            pos = pos_wrap.get("position", {})
            direction = str(pos.get("direction", "")).upper()
            size = float(pos.get("size") or ENERGETIC_EXECUTION_CONFIG.get("size", 1.0))
            close_dir = "SELL" if direction == "BUY" else "BUY"
            elapsed_min = (now_ts - pd.Timestamp(self.state.open_entry_time)).total_seconds() / 60.0 if self.state.open_entry_time else 0
            self.logger.info(
                f"HORIZON TIMEOUT source={self.state.open_position_source} "
                f"elapsed={elapsed_min:.1f}m"
            )
            self.broker.close_position(
                deal_id=self.state.open_deal_id, direction=close_dir, size=size
            )
        except Exception as e:
            self.logger.error(f"Horizon timeout error: {e}")

    def _resync_levels_to_backtest(self, df: pd.DataFrame) -> None:
        pending = self.state.pending_level_resync_bar
        if not pending or not self.state.open_deal_id:
            return
        try:
            target_ts = pd.Timestamp(pending)
            if target_ts.tzinfo is None:
                target_ts = target_ts.tz_localize("UTC")
        except Exception:
            self.state.pending_level_resync_bar = None
            self._save_state()
            return
        if target_ts not in df.index:
            return
        pos_wrap = self.broker.get_position_by_deal_id(self.state.open_deal_id)
        if not pos_wrap:
            self.state.pending_level_resync_bar = None
            self._save_state()
            return
        direction = str(pos_wrap.get("position", {}).get("direction", "")).upper()
        side = 1 if direction == "BUY" else -1
        bar_open_mid = float(df.loc[target_ts, "open"])
        spread = float(ENERGETIC_EXECUTION_CONFIG.get("spread_default", 0.25))
        bar_open_ask = bar_open_mid + spread
        bar_open_bid = bar_open_mid - spread
        tp = float(self.state.open_tp or ENERGETIC_EXECUTION_CONFIG["tp"])
        sl = float(self.state.open_sl or ENERGETIC_EXECUTION_CONFIG["sl"])
        ep = bar_open_ask if side == 1 else bar_open_bid
        stop_level = ep - sl if side == 1 else ep + sl
        limit_level = ep + tp if side == 1 else ep - tp
        try:
            self.broker.amend_position_levels(
                deal_id=self.state.open_deal_id,
                stop_level=round(stop_level, 2),
                limit_level=round(limit_level, 2),
            )
            self.state.pending_level_resync_bar = None
            self._save_state()
        except Exception as e:
            self.logger.error(f"Level resync failed: {e}")

    def _get_feature_df(self) -> pd.DataFrame:
        if self.cached_df.empty:
            return pd.DataFrame()
        end_ts = self.cached_df.index.max()
        if self._feature_df is not None and self._feature_df_end == end_ts:
            return self._feature_df
        self._feature_df = self.scorer.build_feature_df_from_ohlcv(self.cached_df)
        self._feature_df_end = end_ts
        return self._feature_df

    def _close_position_market(self, reason: str) -> bool:
        if not self.state.open_deal_id:
            return False
        try:
            pos_wrap = self.broker.get_position_by_deal_id(self.state.open_deal_id)
            if not pos_wrap:
                return False
            pos = pos_wrap.get("position", {})
            direction = str(pos.get("direction", "")).upper()
            size = float(pos.get("size") or ENERGETIC_EXECUTION_CONFIG.get("size", 1.0))
            close_dir = "SELL" if direction == "BUY" else "BUY"
            self.broker.close_position(
                deal_id=self.state.open_deal_id, direction=close_dir, size=size
            )
            return True
        except Exception as e:
            self.logger.error(f"Close position failed: {e}")
            return False

    def _submit_signal(self, sig: LiveSignal, latest_ts: pd.Timestamp, entry_price: float) -> None:
        try:
            if self._feature_df is not None and latest_ts in self._feature_df.index:
                row = self._feature_df.loc[latest_ts]
                features_json = json.dumps(row.to_dict(), default=str)
            else:
                features_json = None
        except Exception:
            features_json = None

        bar_ts = pd.Timestamp(latest_ts).tz_convert("UTC")
        if self._last_submitted_bar_ts and bar_ts <= self._last_submitted_bar_ts:
            self.logger.error(
                f"CRITICAL: _submit_signal BLOCKED — already submitted for bar {bar_ts}"
            )
            return

        try:
            open_positions = fetch_open_positions(self.service)
            if open_positions:
                deal_ids = [p.get("position", {}).get("dealId", "?") for p in open_positions]
                self.logger.error(
                    f"CRITICAL: _submit_signal BLOCKED — IG has {len(open_positions)} "
                    f"open position(s) deal_ids={deal_ids}."
                )
                return
        except Exception as e:
            self.logger.error(f"_submit_signal: fetch_open_positions failed: {e} — BLOCKING entry")
            return

        bar_iso = pd.Timestamp(latest_ts).tz_convert("UTC").isoformat()
        if is_blocked_entry(latest_ts, self.weak_period_cells):
            self.logger.info(f"Signal blocked by time filter at {latest_ts}")
            self.journal.record_score(
                bar_iso,
                pattern_name=sig.pattern_name,
                pattern_side=sig.side if sig.source == "pattern" else 0,
                pattern_prob=sig.probability if sig.source == "pattern" else None,
                energetic_side=sig.side if sig.source == "energetic" else 0,
                energetic_prob=sig.probability if sig.source == "energetic" else None,
                action="blocked_time_filter",
                open_source=self.state.open_position_source,
                features_json=features_json,
            )
            return

        side_str = "buy" if sig.side == 1 else "sell"
        self.logger.info(
            f"Triggering {sig.source} signal {side_str} pattern={sig.pattern_name} "
            f"tp={sig.tp} sl={sig.sl} h={sig.horizon} prob={sig.probability:.3f}"
        )
        request = OrderRequest(
            symbol="gold",
            side=side_str,
            size=float(ENERGETIC_EXECUTION_CONFIG.get("size", 1.0)),
            signal_time_utc=pd.Timestamp(latest_ts).tz_convert("UTC").isoformat(),
            entry_time_utc=pd.Timestamp(latest_ts).tz_convert("UTC").isoformat(),
            entry_price=float(entry_price),
            probability=float(sig.probability),
            signal_model_family="v15_hybrid",
            metadata={"stop_distance": sig.sl, "limit_distance": sig.tp, "source": sig.source},
        )
        exec_res = self.broker.submit_order(request)
        if exec_res.submitted:
            self._last_submitted_bar_ts = bar_ts
            deal_id = exec_res.deal_id
            self.state.open_deal_id = deal_id
            signal_ts = pd.Timestamp(latest_ts).tz_convert("UTC")
            entry_bar_ts = signal_ts + pd.Timedelta(minutes=1)
            self.state.open_entry_time = signal_ts.isoformat()
            self.state.open_position_source = sig.source
            self.state.open_pattern_name = sig.pattern_name
            self.state.open_tp = sig.tp
            self.state.open_sl = sig.sl
            self.state.open_horizon = sig.horizon
            self.state.open_side = sig.side
            self.state.entry_tp = sig.tp
            self.state.entry_horizon = sig.horizon
            next_bar = pd.Timestamp(latest_ts) + pd.Timedelta(minutes=1)
            if next_bar.tzinfo is None:
                next_bar = next_bar.tz_localize("UTC")
            self._set_horizon_deadline(next_bar, sig.horizon)
            self.state.pending_level_resync_bar = next_bar.isoformat()
            self._save_state()
            self.logger.info(f"Order submitted deal_id={deal_id} source={sig.source}")
            try:
                self.journal.open_trade({
                    "deal_id": deal_id, "source": sig.source, "pattern_name": sig.pattern_name,
                    "side": sig.side, "signal_time": signal_ts.isoformat(),
                    "entry_time": entry_bar_ts.isoformat(), "entry_price": float(entry_price),
                    "tp": sig.tp, "sl": sig.sl, "horizon": sig.horizon,
                    "probability": sig.probability,
                    "horizon_deadline": self.state.open_horizon_deadline,
                })
                self.journal.record_score(
                    bar_iso, pattern_name=sig.pattern_name,
                    pattern_side=sig.side if sig.source == "pattern" else 0,
                    energetic_side=sig.side if sig.source == "energetic" else 0,
                    energetic_prob=sig.probability, action="entry",
                    detail=f"deal_id={deal_id}", open_source=sig.source,
                    features_json=features_json,
                )
            except Exception as je:
                self.logger.error(f"Journal entry failed: {je}")
        else:
            self.logger.error(f"Order failed: {exec_res.reason}")

    def poll_trade(self) -> None:
        now = datetime.now(UTC)
        now_pd = pd.Timestamp(now)
        current_label = trading_day_label(trading_day_start_utc(now))
        last_recon_str = self.state.last_reconciliation_day
        if last_recon_str and current_label > last_recon_str:
            recon_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "daily_reconciliation.py"),
                last_recon_str,
            ]
            try:
                subprocess.Popen(recon_cmd)
            except Exception as e:
                self.logger.error(f"Reconciliation trigger failed: {e}")
        if self.state.last_reconciliation_day != current_label:
            self.state.last_reconciliation_day = current_label
            self._save_state()
        self._sync_trade_results()
        self._check_horizon_timeout()
        try:
            prices = fetch_prices(
                self.service, Price.Gold,
                start_time=now - pd.Timedelta(minutes=5),
                end_time=now,
            )
            if not prices:
                return
            new_df = pd.DataFrame(prices)
            new_df.index = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
            new_df = new_df.rename(
                columns={
                    "openPrice": "open", "highPrice": "high", "lowPrice": "low",
                    "closePrice": "close", "lastTradedVolume": "volume",
                }
            ).sort_index()
            self.cached_df = pd.concat([self.cached_df, new_df])
            self.cached_df = self.cached_df[~self.cached_df.index.duplicated(keep="last")].sort_index()
            max_ts = self.cached_df.index.max()
            cutoff = max_ts - pd.Timedelta(days=self.feature_warmup_days)
            self.cached_df = self.cached_df[self.cached_df.index >= cutoff]
        except Exception as e:
            self.logger.error(f"Fetch failed: {e}")
            return
        current_minute_floor = pd.Timestamp(now).floor("1min")
        df = self.cached_df[self.cached_df.index < current_minute_floor].copy()
        if df.empty:
            return
        raw_latest_ts = df.index[-1]
        self._resync_levels_to_backtest(df)
        if self.last_predicted_bar_ts == raw_latest_ts:
            return
        feat_df = self._get_feature_df()
        bar_iso = pd.Timestamp(raw_latest_ts).tz_convert("UTC").isoformat()
        if feat_df.empty or raw_latest_ts not in feat_df.index:
            self.last_predicted_bar_ts = raw_latest_ts
            return
        latest_ts = raw_latest_ts
        try:
            snap = self.scorer.bar_score_snapshot(
                feat_df, latest_ts, consecutive_losses=self.state.consecutive_losses
            )
            pat_sig, en_sig = self.scorer.score_bar(
                feat_df, latest_ts, consecutive_losses=self.state.consecutive_losses
            )
            try:
                row = feat_df.loc[latest_ts]
                self.journal.record_bar_feature(
                    pd.Timestamp(latest_ts).tz_convert("UTC").isoformat(),
                    json.dumps(row.to_dict(), default=str),
                )
            except Exception:
                pass
        except Exception as e:
            self.logger.error(f"Score failed at {latest_ts}: {e}")
            self.last_predicted_bar_ts = raw_latest_ts
            return
        self.last_predicted_bar_ts = raw_latest_ts
        routed = snap.routed_pattern
        self.logger.info(
            f"[{latest_ts}] pattern={routed or '—'} prob={snap.pattern_prob or '—'} "
            f"en={en_sig.side if en_sig else 0} "
            f"open={self.state.open_position_source or 'flat'}"
        )
        try:
            self.journal.record_score(
                pd.Timestamp(latest_ts).tz_convert("UTC").isoformat(),
                pattern_name=routed, pattern_side=snap.pattern_side,
                pattern_prob=snap.pattern_prob,
                energetic_side=snap.energetic_side,
                energetic_prob=snap.energetic_prob,
                action="score", open_source=self.state.open_position_source,
            )
        except Exception as je:
            self.logger.error(f"Journal score failed: {je}")
        close_price = float(df.loc[latest_ts, "close"])
        if self._manage_open_position(latest_ts, close_price, pat_sig, en_sig):
            return
        entry_sig = pat_sig if pat_sig else en_sig
        if not entry_sig:
            return
        self._submit_signal(entry_sig, latest_ts, close_price)

    def poll_db(self) -> None:
        import time as _t
        for inst in [Price.Gold, Price.Oil, Price.AUD]:
            try:
                fetch_and_store_prices_from_latest(self.service, inst)
            except Exception as e:
                self.logger.error(f"DB store failed for {inst.name}: {e}")
            _t.sleep(2)

    def run(self):
        self.logger.info("v15 Hybrid bot execution loop started (5s=trade  30s=db-only).")
        while True:
            try:
                self._maybe_weekend_retrain()
                now = datetime.now(UTC)
                sec = now.second
                if sec == 5:
                    self.poll_trade()
                    time.sleep(1.2)
                elif sec == 30:
                    self.poll_db()
                    time.sleep(1.2)
                else:
                    time.sleep(0.5)
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(1)


if __name__ == "__main__":
    pid_file = PROJECT_ROOT / "runtime" / "trading_bot_hybrid_v15.pid"
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            try:
                os.kill(old_pid, 0)
                print(f"ERROR: Another instance is already running (PID {old_pid}). Exiting.", file=sys.stderr)
                sys.exit(1)
            except OSError:
                pid_file.unlink()
        except Exception:
            pid_file.unlink()
    pid_file.write_text(str(os.getpid()))
    atexit.register(lambda: pid_file.unlink() if pid_file.exists() else None)
    AlphaGoldHybridV15Bot().run()
