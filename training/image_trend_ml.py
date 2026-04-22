#!/usr/bin/env python3
"""Image-like trend prediction from candle windows.

Pipeline:
1) Load gold minute bars from MySQL.
2) Resample to 1-minute candles by default.
3) Apply time-based no-trade filters (HKT open, London session, NY open).
4) Filter to keep only windows with price range > min_window_range.
5) Convert rolling windows into image-like tensors.
6) Optionally grid-search best min_window_range on train split.
7) Two-stage model:
   - Stage 1: flat vs trend (any direction)
   - Stage 2: up vs down (only on trend-predicted samples)
8) Evaluate out-of-sample: confusion matrix per stage + full 3-class matrix.
9) Directional PnL check with n_days guard.
10) Save model artifact (.joblib) and JSON report.

No-trade windows (UTC):
   HKT 09:00-09:30  -> UTC 01:00-01:30  (HK market open)
   HKT 15:00 to London afternoon fix (~15:00 London)
                    -> UTC 07:00-15:00
   NY  09:30-10:00  -> UTC 13:30-15:00  (covers EST UTC-5 and EDT UTC-4)
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import numbers
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import sys
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import DataLoader


# ── Progress helpers ──────────────────────────────────────────────────────────

_RUN_START: float = time.time()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _elapsed(since: float) -> str:
    s = time.time() - since
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(int(s), 60)
    return f"{m}m{s:02d}s"


def _log(msg: str, *, t0: float | None = None) -> float:
    """Print a timestamped message; return current time for chaining."""
    now = time.time()
    extra = f"  (+{_elapsed(t0)})" if t0 is not None else ""
    print(f"[{_now()}] {msg}{extra}", flush=True)
    return now


# ── No-trade windows in UTC ───────────────────────────────────────────────────
# Each entry: (start_hour, start_min, end_hour, end_min)  [end is exclusive]
BLOCKED_UTC_WINDOWS: list[tuple[int, int, int, int]] = [
    (1,  0,  1, 30),   # HKT 09:00-09:30   (HK market open)
    (7,  0, 15,  0),   # HKT 15:00 → London afternoon gold fix (~UTC 07:00-15:00)
    (13, 30, 15,  0),  # NY  09:30-10:00    (covers EST UTC-5 and EDT UTC-4)
]

DEFAULT_REVERSE_EXIT_PROB = 0.70
HK_TZ = ZoneInfo("Asia/Hong_Kong")
LONDON_TZ = ZoneInfo("Europe/London")
NY_TZ = ZoneInfo("America/New_York")
TRADING_DAY_CUTOFF_HOUR_NY = 17
ASIA_SESSION_START = 6
ASIA_SESSION_END = 19
NY_SESSION_START = 6
NY_SESSION_END = 17
SESSION_PERIOD_SPECS: dict[str, dict[str, object]] = {
    "hkt": {
        "timezone": HK_TZ,
        "start_hour": 8,
        "start_minute": 0,
        "end_hour": 16,
        "end_minute": 0,
    },
    "london": {
        "timezone": LONDON_TZ,
        "start_hour": 8,
        "start_minute": 0,
        "end_hour": 16,
        "end_minute": 30,
    },
    "ny": {
        "timezone": NY_TZ,
        "start_hour": 9,
        "start_minute": 30,
        "end_hour": 16,
        "end_minute": 0,
    },
}
SIGNAL_REFERENCE_MODE = "signal_bar_close"
ENTRY_EXECUTION_MODE = "next_bar_open"
PREP_CACHE_SCHEMA_VERSION = 2  # bumped: bars cache now includes bars_ask / bars_bid

LABEL_DOWN = 0
LABEL_FLAT = 1
LABEL_UP = 2
LABEL_RISKY = 3

TREND_LABELS = (LABEL_DOWN, LABEL_UP)
NO_TRADE_LABELS = (LABEL_FLAT, LABEL_RISKY)

BASE_IMAGE_CHANNEL_NAMES = [
    "open_rel",
    "high_rel",
    "low_rel",
    "close_rel",
    "body_rel",
    "range_rel",
    "vol_z",
    "vol_rel",
    "vol_diff_norm",
]
WICK_IMAGE_CHANNEL_NAMES = [
    "long_lower_wick_flag",
    "long_upper_wick_flag",
]


def make_image_channel_names(use_15m_wick_features: bool = False) -> list[str]:
    names = list(BASE_IMAGE_CHANNEL_NAMES)
    if use_15m_wick_features:
        names.extend(WICK_IMAGE_CHANNEL_NAMES)
    return names


def _is_blocked(ts: pd.Timestamp) -> bool:
    """Return True if UTC timestamp falls inside a no-trade window."""
    t = ts.hour * 60 + ts.minute
    for sh, sm, eh, em in BLOCKED_UTC_WINDOWS:
        if sh * 60 + sm <= t < eh * 60 + em:
            return True
    return False


def _session_hour_label(session_key: str, hour: int) -> str:
    if session_key == "ny" and int(hour) == 9:
        return "09:30"
    return f"{int(hour):02d}:00"


def _normalize_weak_period_cells(payload: object) -> list[dict[str, str]]:
    if isinstance(payload, dict) and isinstance(payload.get("weak_cells"), list):
        raw_cells = payload["weak_cells"]
    elif isinstance(payload, list):
        raw_cells = payload
    else:
        return []
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw_cells:
        if not isinstance(item, dict):
            continue
        session = str(item.get("session", "")).strip().lower()
        day = str(item.get("day", "")).strip()
        hour = str(item.get("hour", "")).strip()
        if session not in SESSION_PERIOD_SPECS or not day or not hour:
            continue
        key = (session, day, hour)
        if key in seen:
            continue
        seen.add(key)
        out.append({"session": session, "day": day, "hour": hour})
    return out


def _load_weak_period_cells(path: Optional[str]) -> list[dict[str, str]]:
    if not path:
        return []
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return _normalize_weak_period_cells(payload)


def _matches_weak_period_cell(ts: pd.Timestamp, cell: dict[str, str]) -> bool:
    ts_utc = pd.Timestamp(ts)
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    else:
        ts_utc = ts_utc.tz_convert("UTC")
    session = cell["session"]
    spec = SESSION_PERIOD_SPECS[session]
    local_ts = ts_utc.tz_convert(spec["timezone"])
    minute_of_day = local_ts.hour * 60 + local_ts.minute
    start_min = int(spec["start_hour"]) * 60 + int(spec["start_minute"])
    end_min = int(spec["end_hour"]) * 60 + int(spec["end_minute"])
    if not (start_min <= minute_of_day < end_min):
        return False
    return local_ts.day_name() == cell["day"] and _session_hour_label(session, int(local_ts.hour)) == cell["hour"]


def _is_weak_period_entry(ts: pd.Timestamp, weak_period_cells: Optional[list[dict[str, str]]]) -> bool:
    if not weak_period_cells:
        return False
    return any(_matches_weak_period_cell(ts, cell) for cell in weak_period_cells)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    table: str
    start_date: str
    end_date: str
    timeframe: str
    disable_time_filter: bool
    window: int
    window_15m: int
    min_window_range: float
    min_15m_drop: float
    min_15m_rise: float
    last_bar_wr90_high: Optional[float]
    last_bar_wr90_low: Optional[float]
    horizon: int
    trend_threshold: float
    adverse_limit: float
    long_target_threshold: Optional[float]
    short_target_threshold: Optional[float]
    long_adverse_limit: Optional[float]
    short_adverse_limit: Optional[float]
    test_start_date: Optional[str]
    test_size: float
    max_samples: Optional[int]
    optimize: bool
    optimize_prob: bool
    two_branch: bool
    two_branch_stage: str   # "both" | "stage2" | "stage1"
    max_flat_ratio: float
    classifier: str
    stage1_min_prob: float
    stage2_min_prob: float
    stage2_min_prob_up: Optional[float]
    stage2_min_prob_down: Optional[float]
    stage1_min_prob_1m: Optional[float]
    stage1_min_prob_15m: Optional[float]
    stage2_min_prob_1m: Optional[float]
    stage2_min_prob_15m: Optional[float]
    use_state_features: bool
    use_15m_wick_features: bool
    wick_feature_min_range: float
    wick_feature_min_pct: float
    wick_feature_min_volume: float
    use_stage1_day_ohl_utc2: bool
    state_oof_splits: int
    pred_history_len: int
    allow_overlap_backtest: bool
    reverse_exit_prob: float
    max_hold_minutes: Optional[float]
    weak_periods_json: Optional[str]
    eval_mode: str
    wf_init_train_months: int
    wf_retrain_days: int
    wf_max_train_days: int
    wf_min_train_samples: int
    wf_disable_sweep: bool
    wf_sweep_flat_ratios: list[float]
    wf_sweep_stage1_probs: list[float]
    wf_sweep_stage2_probs: list[float]
    wf_sweep_stage2_long_probs: list[float]
    wf_sweep_stage2_short_probs: list[float]
    wf_sweep_val_ratio: float
    wf_sweep_min_val_samples: int
    wf_anchor_mode: str
    wf_cycle_model_dir: Optional[str]
    wf_save_cycle_models: bool
    prep_cache_dir: Optional[str]
    refresh_prep_cache: bool
    random_state: int
    model_in: Optional[str]
    model_out: str
    report_out: str
    trades_out: Optional[str]


@dataclass
class WalkForwardWindow:
    cycle_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    window_mode: str
    train_start_day: Optional[pd.Timestamp] = None
    train_end_day: Optional[pd.Timestamp] = None
    test_start_day: Optional[pd.Timestamp] = None
    test_end_day: Optional[pd.Timestamp] = None


def _resolve_directional_targets(
    trend_threshold: float,
    long_target_threshold: Optional[float] = None,
    short_target_threshold: Optional[float] = None,
) -> tuple[float, float]:
    long_thr = float(trend_threshold if long_target_threshold is None else long_target_threshold)
    short_thr = float(trend_threshold if short_target_threshold is None else short_target_threshold)
    return long_thr, short_thr


def _resolve_directional_stops(
    adverse_limit: float,
    long_adverse_limit: Optional[float] = None,
    short_adverse_limit: Optional[float] = None,
) -> tuple[float, float]:
    long_stop = float(adverse_limit if long_adverse_limit is None else long_adverse_limit)
    short_stop = float(adverse_limit if short_adverse_limit is None else short_adverse_limit)
    return long_stop, short_stop


def _resolve_stage2_directional_probs(
    stage2_min_prob: float,
    stage2_min_prob_up: Optional[float],
    stage2_min_prob_down: Optional[float],
) -> tuple[float, float]:
    up_gate = float(stage2_min_prob if stage2_min_prob_up is None else stage2_min_prob_up)
    down_gate = float(stage2_min_prob if stage2_min_prob_down is None else stage2_min_prob_down)
    return up_gate, down_gate


def _public_config_dict(cfg: Config) -> dict[str, object]:
    return asdict(cfg)


def _execution_semantics() -> dict[str, object]:
    return {
        "signal_reference": SIGNAL_REFERENCE_MODE,
        "entry_execution": ENTRY_EXECUTION_MODE,
        "signal_price_field": "curr_close",
        "entry_price_field": "next_bar_open",
        "signal_time_field": "ts",
        "entry_time_field": "entry_time",
        "notes": (
            "Labels and features stay anchored to the signal bar close, while simulated "
            "trade entries execute on the next bar open."
        ),
    }


def _trading_day_label(ts: pd.Timestamp) -> pd.Timestamp:
    day = (pd.Timestamp(ts).tz_convert(NY_TZ) - pd.Timedelta(hours=TRADING_DAY_CUTOFF_HOUR_NY)).floor("D")
    return pd.Timestamp(day)


def _trading_day_labels(ts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex((ts.tz_convert(NY_TZ) - pd.Timedelta(hours=TRADING_DAY_CUTOFF_HOUR_NY)).floor("D"))


STAGE1_DAY_OHL_OFFSET_HOURS = 2
STAGE1_DAY_OHL_FEATURE_NAMES = ["Dopen_utc2_rel", "Dhigh_utc2_rel", "Dlow_utc2_rel"]


def _build_stage1_day_ohl_features(
    bars: pd.DataFrame,
    sample_ts: pd.DatetimeIndex,
    curr_close: np.ndarray,
) -> np.ndarray:
    """Build causal UTC+2 day-open/high/low features for stage 1 only.

    Day boundary is fixed at 00:00 UTC+2. The first bar after that boundary is
    the day's Dopen. DHigh/DLow are cumulative intraday high/low up to the
    current signal bar, so they do not use future information.

    Features are normalised by DRange = (DHigh - DLow) so each value expresses
    how far the price level is from the current bar's close, scaled by today's
    intraday volatility:
      Dopen_utc2_rel = (Dopen - curr_close) / DRange
      Dhigh_utc2_rel = (Dhigh - curr_close) / DRange   (upside to day high)
      Dlow_utc2_rel  = (Dlow  - curr_close) / DRange   (downside to day low)
    """
    if len(sample_ts) != len(curr_close):
        raise ValueError("stage1 day OHLC feature inputs must have matching lengths")

    session_day = (pd.DatetimeIndex(bars.index) + pd.Timedelta(hours=STAGE1_DAY_OHL_OFFSET_HOURS)).floor("D")
    day_open = bars["open"].groupby(session_day).transform("first")
    day_high = bars["high"].groupby(session_day).cummax()
    day_low = bars["low"].groupby(session_day).cummin()
    day_frame = pd.DataFrame(
        {
            "Dopen": day_open,
            "Dhigh": day_high,
            "Dlow": day_low,
        },
        index=bars.index,
    )
    aligned = day_frame.reindex(sample_ts)
    if aligned.isna().any().any():
        raise ValueError("Could not align UTC+2 stage1 day OHLC features to sample timestamps")

    ref = np.asarray(curr_close, dtype=np.float64)
    drange = (aligned["Dhigh"].to_numpy(dtype=np.float64) - aligned["Dlow"].to_numpy(dtype=np.float64))
    drange_safe = np.where(np.abs(drange) < 1e-9, 1.0, drange)
    return np.column_stack([
        (aligned["Dopen"].to_numpy(dtype=np.float64) - ref) / drange_safe,
        (aligned["Dhigh"].to_numpy(dtype=np.float64) - ref) / drange_safe,
        (aligned["Dlow"].to_numpy(dtype=np.float64) - ref) / drange_safe,
    ]).astype(np.float64)


def _augment_stage1_input(X: np.ndarray, stage1_extra: Optional[np.ndarray]) -> np.ndarray:
    if stage1_extra is None:
        return X
    extra = np.asarray(stage1_extra, dtype=np.float64)
    if extra.ndim == 1:
        extra = extra.reshape(-1, 1)
    if extra.shape[0] != X.shape[0]:
        raise ValueError(
            "Stage1 extra feature row count mismatch: "
            f"X={X.shape[0]}, extra={extra.shape[0]}"
        )
    if extra.shape[1] == 0:
        return X
    return np.hstack([X, extra])


def _trading_day_iso(day: Optional[pd.Timestamp]) -> Optional[str]:
    return None if day is None else pd.Timestamp(day).date().isoformat()


def _resolve_split_index(
    idx: pd.DatetimeIndex,
    test_size: float,
    test_start_date: Optional[str],
) -> tuple[int, str]:
    """Resolve chronological split index.

    Prefers fixed UTC test_start_date when provided; falls back to test_size.
    """
    if test_start_date:
        cutoff = pd.Timestamp(test_start_date, tz="UTC")
        split = int(idx.searchsorted(cutoff, side="left"))
        if 100 <= split < (len(idx) - 50):
            return split, f"date:{cutoff.isoformat()}"
    split = int(len(idx) * (1.0 - test_size))
    return split, f"ratio:{test_size}"


def build_walkforward_windows(
    ts: pd.DatetimeIndex,
    init_train_months: int,
    retrain_days: int,
    max_train_days: int,
    min_train_samples: int,
    anchor_mode: str = "elapsed_days",
) -> list[WalkForwardWindow]:
    """Build chronological walk-forward windows.

    - Start with `init_train_months` of training history.
    - Retrain every `retrain_days`.
    - Expand train start from dataset start until train span exceeds
      `max_train_days`, then switch to rolling latest `max_train_days`.
    """
    if len(ts) < max(min_train_samples + 1, 2):
        raise ValueError("Not enough samples for walk-forward evaluation.")
    if init_train_months < 1:
        raise ValueError("--wf-init-train-months must be >= 1")
    if retrain_days < 1:
        raise ValueError("--wf-retrain-days must be >= 1")
    if max_train_days < 30:
        raise ValueError("--wf-max-train-days must be >= 30")

    sample_days = _trading_day_labels(ts)

    def _align_anchor(x: pd.Timestamp) -> pd.Timestamp:
        if anchor_mode != "weekend_fri_close":
            return x
        # Friday-close policy: retrain boundary is Friday 17:00 New York time
        # (DST-aware), matching common FX/CFD trading-day close.
        x_ny = x.tz_convert(NY_TZ)
        d0 = x_ny.floor("D")
        add_days = (4 - d0.weekday()) % 7  # Friday=4
        out_ny = d0 + pd.Timedelta(days=int(add_days), hours=TRADING_DAY_CUTOFF_HOUR_NY)
        if out_ny < x_ny:
            out_ny += pd.Timedelta(days=7)
        return out_ny.tz_convert("UTC")

    first_cutoff = _align_anchor(ts[0] + pd.DateOffset(months=init_train_months))
    train_end = int(ts.searchsorted(first_cutoff, side="left"))
    train_end = max(train_end, int(min_train_samples))
    if train_end >= len(ts):
        raise ValueError("Initial walk-forward train window consumes all samples.")
    current_anchor = first_cutoff

    windows: list[WalkForwardWindow] = []
    cycle_id = 1
    while train_end < len(ts):
        train_span_days = (ts[train_end - 1] - ts[0]) / pd.Timedelta(days=1)
        if float(train_span_days) > float(max_train_days):
            rolling_start_ts = ts[train_end - 1] - pd.Timedelta(days=max_train_days)
            train_start = int(ts.searchsorted(rolling_start_ts, side="left"))
            mode = "rolling_1y"
        else:
            train_start = 0
            mode = "expanding"

        if anchor_mode == "weekend_fri_close":
            train_end_day = pd.Timestamp(_trading_day_label(current_anchor))
            test_start_day = pd.Timestamp(train_end_day + pd.Timedelta(days=3))
            next_anchor = _align_anchor(current_anchor + pd.Timedelta(days=retrain_days))
            test_end_day = pd.Timestamp(_trading_day_label(next_anchor))
            test_start = int(sample_days.searchsorted(test_start_day, side="left"))
            test_end = int(sample_days.searchsorted(test_end_day, side="right"))
        else:
            test_start = train_end
            next_anchor = ts[test_start] + pd.Timedelta(days=retrain_days)
            test_end = int(ts.searchsorted(next_anchor, side="left"))
            train_end_day = _trading_day_label(ts[train_end - 1])
            test_start_day = _trading_day_label(ts[test_start])
            test_end_idx = min(len(ts) - 1, max(test_start, test_end - 1))
            test_end_day = _trading_day_label(ts[test_end_idx])
        if test_start >= len(ts):
            break
        if test_end <= test_start:
            test_end = min(len(ts), test_start + 1)
            test_end_day = _trading_day_label(ts[test_end - 1])

        windows.append(
            WalkForwardWindow(
                cycle_id=cycle_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                window_mode=mode,
                train_start_day=_trading_day_label(ts[train_start]),
                train_end_day=train_end_day,
                test_start_day=test_start_day,
                test_end_day=test_end_day,
            )
        )
        cycle_id += 1
        train_end = test_end
        current_anchor = next_anchor

    return windows


# ── Data preparation ──────────────────────────────────────────────────────────

def _prepare_ohlcv(raw: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    idx = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
    frame = pd.DataFrame(
        {
            "open":   raw["openPrice"].astype(float).to_numpy(),
            "high":   raw["highPrice"].astype(float).to_numpy(),
            "low":    raw["lowPrice"].astype(float).to_numpy(),
            "close":  raw["closePrice"].astype(float).to_numpy(),
            "volume": raw["lastTradedVolume"].fillna(0.0).astype(float).to_numpy(),
        },
        index=idx,
    ).sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = frame.resample(timeframe, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"]).copy()


def _prepare_ask_bid_ohlcv(raw: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample ask and bid OHLCV from raw data.  Returns (df_ask, df_bid)."""
    idx = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}

    def _build(suffix: str) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "open":  raw[f"openPrice_{suffix}"].astype(float).to_numpy(),
                "high":  raw[f"highPrice_{suffix}"].astype(float).to_numpy(),
                "low":   raw[f"lowPrice_{suffix}"].astype(float).to_numpy(),
                "close": raw[f"closePrice_{suffix}"].astype(float).to_numpy(),
            },
            index=idx,
        ).sort_index()
        out = frame.resample(timeframe, label="left", closed="left").agg(agg)
        return out.dropna(subset=["open", "high", "low", "close"]).copy()

    return _build("ask"), _build("bid")


def _build_spread_price_arrays(
    ts_a: pd.DatetimeIndex,
    entry_ts_a: pd.DatetimeIndex,
    fut_ts_a: pd.DatetimeIndex,
    bars_ask: pd.DataFrame,
    bars_bid: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (entry_ask, entry_bid, curr_ask, curr_bid, fut_ask, fut_bid).

    - entry_ask/bid : ask/bid open of the entry bar (next bar after signal)
    - curr_ask/bid  : ask/bid close of the signal bar (used for stop/target monitoring)
    - fut_ask/bid   : ask/bid close of the horizon bar (used for planned/timeout exit)
    """
    def _align_open(df: pd.DataFrame, ts: pd.DatetimeIndex) -> np.ndarray:
        idx = df.index.get_indexer(ts, method="nearest")
        return df["open"].to_numpy(dtype=np.float64)[idx]

    def _align_close(df: pd.DataFrame, ts: pd.DatetimeIndex) -> np.ndarray:
        idx = df.index.get_indexer(ts, method="nearest")
        return df["close"].to_numpy(dtype=np.float64)[idx]

    entry_ask = _align_open(bars_ask, entry_ts_a)
    entry_bid = _align_open(bars_bid, entry_ts_a)
    curr_ask  = _align_close(bars_ask, ts_a)
    curr_bid  = _align_close(bars_bid, ts_a)
    fut_ask   = _align_close(bars_ask, fut_ts_a)
    fut_bid   = _align_close(bars_bid, fut_ts_a)
    return entry_ask, entry_bid, curr_ask, curr_bid, fut_ask, fut_bid


def _resolve_prep_cache_dir(raw: Optional[str]) -> Optional[Path]:
    if raw is None or not str(raw).strip():
        return None
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _cache_key(payload: dict[str, object]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _bars_cache_key_payload(cfg: Config) -> dict[str, object]:
    return {
        "schema": PREP_CACHE_SCHEMA_VERSION,
        "kind": "bars",
        "table": cfg.table,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "timeframe": cfg.timeframe,
    }


def _dataset_cache_key_payload(
    cfg: Config,
    min_window_range: float,
    threshold: float,
    long_target_threshold: float,
    short_target_threshold: float,
    long_adverse_limit: float,
    short_adverse_limit: float,
) -> dict[str, object]:
    use_15m_wick_features = bool(getattr(cfg, "use_15m_wick_features", False))
    wick_feature_min_range = float(getattr(cfg, "wick_feature_min_range", 40.0))
    wick_feature_min_pct = float(getattr(cfg, "wick_feature_min_pct", 35.0))
    wick_feature_min_volume = float(getattr(cfg, "wick_feature_min_volume", 3000.0))
    return {
        "schema": PREP_CACHE_SCHEMA_VERSION,
        "kind": "supervised_dataset",
        "table": cfg.table,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "timeframe": cfg.timeframe,
        "disable_time_filter": bool(cfg.disable_time_filter),
        "window": int(cfg.window),
        "window_15m": int(cfg.window_15m),
        "use_15m_wick_features": use_15m_wick_features,
        "wick_feature_min_range": wick_feature_min_range,
        "wick_feature_min_pct": wick_feature_min_pct,
        "wick_feature_min_volume": wick_feature_min_volume,
        "min_window_range": float(min_window_range),
        "min_15m_drop": float(cfg.min_15m_drop),
        "min_15m_rise": float(cfg.min_15m_rise),
        "last_bar_wr90_high": None if cfg.last_bar_wr90_high is None else float(cfg.last_bar_wr90_high),
        "last_bar_wr90_low": None if cfg.last_bar_wr90_low is None else float(cfg.last_bar_wr90_low),
        "horizon": int(cfg.horizon),
        "trend_threshold": float(threshold),
        "adverse_limit": float(cfg.adverse_limit),
        "long_target_threshold": float(long_target_threshold),
        "short_target_threshold": float(short_target_threshold),
        "long_adverse_limit": float(long_adverse_limit),
        "short_adverse_limit": float(short_adverse_limit),
    }


def _load_or_build_bars(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return (bars_mid, bars_ask, bars_bid, cache_info)."""
    cache_root = _resolve_prep_cache_dir(cfg.prep_cache_dir)
    if cache_root is None:
        raw = DataLoader().load_data(cfg.table, start_date=cfg.start_date, end_date=cfg.end_date)
        bars = _prepare_ohlcv(raw, cfg.timeframe)
        bars_ask, bars_bid = _prepare_ask_bid_ohlcv(raw, cfg.timeframe)
        return bars, bars_ask, bars_bid, {"enabled": False, "kind": "bars"}

    key = _cache_key(_bars_cache_key_payload(cfg))
    cache_path = cache_root / "bars" / f"{key}.joblib"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "enabled": True,
        "kind": "bars",
        "key": key,
        "path": str(cache_path),
        "hit": False,
    }
    if cache_path.exists() and not cfg.refresh_prep_cache:
        payload = joblib.load(cache_path)
        # Invalidate old caches that don't have bars_ask/bars_bid.
        if "bars_ask" in payload and "bars_bid" in payload:
            status["hit"] = True
            _log(f"Prep cache HIT (bars): {cache_path.relative_to(PROJECT_ROOT)}")
            return payload["bars"], payload["bars_ask"], payload["bars_bid"], status
        _log(f"Prep cache stale (no ask/bid bars): rebuilding {cache_path.relative_to(PROJECT_ROOT)}")

    raw = DataLoader().load_data(cfg.table, start_date=cfg.start_date, end_date=cfg.end_date)
    bars = _prepare_ohlcv(raw, cfg.timeframe)
    bars_ask, bars_bid = _prepare_ask_bid_ohlcv(raw, cfg.timeframe)
    joblib.dump(
        {
            "schema": PREP_CACHE_SCHEMA_VERSION,
            "kind": "bars",
            "cache_key": key,
            "bars": bars,
            "bars_ask": bars_ask,
            "bars_bid": bars_bid,
        },
        cache_path,
        compress=3,
    )
    _log(f"Prep cache MISS (bars): saved {cache_path.relative_to(PROJECT_ROOT)}")
    return bars, bars_ask, bars_bid, status


def _load_or_build_supervised_dataset(
    bars: pd.DataFrame,
    cfg: Config,
    min_window_range: float,
    threshold: float,
    long_target_threshold: float,
    short_target_threshold: float,
    long_adverse_limit: float,
    short_adverse_limit: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    int,
    int,
    dict[str, object],
]:
    use_15m_wick_features = bool(getattr(cfg, "use_15m_wick_features", False))
    wick_feature_min_range = float(getattr(cfg, "wick_feature_min_range", 40.0))
    wick_feature_min_pct = float(getattr(cfg, "wick_feature_min_pct", 35.0))
    wick_feature_min_volume = float(getattr(cfg, "wick_feature_min_volume", 3000.0))
    cache_root = _resolve_prep_cache_dir(cfg.prep_cache_dir)
    if cache_root is None:
        X_tensor, y, ts, curr_a, entry_a, fut_a, entry_ts_a, fut_ts_a, kept, skipped = build_dataset(
            bars,
            cfg.window,
            min_window_range,
            cfg.horizon,
            threshold,
            cfg.adverse_limit,
            long_target_threshold=long_target_threshold,
            short_target_threshold=short_target_threshold,
            long_adverse_limit=long_adverse_limit,
            short_adverse_limit=short_adverse_limit,
            min_15m_drop=cfg.min_15m_drop,
            min_15m_rise=cfg.min_15m_rise,
            last_bar_wr90_high=cfg.last_bar_wr90_high,
            last_bar_wr90_low=cfg.last_bar_wr90_low,
            window_15m=cfg.window_15m,
            apply_time_filter=not cfg.disable_time_filter,
            use_15m_wick_features=use_15m_wick_features,
            wick_feature_min_range=wick_feature_min_range,
            wick_feature_min_pct=wick_feature_min_pct,
            wick_feature_min_volume=wick_feature_min_volume,
        )
        return X_tensor, y, ts, curr_a, entry_a, fut_a, entry_ts_a, fut_ts_a, kept, skipped, {
            "enabled": False,
            "kind": "supervised_dataset",
        }

    key = _cache_key(
        _dataset_cache_key_payload(
            cfg,
            min_window_range,
            threshold,
            long_target_threshold,
            short_target_threshold,
            long_adverse_limit,
            short_adverse_limit,
        )
    )
    cache_path = cache_root / "datasets" / f"{key}.joblib"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "enabled": True,
        "kind": "supervised_dataset",
        "key": key,
        "path": str(cache_path),
        "hit": False,
    }
    if cache_path.exists() and not cfg.refresh_prep_cache:
        payload = joblib.load(cache_path)
        status["hit"] = True
        _log(f"Prep cache HIT (dataset): {cache_path.relative_to(PROJECT_ROOT)}")
        data = payload["data"]
        return (
            data["X_tensor"],
            data["y"],
            data["ts"],
            data["curr_a"],
            data["entry_a"],
            data["fut_a"],
            data["entry_ts_a"],
            data["fut_ts_a"],
            int(data["kept"]),
            int(data["skipped"]),
            status,
        )

    X_tensor, y, ts, curr_a, entry_a, fut_a, entry_ts_a, fut_ts_a, kept, skipped = build_dataset(
        bars,
        cfg.window,
        min_window_range,
        cfg.horizon,
        threshold,
        cfg.adverse_limit,
        long_target_threshold=long_target_threshold,
        short_target_threshold=short_target_threshold,
        long_adverse_limit=long_adverse_limit,
        short_adverse_limit=short_adverse_limit,
        min_15m_drop=cfg.min_15m_drop,
        min_15m_rise=cfg.min_15m_rise,
        last_bar_wr90_high=cfg.last_bar_wr90_high,
        last_bar_wr90_low=cfg.last_bar_wr90_low,
        window_15m=cfg.window_15m,
        apply_time_filter=not cfg.disable_time_filter,
        use_15m_wick_features=use_15m_wick_features,
        wick_feature_min_range=wick_feature_min_range,
        wick_feature_min_pct=wick_feature_min_pct,
        wick_feature_min_volume=wick_feature_min_volume,
    )
    joblib.dump(
        {
            "schema": PREP_CACHE_SCHEMA_VERSION,
            "kind": "supervised_dataset",
            "cache_key": key,
            "data": {
                "X_tensor": X_tensor,
                "y": y,
                "ts": ts,
                "curr_a": curr_a,
                "entry_a": entry_a,
                "fut_a": fut_a,
                "entry_ts_a": entry_ts_a,
                "fut_ts_a": fut_ts_a,
                "kept": int(kept),
                "skipped": int(skipped),
            },
        },
        cache_path,
        compress=3,
    )
    _log(f"Prep cache MISS (dataset): saved {cache_path.relative_to(PROJECT_ROOT)}")
    return X_tensor, y, ts, curr_a, entry_a, fut_a, entry_ts_a, fut_ts_a, kept, skipped, status


# ── Image construction ────────────────────────────────────────────────────────

def _constant_feature_channels(width: int, values: list[float]) -> list[np.ndarray]:
    return [np.full(width, float(v), dtype=np.float64) for v in values]


def _compute_long_wick_flags_from_ohlcv(
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    volume: float,
    *,
    min_range: float,
    min_wick_pct: float,
    min_volume: float,
) -> tuple[float, float]:
    range_val = float(high_px) - float(low_px)
    if range_val <= 0.0:
        return 0.0, 0.0
    lower_wick = min(float(open_px), float(close_px)) - float(low_px)
    upper_wick = float(high_px) - max(float(open_px), float(close_px))
    lower_wick_pct = (lower_wick / range_val) * 100.0
    upper_wick_pct = (upper_wick / range_val) * 100.0
    range_ok = range_val >= float(min_range)
    volume_ok = float(volume) > float(min_volume)
    lower_flag = 1.0 if (lower_wick_pct > float(min_wick_pct) and range_ok and volume_ok) else 0.0
    upper_flag = 1.0 if (upper_wick_pct > float(min_wick_pct) and range_ok and volume_ok) else 0.0
    return lower_flag, upper_flag


def _last_completed_15m_wick_flags(
    w: pd.DataFrame,
    *,
    min_range: float,
    min_wick_pct: float,
    min_volume: float,
) -> tuple[float, float]:
    if len(w) < 15:
        return 0.0, 0.0
    bars_15m = _resample_15m(w)
    if bars_15m.empty:
        return 0.0, 0.0
    signal_ts = pd.Timestamp(w.index[-1])
    cutoff = (signal_ts - pd.Timedelta(minutes=15)).floor("15min")
    eligible = bars_15m.loc[bars_15m.index <= cutoff]
    if eligible.empty:
        return 0.0, 0.0
    last_bar = eligible.iloc[-1]
    return _compute_long_wick_flags_from_ohlcv(
        float(last_bar["open"]),
        float(last_bar["high"]),
        float(last_bar["low"]),
        float(last_bar["close"]),
        float(last_bar.get("volume", 0.0)),
        min_range=min_range,
        min_wick_pct=min_wick_pct,
        min_volume=min_volume,
    )


def _window_to_image(w: pd.DataFrame, *, extra_channels: Optional[list[np.ndarray]] = None) -> np.ndarray:
    """Encode a candle window as an image-like tensor [channels, width].

    Base channels:
      0 open_rel       relative open  to first close
      1 high_rel       relative high  to first close
      2 low_rel        relative low   to first close
      3 close_rel      relative close to first close
      4 body_rel       (close - open) / first close
      5 range_rel      (high  - low)  / first close
      6 vol_z          volume z-score within window
      7 vol_rel        volume relative to first bar in window
      8 vol_diff_norm  normalised bar-to-bar volume change
    """
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

    channels: list[np.ndarray] = [
        open_rel,
        high_rel,
        low_rel,
        close_rel,
        body_rel,
        range_rel,
        vol_z,
        vol_rel,
        vol_diff_norm,
    ]
    if extra_channels:
        channels.extend(np.asarray(ch, dtype=np.float64) for ch in extra_channels)
    return np.stack(channels, axis=0).astype(np.float32)


# ── Label ─────────────────────────────────────────────────────────────────────

def _label(curr_close: float, future_high: float, future_low: float,
           threshold: float, adverse_limit: float,
           long_target_threshold: Optional[float] = None,
           short_target_threshold: Optional[float] = None,
           long_adverse_limit: Optional[float] = None,
           short_adverse_limit: Optional[float] = None) -> int:
    """
    up     (2): target hit without violating long stop
    down   (0): target hit without violating short stop
    risky  (3): volatile/ambiguous path (target+stop hit for a side, or both targets hit)
    flat   (1): otherwise
    """
    up_move      = (future_high - curr_close) / curr_close
    up_adverse   =  curr_close  - future_low
    down_move    = (curr_close  - future_low) / curr_close
    down_adverse =  future_high - curr_close

    long_thr, short_thr = _resolve_directional_targets(threshold, long_target_threshold, short_target_threshold)
    long_stop, short_stop = _resolve_directional_stops(adverse_limit, long_adverse_limit, short_adverse_limit)

    long_target_hit = up_move > long_thr
    short_target_hit = down_move > short_thr
    long_stop_hit = up_adverse >= long_stop
    short_stop_hit = down_adverse >= short_stop

    # Mark windows as risky when targets and stops conflict or both directional
    # targets are touched within the same horizon.
    risky = (
        (long_target_hit and long_stop_hit)
        or (short_target_hit and short_stop_hit)
        or (long_target_hit and short_target_hit)
    )
    if risky:
        return LABEL_RISKY

    up_ok = long_target_hit and not long_stop_hit
    down_ok = short_target_hit and not short_stop_hit

    if up_ok and down_ok:
        return LABEL_RISKY
    if up_ok:
        return LABEL_UP
    if down_ok:
        return LABEL_DOWN
    return LABEL_FLAT


# ── 15-min bar helpers ────────────────────────────────────────────────────────

def _resample_15m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a bar DataFrame to 15-minute OHLCV bars (no lookahead)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample("15min", label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"]).copy()


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    df: pd.DataFrame,
    window: int,
    min_window_range: float,
    horizon: int,
    threshold: float,
    adverse_limit: float,
    long_target_threshold: Optional[float] = None,
    short_target_threshold: Optional[float] = None,
    long_adverse_limit: Optional[float] = None,
    short_adverse_limit: Optional[float] = None,
    min_15m_drop: float = 0.0,
    min_15m_rise: float = 0.0,
    last_bar_wr90_high: Optional[float] = None,
    last_bar_wr90_low: Optional[float] = None,
    window_15m: int = 40,
    apply_time_filter: bool = True,
    use_15m_wick_features: bool = False,
    wick_feature_min_range: float = 40.0,
    wick_feature_min_pct: float = 35.0,
    wick_feature_min_volume: float = 3000.0,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    int,
    int,
]:
    tensors:   list[np.ndarray]   = []
    labels:    list[int]          = []
    ts:        list[pd.Timestamp] = []
    curr_list: list[float]        = []
    entry_list:list[float]        = []
    fut_list:  list[float]        = []
    entry_ts:  list[pd.Timestamp] = []
    fut_ts:    list[pd.Timestamp] = []
    skipped = 0
    total_iters = len(df) - horizon - (window - 1)
    t_ds = time.time()
    report_every = max(1, total_iters // 20)   # ~5% increments

    # ── Pre-build 15-min bars for no-lookahead 15-min image feature ──────────
    _use_15m_img = window_15m > 0
    if _use_15m_img:
        bars_15m = _resample_15m(df)
        ts_15m_i64: np.ndarray = bars_15m.index.asi8  # int64 nanoseconds
        o15 = bars_15m["open"].to_numpy(dtype=np.float64)
        h15 = bars_15m["high"].to_numpy(dtype=np.float64)
        l15 = bars_15m["low"].to_numpy(dtype=np.float64)
        c15 = bars_15m["close"].to_numpy(dtype=np.float64)
        v15 = bars_15m["volume"].to_numpy(dtype=np.float64)
        # For each 1-min bar at position i, find the last 15-min bar that is
        # fully complete by signal_ts.  A 15-min bar labelled T covers
        # [T, T+15min); it is complete only once we reach T+15min.
        # Therefore the cutoff is:  T_start <= signal_ts - 15min
        _15m_ns = int(15 * 60 * 1e9)  # 15 minutes in nanoseconds
        cutoff_i64 = df.index.asi8 - _15m_ns
        # all_idx_15m[i] = index of last complete 15-min bar for 1-min bar i
        all_idx_15m: np.ndarray = np.searchsorted(ts_15m_i64, cutoff_i64, side="right") - 1
        print(
            f"  [{_now()}] 15-min image: resampled {len(bars_15m):,} bars, "
            f"window_15m={window_15m}",
            flush=True,
        )
    else:
        all_idx_15m = np.empty(0, dtype=np.int64)  # unused placeholder

    for idx_i, i in enumerate(range(window - 1, len(df) - horizon)):
        signal_ts = df.index[i]

        if idx_i > 0 and idx_i % report_every == 0:
            pct = idx_i / total_iters * 100
            kept_so_far = len(tensors)
            elapsed = _elapsed(t_ds)
            eta_s = (time.time() - t_ds) / idx_i * (total_iters - idx_i)
            eta = f"{eta_s:.0f}s" if eta_s < 60 else f"{eta_s/60:.1f}m"
            print(
                f"  [{_now()}] {pct:>5.1f}%  bars={idx_i:>7,}/{total_iters:,}"
                f"  kept={kept_so_far:,}  skipped={skipped:,}"
                f"  elapsed={elapsed}  eta≈{eta}",
                flush=True,
            )

        # ── time filter ──────────────────────────────────────────────────────
        if apply_time_filter and _is_blocked(signal_ts):
            skipped += 1
            continue

        # ── range filter ─────────────────────────────────────────────────────────
        w = df.iloc[i - window + 1 : i + 1]
        #print('window: ', window)
        if float(w["high"].max() - w["low"].min()) <= min_window_range:
            skipped += 1
            continue

        # ── 15-min directional move filter ───────────────────────────────────
        # Keep windows that contain at least one 15-bar interval with:
        #   - high-low swing >= min_15m_drop (legacy behavior), OR
        #   - upside excursion from interval start close >= min_15m_rise.
        if (min_15m_drop > 0 or min_15m_rise > 0) and len(w) >= 15:
            h = w["high"].to_numpy()
            l = w["low"].to_numpy()
            c = w["close"].to_numpy()
            roll_h = np.lib.stride_tricks.sliding_window_view(h, 15).max(axis=1)
            roll_l = np.lib.stride_tricks.sliding_window_view(l, 15).min(axis=1)
            roll_c0 = np.lib.stride_tricks.sliding_window_view(c, 15)[:, 0]
            drop_ok = False if min_15m_drop <= 0 else ((roll_h - roll_l) >= min_15m_drop).any()
            rise_ok = False if min_15m_rise <= 0 else ((roll_h - roll_c0) >= min_15m_rise).any()
            if not (drop_ok or rise_ok):
                skipped += 1
                continue

        # ── last-bar WR90 extreme filter ──────────────────────────────────────
        if last_bar_wr90_high is not None or last_bar_wr90_low is not None:
            if len(w) < 90:
                skipped += 1
                continue
            wr_src = w.iloc[-90:]
            wr_high = float(wr_src["high"].max())
            wr_low = float(wr_src["low"].min())
            wr_close = float(wr_src["close"].iloc[-1])
            wr_span = wr_high - wr_low
            if wr_span <= 0.0:
                skipped += 1
                continue
            wr90_last = -100.0 * ((wr_high - wr_close) / wr_span)
            high_ok = False if last_bar_wr90_high is None else (wr90_last >= float(last_bar_wr90_high))
            low_ok = False if last_bar_wr90_low is None else (wr90_last <= float(last_bar_wr90_low))
            if not (high_ok or low_ok):
                skipped += 1
                continue

        curr_close   = float(df["close"].iloc[i])
        entry_open   = float(df["open"].iloc[i + 1])
        next_bar_ts  = df.index[i + 1]
        fut_w        = df.iloc[i + 1 : i + horizon + 1]
        future_high  = float(fut_w["high"].max())
        future_low   = float(fut_w["low"].min())
        future_close = float(df["close"].iloc[i + horizon])
        future_ts    = df.index[i + horizon]

        # ── 15-min image (no look-forward) ───────────────────────────────────
        # Must be checked before we accept this sample, so skipped count stays
        # consistent. A 15-min bar labelled T is only included once T+15min has
        # passed (all_idx_15m already encodes this cutoff).
        if _use_15m_img:
            j15 = int(all_idx_15m[i])
            if j15 < window_15m - 1:
                # Not enough completed 15-min history yet – skip this sample.
                skipped += 1
                continue
            s15 = j15 - window_15m + 1
            w_15m_df = pd.DataFrame(
                {
                    "open":   o15[s15 : j15 + 1],
                    "high":   h15[s15 : j15 + 1],
                    "low":    l15[s15 : j15 + 1],
                    "close":  c15[s15 : j15 + 1],
                    "volume": v15[s15 : j15 + 1],
                }
            )
            wick_flags = [0.0, 0.0]
            if use_15m_wick_features:
                wick_flags = list(_last_completed_15m_wick_flags(
                    w,
                    min_range=wick_feature_min_range,
                    min_wick_pct=wick_feature_min_pct,
                    min_volume=wick_feature_min_volume,
                ))
            img_1m = _window_to_image(
                w,
                extra_channels=_constant_feature_channels(len(w), wick_flags) if use_15m_wick_features else None,
            )
            img_15m = _window_to_image(
                w_15m_df,
                extra_channels=_constant_feature_channels(len(w_15m_df), wick_flags) if use_15m_wick_features else None,
            )
            combined = np.concatenate([img_1m, img_15m], axis=1)
            tensors.append(combined)
        else:
            wick_extra = None
            if use_15m_wick_features:
                wick_extra = _constant_feature_channels(
                    len(w),
                    list(_last_completed_15m_wick_flags(
                        w,
                        min_range=wick_feature_min_range,
                        min_wick_pct=wick_feature_min_pct,
                        min_volume=wick_feature_min_volume,
                    )),
                )
            tensors.append(_window_to_image(w, extra_channels=wick_extra))
        labels.append(
            _label(
                curr_close,
                future_high,
                future_low,
                threshold,
                adverse_limit,
                long_target_threshold=long_target_threshold,
                short_target_threshold=short_target_threshold,
                long_adverse_limit=long_adverse_limit,
                short_adverse_limit=short_adverse_limit,
            )
        )
        ts.append(signal_ts)
        curr_list.append(curr_close)
        entry_list.append(entry_open)
        fut_list.append(future_close)
        entry_ts.append(next_bar_ts)
        fut_ts.append(future_ts)

    if not tensors:
        raise ValueError("Dataset is empty – loosen filters or expand date range.")

    print(f"  [{_now()}] Dataset build done — kept={len(tensors):,}  skipped={skipped:,}  took={_elapsed(t_ds)}", flush=True)

    X      = np.stack(tensors, axis=0)
    y      = np.asarray(labels,    dtype=np.int64)
    t      = pd.DatetimeIndex(ts)
    curr_a = np.asarray(curr_list, dtype=np.float64)
    entry_a= np.asarray(entry_list,dtype=np.float64)
    fut_a  = np.asarray(fut_list,  dtype=np.float64)
    entry_t= pd.DatetimeIndex(entry_ts)
    fut_t  = pd.DatetimeIndex(fut_ts)
    return X, y, t, curr_a, entry_a, fut_a, entry_t, fut_t, int(len(y)), skipped


def flatten_tensors(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


BASE_STATE_FEATURE_NAMES = [
    "pos_short",
    "pos_flat",
    "pos_long",
    "bars_in_pos_norm",
    "unrealized_return",
    "status_target_hit",
    "status_stop_hit",
]


def make_state_feature_names(pred_history_len: int) -> list[str]:
    names: list[str] = []
    for lag in range(1, pred_history_len + 1):
        names.extend([
            f"pred_lag{lag}_down",
            f"pred_lag{lag}_flat",
            f"pred_lag{lag}_up",
        ])
    names.extend(BASE_STATE_FEATURE_NAMES)
    return names


def _compute_state_features_from_pred(
    pred: np.ndarray,
    curr: np.ndarray,
    entry_px: np.ndarray,
    adverse_limit: float,
    trend_threshold: float,
    long_target_threshold: Optional[float] = None,
    short_target_threshold: Optional[float] = None,
    long_adverse_limit: Optional[float] = None,
    short_adverse_limit: Optional[float] = None,
    pred_history_len: int = 150,
    max_bars: int = 120,
) -> np.ndarray:
    """Build causal state features from historical predictions and next-bar entry price.

    Features at index i are computed from state available up to bar i only.
    """
    n = len(pred)
    hist = max(0, int(pred_history_len))
    n_hist_feats = hist * 3
    n_total_feats = n_hist_feats + len(BASE_STATE_FEATURE_NAMES)
    out = np.zeros((n, n_total_feats), dtype=np.float64)
    long_thr, short_thr = _resolve_directional_targets(trend_threshold, long_target_threshold, short_target_threshold)
    long_stop, short_stop = _resolve_directional_stops(adverse_limit, long_adverse_limit, short_adverse_limit)

    pos = 0        # -1 short, 0 flat, +1 long
    entry = 0.0
    bars_in_pos = 0
    target_hit_flag = 0.0
    stop_hit_flag = 0.0

    for i in range(n):
        # Encode state available at decision time for this bar.
        for lag in range(1, hist + 1):
            j = i - lag
            if j < 0:
                continue
            cls = int(pred[j])
            if cls == LABEL_RISKY:
                cls = LABEL_FLAT
            if cls in (LABEL_DOWN, LABEL_FLAT, LABEL_UP):
                out[i, (lag - 1) * 3 + cls] = 1.0

        b = n_hist_feats
        out[i, b + (pos + 1)] = 1.0
        out[i, b + 3] = min(float(bars_in_pos), float(max_bars)) / float(max_bars)
        if pos != 0 and entry > 0.0:
            out[i, b + 4] = ((float(curr[i]) - entry) * float(pos)) / entry
        out[i, b + 5] = target_hit_flag
        out[i, b + 6] = stop_hit_flag

        # One-bar event flags.
        target_hit_flag = 0.0
        stop_hit_flag = 0.0

        # Update active position status with current bar close.
        if pos != 0 and entry > 0.0:
            move_abs = (float(curr[i]) - entry) * float(pos)
            target_abs = abs(entry) * float(long_thr if pos > 0 else short_thr)
            stop_abs = float(long_stop if pos > 0 else short_stop)
            if move_abs >= target_abs:
                pos = 0
                entry = 0.0
                bars_in_pos = 0
                target_hit_flag = 1.0
            elif move_abs <= -stop_abs:
                pos = 0
                entry = 0.0
                bars_in_pos = 0
                stop_hit_flag = 1.0
            else:
                bars_in_pos += 1

        # Open a new simulated position only when currently flat.
        if pos == 0 and int(pred[i]) in TREND_LABELS:
            pos = 1 if int(pred[i]) == LABEL_UP else -1
            entry = float(entry_px[i])
            bars_in_pos = 0

    return out


def _build_oof_state_features(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    curr_tr: np.ndarray,
    entry_tr: np.ndarray,
    cfg: Config,
    n_1m_feats: int,
    two_branch: bool,
    stage1_extra_tr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build train-time state features with time-series OOF predictions."""
    n = len(X_tr)
    out = np.zeros((n, len(make_state_feature_names(cfg.pred_history_len))), dtype=np.float64)
    tscv = TimeSeriesSplit(n_splits=cfg.state_oof_splits)

    for fold_i, (fit_idx, val_idx) in enumerate(tscv.split(X_tr), 1):
        if len(fit_idx) < 200 or len(val_idx) < 50:
            continue
        try:
            m1_f, m2_f = train_two_stage(
                X_tr[fit_idx],
                y_tr[fit_idx],
                cfg.random_state + fold_i,
                cfg.max_flat_ratio,
                cfg.classifier,
                two_branch=two_branch,
                n_1m_feats=n_1m_feats,
                two_branch_stage=cfg.two_branch_stage,
                stage1_extra=None if stage1_extra_tr is None else stage1_extra_tr[fit_idx],
            )
            pred_val = predict_two_stage(
                X_tr[val_idx],
                m1_f,
                m2_f,
                stage1_min_prob=cfg.stage1_min_prob,
                stage2_min_prob=cfg.stage2_min_prob,
                stage2_min_prob_up=cfg.stage2_min_prob_up,
                stage2_min_prob_down=cfg.stage2_min_prob_down,
                stage1_min_prob_1m=cfg.stage1_min_prob_1m,
                stage1_min_prob_15m=cfg.stage1_min_prob_15m,
                stage2_min_prob_1m=cfg.stage2_min_prob_1m,
                stage2_min_prob_15m=cfg.stage2_min_prob_15m,
                stage1_extra=None if stage1_extra_tr is None else stage1_extra_tr[val_idx],
            )
            out[val_idx] = _compute_state_features_from_pred(
                pred_val,
                curr_tr[val_idx],
                entry_tr[val_idx],
                cfg.adverse_limit,
                cfg.trend_threshold,
                long_target_threshold=cfg.long_target_threshold,
                short_target_threshold=cfg.short_target_threshold,
                long_adverse_limit=cfg.long_adverse_limit,
                short_adverse_limit=cfg.short_adverse_limit,
                pred_history_len=cfg.pred_history_len,
            )
        except ValueError:
            continue

    return out


# ── Two-stage model ───────────────────────────────────────────────────────────

def _make_single_model(classifier: str = "gradient_boosting", random_state: int = 42):
    """Return a fresh, unfitted estimator for one stage."""
    if classifier == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=random_state,
            )),
        ])
    # Default: HistGradientBoostingClassifier — fast, handles imbalance,
    # non-linear, no need for feature scaling.
    return HistGradientBoostingClassifier(
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        learning_rate=0.05,
        random_state=random_state,
    )


class TwoBranchClassifier:
    """Two independent GBMs — one on the 1-min image slice, one on the 15-min
    image slice — whose predict_proba outputs are averaged at inference time.

    The input X must be a pre-flattened array with layout:
        [ 1m_features (n_channels * window)  |  15m_features (n_channels * window_15m) ]

    ``n_1m_feats`` is the split point (= n_channels * window).
    """

    def __init__(
        self,
        n_1m_feats: int,
        classifier: str = "gradient_boosting",
        random_state: int = 42,
    ) -> None:
        self.n_1m_feats  = n_1m_feats
        self.classifier  = classifier
        self.random_state = random_state
        self.model_1m_:  object = None
        self.model_15m_: object = None
        self.classes_:   np.ndarray = np.array([0, 1])

    # ------------------------------------------------------------------
    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X[:, :self.n_1m_feats], X[:, self.n_1m_feats:]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "TwoBranchClassifier":
        X_1m, X_15m = self._split(X)
        self.model_1m_  = _make_single_model(self.classifier, self.random_state)
        self.model_15m_ = _make_single_model(self.classifier, self.random_state)
        if isinstance(self.model_1m_, Pipeline):
            self.model_1m_.fit(X_1m, y)
            self.model_15m_.fit(X_15m, y)
        else:
            kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
            self.model_1m_.fit(X_1m, y, **kw)
            self.model_15m_.fit(X_15m, y, **kw)
        self.classes_ = self.model_1m_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_1m, X_15m = self._split(X)
        p1 = self.model_1m_.predict_proba(X_1m)
        p2 = self.model_15m_.predict_proba(X_15m)
        return (p1 + p2) / 2.0

    def predict_proba_branches(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-branch probabilities: (1m_probs, 15m_probs)."""
        X_1m, X_15m = self._split(X)
        return self.model_1m_.predict_proba(X_1m), self.model_15m_.predict_proba(X_15m)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class SlicedClassifier:
    """Single GBM that only uses the first ``n_feats`` columns of X.

    Used when ``--two-branch-stage stage2``: stage 1 sees only the 1-min
    image slice (n_feats = n_channels * window), so it exactly replicates the baseline
    1m-only behaviour while stage 2 uses a TwoBranchClassifier.
    Similarly used for stage 2 when ``--two-branch-stage stage1``.
    """

    def __init__(
        self,
        n_feats: int,
        classifier: str = "gradient_boosting",
        random_state: int = 42,
    ) -> None:
        self.n_feats     = n_feats
        self.classifier  = classifier
        self.random_state = random_state
        self.model_:  object = None
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "SlicedClassifier":
        self.model_ = _make_single_model(self.classifier, self.random_state)
        Xs = X[:, :self.n_feats]
        if isinstance(self.model_, Pipeline):
            self.model_.fit(Xs, y)
        else:
            kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
            self.model_.fit(Xs, y, **kw)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X[:, :self.n_feats])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def train_two_stage(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    max_flat_ratio: float = 4.0,
    classifier: str = "gradient_boosting",
    two_branch: bool = False,
    n_1m_feats: int = 0,
    two_branch_stage: str = "both",
    stage1_extra: Optional[np.ndarray] = None,
) -> tuple:
    """
    Stage 1: flat(0) vs trend(1).
      - Flat class is subsampled to max_flat_ratio × n_trend before fitting
        to prevent the model being overwhelmed by the majority class.
      - Sample weights are passed to balance remaining class imbalance.
    Stage 2: down(0) vs up(1) on trend-only samples.

    When ``two_branch=True`` and ``n_1m_feats > 0`` each affected stage uses a
    TwoBranchClassifier that trains one GBM on the 1-min image slice and a
    second GBM on the 15-min image slice, then averages their probabilities.
    ``two_branch_stage`` controls which stages: 'both' | 'stage2' | 'stage1'.
    """
    rng = np.random.default_rng(random_state)

    use_2b_s1 = two_branch and n_1m_feats > 0 and two_branch_stage in ("both", "stage1")
    use_2b_s2 = two_branch and n_1m_feats > 0 and two_branch_stage in ("both", "stage2")
    uses_sliced_stage1 = two_branch and n_1m_feats > 0 and two_branch_stage == "stage2"

    if stage1_extra is not None and (use_2b_s1 or uses_sliced_stage1):
        raise ValueError("UTC+2 stage1 day OHLC features are not supported with two-branch stage1 modes")

    def _new_s1() -> object:
        if use_2b_s1:
            return TwoBranchClassifier(n_1m_feats, classifier, random_state)
        if two_branch and n_1m_feats > 0 and two_branch_stage == "stage2":
            # In stage2-only mode, keep stage1 identical to baseline by using
            # only the 1-min feature slice.
            return SlicedClassifier(n_1m_feats, classifier, random_state)
        return _make_single_model(classifier, random_state)

    def _new_s2() -> object:
        if use_2b_s2:
            return TwoBranchClassifier(n_1m_feats, classifier, random_state)
        if two_branch and n_1m_feats > 0 and two_branch_stage == "stage1":
            # Symmetric behavior for stage1-only mode.
            return SlicedClassifier(n_1m_feats, classifier, random_state)
        return _make_single_model(classifier, random_state)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    trend_mask = np.isin(y, TREND_LABELS)
    n_trend    = int(trend_mask.sum())
    flat_idx   = np.where(~trend_mask)[0]
    trend_idx  = np.where(trend_mask)[0]

    if n_trend < 10:
        raise ValueError("Too few trend samples to train stage 1.")

    # Subsample flat class
    max_flat   = min(len(flat_idx), int(max_flat_ratio * n_trend))
    keep_flat  = rng.choice(flat_idx, size=max_flat, replace=False)
    s1_idx     = np.sort(np.concatenate([keep_flat, trend_idx]))
    X_s1       = _augment_stage1_input(
        X[s1_idx],
        None if stage1_extra is None else np.asarray(stage1_extra)[s1_idx],
    )
    y_s1       = trend_mask[s1_idx].astype(np.int64)

    sw_s1 = compute_sample_weight("balanced", y_s1)
    m1    = _new_s1()
    if isinstance(m1, Pipeline):
        m1.fit(X_s1, y_s1)
    else:
        m1.fit(X_s1, y_s1, sample_weight=sw_s1)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    if n_trend < 10:
        raise ValueError("Too few trend samples to train stage 2.")
    X_s2  = X[trend_mask]
    y_s2  = (y[trend_mask] == LABEL_UP).astype(np.int64)
    sw_s2 = compute_sample_weight("balanced", y_s2)
    m2    = _new_s2()
    if isinstance(m2, Pipeline):
        m2.fit(X_s2, y_s2)
    else:
        m2.fit(X_s2, y_s2, sample_weight=sw_s2)

    return m1, m2


def predict_two_stage_details(
    X: np.ndarray,
    m1,
    m2,
    stage1_min_prob: float = 0.55,
    stage2_min_prob: float = 0.55,
    stage2_min_prob_up: Optional[float] = None,
    stage2_min_prob_down: Optional[float] = None,
    stage1_min_prob_1m: Optional[float] = None,
    stage1_min_prob_15m: Optional[float] = None,
    stage2_min_prob_1m: Optional[float] = None,
    stage2_min_prob_15m: Optional[float] = None,
    stage1_extra: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Return 3-class predictions plus per-signal confidence metadata.

    - Stage 1 must pass P(trend) >= stage1_min_prob.
    - Stage 2 must pass directional confidence with optional side-specific gates.
    - Otherwise keep class=flat.
    """
    result = np.full(len(X), LABEL_FLAT, dtype=np.int64)
    trend_prob = np.zeros(len(X), dtype=np.float64)
    signal_prob = np.zeros(len(X), dtype=np.float64)
    up_prob_full = np.full(len(X), np.nan, dtype=np.float64)

    if stage1_extra is not None and hasattr(m1, "predict_proba_branches"):
        raise ValueError("UTC+2 stage1 day OHLC features are not supported with multi-branch stage1 inference")

    X_s1 = _augment_stage1_input(X, stage1_extra)
    p1 = m1.predict_proba(X_s1)
    c1 = list(getattr(m1, "classes_", [0, 1]))
    idx_trend = c1.index(1) if 1 in c1 else -1
    if idx_trend < 0:
        return {
            "pred": result,
            "trend_prob": trend_prob,
            "signal_prob": signal_prob,
            "up_prob": up_prob_full,
        }
    trend_prob = p1[:, idx_trend]
    trend_mask = trend_prob >= stage1_min_prob

    if hasattr(m1, "predict_proba_branches"):
        p1_1m, p1_15m = m1.predict_proba_branches(X)
        g1m = stage1_min_prob if stage1_min_prob_1m is None else stage1_min_prob_1m
        g15 = stage1_min_prob if stage1_min_prob_15m is None else stage1_min_prob_15m
        trend_mask &= p1_1m[:, idx_trend] >= g1m
        trend_mask &= p1_15m[:, idx_trend] >= g15

    if trend_mask.any():
        x2 = X[trend_mask]
        p2 = m2.predict_proba(x2)
        c2 = list(getattr(m2, "classes_", [0, 1]))
        idx_up = c2.index(1) if 1 in c2 else -1
        if idx_up >= 0:
            up_prob = p2[:, idx_up]
            down_prob = 1.0 - up_prob
            up_gate, down_gate = _resolve_stage2_directional_probs(
                stage2_min_prob,
                stage2_min_prob_up,
                stage2_min_prob_down,
            )
            is_up_side = up_prob >= 0.5
            chosen_prob = np.where(is_up_side, up_prob, down_prob)
            side_gate = np.where(is_up_side, up_gate, down_gate)
            sure = chosen_prob >= side_gate

            if hasattr(m2, "predict_proba_branches"):
                p2_1m, p2_15m = m2.predict_proba_branches(x2)
                up_1m = p2_1m[:, idx_up]
                up_15m = p2_15m[:, idx_up]
                conf_1m = np.maximum(up_1m, 1.0 - up_1m)
                conf_15m = np.maximum(up_15m, 1.0 - up_15m)
                g1m = stage2_min_prob if stage2_min_prob_1m is None else stage2_min_prob_1m
                g15 = stage2_min_prob if stage2_min_prob_15m is None else stage2_min_prob_15m
                sure &= conf_1m >= g1m
                sure &= conf_15m >= g15

            idx_global = np.where(trend_mask)[0]
            up_prob_full[idx_global] = up_prob
            idx_sure = idx_global[sure]
            is_up = up_prob[sure] >= 0.5
            result[idx_sure] = np.where(is_up, LABEL_UP, LABEL_DOWN)
            signal_prob[idx_sure] = np.where(is_up, up_prob[sure], down_prob[sure])

    return {
        "pred": result,
        "trend_prob": trend_prob,
        "signal_prob": signal_prob,
        "up_prob": up_prob_full,
    }


def predict_two_stage(
    X: np.ndarray,
    m1,
    m2,
    stage1_min_prob: float = 0.55,
    stage2_min_prob: float = 0.55,
    stage2_min_prob_up: Optional[float] = None,
    stage2_min_prob_down: Optional[float] = None,
    stage1_min_prob_1m: Optional[float] = None,
    stage1_min_prob_15m: Optional[float] = None,
    stage2_min_prob_1m: Optional[float] = None,
    stage2_min_prob_15m: Optional[float] = None,
    stage1_extra: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns tradable predictions: 0=down, 1=no-trade(flat/risky), 2=up."""
    return predict_two_stage_details(
        X,
        m1,
        m2,
        stage1_min_prob=stage1_min_prob,
        stage2_min_prob=stage2_min_prob,
        stage2_min_prob_up=stage2_min_prob_up,
        stage2_min_prob_down=stage2_min_prob_down,
        stage1_min_prob_1m=stage1_min_prob_1m,
        stage1_min_prob_15m=stage1_min_prob_15m,
        stage2_min_prob_1m=stage2_min_prob_1m,
        stage2_min_prob_15m=stage2_min_prob_15m,
        stage1_extra=stage1_extra,
    )["pred"]


def predict_saved_bundle_details(
    X: np.ndarray,
    curr: np.ndarray,
    entry: np.ndarray,
    bundle: dict[str, object],
    *,
    stage1_min_prob: float,
    stage2_min_prob: float,
    stage2_min_prob_up: Optional[float] = None,
    stage2_min_prob_down: Optional[float] = None,
    stage1_extra: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    m1 = bundle["stage1"]
    m2 = bundle["stage2"]
    saved_cfg = bundle.get("config") if isinstance(bundle.get("config"), dict) else {}
    use_state = bool(bundle.get("use_state_features", saved_cfg.get("use_state_features", False)))
    if not use_state:
        return predict_two_stage_details(
            X,
            m1,
            m2,
            stage1_min_prob=stage1_min_prob,
            stage2_min_prob=stage2_min_prob,
            stage2_min_prob_up=stage2_min_prob_up,
            stage2_min_prob_down=stage2_min_prob_down,
            stage1_min_prob_1m=saved_cfg.get("stage1_min_prob_1m"),
            stage1_min_prob_15m=saved_cfg.get("stage1_min_prob_15m"),
            stage2_min_prob_1m=saved_cfg.get("stage2_min_prob_1m"),
            stage2_min_prob_15m=saved_cfg.get("stage2_min_prob_15m"),
            stage1_extra=stage1_extra,
        )

    n = int(len(X))
    pred = np.full(n, LABEL_FLAT, dtype=np.int64)
    signal_prob = np.zeros(n, dtype=np.float64)
    loop_start = time.time()
    report_every = max(1, n // 20)
    hist = int(saved_cfg.get("pred_history_len", 150))
    n_hist_feats = hist * 3
    max_bars = 120
    long_thr = float(saved_cfg.get("long_target_threshold") or saved_cfg.get("trend_threshold", 0.008))
    short_thr = float(saved_cfg.get("short_target_threshold") or saved_cfg.get("trend_threshold", 0.008))
    long_stop = float(saved_cfg.get("long_adverse_limit") or saved_cfg.get("adverse_limit", 15.0))
    short_stop = float(saved_cfg.get("short_adverse_limit") or saved_cfg.get("adverse_limit", 15.0))
    pos = 0
    entry_px = 0.0
    bars_in_pos = 0
    target_hit_flag = 0.0
    stop_hit_flag = 0.0

    for i in range(n):
        if i > 0 and i % report_every == 0:
            pct = (i / n) * 100.0
            elapsed = time.time() - loop_start
            eta_s = (elapsed / i) * (n - i) if i > 0 else 0.0
            eta = f"{eta_s:.0f}s" if eta_s < 60 else f"{eta_s/60:.1f}m"
            print(
                f"  [{_now()}] saved-model predict {pct:>5.1f}%  samples={i:>7,}/{n:,}  elapsed={_elapsed(loop_start)}  eta≈{eta}",
                flush=True,
            )
        state_vec = np.zeros((1, n_hist_feats + 7), dtype=np.float64)
        for lag in range(1, hist + 1):
            j = i - lag
            if j < 0:
                continue
            cls = int(pred[j])
            if cls == LABEL_RISKY:
                cls = LABEL_FLAT
            if cls in (LABEL_DOWN, LABEL_FLAT, LABEL_UP):
                state_vec[0, (lag - 1) * 3 + cls] = 1.0
        b = n_hist_feats
        state_vec[0, b + (pos + 1)] = 1.0
        state_vec[0, b + 3] = min(float(bars_in_pos), float(max_bars)) / float(max_bars)
        if pos != 0 and entry_px > 0.0:
            state_vec[0, b + 4] = ((float(curr[i]) - entry_px) * float(pos)) / entry_px
        state_vec[0, b + 5] = target_hit_flag
        state_vec[0, b + 6] = stop_hit_flag
        X_i = np.hstack([X[i : i + 1], state_vec])
        pm = predict_two_stage_details(
            X_i,
            m1,
            m2,
            stage1_min_prob=stage1_min_prob,
            stage2_min_prob=stage2_min_prob,
            stage2_min_prob_up=stage2_min_prob_up,
            stage2_min_prob_down=stage2_min_prob_down,
            stage1_min_prob_1m=saved_cfg.get("stage1_min_prob_1m"),
            stage1_min_prob_15m=saved_cfg.get("stage1_min_prob_15m"),
            stage2_min_prob_1m=saved_cfg.get("stage2_min_prob_1m"),
            stage2_min_prob_15m=saved_cfg.get("stage2_min_prob_15m"),
            stage1_extra=None if stage1_extra is None else stage1_extra[i : i + 1],
        )
        pred_i = int(pm["pred"][0])
        pred[i] = pred_i
        signal_prob[i] = float(pm["signal_prob"][0])
        target_hit_flag = 0.0
        stop_hit_flag = 0.0
        if pos != 0 and entry_px > 0.0:
            move_abs = (float(curr[i]) - entry_px) * float(pos)
            target_abs = abs(entry_px) * float(long_thr if pos > 0 else short_thr)
            stop_abs = float(long_stop if pos > 0 else short_stop)
            if move_abs >= target_abs:
                pos = 0
                entry_px = 0.0
                bars_in_pos = 0
                target_hit_flag = 1.0
            elif move_abs <= -stop_abs:
                pos = 0
                entry_px = 0.0
                bars_in_pos = 0
                stop_hit_flag = 1.0
            else:
                bars_in_pos += 1
        if pos == 0 and pred_i in TREND_LABELS:
            pos = 1 if pred_i == LABEL_UP else -1
            entry_px = float(entry[i])
            bars_in_pos = 0

    print(f"  [{_now()}] saved-model predict done — samples={n:,}  took={_elapsed(loop_start)}", flush=True)
    return {"pred": pred, "signal_prob": signal_prob}


# ── Optimiser  (sweeps min_window_range only; threshold & horizon are fixed) ──

_OPT_RANGES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]


def optimize_min_range(
    bars: pd.DataFrame,
    window: int,
    threshold: float,
    horizon: int,
    adverse_limit: float,
    long_target_threshold: Optional[float],
    short_target_threshold: Optional[float],
    long_adverse_limit: Optional[float],
    short_adverse_limit: Optional[float],
    max_flat_ratio: float,
    classifier: str,
    random_state: int,
    min_15m_drop: float = 0.0,
    min_15m_rise: float = 0.0,
    last_bar_wr90_high: Optional[float] = None,
    last_bar_wr90_low: Optional[float] = None,
    window_15m: int = 40,
    two_branch: bool = False,
    two_branch_stage: str = "both",
    use_stage1_day_ohl_utc2: bool = False,
) -> float:
    """Grid-search best min_window_range on train bars.

    Score = total_pnl * sqrt(n_signals / 50)
      - Rewards high total PnL AND sufficient signal count.
      - Avoids picking rare signals that don't generalise.
    """
    best_score, best_range = -1e18, _OPT_RANGES[0]
    rows    = []
    n_combos = len(_OPT_RANGES)
    t_opt    = time.time()

    for combo_i, rng_val in enumerate(_OPT_RANGES, 1):
        t_combo = time.time()
        print(
            f"  [{_now()}] combo {combo_i:>2}/{n_combos}  "
            f"min_range={rng_val:>5}  thr={threshold}  hor={horizon}",
            end="  ", flush=True,
        )
        try:
            X, y, ts, curr_a, entry_a, fut_a, entry_ts_a, fut_ts_a, kept, skipped = build_dataset(
                bars, window, rng_val, horizon, threshold, adverse_limit,
                long_target_threshold=long_target_threshold,
                short_target_threshold=short_target_threshold,
                long_adverse_limit=long_adverse_limit,
                short_adverse_limit=short_adverse_limit,
                min_15m_drop=min_15m_drop,
                min_15m_rise=min_15m_rise,
                last_bar_wr90_high=last_bar_wr90_high,
                last_bar_wr90_low=last_bar_wr90_low,
                window_15m=window_15m,
                use_15m_wick_features=False,
            )
        except ValueError:
            print("skip (empty)", flush=True)
            continue

        n_trend    = int(np.isin(y, TREND_LABELS).sum())
        trend_pct  = n_trend / max(kept, 1) * 100
        if kept < 100 or trend_pct < 2.0:
            print(f"skip  kept={kept}  trend={trend_pct:.1f}%%", flush=True)
            continue

        sp = int(kept * 0.7)
        if sp < 60 or (kept - sp) < 30:
            print("skip (split too small)", flush=True)
            continue

        Xf = flatten_tensors(X)
        n_1m_feats = int(X.shape[1]) * window
        stage1_extra = _build_stage1_day_ohl_features(bars, ts, curr_a) if use_stage1_day_ohl_utc2 else None
        try:
            m1, m2 = train_two_stage(
                Xf[:sp], y[:sp], random_state, max_flat_ratio, classifier,
                two_branch=two_branch, n_1m_feats=n_1m_feats,
                two_branch_stage=two_branch_stage,
                stage1_extra=None if stage1_extra is None else stage1_extra[:sp],
            )
        except ValueError as e:
            print(f"skip ({e})", flush=True)
            continue

        pred_meta = predict_two_stage_details(
            Xf[sp:],
            m1,
            m2,
            stage1_extra=None if stage1_extra is None else stage1_extra[sp:],
        )
        trades_df = _backtest_trades_df(
            ts[sp:],
            entry_ts_a[sp:],
            fut_ts_a[sp:],
            pred_meta["pred"],
            curr_a[sp:],
            entry_a[sp:],
            fut_a[sp:],
            signal_prob=pred_meta["signal_prob"],
            adverse_limit=adverse_limit,
            long_target_threshold=long_target_threshold,
            short_target_threshold=short_target_threshold,
            long_adverse_limit=long_adverse_limit,
            short_adverse_limit=short_adverse_limit,
            reverse_exit_prob=DEFAULT_REVERSE_EXIT_PROB,
        )
        n_sig = int(len(trades_df))
        if n_sig == 0:
            print("skip (no signals)", flush=True)
            continue

        pnl = float(trades_df["pnl"].sum())
        score = pnl * np.sqrt(max(n_sig, 1) / 50.0)

        rows.append((rng_val, score, n_sig, pnl, trend_pct, kept))
        marker = " ◄" if (score > best_score and n_sig >= 10) else ""
        print(
            f"score={score:>+9.2f}  pnl=${pnl:>8.2f}  "
            f"signals={n_sig:>5}  trend={trend_pct:>5.1f}%%  "
            f"kept={kept:,}  ({_elapsed(t_combo)}){marker}",
            flush=True,
        )
        if score > best_score and n_sig >= 10:
            best_score, best_range = score, rng_val

    print(f"  [{_now()}] Optimisation done — total={_elapsed(t_opt)}", flush=True)

    print(f"\n  {'min_range':>10} {'score':>10} {'signals':>8} {'total_pnl':>10} {'trend%%':>8} {'kept':>8}")
    for r in sorted(rows, key=lambda x: -x[1]):
        mark = " ◄ best" if r[0] == best_range else ""
        print(f"  {r[0]:>10} {r[1]:>10.2f} {r[2]:>8} {r[3]:>10.2f} {r[4]:>7.1f}%% {r[5]:>8,}{mark}")

    if not rows:
        print("\n  → No valid min_window_range candidate found; falling back to 40")
        return 40.0

    print(f"\n  → Best min_window_range={best_range}  score={best_score:.2f}")
    return best_range


_OPT_STAGE1_PROBS = [0.50, 0.55, 0.60, 0.65, 0.70]
_OPT_STAGE2_PROBS = [0.50, 0.55, 0.60, 0.65, 0.70]


def optimize_prob_thresholds(
    X_val: np.ndarray,
    ts_val: pd.DatetimeIndex,
    entry_ts_val: pd.DatetimeIndex,
    fut_ts_val: pd.DatetimeIndex,
    curr_val: np.ndarray,
    entry_val: np.ndarray,
    fut_val: np.ndarray,
    m1,
    m2,
    adverse_limit: float,
    long_target_threshold: Optional[float],
    short_target_threshold: Optional[float],
    long_adverse_limit: Optional[float],
    short_adverse_limit: Optional[float],
    stage2_min_prob_up: Optional[float] = None,
    stage2_min_prob_down: Optional[float] = None,
    stage1_min_prob_1m: Optional[float] = None,
    stage1_min_prob_15m: Optional[float] = None,
    stage2_min_prob_1m: Optional[float] = None,
    stage2_min_prob_15m: Optional[float] = None,
    stage1_extra_val: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Tune stage probability gates on validation data using PnL-driven score."""
    best_score = -1e18
    best_pair = (_OPT_STAGE1_PROBS[0], _OPT_STAGE2_PROBS[0])
    rows: list[tuple[float, float, float, int, float]] = []

    combos = list(itertools.product(_OPT_STAGE1_PROBS, _OPT_STAGE2_PROBS))
    t_opt = time.time()
    for idx, (p1, p2) in enumerate(combos, 1):
        pred_meta = predict_two_stage_details(
            X_val,
            m1,
            m2,
            stage1_min_prob=p1,
            stage2_min_prob=p2,
            stage2_min_prob_up=stage2_min_prob_up,
            stage2_min_prob_down=stage2_min_prob_down,
            stage1_min_prob_1m=stage1_min_prob_1m,
            stage1_min_prob_15m=stage1_min_prob_15m,
            stage2_min_prob_1m=stage2_min_prob_1m,
            stage2_min_prob_15m=stage2_min_prob_15m,
            stage1_extra=stage1_extra_val,
        )
        trades_df = _backtest_trades_df(
            ts_val,
            entry_ts_val,
            fut_ts_val,
            pred_meta["pred"],
            curr_val,
            entry_val,
            fut_val,
            signal_prob=pred_meta["signal_prob"],
            adverse_limit=adverse_limit,
            long_target_threshold=long_target_threshold,
            short_target_threshold=short_target_threshold,
            long_adverse_limit=long_adverse_limit,
            short_adverse_limit=short_adverse_limit,
            reverse_exit_prob=DEFAULT_REVERSE_EXIT_PROB,
        )
        n_sig = int(len(trades_df))
        if n_sig == 0:
            rows.append((p1, p2, -1e18, 0, 0.0))
            continue

        pnl = float(trades_df["pnl"].sum())
        score = pnl * np.sqrt(max(n_sig, 1) / 100.0)
        rows.append((p1, p2, score, n_sig, pnl))

        if n_sig >= 20 and score > best_score:
            best_score = score
            best_pair = (p1, p2)

        if idx % 5 == 0 or idx == len(combos):
            _log(f"Prob-grid progress: {idx}/{len(combos)}")

    print(f"\n  {'stage1':>8} {'stage2':>8} {'score':>10} {'signals':>8} {'total_pnl':>10}")
    for r in sorted(rows, key=lambda x: -x[2])[:10]:
        mark = " ◄ best" if (r[0], r[1]) == best_pair else ""
        score_disp = r[2] if r[2] > -1e17 else float('-inf')
        print(f"  {r[0]:>8.2f} {r[1]:>8.2f} {score_disp:>10.2f} {r[3]:>8} {r[4]:>10.2f}{mark}")
    _log(f"Probability optimisation done — total={_elapsed(t_opt)}")
    return best_pair


def select_wf_cycle_config(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    ts_tr: pd.DatetimeIndex,
    entry_ts_tr: pd.DatetimeIndex,
    fut_ts_tr: pd.DatetimeIndex,
    curr_tr: np.ndarray,
    entry_tr: np.ndarray,
    fut_tr: np.ndarray,
    cfg: Config,
    n_1m_feats: int,
    two_branch: bool,
    long_adverse_limit: Optional[float],
    short_adverse_limit: Optional[float],
    stage1_extra_tr: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Sweep per-cycle train config on a trailing validation split."""
    n = len(X_tr)
    split_fit = int(n * (1.0 - cfg.wf_sweep_val_ratio))
    if split_fit < 100 or (n - split_fit) < cfg.wf_sweep_min_val_samples:
        return {
            "used_sweep": False,
            "reason": "validation_too_small",
            "max_flat_ratio": float(cfg.max_flat_ratio),
            "stage1_min_prob": float(cfg.stage1_min_prob),
            "stage2_min_prob": float(cfg.stage2_min_prob),
            "stage2_min_prob_up": float(cfg.stage2_min_prob if cfg.stage2_min_prob_up is None else cfg.stage2_min_prob_up),
            "stage2_min_prob_down": float(cfg.stage2_min_prob if cfg.stage2_min_prob_down is None else cfg.stage2_min_prob_down),
            "score": None,
            "signals": 0,
            "total_pnl": 0.0,
            "candidate_count": 0,
        }

    X_fit, y_fit = X_tr[:split_fit], y_tr[:split_fit]
    X_val = X_tr[split_fit:]
    ts_val = ts_tr[split_fit:]
    entry_ts_val = entry_ts_tr[split_fit:]
    fut_ts_val = fut_ts_tr[split_fit:]
    curr_val = curr_tr[split_fit:]
    entry_val = entry_tr[split_fit:]
    fut_val = fut_tr[split_fit:]
    stage1_extra_fit = None if stage1_extra_tr is None else stage1_extra_tr[:split_fit]
    stage1_extra_val = None if stage1_extra_tr is None else stage1_extra_tr[split_fit:]

    stage1_prob_grid = cfg.wf_sweep_stage1_probs if cfg.wf_sweep_stage1_probs else _OPT_STAGE1_PROBS
    stage2_prob_grid = (
        cfg.wf_sweep_stage2_probs
        if cfg.wf_sweep_stage2_probs
        else [float(cfg.stage2_min_prob)]
        if (cfg.wf_sweep_stage2_long_probs or cfg.wf_sweep_stage2_short_probs)
        else _OPT_STAGE2_PROBS
    )
    long_prob_grid = (
        cfg.wf_sweep_stage2_long_probs
        if cfg.wf_sweep_stage2_long_probs
        else [float(cfg.stage2_min_prob if cfg.stage2_min_prob_up is None else cfg.stage2_min_prob_up)]
    )
    short_prob_grid = (
        cfg.wf_sweep_stage2_short_probs
        if cfg.wf_sweep_stage2_short_probs
        else [float(cfg.stage2_min_prob if cfg.stage2_min_prob_down is None else cfg.stage2_min_prob_down)]
    )

    best_any: Optional[dict[str, object]] = None
    best_strict: Optional[dict[str, object]] = None
    candidate_count = 0

    for flat_ratio in cfg.wf_sweep_flat_ratios:
        try:
            m1_tmp, m2_tmp = train_two_stage(
                X_fit,
                y_fit,
                cfg.random_state,
                float(flat_ratio),
                cfg.classifier,
                two_branch=two_branch,
                n_1m_feats=n_1m_feats,
                two_branch_stage=cfg.two_branch_stage,
                stage1_extra=stage1_extra_fit,
            )
        except ValueError:
            continue

        for p1, p2, p2_up, p2_down in itertools.product(
            stage1_prob_grid,
            stage2_prob_grid,
            long_prob_grid,
            short_prob_grid,
        ):
            candidate_count += 1
            pred_meta = predict_two_stage_details(
                X_val,
                m1_tmp,
                m2_tmp,
                stage1_min_prob=p1,
                stage2_min_prob=p2,
                stage2_min_prob_up=float(p2_up),
                stage2_min_prob_down=float(p2_down),
                stage1_min_prob_1m=cfg.stage1_min_prob_1m,
                stage1_min_prob_15m=cfg.stage1_min_prob_15m,
                stage2_min_prob_1m=cfg.stage2_min_prob_1m,
                stage2_min_prob_15m=cfg.stage2_min_prob_15m,
                stage1_extra=stage1_extra_val,
            )
            trades_df = _backtest_trades_df(
                ts_val,
                entry_ts_val,
                fut_ts_val,
                pred_meta["pred"],
                curr_val,
                entry_val,
                fut_val,
                signal_prob=pred_meta["signal_prob"],
                adverse_limit=cfg.adverse_limit,
                long_target_threshold=cfg.long_target_threshold,
                short_target_threshold=cfg.short_target_threshold,
                long_adverse_limit=long_adverse_limit,
                short_adverse_limit=short_adverse_limit,
                allow_overlap=cfg.allow_overlap_backtest,
                reverse_exit_prob=cfg.reverse_exit_prob,
            )
            n_sig = int(len(trades_df))
            pnl = float(trades_df["pnl"].sum()) if n_sig > 0 else 0.0
            score = pnl * np.sqrt(max(n_sig, 1) / 100.0) if n_sig > 0 else -1e18

            if best_any is None or float(score) > float(best_any["score"]):
                best_any = {
                    "used_sweep": True,
                    "reason": "ok",
                    "max_flat_ratio": float(flat_ratio),
                    "stage1_min_prob": float(p1),
                    "stage2_min_prob": float(p2),
                    "stage2_min_prob_up": float(p2_up),
                    "stage2_min_prob_down": float(p2_down),
                    "score": float(score),
                    "signals": n_sig,
                    "total_pnl": pnl,
                    "candidate_count": candidate_count,
                }
            if n_sig >= 20 and (best_strict is None or float(score) > float(best_strict["score"])):
                best_strict = {
                    "used_sweep": True,
                    "reason": "ok",
                    "max_flat_ratio": float(flat_ratio),
                    "stage1_min_prob": float(p1),
                    "stage2_min_prob": float(p2),
                    "stage2_min_prob_up": float(p2_up),
                    "stage2_min_prob_down": float(p2_down),
                    "score": float(score),
                    "signals": n_sig,
                    "total_pnl": pnl,
                    "candidate_count": candidate_count,
                }

    if best_strict is not None:
        best_strict["candidate_count"] = candidate_count
        return best_strict
    if best_any is not None:
        best_any["candidate_count"] = candidate_count
        best_any["reason"] = "fallback_low_signal"
        return best_any
    return {
        "used_sweep": False,
        "reason": "no_valid_candidates",
        "max_flat_ratio": float(cfg.max_flat_ratio),
        "stage1_min_prob": float(cfg.stage1_min_prob),
        "stage2_min_prob": float(cfg.stage2_min_prob),
        "stage2_min_prob_up": float(cfg.stage2_min_prob if cfg.stage2_min_prob_up is None else cfg.stage2_min_prob_up),
        "stage2_min_prob_down": float(cfg.stage2_min_prob if cfg.stage2_min_prob_down is None else cfg.stage2_min_prob_down),
        "score": None,
        "signals": 0,
        "total_pnl": 0.0,
        "candidate_count": candidate_count,
    }


# ── PnL check ─────────────────────────────────────────────────────────────────

def _backtest_trades_df(
    ts: pd.DatetimeIndex,
    entry_ts: pd.DatetimeIndex,
    fut_ts: pd.DatetimeIndex,
    pred: np.ndarray,
    curr: np.ndarray,
    entry_px: np.ndarray,
    fut: np.ndarray,
    signal_prob: Optional[np.ndarray] = None,
    adverse_limit: float = 15.0,
    long_target_threshold: Optional[float] = None,
    short_target_threshold: Optional[float] = None,
    long_adverse_limit: Optional[float] = None,
    short_adverse_limit: Optional[float] = None,
    allow_overlap: bool = False,
    reverse_exit_prob: float = DEFAULT_REVERSE_EXIT_PROB,
    max_hold_minutes: Optional[float] = None,
    weak_period_cells: Optional[list[dict[str, str]]] = None,
    # Ask/bid spread arrays – when provided, entries use ask/bid open and
    # exits use bid/ask close (long buys at ask, exits at bid; short sells at bid, exits at ask).
    entry_px_ask: Optional[np.ndarray] = None,
    entry_px_bid: Optional[np.ndarray] = None,
    curr_ask: Optional[np.ndarray] = None,
    curr_bid: Optional[np.ndarray] = None,
    fut_ask: Optional[np.ndarray] = None,
    fut_bid: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    columns = [
        "signal_idx",
        "ts",
        "entry_time",
        "exit_time",
        "pred",
        "side",
        "entry_price",
        "exit_price",
        "pnl",
        "entry_signal_prob",
        "last_signal_prob",
        "target_updates",
        "last_target_signal_idx",
        "last_target_time",
        "signal_ts_close",
        "update_ts_close",
        "last_target_price",
        "exit_reason",
    ]
    rows: list[dict[str, object]] = []
    next_entry_allowed_ts: Optional[pd.Timestamp] = None
    long_stop_abs, short_stop_abs = _resolve_directional_stops(adverse_limit, long_adverse_limit, short_adverse_limit)
    max_hold_delta: Optional[pd.Timedelta] = None
    if max_hold_minutes is not None:
        max_hold_delta = pd.Timedelta(minutes=float(max_hold_minutes))
        if max_hold_delta <= pd.Timedelta(0):
            max_hold_delta = None

    if allow_overlap and max_hold_delta is not None:
        raise ValueError("max_hold_minutes is not supported with allow_overlap=True")

    def _stop_abs_for_pred(pred_cls: int) -> float:
        return max(float(long_stop_abs if int(pred_cls) == LABEL_UP else short_stop_abs), 0.0)

    def _target_abs_for_pred(pred_cls: int, entry_price: float) -> float:
        if entry_price == 0.0:
            return 0.0
        thr = long_target_threshold if int(pred_cls) == LABEL_UP else short_target_threshold
        if thr is None:
            return 0.0
        return abs(float(entry_price)) * float(thr)

    def _sig_prob(i: int) -> float:
        if signal_prob is None:
            return 0.0
        return float(signal_prob[i])

    def _spread_entry(i: int, side_num: float) -> float:
        """Long buys at ask; short sells at bid."""
        if side_num > 0 and entry_px_ask is not None:
            return float(entry_px_ask[i])
        if side_num < 0 and entry_px_bid is not None:
            return float(entry_px_bid[i])
        return float(entry_px[i])

    def _spread_curr(i: int, side_num: float) -> float:
        """Long position monitored/closed at bid; short at ask."""
        if side_num > 0 and curr_bid is not None:
            return float(curr_bid[i])
        if side_num < 0 and curr_ask is not None:
            return float(curr_ask[i])
        return float(curr[i])

    def _spread_fut(i: int, side_num: float) -> float:
        """Long exits at horizon bid; short exits at horizon ask."""
        if side_num > 0 and fut_bid is not None:
            return float(fut_bid[i])
        if side_num < 0 and fut_ask is not None:
            return float(fut_ask[i])
        return float(fut[i])

    def _append_trade(trade: dict[str, object], exit_ts: pd.Timestamp, exit_price: float, exit_reason: str) -> None:
        side = float(trade["side_num"])
        entry_price = float(trade["entry_price"])
        effective_exit_price = float(exit_price)
        stop_abs = float(trade["stop_abs"])
        if stop_abs > 0.0:
            stop_price = entry_price - side * stop_abs
            raw_pnl = (effective_exit_price - entry_price) * side
            if raw_pnl < -stop_abs:
                effective_exit_price = float(stop_price)
                exit_reason = "stop_loss"
        final_pnl = float((effective_exit_price - entry_price) * side)
        if exit_reason == "signal_target":
            target_abs = float(trade.get("target_abs", 0.0))
            updated_targets = int(trade.get("target_updates", 0) or 0)
            if updated_targets > 0:
                # A rolled/updated signal target closed as planned, not a pure timeout.
                exit_reason = "target_hit"
            else:
                exit_reason = "target_hit" if target_abs > 0.0 and final_pnl >= target_abs else "timeout"
        elif exit_reason == "horizon":
            exit_reason = "timeout"
        rows.append(
            {
                "signal_idx": int(trade["signal_idx"]),
                "ts": pd.Timestamp(trade.get("signal_time", trade["entry_time"])).isoformat(),
                "entry_time": pd.Timestamp(trade["entry_time"]).isoformat(),
                "exit_time": exit_ts.isoformat(),
                "pred": int(trade["pred"]),
                "side": str(trade["side"]),
                "entry_price": entry_price,
                "exit_price": effective_exit_price,
                "pnl": final_pnl,
                "entry_signal_prob": float(trade["entry_signal_prob"]),
                "last_signal_prob": float(trade["last_signal_prob"]),
                "target_updates": int(trade["target_updates"]),
                "last_target_signal_idx": int(trade["last_target_signal_idx"]),
                "last_target_time": pd.Timestamp(trade["last_target_time"]).isoformat(),
                "signal_ts_close": float(trade.get("signal_ts_close", np.nan)),
                "update_ts_close": float(trade.get("update_ts_close", np.nan)),
                "last_target_price": float(trade["last_target_price"]),
                "exit_reason": exit_reason,
            }
        )

    if allow_overlap:
        for i in range(len(pred)):
            if int(pred[i]) not in TREND_LABELS:
                continue

            signal_ts = pd.Timestamp(ts[i])
            trade_entry_ts = pd.Timestamp(entry_ts[i])
            exit_ts = pd.Timestamp(fut_ts[i])
            if _is_weak_period_entry(trade_entry_ts, weak_period_cells):
                continue
            if next_entry_allowed_ts is not None and trade_entry_ts <= next_entry_allowed_ts:
                continue

            side = 1.0 if int(pred[i]) == LABEL_UP else -1.0
            stop_abs = _stop_abs_for_pred(int(pred[i]))
            adj_entry = _spread_entry(i, side)
            raw_exit = _spread_fut(i, side)
            exit_reason = "timeout"
            if stop_abs > 0.0 and (raw_exit - adj_entry) * side < -stop_abs:
                raw_exit = float(adj_entry - side * stop_abs)
                exit_reason = "stop_loss"
            rows.append(
                {
                    "signal_idx": i,
                    "ts": signal_ts.isoformat(),
                    "entry_time": trade_entry_ts.isoformat(),
                    "exit_time": exit_ts.isoformat(),
                    "pred": int(pred[i]),
                    "side": "up" if pred[i] == 2 else "down",
                    "entry_price": adj_entry,
                    "exit_price": raw_exit,
                    "pnl": float((raw_exit - adj_entry) * side),
                    "stop_abs": stop_abs,
                    "target_abs": _target_abs_for_pred(int(pred[i]), adj_entry),
                    "entry_signal_prob": _sig_prob(i),
                    "last_signal_prob": _sig_prob(i),
                    "target_updates": 0,
                    "last_target_signal_idx": i,
                    "last_target_time": exit_ts.isoformat(),
                    "signal_ts_close": _spread_curr(i, side),
                    "update_ts_close": _spread_fut(i, side),
                    "last_target_price": _spread_fut(i, side),
                    "exit_reason": exit_reason,
                }
            )
            next_entry_allowed_ts = exit_ts
        return pd.DataFrame(rows, columns=columns)

    open_trade: Optional[dict[str, object]] = None
    for i in range(len(pred)):
        signal_ts = pd.Timestamp(ts[i])
        trade_entry_ts = pd.Timestamp(entry_ts[i])
        exit_ts = pd.Timestamp(fut_ts[i])

        if open_trade is not None:
            stop_abs = float(open_trade["stop_abs"])
            target_abs = float(open_trade["target_abs"])
            side_num = float(open_trade["side_num"])
            entry_price = float(open_trade["entry_price"])
            stop_px = entry_price - side_num * stop_abs
            target_px = entry_price + side_num * target_abs
            curr_exit_px = _spread_curr(i, side_num)
            if stop_abs > 0.0 and (curr_exit_px - entry_price) * side_num <= -stop_abs:
                _append_trade(open_trade, signal_ts, float(stop_px), "stop_loss")
                next_entry_allowed_ts = signal_ts
                open_trade = None
            elif target_abs > 0.0 and (curr_exit_px - entry_price) * side_num >= target_abs:
                _append_trade(open_trade, signal_ts, float(target_px), "target_hit")
                next_entry_allowed_ts = signal_ts
                open_trade = None

        same_direction_signal = False
        # Check reverse/same-direction signal BEFORE planned-exit-time so a signal
        # bar can update state first instead of being prematurely labelled timeout.
        if open_trade is not None and int(pred[i]) not in NO_TRADE_LABELS:
            if int(pred[i]) != int(open_trade["pred"]) and _sig_prob(i) > reverse_exit_prob:
                _reverse_exit_px = _spread_entry(i, -float(open_trade["side_num"]))
                _append_trade(open_trade, trade_entry_ts, _reverse_exit_px, "reverse_signal")
                next_entry_allowed_ts = trade_entry_ts
                open_trade = None
            elif int(pred[i]) == int(open_trade["pred"]):
                same_direction_signal = True
                open_trade["last_signal_prob"] = _sig_prob(i)
                side_num = float(open_trade["side_num"])
                entry_price = float(open_trade["entry_price"])
                target_abs = float(open_trade.get("target_abs", 0.0))
                current_target_profit = target_abs  # Original target in absolute points
                
                # New signal bar close price
                signal_bar_close = _spread_curr(i, side_num)
                # Calculate new target: signal_bar_close + (signal_bar_close * threshold)
                new_target_abs = _target_abs_for_pred(int(pred[i]), signal_bar_close)
                new_target_profit = new_target_abs
                
                # Dynamic hold: roll timeout on improving updates, capped by max-hold when set.
                timeout_cap_time = open_trade.get("timeout_cap_time")
                if new_target_profit > current_target_profit:
                    rolled_deadline = exit_ts
                    if timeout_cap_time is not None:
                        cap_ts = pd.Timestamp(timeout_cap_time)
                        if rolled_deadline > cap_ts:
                            rolled_deadline = cap_ts
                    if signal_ts < rolled_deadline:
                        new_planned_exit = signal_bar_close + side_num * new_target_abs
                        open_trade["target_updates"] = int(open_trade["target_updates"]) + 1
                        open_trade["last_target_signal_idx"] = i
                        open_trade["last_target_time"] = signal_ts
                        open_trade["update_ts_close"] = _spread_fut(i, side_num)
                        open_trade["last_target_price"] = new_planned_exit
                        open_trade["planned_exit_time"] = rolled_deadline
                        open_trade["planned_exit_price"] = new_planned_exit

        if open_trade is not None and signal_ts >= pd.Timestamp(open_trade["planned_exit_time"]):
            planned_exit_ts = pd.Timestamp(open_trade["planned_exit_time"])
            _append_trade(open_trade, planned_exit_ts, _spread_curr(i, float(open_trade["side_num"])), "timeout")
            next_entry_allowed_ts = planned_exit_ts
            open_trade = None

        if int(pred[i]) in NO_TRADE_LABELS:
            continue

        if open_trade is None:
            if next_entry_allowed_ts is not None and trade_entry_ts <= next_entry_allowed_ts:
                continue
            if _is_weak_period_entry(trade_entry_ts, weak_period_cells):
                continue
            open_trade = {
                "signal_idx": i,
                "signal_time": signal_ts,
                "entry_time": trade_entry_ts,
                "pred": int(pred[i]),
                "side": "up" if int(pred[i]) == LABEL_UP else "down",
                "side_num": 1.0 if int(pred[i]) == LABEL_UP else -1.0,
                "stop_abs": _stop_abs_for_pred(int(pred[i])),
            }
            _new_side_num = float(open_trade["side_num"])
            _adj_entry = _spread_entry(i, _new_side_num)
            _adj_fut   = _spread_fut(i, _new_side_num)
            _signal_close = _spread_curr(i, _new_side_num)
            _target_abs = _target_abs_for_pred(int(pred[i]), _adj_entry)
            _planned_exit = _adj_entry + _new_side_num * _target_abs
            _timeout_deadline = (trade_entry_ts + max_hold_delta) if max_hold_delta is not None else exit_ts
            open_trade.update({
                "target_abs": _target_abs,
                "entry_price": _adj_entry,
                "entry_signal_prob": _sig_prob(i),
                "last_signal_prob": _sig_prob(i),
                "target_updates": 0,
                "last_target_signal_idx": i,
                "last_target_time": signal_ts,
                "signal_ts_close": _signal_close,
                "update_ts_close": _adj_fut,
                "last_target_price": _planned_exit,
                "planned_exit_time": _timeout_deadline,
                "planned_exit_price": _planned_exit,
                "timeout_cap_time": (trade_entry_ts + max_hold_delta) if max_hold_delta is not None else None,
            })
            continue

        if open_trade is not None and same_direction_signal:
            continue
        # Opposite-direction signal with prob <= reverse_exit_prob: hold open trade.

    if open_trade is not None:
        planned_exit_ts = pd.Timestamp(open_trade["planned_exit_time"])
        _append_trade(open_trade, planned_exit_ts, float(open_trade["planned_exit_price"]), "signal_target")

    return pd.DataFrame(rows, columns=columns)


def directional_pnl_report(
    ts: pd.DatetimeIndex,
    entry_ts: pd.DatetimeIndex,
    fut_ts: pd.DatetimeIndex,
    pred: np.ndarray,
    curr: np.ndarray,
    entry_px: np.ndarray,
    fut: np.ndarray,
    signal_prob: Optional[np.ndarray] = None,
    adverse_limit: float = 15.0,
    long_target_threshold: Optional[float] = None,
    short_target_threshold: Optional[float] = None,
    long_adverse_limit: Optional[float] = None,
    short_adverse_limit: Optional[float] = None,
    allow_overlap: bool = False,
    reverse_exit_prob: float = DEFAULT_REVERSE_EXIT_PROB,
    max_hold_minutes: Optional[float] = None,
    weak_period_cells: Optional[list[dict[str, str]]] = None,
    entry_px_ask: Optional[np.ndarray] = None,
    entry_px_bid: Optional[np.ndarray] = None,
    curr_ask: Optional[np.ndarray] = None,
    curr_bid: Optional[np.ndarray] = None,
    fut_ask: Optional[np.ndarray] = None,
    fut_bid: Optional[np.ndarray] = None,
) -> tuple[dict, pd.DataFrame]:
    pdf = _backtest_trades_df(
        ts,
        entry_ts,
        fut_ts,
        pred,
        curr,
        entry_px,
        fut,
        signal_prob=signal_prob,
        adverse_limit=adverse_limit,
        long_target_threshold=long_target_threshold,
        short_target_threshold=short_target_threshold,
        long_adverse_limit=long_adverse_limit,
        short_adverse_limit=short_adverse_limit,
        allow_overlap=allow_overlap,
        reverse_exit_prob=reverse_exit_prob,
        max_hold_minutes=max_hold_minutes,
        weak_period_cells=weak_period_cells,
        entry_px_ask=entry_px_ask,
        entry_px_bid=entry_px_bid,
        curr_ask=curr_ask,
        curr_bid=curr_bid,
        fut_ask=fut_ask,
        fut_bid=fut_bid,
    )
    def _streak_stats(pnl_vals: np.ndarray) -> dict[str, int]:
        max_win = 0
        max_loss = 0
        cur_win = 0
        cur_loss = 0
        for v in pnl_vals:
            if v > 0:
                cur_win += 1
                cur_loss = 0
                max_win = max(max_win, cur_win)
            elif v < 0:
                cur_loss += 1
                cur_win = 0
                max_loss = max(max_loss, cur_loss)
            else:
                cur_win = 0
                cur_loss = 0
        current_streak = cur_win if cur_win > 0 else cur_loss
        return {
            "max_win_streak": int(max_win),
            "max_loss_streak": int(max_loss),
            "current_win_streak": int(cur_win),
            "current_loss_streak": int(cur_loss),
            "current_streak": int(current_streak),
        }

    def _empty_bucket() -> dict[str, object]:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "avg_trade": 0.0,
            "median_trade": 0.0,
            "win_rate_pct": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": None,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_win": None,
            "avg_loss": None,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "current_streak": 0,
            "current_win_streak": 0,
            "current_loss_streak": 0,
            "avg_duration_min": 0.0,
            "median_duration_min": 0.0,
            "min_duration_min": 0.0,
            "max_duration_min": 0.0,
            "n_days": 0,
            "avg_trades_per_day": 0.0,
            "avg_day": None,
            "median_day": None,
            "best_day": None,
            "worst_day": None,
            "positive_days_pct": None,
            "trade_max_drawdown": 0.0,
            "daily_max_drawdown": 0.0,
            "exit_reason_counts": {},
            "reverse_signal_stats": {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "breakeven": 0,
                "avg_pnl": None,
                "win_rate_pct": None,
                "loss_rate_pct": None,
            },
            "target_hit_stats": {
                "trades": 0,
                "avg_pnl": None,
            },
            "timeout_stats": {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "breakeven": 0,
                "avg_pnl": None,
                "win_rate_pct": None,
                "loss_rate_pct": None,
            },
            "target_updates_mean": 0.0,
            "target_updates_median": 0.0,
            "target_updates_max": 0,
            "time_distribution": {
                "timezone_labels": {
                    "hkt": "Asia/Hong_Kong",
                    "ny": "America/New_York",
                },
                "by_weekday_hkt": [],
                "by_hour_hkt": [],
                "by_hour_ny": [],
                "by_session": [],
                "weekday_hour_hkt_heatmap": {},
            },
        }

    def _time_distribution_stats(x: pd.DataFrame) -> dict[str, object]:
        if x.empty:
            return dict(_empty_bucket()["time_distribution"])

        tdf = x.copy()
        hkt_time = tdf["entry_time"].dt.tz_convert(HK_TZ)
        ny_time = tdf["entry_time"].dt.tz_convert(NY_TZ)
        tdf["weekday_hkt"] = hkt_time.dt.day_name()
        tdf["hour_hkt"] = hkt_time.dt.hour
        tdf["hour_ny"] = ny_time.dt.hour

        def _session_name(row: pd.Series) -> str:
            h_hkt = int(row["hour_hkt"])
            h_ny = int(row["hour_ny"])
            if ASIA_SESSION_START <= h_hkt <= ASIA_SESSION_END:
                return "Asia"
            if NY_SESSION_START <= h_ny <= NY_SESSION_END:
                return "NY"
            return "Other"

        tdf["session"] = tdf[["hour_hkt", "hour_ny"]].apply(_session_name, axis=1)

        def _summarize(group_cols: list[str], *, sort_key=None) -> list[dict[str, object]]:
            grouped = (
                tdf.groupby(group_cols, dropna=False)["pnl"]
                .agg(["size", "sum", "mean", lambda s: float((s > 0).mean() * 100.0)])
                .reset_index()
            )
            grouped.columns = [*group_cols, "trades", "total_pnl", "avg_trade", "win_rate_pct"]
            rows: list[dict[str, object]] = []
            for _, row in grouped.iterrows():
                item = {col: (None if pd.isna(row[col]) else (int(row[col]) if str(col).startswith("hour_") else str(row[col]))) for col in group_cols}
                item.update(
                    {
                        "trades": int(row["trades"]),
                        "total_pnl": float(row["total_pnl"]),
                        "avg_trade": float(row["avg_trade"]),
                        "win_rate_pct": float(row["win_rate_pct"]),
                    }
                )
                rows.append(item)
            if sort_key is not None:
                rows.sort(key=sort_key)
            return rows

        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_weekday_hkt = _summarize(["weekday_hkt"], sort_key=lambda r: weekday_order.index(r["weekday_hkt"]) if r["weekday_hkt"] in weekday_order else 99)
        by_hour_hkt = _summarize(["hour_hkt"], sort_key=lambda r: int(r["hour_hkt"]))
        by_hour_ny = _summarize(["hour_ny"], sort_key=lambda r: int(r["hour_ny"]))
        by_session = _summarize(["session"], sort_key=lambda r: {"Asia": 0, "NY": 1, "Other": 2}.get(str(r["session"]), 99))
        heatmap_rows = _summarize(
            ["weekday_hkt", "hour_hkt"],
            sort_key=lambda r: (
                weekday_order.index(r["weekday_hkt"]) if r["weekday_hkt"] in weekday_order else 99,
                int(r["hour_hkt"]),
            ),
        )
        weekday_hour_hkt_heatmap: dict[str, dict[str, dict[str, object]]] = {}
        for row in heatmap_rows:
            day = str(row["weekday_hkt"])
            hour_key = f"{int(row['hour_hkt']):02d}:00"
            weekday_hour_hkt_heatmap.setdefault(day, {})[hour_key] = {
                "trades": int(row["trades"]),
                "total_pnl": float(row["total_pnl"]),
                "avg_trade": float(row["avg_trade"]),
                "win_rate_pct": float(row["win_rate_pct"]),
            }

        return {
            "timezone_labels": {
                "hkt": "Asia/Hong_Kong",
                "ny": "America/New_York",
            },
            "by_weekday_hkt": by_weekday_hkt,
            "by_hour_hkt": by_hour_hkt,
            "by_hour_ny": by_hour_ny,
            "by_session": by_session,
            "weekday_hour_hkt_heatmap": weekday_hour_hkt_heatmap,
        }

    def _bucket_stats(df: pd.DataFrame) -> dict[str, object]:
        if df.empty:
            return _empty_bucket()

        x = df.copy()
        x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True)
        x["exit_time"] = pd.to_datetime(x["exit_time"], utc=True)
        x["duration_min"] = (x["exit_time"] - x["entry_time"]).dt.total_seconds() / 60.0
        x["trading_day"] = (x["entry_time"].dt.tz_convert(NY_TZ) - pd.Timedelta(hours=TRADING_DAY_CUTOFF_HOUR_NY)).dt.floor("D")

        pnl = x["pnl"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        gross_profit = float(wins.sum())
        gross_loss = float(losses.sum())
        pf = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None

        daily = x.groupby("trading_day")["pnl"].sum()
        equity_trade = pnl.cumsum()
        trade_dd = float((equity_trade - equity_trade.cummax()).min())
        equity_day = daily.cumsum()
        daily_dd = float((equity_day - equity_day.cummax()).min()) if len(daily) > 0 else 0.0

        streak = _streak_stats(pnl.to_numpy(dtype=np.float64))
        target_updates = pd.to_numeric(x["target_updates"], errors="coerce").fillna(0.0)
        x["exit_reason_norm"] = x["exit_reason"].astype(str)
        legacy_sig = x["exit_reason_norm"] == "signal_target"
        x.loc[legacy_sig & (x["pnl"] > 0.0), "exit_reason_norm"] = "target_hit"
        x.loc[legacy_sig & (x["pnl"] <= 0.0), "exit_reason_norm"] = "timeout"
        x.loc[x["exit_reason_norm"] == "horizon", "exit_reason_norm"] = "timeout"
        x.loc[(x["exit_reason_norm"] == "target_hit") & (x["pnl"] <= 0.0), "exit_reason_norm"] = "timeout"

        reverse_df = x[x["exit_reason_norm"] == "reverse_signal"]
        target_hit_df = x[x["exit_reason_norm"] == "target_hit"]
        timeout_df = x[x["exit_reason_norm"] == "timeout"]
        rev_n = int(len(reverse_df))
        rev_wins = int((reverse_df["pnl"] > 0).sum()) if rev_n else 0
        rev_losses = int((reverse_df["pnl"] < 0).sum()) if rev_n else 0
        rev_be = int((reverse_df["pnl"] == 0).sum()) if rev_n else 0
        th_n = int(len(target_hit_df))
        to_n = int(len(timeout_df))
        to_wins = int((timeout_df["pnl"] > 0).sum()) if to_n else 0
        to_losses = int((timeout_df["pnl"] < 0).sum()) if to_n else 0
        to_be = int((timeout_df["pnl"] == 0).sum()) if to_n else 0

        return {
            "trades": int(len(x)),
            "total_pnl": float(pnl.sum()),
            "avg_trade": float(pnl.mean()),
            "median_trade": float(pnl.median()),
            "win_rate_pct": float((pnl > 0).mean() * 100.0),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": float(pf) if pf is not None else None,
            "best_trade": float(pnl.max()),
            "worst_trade": float(pnl.min()),
            "avg_win": float(wins.mean()) if len(wins) else None,
            "avg_loss": float(losses.mean()) if len(losses) else None,
            "max_win_streak": int(streak["max_win_streak"]),
            "max_loss_streak": int(streak["max_loss_streak"]),
            "current_streak": int(streak["current_streak"]),
            "current_win_streak": int(streak["current_win_streak"]),
            "current_loss_streak": int(streak["current_loss_streak"]),
            "avg_duration_min": float(x["duration_min"].mean()),
            "median_duration_min": float(x["duration_min"].median()),
            "min_duration_min": float(x["duration_min"].min()),
            "max_duration_min": float(x["duration_min"].max()),
            "n_days": int(len(daily)),
            "avg_trades_per_day": float(len(x) / len(daily)) if len(daily) else 0.0,
            "avg_day": float(daily.mean()) if len(daily) else None,
            "median_day": float(daily.median()) if len(daily) else None,
            "best_day": float(daily.max()) if len(daily) else None,
            "worst_day": float(daily.min()) if len(daily) else None,
            "positive_days_pct": float((daily > 0).mean() * 100.0) if len(daily) else None,
            "trade_max_drawdown": trade_dd,
            "daily_max_drawdown": daily_dd,
            "exit_reason_counts": {str(k): int(v) for k, v in x["exit_reason_norm"].value_counts().items()},
            "reverse_signal_stats": {
                "trades": rev_n,
                "wins": rev_wins,
                "losses": rev_losses,
                "breakeven": rev_be,
                "avg_pnl": float(reverse_df["pnl"].mean()) if rev_n else None,
                "win_rate_pct": float((rev_wins / rev_n) * 100.0) if rev_n else None,
                "loss_rate_pct": float((rev_losses / rev_n) * 100.0) if rev_n else None,
            },
            "target_hit_stats": {
                "trades": th_n,
                "avg_pnl": float(target_hit_df["pnl"].mean()) if th_n else None,
            },
            "timeout_stats": {
                "trades": to_n,
                "wins": to_wins,
                "losses": to_losses,
                "breakeven": to_be,
                "avg_pnl": float(timeout_df["pnl"].mean()) if to_n else None,
                "win_rate_pct": float((to_wins / to_n) * 100.0) if to_n else None,
                "loss_rate_pct": float((to_losses / to_n) * 100.0) if to_n else None,
            },
            "target_updates_mean": float(target_updates.mean()) if len(target_updates) else 0.0,
            "target_updates_median": float(target_updates.median()) if len(target_updates) else 0.0,
            "target_updates_max": int(target_updates.max()) if len(target_updates) else 0,
            "time_distribution": _time_distribution_stats(x),
        }

    all_stats = _bucket_stats(pdf)
    long_stats = _bucket_stats(pdf[pdf["side"] == "up"])
    short_stats = _bucket_stats(pdf[pdf["side"] == "down"])

    n_days = int(all_stats["n_days"])
    report = {
        # Backward-compatible top-level summary keys.
        "trades": int(all_stats["trades"]),
        "total_pnl": float(all_stats["total_pnl"]),
        "avg_trade": float(all_stats["avg_trade"]),
        "n_days": n_days,
        "avg_trades_per_day": float(all_stats["avg_trades_per_day"]),
        "avg_day": float(all_stats["avg_day"]) if (n_days >= 5 and all_stats["avg_day"] is not None) else None,
        "positive_days_pct": float(all_stats["positive_days_pct"]) if (n_days >= 5 and all_stats["positive_days_pct"] is not None) else None,
        "max_drawdown": float(all_stats["trade_max_drawdown"]),
        "long": {
            "trades": int(long_stats["trades"]),
            "wins": int((pdf[(pdf["side"] == "up")]["pnl"] > 0).sum()) if long_stats["trades"] else 0,
            "losses": int((pdf[(pdf["side"] == "up")]["pnl"] < 0).sum()) if long_stats["trades"] else 0,
            "win_rate": float(long_stats["win_rate_pct"]) if long_stats["trades"] else None,
            "total_pnl": float(long_stats["total_pnl"]),
            "avg_trade": float(long_stats["avg_trade"]),
        },
        "short": {
            "trades": int(short_stats["trades"]),
            "wins": int((pdf[(pdf["side"] == "down")]["pnl"] > 0).sum()) if short_stats["trades"] else 0,
            "losses": int((pdf[(pdf["side"] == "down")]["pnl"] < 0).sum()) if short_stats["trades"] else 0,
            "win_rate": float(short_stats["win_rate_pct"]) if short_stats["trades"] else None,
            "total_pnl": float(short_stats["total_pnl"]),
            "avg_trade": float(short_stats["avg_trade"]),
        },
        "streaks": {
            "max_win_streak": int(all_stats["max_win_streak"]),
            "max_loss_streak": int(all_stats["max_loss_streak"]),
            "current_win_streak": int(all_stats["current_win_streak"]),
            "current_loss_streak": int(all_stats["current_loss_streak"]),
        },
        "reverse_signal": dict(all_stats.get("reverse_signal_stats", {})),
        "target_hit": dict(all_stats.get("target_hit_stats", {})),
        "timeout": dict(all_stats.get("timeout_stats", {})),
        # Full-stat style payload.
        "all": all_stats,
        "long_up": long_stats,
        "short_down": short_stats,
    }
    return (report, pdf)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Image-like candle trend ML model.")
    p.add_argument("--table",            default="gold_prices")
    p.add_argument("--start-date",       default="2025-05-20")
    p.add_argument("--end-date",         default="2026-04-10")
    p.add_argument("--timeframe",        default="1min")
    p.add_argument("--disable-time-filter", action="store_true",
                   help="Disable UTC no-trade window filter.")
    p.add_argument("--window",           type=int,   default=150)
    p.add_argument("--window-15m",       type=int,   default=0,
                   help="Number of completed 15-min bars to include as a second image channel. "
                        "Disabled by default; set >0 only when using --two-branch.")
    p.add_argument("--min-window-range", type=float, default=40.0,
                   help="Keep windows where high-low > this value (default: 40)")
    p.add_argument("--min-15m-drop",    type=float, default=20.0,
                   help="Keep windows containing a 15-bar downside move >= this value in $ (default: 20, 0=off)")
    p.add_argument("--min-15m-rise",    type=float, default=0.0,
                   help="Also keep windows containing a 15-bar upside move >= this value in $ (default: 0, off)")
    p.add_argument("--last-bar-wr90-high", type=float, default=None,
                   help="Optional last-bar WR90 upper-extreme filter. Keep samples when WR90 >= this value.")
    p.add_argument("--last-bar-wr90-low", type=float, default=None,
                   help="Optional last-bar WR90 lower-extreme filter. Keep samples when WR90 <= this value.")
    p.add_argument("--horizon",          type=int,   default=25,
                   help="Prediction horizon in bars (default: 25)")
    p.add_argument("--trend-threshold",  type=float, default=0.004,
                   help="Move threshold for up/down labels (0.004 = 0.4%%)")
    p.add_argument("--adverse-limit",    type=float, default=15.0,
                   help="Absolute adverse move cap in price units (default: 15)")
    p.add_argument("--long-target-threshold", type=float, default=None,
                   help="Optional long-only target threshold override. Defaults to --trend-threshold.")
    p.add_argument("--short-target-threshold", type=float, default=None,
                   help="Optional short-only target threshold override. Defaults to --trend-threshold.")
    p.add_argument("--long-adverse-limit", type=float, default=None,
                   help="Optional long-only stop/adverse limit override. Defaults to --adverse-limit.")
    p.add_argument("--short-adverse-limit", type=float, default=None,
                   help="Optional short-only stop/adverse limit override. Defaults to --adverse-limit.")
    p.add_argument("--test-start-date", default="2026-02-06",
                   help="Optional UTC date/time to start test set (e.g. 2026-02-06). "
                        "If invalid for current data window, falls back to --test-size.")
    p.add_argument("--test-size",        type=float, default=0.30)
    p.add_argument("--max-samples",      type=int,   default=None)
    p.add_argument("--optimize",         action="store_true",
                   help="Grid-search min-window-range on train split (threshold and horizon fixed)")
    p.add_argument("--optimize-prob",    action="store_true",
                   help="Grid-search stage1/stage2 probability gates on a train validation split")
    p.add_argument("--two-branch",       action="store_true",
                   help="Train separate GBMs on the 1-min and 15-min image slices per stage, "
                        "then average their predict_proba (requires --window-15m > 0)")
    p.add_argument("--two-branch-stage", default="both",
                   choices=["both", "stage2", "stage1"],
                   help="Which stage(s) use the two-branch model: "
                        "'both' (default), 'stage2' (1m-only for stage1, two-branch for stage2), "
                        "'stage1' (two-branch for stage1, 1m-only for stage2)")
    p.add_argument("--max-flat-ratio",   type=float, default=4.0,
                   help="Max ratio flat:trend samples in stage-1 training (default: 4.0)")
    p.add_argument("--classifier",       default="gradient_boosting",
                   choices=["gradient_boosting", "logistic"],
                   help="Model type for both stages (default: gradient_boosting)")
    p.add_argument("--stage1-min-prob",  type=float, default=0.55,
                   help="Stage-1 minimum P(trend) to enable stage-2 (default: 0.55)")
    p.add_argument("--stage1-min-prob-1m", type=float, default=None,
                   help="Optional branch gate for stage-1 1m model. If set, requires P1m(trend) >= this.")
    p.add_argument("--stage1-min-prob-15m", type=float, default=None,
                   help="Optional branch gate for stage-1 15m model. If set, requires P15m(trend) >= this.")
    p.add_argument("--stage2-min-prob",  type=float, default=0.55,
                   help="Stage-2 minimum directional confidence max(P(up),P(down)) (default: 0.55)")
    p.add_argument("--stage2-min-prob-up", "--stage2-min-prob-long", dest="stage2_min_prob_up", type=float, default=None,
                   help="Optional Stage-2 confidence gate for UP/LONG predictions. Defaults to --stage2-min-prob.")
    p.add_argument("--stage2-min-prob-down", "--stage2-min-prob-short", dest="stage2_min_prob_down", type=float, default=None,
                   help="Optional Stage-2 confidence gate for DOWN/SHORT predictions. Defaults to --stage2-min-prob.")
    p.add_argument("--stage2-min-prob-1m", type=float, default=None,
                   help="Optional branch gate for stage-2 1m model confidence.")
    p.add_argument("--stage2-min-prob-15m", type=float, default=None,
                   help="Optional branch gate for stage-2 15m model confidence.")
    p.add_argument("--use-state-features", action="store_true",
                   help="Append causal state features (prev prediction continuity + trade status).")
    p.add_argument("--use-15m-wick-features", action="store_true",
                   help="Append lower/upper long-wick channels derived from the last completed 15-minute bar using the support.py-style definition.")
    p.add_argument("--wick-feature-min-range", type=float, default=40.0,
                   help="Minimum 15-minute bar range required for long-wick features (default: 40.0).")
    p.add_argument("--wick-feature-min-pct", type=float, default=35.0,
                   help="Minimum wick percentage required for long-wick features (default: 35.0).")
    p.add_argument("--wick-feature-min-volume", type=float, default=3000.0,
                   help="Minimum 15-minute volume required for long-wick features (default: 3000.0, matching support.py).")
    p.add_argument("--use-stage1-day-ohl-utc2", action="store_true",
                   help="Append UTC+2 day-context features (Dopen/DHigh/DLow, relative to current close) to stage-1 only.")
    p.add_argument("--state-oof-splits", type=int, default=5,
                   help="Number of TimeSeriesSplit folds for train OOF state features (default: 5).")
    p.add_argument("--pred-history-len", type=int, default=150,
                   help="Number of previous predictions to encode as state inputs (default: 150).")
    p.add_argument("--allow-overlap-backtest", action="store_true",
                   help="Allow overlapping trades in backtest accounting/export (default: off / single-position).")
    p.add_argument("--reverse-exit-prob", type=float, default=DEFAULT_REVERSE_EXIT_PROB,
                   help="If a reverse-direction signal arrives while a position is open, exit when its side probability exceeds this value (default: 0.70).")
    p.add_argument("--max-hold-minutes", type=float, default=None,
                   help="Optional backtest-only hard timeout in minutes. Keeps model horizon unchanged and only caps simulated hold length.")
    p.add_argument("--eval-mode", default="walk_forward", choices=["single_split", "walk_forward"],
                   help="Evaluation mode: single chronological split or walk-forward retraining (default: walk_forward).")
    p.add_argument("--wf-init-train-months", type=int, default=6,
                   help="Walk-forward initial training window in months (default: 6).")
    p.add_argument("--wf-retrain-days", type=int, default=14,
                   help="Walk-forward retraining cadence in days (default: 14).")
    p.add_argument("--wf-max-train-days", type=int, default=365,
                   help="Walk-forward max train span in days before switching to rolling window (default: 365).")
    p.add_argument("--wf-min-train-samples", type=int, default=300,
                   help="Minimum samples required in each walk-forward training window (default: 300).")
    p.add_argument("--wf-disable-sweep", action="store_true",
                   help="Disable per-retrain config sweep in walk-forward mode.")
    p.add_argument("--wf-sweep-flat-ratios", default="3,4,5",
                   help="Comma-separated max_flat_ratio candidates for each retrain sweep (default: 3,4,5).")
    p.add_argument("--wf-sweep-stage1-probs", default=None,
                   help="Optional comma-separated stage1 probability gates for each retrain sweep. "
                        "Defaults to built-in grid 0.50,0.55,0.60,0.65,0.70.")
    p.add_argument("--wf-sweep-stage2-probs", default=None,
                   help="Optional comma-separated stage2 base probability gates for each retrain sweep. "
                        "Defaults to built-in grid 0.50,0.55,0.60,0.65,0.70.")
    p.add_argument("--wf-sweep-stage2-long-probs", dest="wf_sweep_stage2_long_probs", default=None,
                   help="Optional comma-separated Stage-2 UP/LONG probability gates to sweep per retrain cycle. "
                        "Defaults to fixed --stage2-min-prob-up (or --stage2-min-prob when unset).")
    p.add_argument("--wf-sweep-stage2-short-probs", dest="wf_sweep_stage2_short_probs", default=None,
                   help="Optional comma-separated Stage-2 DOWN/SHORT probability gates to sweep per retrain cycle. "
                        "Defaults to fixed --stage2-min-prob-down (or --stage2-min-prob when unset).")
    p.add_argument("--wf-sweep-val-ratio", type=float, default=0.20,
                   help="Validation tail ratio used by per-retrain sweep (default: 0.20).")
    p.add_argument("--wf-sweep-min-val-samples", type=int, default=50,
                   help="Minimum validation samples required for per-retrain sweep (default: 50).")
    p.add_argument("--wf-anchor-mode", default="weekend_fri_close", choices=["elapsed_days", "weekend_fri_close"],
                   help="Walk-forward retrain anchor policy: elapsed_days or weekend_fri_close (Friday 17:00 New York, default).")
    p.add_argument("--wf-cycle-model-dir", default=None,
                   help="Optional directory to save one model artifact per walk-forward cycle. "
                        "Default: <model-out stem>_cycles")
    p.add_argument("--no-wf-save-cycle-models", action="store_true",
                   help="Disable saving per-cycle walk-forward model artifacts.")
    p.add_argument("--prep-cache-dir", default=None,
                   help="Optional directory for reusable prep cache (bars + supervised dataset).")
    p.add_argument("--refresh-prep-cache", action="store_true",
                   help="Ignore any existing prep cache artifacts and rebuild them.")
    p.add_argument("--random-state",     type=int,   default=42)
    p.add_argument("--model-in",         default=None,
                   help="Optional existing model artifact (.joblib). When set, skip training and only run inference/backtest (single_split only).")
    p.add_argument("--weak-periods-json", default=None,
                   help="Optional JSON file listing weak session/day/hour cells to exclude at backtest entry time. Accepts either a top-level weak_cells array or a raw array of {session, day, hour} objects.")
    p.add_argument("--model-out",        default="training/image_trend_model.joblib")
    p.add_argument("--report-out",       default="training/image_trend_report.json")
    p.add_argument("--trades-out",       default=None,
                   help="Optional CSV path to save test trades (ts, pred, pnl).")
    args = p.parse_args()
    wf_sweep_flat_ratios = [
        float(tok.strip())
        for tok in str(args.wf_sweep_flat_ratios).split(",")
        if tok.strip()
    ]
    wf_sweep_stage1_probs = [
        float(tok.strip())
        for tok in str(args.wf_sweep_stage1_probs).split(",")
        if tok.strip()
    ] if args.wf_sweep_stage1_probs is not None else []
    wf_sweep_stage2_probs = [
        float(tok.strip())
        for tok in str(args.wf_sweep_stage2_probs).split(",")
        if tok.strip()
    ] if args.wf_sweep_stage2_probs is not None else []
    wf_sweep_stage2_long_probs = [
        float(tok.strip())
        for tok in str(args.wf_sweep_stage2_long_probs).split(",")
        if tok.strip()
    ] if args.wf_sweep_stage2_long_probs is not None else []
    wf_sweep_stage2_short_probs = [
        float(tok.strip())
        for tok in str(args.wf_sweep_stage2_short_probs).split(",")
        if tok.strip()
    ] if args.wf_sweep_stage2_short_probs is not None else []
    return Config(
        table=args.table, start_date=args.start_date, end_date=args.end_date,
        timeframe=args.timeframe, disable_time_filter=args.disable_time_filter, window=args.window,
        window_15m=args.window_15m,
        min_window_range=args.min_window_range,
        min_15m_drop=args.min_15m_drop,
        min_15m_rise=args.min_15m_rise,
        last_bar_wr90_high=args.last_bar_wr90_high,
        last_bar_wr90_low=args.last_bar_wr90_low,
        horizon=args.horizon,
        trend_threshold=args.trend_threshold, adverse_limit=args.adverse_limit,
        long_target_threshold=args.long_target_threshold,
        short_target_threshold=args.short_target_threshold,
        long_adverse_limit=args.long_adverse_limit,
        short_adverse_limit=args.short_adverse_limit,
        test_start_date=args.test_start_date,
        test_size=args.test_size, max_samples=args.max_samples,
        optimize=args.optimize, optimize_prob=args.optimize_prob,
        two_branch=args.two_branch,
        two_branch_stage=args.two_branch_stage,
        max_flat_ratio=args.max_flat_ratio,
        classifier=args.classifier,
        stage1_min_prob=args.stage1_min_prob,
        stage2_min_prob=args.stage2_min_prob,
        stage2_min_prob_up=args.stage2_min_prob_up,
        stage2_min_prob_down=args.stage2_min_prob_down,
        stage1_min_prob_1m=args.stage1_min_prob_1m,
        stage1_min_prob_15m=args.stage1_min_prob_15m,
        stage2_min_prob_1m=args.stage2_min_prob_1m,
        stage2_min_prob_15m=args.stage2_min_prob_15m,
        use_state_features=args.use_state_features,
        use_15m_wick_features=args.use_15m_wick_features,
        wick_feature_min_range=args.wick_feature_min_range,
        wick_feature_min_pct=args.wick_feature_min_pct,
        wick_feature_min_volume=args.wick_feature_min_volume,
        use_stage1_day_ohl_utc2=args.use_stage1_day_ohl_utc2,
        state_oof_splits=args.state_oof_splits,
        pred_history_len=args.pred_history_len,
        allow_overlap_backtest=args.allow_overlap_backtest,
        reverse_exit_prob=args.reverse_exit_prob,
        max_hold_minutes=args.max_hold_minutes,
        weak_periods_json=args.weak_periods_json,
        eval_mode=args.eval_mode,
        wf_init_train_months=args.wf_init_train_months,
        wf_retrain_days=args.wf_retrain_days,
        wf_max_train_days=args.wf_max_train_days,
        wf_min_train_samples=args.wf_min_train_samples,
        wf_disable_sweep=args.wf_disable_sweep,
        wf_sweep_flat_ratios=wf_sweep_flat_ratios,
        wf_sweep_stage1_probs=wf_sweep_stage1_probs,
        wf_sweep_stage2_probs=wf_sweep_stage2_probs,
        wf_sweep_stage2_long_probs=wf_sweep_stage2_long_probs,
        wf_sweep_stage2_short_probs=wf_sweep_stage2_short_probs,
        wf_sweep_val_ratio=args.wf_sweep_val_ratio,
        wf_sweep_min_val_samples=args.wf_sweep_min_val_samples,
        wf_anchor_mode=args.wf_anchor_mode,
        wf_cycle_model_dir=args.wf_cycle_model_dir,
        wf_save_cycle_models=not args.no_wf_save_cycle_models,
        prep_cache_dir=args.prep_cache_dir,
        refresh_prep_cache=args.refresh_prep_cache,
        random_state=args.random_state,
        model_in=args.model_in,
        model_out=args.model_out, report_out=args.report_out,
        trades_out=args.trades_out,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = parse_args()
    t_main = _log("Starting image-trend pipeline")
    weak_period_cells = _load_weak_period_cells(cfg.weak_periods_json)
    if weak_period_cells:
        _log(f"Loaded weak-period exclusions: {len(weak_period_cells)} cells from {cfg.weak_periods_json}")

    if cfg.window_15m < 0:
        raise ValueError("--window-15m must be >= 0")
    if cfg.state_oof_splits < 2:
        raise ValueError("--state-oof-splits must be >= 2")
    if cfg.pred_history_len < 1:
        raise ValueError("--pred-history-len must be >= 1")
    if cfg.last_bar_wr90_high is not None and not (-100.0 <= cfg.last_bar_wr90_high <= 0.0):
        raise ValueError("--last-bar-wr90-high must be within [-100, 0]")
    if cfg.last_bar_wr90_low is not None and not (-100.0 <= cfg.last_bar_wr90_low <= 0.0):
        raise ValueError("--last-bar-wr90-low must be within [-100, 0]")
    if cfg.last_bar_wr90_high is not None and cfg.last_bar_wr90_low is not None and cfg.last_bar_wr90_high <= cfg.last_bar_wr90_low:
        raise ValueError("--last-bar-wr90-high must be greater than --last-bar-wr90-low")
    if cfg.wf_min_train_samples < 100:
        raise ValueError("--wf-min-train-samples must be >= 100")
    if not cfg.wf_sweep_flat_ratios:
        raise ValueError("--wf-sweep-flat-ratios must not be empty")
    if cfg.wf_sweep_stage1_probs and any(p <= 0.0 or p >= 1.0 for p in cfg.wf_sweep_stage1_probs):
        raise ValueError("--wf-sweep-stage1-probs values must be in (0, 1)")
    if cfg.wf_sweep_stage2_probs and any(p <= 0.0 or p >= 1.0 for p in cfg.wf_sweep_stage2_probs):
        raise ValueError("--wf-sweep-stage2-probs values must be in (0, 1)")
    if cfg.wf_sweep_stage2_long_probs and any(p <= 0.0 or p >= 1.0 for p in cfg.wf_sweep_stage2_long_probs):
        raise ValueError("--wf-sweep-stage2-long-probs values must be in (0, 1)")
    if cfg.wf_sweep_stage2_short_probs and any(p <= 0.0 or p >= 1.0 for p in cfg.wf_sweep_stage2_short_probs):
        raise ValueError("--wf-sweep-stage2-short-probs values must be in (0, 1)")
    if cfg.wf_sweep_val_ratio <= 0.0 or cfg.wf_sweep_val_ratio >= 0.5:
        raise ValueError("--wf-sweep-val-ratio must be in (0, 0.5)")
    if cfg.wf_sweep_min_val_samples < 30:
        raise ValueError("--wf-sweep-min-val-samples must be >= 30")

    cache_info: dict[str, object] = {}

    t = _log("Preparing bars...")
    bars, bars_ask, bars_bid, bars_cache_info = _load_or_build_bars(cfg)
    cache_info["bars"] = bars_cache_info
    _log(f"Bars after resample: {len(bars):,}", t0=t)

    bar_split, split_mode = _resolve_split_index(bars.index, cfg.test_size, cfg.test_start_date)
    _log(f"Bar split boundary: train_bars={bar_split:,}  test_bars={len(bars)-bar_split:,}  ({split_mode})")
    train_bars = bars.iloc[:bar_split].copy()

    threshold = cfg.trend_threshold
    horizon   = cfg.horizon
    min_window_range = cfg.min_window_range
    two_branch = cfg.two_branch
    effective_window_15m = cfg.window_15m
    long_target_threshold, short_target_threshold = _resolve_directional_targets(
        cfg.trend_threshold,
        cfg.long_target_threshold,
        cfg.short_target_threshold,
    )
    long_adverse_limit, short_adverse_limit = _resolve_directional_stops(
        cfg.adverse_limit,
        cfg.long_adverse_limit,
        cfg.short_adverse_limit,
    )
    # When --model-in is used the model is not retrained, so the dataset cache
    # key does not need to vary with la/sa — pin both to adverse_limit so all
    # stop-sweep combos share a single cached dataset.
    if cfg.model_in is not None:
        dataset_long_adverse_limit  = float(cfg.adverse_limit)
        dataset_short_adverse_limit = float(cfg.adverse_limit)
    else:
        dataset_long_adverse_limit  = long_adverse_limit
        dataset_short_adverse_limit = short_adverse_limit
    if cfg.optimize:
        min_window_range = optimize_min_range(
            train_bars,
            cfg.window,
            threshold,
            horizon,
            cfg.adverse_limit,
            long_target_threshold,
            short_target_threshold,
            long_adverse_limit,
            short_adverse_limit,
            cfg.max_flat_ratio,
            cfg.classifier,
            cfg.random_state,
            min_15m_drop=cfg.min_15m_drop,
            min_15m_rise=cfg.min_15m_rise,
            last_bar_wr90_high=cfg.last_bar_wr90_high,
            last_bar_wr90_low=cfg.last_bar_wr90_low,
            window_15m=cfg.window_15m,
            two_branch=two_branch,
            two_branch_stage=cfg.two_branch_stage,
            use_stage1_day_ohl_utc2=cfg.use_stage1_day_ohl_utc2,
        )
        _log(f"Optimized min_window_range={min_window_range}")

    t = _log("Preparing supervised dataset...")
    X_tensor, y, ts, curr_a, entry_a, fut_a, entry_ts_a, fut_ts_a, kept, skipped, dataset_cache_info = _load_or_build_supervised_dataset(
        bars,
        cfg,
        min_window_range,
        threshold,
        long_target_threshold,
        short_target_threshold,
        dataset_long_adverse_limit,
        dataset_short_adverse_limit,
    )
    cache_info["dataset"] = dataset_cache_info
    _log(f"Dataset ready: kept={kept:,} skipped={skipped:,}", t0=t)

    if cfg.max_samples is not None and kept > cfg.max_samples:
        start_idx = kept - cfg.max_samples
        X_tensor = X_tensor[start_idx:]
        y = y[start_idx:]
        ts = ts[start_idx:]
        curr_a = curr_a[start_idx:]
        entry_a = entry_a[start_idx:]
        fut_a = fut_a[start_idx:]
        entry_ts_a = entry_ts_a[start_idx:]
        fut_ts_a = fut_ts_a[start_idx:]
        kept = int(len(y))
        _log(f"Applied --max-samples: using latest {kept:,} samples")

    if kept < 200:
        raise ValueError("Not enough supervised samples after filtering.")

    if cfg.window_15m <= 0 and two_branch:
        _log("Disabling --two-branch because --window-15m <= 0")
        two_branch = False

    image_channel_names = make_image_channel_names(cfg.use_15m_wick_features)
    n_1m_feats = int(X_tensor.shape[1]) * cfg.window
    X_flat = flatten_tensors(X_tensor)
    n_total = len(y)
    stage1_extra_feature_names = list(STAGE1_DAY_OHL_FEATURE_NAMES) if cfg.use_stage1_day_ohl_utc2 else []
    stage1_day_ohl_full = (
        _build_stage1_day_ohl_features(bars, ts, curr_a)
        if cfg.use_stage1_day_ohl_utc2 else None
    )

    sample_split_mode = "walk_forward"
    split = 0
    walkforward_cycles_report: list[dict[str, object]] = []

    stage1_prob = cfg.stage1_min_prob
    stage2_prob = cfg.stage2_min_prob
    stage2_prob_up = cfg.stage2_min_prob_up
    stage2_prob_down = cfg.stage2_min_prob_down
    model_meta: dict[str, object] = {}

    reuse_model_path = Path(cfg.model_in) if cfg.model_in else None
    if reuse_model_path is not None:
        if cfg.eval_mode != "single_split":
            raise ValueError("--model-in currently supports only --eval-mode single_split")
        if cfg.optimize or cfg.optimize_prob:
            raise ValueError("--model-in cannot be combined with --optimize or --optimize-prob")

    state_feature_names = make_state_feature_names(cfg.pred_history_len) if cfg.use_state_features else []

    if cfg.eval_mode == "single_split":
        split, sample_split_mode = _resolve_split_index(ts, cfg.test_size, cfg.test_start_date)
        if split < 100 or (n_total - split) < 50:
            raise ValueError("Not enough samples. Increase date range.")

        X_tr, X_te = X_flat[:split], X_flat[split:]
        y_tr, y_te = y[:split], y[split:]
        ts_te = ts[split:]
        curr_tr = curr_a[:split]
        curr_te = curr_a[split:]
        entry_tr = entry_a[:split]
        entry_te = entry_a[split:]
        entry_ts_te = entry_ts_a[split:]
        fut_te = fut_a[split:]
        fut_ts_te = fut_ts_a[split:]
        stage1_extra_tr = None if stage1_day_ohl_full is None else stage1_day_ohl_full[:split]
        stage1_extra_te = None if stage1_day_ohl_full is None else stage1_day_ohl_full[split:]

        _log(f"Split: train={split:,}  test={n_total - split:,}  ({sample_split_mode})")
        print("Class balance train (down/flat/up/risky):", np.bincount(y_tr, minlength=4).tolist())
        print("Class balance test  (down/flat/up/risky):", np.bincount(y_te, minlength=4).tolist())

        if cfg.use_state_features and reuse_model_path is None:
            t = _log("Building causal state features (OOF train + base test)...")
            meta_tr = _build_oof_state_features(
                X_tr,
                y_tr,
                curr_tr,
                entry_tr,
                cfg,
                n_1m_feats,
                two_branch,
                stage1_extra_tr=stage1_extra_tr,
            )
            m1_base, m2_base = train_two_stage(
                X_tr,
                y_tr,
                cfg.random_state,
                cfg.max_flat_ratio,
                cfg.classifier,
                two_branch=two_branch,
                n_1m_feats=n_1m_feats,
                two_branch_stage=cfg.two_branch_stage,
                stage1_extra=stage1_extra_tr,
            )
            pred_te_base = predict_two_stage(
                X_te,
                m1_base,
                m2_base,
                stage1_min_prob=stage1_prob,
                stage2_min_prob=stage2_prob,
                stage2_min_prob_up=stage2_prob_up,
                stage2_min_prob_down=stage2_prob_down,
                stage1_min_prob_1m=cfg.stage1_min_prob_1m,
                stage1_min_prob_15m=cfg.stage1_min_prob_15m,
                stage2_min_prob_1m=cfg.stage2_min_prob_1m,
                stage2_min_prob_15m=cfg.stage2_min_prob_15m,
                stage1_extra=stage1_extra_te,
            )
            meta_te_base = _compute_state_features_from_pred(
                pred_te_base,
                curr_te,
                entry_te,
                cfg.adverse_limit,
                cfg.trend_threshold,
                long_target_threshold=long_target_threshold,
                short_target_threshold=short_target_threshold,
                long_adverse_limit=long_adverse_limit,
                short_adverse_limit=short_adverse_limit,
                pred_history_len=cfg.pred_history_len,
            )
            X_tr = np.hstack([X_tr, meta_tr])
            X_te = np.hstack([X_te, meta_te_base])
            _log(f"State features added: +{meta_tr.shape[1]} columns", t0=t)
        elif cfg.use_state_features and reuse_model_path is not None:
            _log("Skipping train-time state feature build because --model-in will generate saved state features causally")
        else:
            state_feature_names = []

        if cfg.optimize_prob:
            t = _log("Optimising stage probability thresholds on train-validation split...")
            split_fit = int(len(X_tr) * 0.8)
            if split_fit < 100 or (len(X_tr) - split_fit) < 50:
                _log("Not enough train samples for prob optimization; using provided thresholds", t0=t)
            else:
                X_fit, y_fit = X_tr[:split_fit], y_tr[:split_fit]
                X_val = X_tr[split_fit:]
                curr_fit = curr_tr[:split_fit]
                curr_val = curr_tr[split_fit:]
                entry_fit = entry_tr[:split_fit]
                entry_val = entry_tr[split_fit:]
                entry_ts_fit = entry_ts_a[:split][:split_fit]
                entry_ts_val = entry_ts_a[:split][split_fit:]
                fut_fit = fut_a[:split][:split_fit]
                fut_val = fut_a[:split][split_fit:]
                fut_ts_fit = fut_ts_a[:split][:split_fit]
                fut_ts_val = fut_ts_a[:split][split_fit:]
                _ = (curr_fit, entry_fit, entry_ts_fit, fut_fit, fut_ts_fit)  # keep fit slices explicit for readability

                m1_tmp, m2_tmp = train_two_stage(
                    X_fit,
                    y_fit,
                    cfg.random_state,
                    cfg.max_flat_ratio,
                    cfg.classifier,
                    two_branch=two_branch,
                    n_1m_feats=n_1m_feats,
                    two_branch_stage=cfg.two_branch_stage,
                )
                stage1_prob, stage2_prob = optimize_prob_thresholds(
                    X_val,
                    ts[split_fit:split],
                    entry_ts_val,
                    fut_ts_val,
                    curr_val,
                    entry_val,
                    fut_val,
                    m1_tmp,
                    m2_tmp,
                    adverse_limit=cfg.adverse_limit,
                    long_target_threshold=long_target_threshold,
                    short_target_threshold=short_target_threshold,
                    long_adverse_limit=long_adverse_limit,
                    short_adverse_limit=short_adverse_limit,
                    stage2_min_prob_up=stage2_prob_up,
                    stage2_min_prob_down=stage2_prob_down,
                    stage1_min_prob_1m=cfg.stage1_min_prob_1m,
                    stage1_min_prob_15m=cfg.stage1_min_prob_15m,
                    stage2_min_prob_1m=cfg.stage2_min_prob_1m,
                    stage2_min_prob_15m=cfg.stage2_min_prob_15m,
                    stage1_extra_val=None if stage1_extra_tr is None else stage1_extra_tr[split_fit:],
                )
                _log(f"→ Best probability gates: stage1={stage1_prob:.2f}, stage2={stage2_prob:.2f}", t0=t)

        if reuse_model_path is not None:
            t = _log(f"Loading trained model from {reuse_model_path}...")
            bundle = joblib.load(reuse_model_path)
            if not isinstance(bundle, dict) or "stage1" not in bundle or "stage2" not in bundle:
                raise ValueError("--model-in artifact must contain 'stage1' and 'stage2'")
            m1 = bundle["stage1"]
            m2 = bundle["stage2"]
            model_meta = {
                "model_in": str(reuse_model_path),
                "saved_stage1_min_prob": bundle.get("stage1_min_prob"),
                "saved_stage2_min_prob": bundle.get("stage2_min_prob"),
            }
            expected_shape = bundle.get("feature_shape")
            if isinstance(expected_shape, (list, tuple)) and len(expected_shape) == 2:
                got_shape = [int(X_tensor.shape[1]), int(X_tensor.shape[2])]
                if [int(expected_shape[0]), int(expected_shape[1])] != got_shape:
                    raise ValueError(
                        "Feature shape mismatch for --model-in: "
                        f"model={expected_shape}, current={got_shape}. "
                        "Use matching --window/--window-15m and data build settings."
                    )
            saved_cfg = bundle.get("config") if isinstance(bundle.get("config"), dict) else {}
            saved_use_state = bool(bundle.get("use_state_features", saved_cfg.get("use_state_features", False)))
            if saved_use_state != bool(cfg.use_state_features):
                raise ValueError(
                    "--model-in feature mismatch: --use-state-features must match the saved artifact "
                    f"(saved={saved_use_state}, current={bool(cfg.use_state_features)})"
                )
            saved_use_wick = bool(bundle.get("use_15m_wick_features", saved_cfg.get("use_15m_wick_features", False)))
            if saved_use_wick != bool(cfg.use_15m_wick_features):
                raise ValueError(
                    "--model-in feature mismatch: --use-15m-wick-features must match the saved artifact "
                    f"(saved={saved_use_wick}, current={bool(cfg.use_15m_wick_features)})"
                )
            state_feat_count = len(make_state_feature_names(int(saved_cfg.get("pred_history_len", cfg.pred_history_len)))) if saved_use_state else 0
            expected_features = getattr(m1, "n_features_in_", None)
            got_s1_features = int(X_te.shape[1]) + state_feat_count + (0 if stage1_extra_te is None else int(stage1_extra_te.shape[1]))
            if isinstance(expected_features, numbers.Integral) and int(expected_features) != got_s1_features:
                raise ValueError(
                    "Stage-1 input feature count mismatch for --model-in: "
                    f"model={int(expected_features)}, current={got_s1_features}."
                )
            expected_s2_features = getattr(m2, "n_features_in_", None)
            got_s2_features = int(X_te.shape[1]) + state_feat_count
            if isinstance(expected_s2_features, numbers.Integral) and int(expected_s2_features) != got_s2_features:
                raise ValueError(
                    "Stage-2 input feature count mismatch for --model-in: "
                    f"model={int(expected_s2_features)}, current={got_s2_features}."
                )
            two_branch = bool(bundle.get("two_branch", two_branch))
            _log("Loaded model artifact (training skipped)", t0=t)
        else:
            t = _log("Training two-stage model...")
            m1, m2 = train_two_stage(
                X_tr,
                y_tr,
                cfg.random_state,
                cfg.max_flat_ratio,
                cfg.classifier,
                two_branch=two_branch,
                n_1m_feats=n_1m_feats,
                two_branch_stage=cfg.two_branch_stage,
                stage1_extra=stage1_extra_tr,
            )
            _log("Training done", t0=t)
        if reuse_model_path is not None:
            pred_meta = predict_saved_bundle_details(
                X_te,
                curr_te,
                entry_te,
                bundle,
                stage1_min_prob=stage1_prob,
                stage2_min_prob=stage2_prob,
                stage2_min_prob_up=stage2_prob_up,
                stage2_min_prob_down=stage2_prob_down,
                stage1_extra=stage1_extra_te,
            )
        else:
            pred_meta = predict_two_stage_details(
                X_te,
                m1,
                m2,
                stage1_min_prob=stage1_prob,
                stage2_min_prob=stage2_prob,
                stage2_min_prob_up=stage2_prob_up,
                stage2_min_prob_down=stage2_prob_down,
                stage1_min_prob_1m=cfg.stage1_min_prob_1m,
                stage1_min_prob_15m=cfg.stage1_min_prob_15m,
                stage2_min_prob_1m=cfg.stage2_min_prob_1m,
                stage2_min_prob_15m=cfg.stage2_min_prob_15m,
                stage1_extra=stage1_extra_te,
            )
        y_pred = pred_meta["pred"]
    else:
        state_feature_names = make_state_feature_names(cfg.pred_history_len) if cfg.use_state_features else []
        cycle_model_dir: Optional[Path] = None
        if cfg.eval_mode == "walk_forward" and cfg.wf_save_cycle_models:
            if cfg.wf_cycle_model_dir:
                cycle_model_dir = Path(cfg.wf_cycle_model_dir)
            else:
                base_model_path = Path(cfg.model_out)
                cycle_model_dir = base_model_path.parent / f"{base_model_path.stem}_cycles"
            cycle_model_dir.mkdir(parents=True, exist_ok=True)
        windows = build_walkforward_windows(
            ts,
            init_train_months=cfg.wf_init_train_months,
            retrain_days=cfg.wf_retrain_days,
            max_train_days=cfg.wf_max_train_days,
            min_train_samples=cfg.wf_min_train_samples,
            anchor_mode=cfg.wf_anchor_mode,
        )
        _log(
            f"Walk-forward schedule ready: cycles={len(windows)}  init_train={cfg.wf_init_train_months}m  "
            f"retrain_every={cfg.wf_retrain_days}d  max_train={cfg.wf_max_train_days}d  "
            f"anchor={cfg.wf_anchor_mode}"
        )

        y_pred_all = np.full(n_total, LABEL_FLAT, dtype=np.int64)
        signal_prob_all = np.zeros(n_total, dtype=np.float64)
        eval_mask = np.zeros(n_total, dtype=bool)
        m1 = None
        m2 = None
        wf_done_cycles = 0
        wf_cum_trades = 0.0
        wf_cum_pnl = 0.0

        for wf in windows:
            tr_sl = slice(wf.train_start, wf.train_end)
            te_sl = slice(wf.test_start, wf.test_end)
            if (wf.train_end - wf.train_start) < cfg.wf_min_train_samples:
                continue

            t_cycle = _log(
                f"WF cycle {wf.cycle_id}/{len(windows)}: train={wf.train_start:,}:{wf.train_end:,} "
                f"test={wf.test_start:,}:{wf.test_end:,} mode={wf.window_mode}"
            )

            cycle_cfg = {
                "used_sweep": False,
                "reason": "disabled",
                "max_flat_ratio": float(cfg.max_flat_ratio),
                "stage1_min_prob": float(stage1_prob),
                "stage2_min_prob": float(stage2_prob),
                "stage2_min_prob_up": float(stage2_prob if stage2_prob_up is None else stage2_prob_up),
                "stage2_min_prob_down": float(stage2_prob if stage2_prob_down is None else stage2_prob_down),
                "score": None,
                "signals": 0,
                "total_pnl": 0.0,
                "candidate_count": 0,
            }
            if not cfg.wf_disable_sweep:
                cycle_cfg = select_wf_cycle_config(
                    X_flat[tr_sl],
                    y[tr_sl],
                    ts[tr_sl],
                    entry_ts_a[tr_sl],
                    fut_ts_a[tr_sl],
                    curr_a[tr_sl],
                    entry_a[tr_sl],
                    fut_a[tr_sl],
                    cfg,
                    n_1m_feats,
                    two_branch,
                    long_adverse_limit,
                    short_adverse_limit,
                    stage1_extra_tr=None if stage1_day_ohl_full is None else stage1_day_ohl_full[tr_sl],
                )
                _log(
                    "WF sweep: "
                    f"flat_ratio={cycle_cfg['max_flat_ratio']:.2f} "
                    f"s1={cycle_cfg['stage1_min_prob']:.2f} "
                    f"s2={cycle_cfg['stage2_min_prob']:.2f} "
                    f"stage2_up={cycle_cfg.get('stage2_min_prob_up', cycle_cfg['stage2_min_prob']):.2f} "
                    f"stage2_down={cycle_cfg.get('stage2_min_prob_down', cycle_cfg['stage2_min_prob']):.2f} "
                    f"reason={cycle_cfg['reason']}"
                )

            X_tr_cycle = X_flat[tr_sl]
            X_te_cycle = X_flat[te_sl]
            stage1_extra_tr_cycle = None if stage1_day_ohl_full is None else stage1_day_ohl_full[tr_sl]
            stage1_extra_te_cycle = None if stage1_day_ohl_full is None else stage1_day_ohl_full[te_sl]

            if cfg.use_state_features:
                t_state = _log(f"WF cycle {wf.cycle_id}: building causal state features...")
                y_tr_cycle = y[tr_sl]
                curr_tr_cycle = curr_a[tr_sl]
                curr_te_cycle = curr_a[te_sl]
                entry_tr_cycle = entry_a[tr_sl]
                entry_te_cycle = entry_a[te_sl]

                meta_tr_cycle = _build_oof_state_features(
                    X_tr_cycle,
                    y_tr_cycle,
                    curr_tr_cycle,
                    entry_tr_cycle,
                    cfg,
                    n_1m_feats,
                    two_branch,
                    stage1_extra_tr=stage1_extra_tr_cycle,
                )
                m1_base, m2_base = train_two_stage(
                    X_tr_cycle,
                    y_tr_cycle,
                    cfg.random_state + wf.cycle_id,
                    float(cycle_cfg["max_flat_ratio"]),
                    cfg.classifier,
                    two_branch=two_branch,
                    n_1m_feats=n_1m_feats,
                    two_branch_stage=cfg.two_branch_stage,
                    stage1_extra=stage1_extra_tr_cycle,
                )
                pred_te_base = predict_two_stage(
                    X_te_cycle,
                    m1_base,
                    m2_base,
                    stage1_min_prob=float(cycle_cfg["stage1_min_prob"]),
                    stage2_min_prob=float(cycle_cfg["stage2_min_prob"]),
                    stage2_min_prob_up=float(cycle_cfg.get("stage2_min_prob_up", cycle_cfg["stage2_min_prob"])),
                    stage2_min_prob_down=float(cycle_cfg.get("stage2_min_prob_down", cycle_cfg["stage2_min_prob"])),
                    stage1_min_prob_1m=cfg.stage1_min_prob_1m,
                    stage1_min_prob_15m=cfg.stage1_min_prob_15m,
                    stage2_min_prob_1m=cfg.stage2_min_prob_1m,
                    stage2_min_prob_15m=cfg.stage2_min_prob_15m,
                    stage1_extra=stage1_extra_te_cycle,
                )
                meta_te_base = _compute_state_features_from_pred(
                    pred_te_base,
                    curr_te_cycle,
                    entry_te_cycle,
                    cfg.adverse_limit,
                    cfg.trend_threshold,
                    long_target_threshold=long_target_threshold,
                    short_target_threshold=short_target_threshold,
                    long_adverse_limit=long_adverse_limit,
                    short_adverse_limit=short_adverse_limit,
                    pred_history_len=cfg.pred_history_len,
                )
                X_tr_cycle = np.hstack([X_tr_cycle, meta_tr_cycle])
                X_te_cycle = np.hstack([X_te_cycle, meta_te_base])
                _log(f"WF cycle {wf.cycle_id}: state features added +{meta_tr_cycle.shape[1]} cols", t0=t_state)

            m1, m2 = train_two_stage(
                X_tr_cycle,
                y[tr_sl],
                cfg.random_state + wf.cycle_id,
                float(cycle_cfg["max_flat_ratio"]),
                cfg.classifier,
                two_branch=two_branch,
                n_1m_feats=n_1m_feats,
                two_branch_stage=cfg.two_branch_stage,
                stage1_extra=stage1_extra_tr_cycle,
            )
            cycle_model_path: Optional[Path] = None
            if cycle_model_dir is not None:
                cycle_model_path = cycle_model_dir / f"cycle_{wf.cycle_id:02d}.joblib"
                joblib.dump(
                    {
                        "stage1": m1,
                        "stage2": m2,
                        "config": _public_config_dict(cfg),
                        "execution_semantics": _execution_semantics(),
                        "evaluation_mode": cfg.eval_mode,
                        "cycle_id": wf.cycle_id,
                        "train_start": ts[wf.train_start].isoformat(),
                        "train_end": ts[wf.train_end - 1].isoformat(),
                        "test_start": ts[wf.test_start].isoformat(),
                        "test_end": ts[wf.test_end - 1].isoformat(),
                        "threshold": threshold,
                        "horizon": horizon,
                        "long_target_threshold": long_target_threshold,
                        "short_target_threshold": short_target_threshold,
                        "long_adverse_limit": long_adverse_limit,
                        "short_adverse_limit": short_adverse_limit,
                        "stage1_min_prob": float(cycle_cfg["stage1_min_prob"]),
                        "stage2_min_prob": float(cycle_cfg["stage2_min_prob"]),
                        "stage2_min_prob_up": float(cycle_cfg.get("stage2_min_prob_up", cycle_cfg["stage2_min_prob"])),
                        "stage2_min_prob_down": float(cycle_cfg.get("stage2_min_prob_down", cycle_cfg["stage2_min_prob"])),
                        "stage1_min_prob_1m": cfg.stage1_min_prob_1m,
                        "stage1_min_prob_15m": cfg.stage1_min_prob_15m,
                        "stage2_min_prob_1m": cfg.stage2_min_prob_1m,
                        "stage2_min_prob_15m": cfg.stage2_min_prob_15m,
                        "window_15m": cfg.window_15m,
                        "min_15m_drop": cfg.min_15m_drop,
                        "min_15m_rise": cfg.min_15m_rise,
                        "two_branch": two_branch,
                        "two_branch_stage": cfg.two_branch_stage,
                        "n_1m_feats": n_1m_feats,
                        "use_state_features": cfg.use_state_features,
                        "use_stage1_day_ohl_utc2": cfg.use_stage1_day_ohl_utc2,
                        "allow_overlap_backtest": cfg.allow_overlap_backtest,
                        "reverse_exit_prob": cfg.reverse_exit_prob,
                        "use_15m_wick_features": cfg.use_15m_wick_features,
                        "wick_feature_min_range": cfg.wick_feature_min_range,
                        "wick_feature_min_pct": cfg.wick_feature_min_pct,
                        "wick_feature_min_volume": cfg.wick_feature_min_volume,
                        "image_channel_names": image_channel_names,
                        "state_feature_names": state_feature_names,
                        "stage1_extra_feature_names": stage1_extra_feature_names,
                        "feature_shape": [int(X_tensor.shape[1]), int(X_tensor.shape[2])],
                        "label_map": {0: "down", 1: "flat", 2: "up", 3: "risky"},
                        "blocked_utc": BLOCKED_UTC_WINDOWS,
                    },
                    cycle_model_path,
                )
                _log(f"Saved WF cycle model: {cycle_model_path}")
            pm = predict_two_stage_details(
                X_te_cycle,
                m1,
                m2,
                stage1_min_prob=float(cycle_cfg["stage1_min_prob"]),
                stage2_min_prob=float(cycle_cfg["stage2_min_prob"]),
                stage2_min_prob_up=float(cycle_cfg.get("stage2_min_prob_up", cycle_cfg["stage2_min_prob"])),
                stage2_min_prob_down=float(cycle_cfg.get("stage2_min_prob_down", cycle_cfg["stage2_min_prob"])),
                stage1_min_prob_1m=cfg.stage1_min_prob_1m,
                stage1_min_prob_15m=cfg.stage1_min_prob_15m,
                stage2_min_prob_1m=cfg.stage2_min_prob_1m,
                stage2_min_prob_15m=cfg.stage2_min_prob_15m,
                stage1_extra=stage1_extra_te_cycle,
            )
            y_pred_all[te_sl] = pm["pred"]
            signal_prob_all[te_sl] = pm["signal_prob"]
            eval_mask[te_sl] = True

            cycle_pnl, _ = directional_pnl_report(
                ts[te_sl],
                entry_ts_a[te_sl],
                fut_ts_a[te_sl],
                pm["pred"],
                curr_a[te_sl],
                entry_a[te_sl],
                fut_a[te_sl],
                signal_prob=pm["signal_prob"],
                adverse_limit=cfg.adverse_limit,
                long_target_threshold=long_target_threshold,
                short_target_threshold=short_target_threshold,
                long_adverse_limit=long_adverse_limit,
                short_adverse_limit=short_adverse_limit,
                allow_overlap=cfg.allow_overlap_backtest,
                reverse_exit_prob=cfg.reverse_exit_prob,
                max_hold_minutes=cfg.max_hold_minutes,
                weak_period_cells=weak_period_cells,
            )
            wf_done_cycles += 1
            wf_cum_trades += int(cycle_pnl["trades"])
            wf_cum_pnl += float(cycle_pnl["total_pnl"])
            avg_day_disp = (
                f"{float(cycle_pnl['avg_day']):+.2f}"
                if cycle_pnl["avg_day"] is not None else "N/A"
            )
            avg_trades_day_disp = f"{float(cycle_pnl.get('avg_trades_per_day', 0.0)):.2f}"
            pos_days_disp = (
                f"{float(cycle_pnl['positive_days_pct']):.1f}%"
                if cycle_pnl["positive_days_pct"] is not None else "N/A"
            )
            _log(
                "WF 2-week stats: "
                f"cycle={wf.cycle_id}  "
                f"window=[{_trading_day_iso(wf.test_start_day)}→{_trading_day_iso(wf.test_end_day)}]  "
                f"samples=[{ts[wf.test_start].date()}→{ts[wf.test_end - 1].date()}]  "
                f"trades={int(cycle_pnl['trades'])}  "
                f"pnl=${float(cycle_pnl['total_pnl']):+.2f}  "
                f"avg_trade=${float(cycle_pnl['avg_trade']):+.2f}  "
                f"days={int(cycle_pnl['n_days'])}  "
                f"trades/day={avg_trades_day_disp}  "
                f"avg_day=${avg_day_disp}  "
                f"pos_days={pos_days_disp}  "
                f"cum_trades={wf_cum_trades}  "
                f"cum_pnl=${wf_cum_pnl:+.2f}"
            )
            walkforward_cycles_report.append(
                {
                    "cycle_id": wf.cycle_id,
                    "window_mode": wf.window_mode,
                    "train_start": _trading_day_iso(wf.train_start_day),
                    "train_end": _trading_day_iso(wf.train_end_day),
                    "test_start": _trading_day_iso(wf.test_start_day),
                    "test_end": _trading_day_iso(wf.test_end_day),
                    "train_start_sample_ts": ts[wf.train_start].isoformat(),
                    "train_end_sample_ts": ts[wf.train_end - 1].isoformat(),
                    "test_start_sample_ts": ts[wf.test_start].isoformat(),
                    "test_end_sample_ts": ts[wf.test_end - 1].isoformat(),
                    "train_samples": int(wf.train_end - wf.train_start),
                    "test_samples": int(wf.test_end - wf.test_start),
                    "selected_config": cycle_cfg,
                    "cycle_model_out": str(cycle_model_path) if cycle_model_path is not None else None,
                    "directional_pnl": cycle_pnl,
                    "progress": {
                        "completed_cycles": wf_done_cycles,
                        "cum_trades": int(wf_cum_trades),
                        "cum_total_pnl": float(wf_cum_pnl),
                    },
                }
            )
            _log("WF cycle done", t0=t_cycle)

        if m1 is None or m2 is None or not eval_mask.any():
            raise ValueError("Walk-forward produced no evaluable test samples.")

        eval_idx = np.where(eval_mask)[0]
        split = int(eval_idx[0])
        sample_split_mode = f"walk_forward:{len(walkforward_cycles_report)}_cycles"
        y_te = y[eval_idx]
        y_pred = y_pred_all[eval_idx]
        ts_te = ts[eval_idx]
        curr_te = curr_a[eval_idx]
        entry_te = entry_a[eval_idx]
        entry_ts_te = entry_ts_a[eval_idx]
        fut_te = fut_a[eval_idx]
        fut_ts_te = fut_ts_a[eval_idx]
        pred_meta = {
            "pred": y_pred,
            "signal_prob": signal_prob_all[eval_idx],
        }
        _log(f"Walk-forward eval samples: {len(eval_idx):,}")
        print("Class balance eval  (down/flat/up/risky):", np.bincount(y_te, minlength=4).tolist())

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    y_te_s1   = np.isin(y_te, TREND_LABELS).astype(np.int64)
    y_pred_s1 = np.isin(y_pred, TREND_LABELS).astype(np.int64)
    s1_acc    = accuracy_score(y_te_s1, y_pred_s1)
    s1_bacc   = balanced_accuracy_score(y_te_s1, y_pred_s1)
    s1_cm     = confusion_matrix(y_te_s1, y_pred_s1, labels=[0, 1])
    s1_report = classification_report(
        y_te_s1, y_pred_s1, target_names=["flat", "trend"],
        output_dict=True, zero_division=0,
    )
    print("\n── Stage 1 : flat vs trend ─────────────────────────────────────")
    print(f"   Accuracy={s1_acc:.4f}   Balanced={s1_bacc:.4f}")
    print("   Confusion matrix  (rows=true, cols=pred)  [flat | trend]")
    print(f"             pred_flat  pred_trend")
    print(f"   true_flat    {s1_cm[0,0]:>5}       {s1_cm[0,1]:>5}")
    print(f"   true_trend   {s1_cm[1,0]:>5}       {s1_cm[1,1]:>5}")
    print(classification_report(
        y_te_s1,
        y_pred_s1,
        labels=[0, 1],
        target_names=["flat", "trend"],
        zero_division=0,
    ))

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    s2_acc = s2_bacc = 0.0
    s2_cm     = np.zeros((2, 2), dtype=int)
    s2_report = {}
    trend_mask = np.isin(y_te, TREND_LABELS)
    if trend_mask.sum() > 0:
        pred_tr   = y_pred[trend_mask]
        true_tr   = y_te[trend_mask]
        dir_mask  = np.isin(pred_tr, TREND_LABELS)
        if dir_mask.sum() > 0:
            yp2 = (pred_tr[dir_mask] == LABEL_UP).astype(np.int64)
            yt2 = (true_tr[dir_mask] == LABEL_UP).astype(np.int64)
            s2_acc  = accuracy_score(yt2, yp2)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"A single label was found in 'y_true' and 'y_pred'.*",
                    category=UserWarning,
                )
                s2_bacc = balanced_accuracy_score(yt2, yp2)
            s2_cm   = confusion_matrix(yt2, yp2, labels=[0, 1])
            s2_report = classification_report(
                yt2, yp2, labels=[0, 1], target_names=["down", "up"],
                output_dict=True, zero_division=0,
            )
            print("── Stage 2 : down vs up (on trend-predicted samples) ───────────")
            print(f"   Accuracy={s2_acc:.4f}   Balanced={s2_bacc:.4f}")
            print("   Confusion matrix  (rows=true, cols=pred)  [down | up]")
            print(f"            pred_down   pred_up")
            print(f"   true_down   {s2_cm[0,0]:>5}     {s2_cm[0,1]:>5}")
            print(f"   true_up     {s2_cm[1,0]:>5}     {s2_cm[1,1]:>5}")
            print(classification_report(
                yt2,
                yp2,
                labels=[0, 1],
                target_names=["down", "up"],
                zero_division=0,
            ))

    # ── Full 4-class (down / flat / up / risky) ─────────────────────────────
    full_cm = confusion_matrix(y_te, y_pred, labels=[LABEL_DOWN, LABEL_FLAT, LABEL_UP, LABEL_RISKY])
    print("── Full 4-class (down / flat / up / risky) ────────────────────")
    print("   Confusion matrix  (rows=true, cols=pred)  [down | flat | up | risky]")
    print("             pred_down  pred_flat  pred_up  pred_risky")
    print(f"   true_down    {full_cm[0,0]:>5}      {full_cm[0,1]:>5}    {full_cm[0,2]:>5}      {full_cm[0,3]:>5}")
    print(f"   true_flat    {full_cm[1,0]:>5}      {full_cm[1,1]:>5}    {full_cm[1,2]:>5}      {full_cm[1,3]:>5}")
    print(f"   true_up      {full_cm[2,0]:>5}      {full_cm[2,1]:>5}    {full_cm[2,2]:>5}      {full_cm[2,3]:>5}")
    print(f"   true_risky   {full_cm[3,0]:>5}      {full_cm[3,1]:>5}    {full_cm[3,2]:>5}      {full_cm[3,3]:>5}")
    print(classification_report(
        y_te,
        y_pred,
        labels=[LABEL_DOWN, LABEL_FLAT, LABEL_UP, LABEL_RISKY],
        target_names=["down", "flat", "up", "risky"],
        zero_division=0,
    ))

    # ── PnL check ─────────────────────────────────────────────────────────────
    # Build ask/bid price arrays so the backtest uses realistic fill prices
    # (longs enter at ask / exit at bid; shorts enter at bid / exit at ask).
    _spread_entry_ask, _spread_entry_bid, _spread_curr_ask, _spread_curr_bid, _spread_fut_ask, _spread_fut_bid = (
        _build_spread_price_arrays(ts_te, entry_ts_te, fut_ts_te, bars_ask, bars_bid)
    )
    pnl, trades_df = directional_pnl_report(
        ts_te,
        entry_ts_te,
        fut_ts_te,
        y_pred,
        curr_te,
        entry_te,
        fut_te,
        signal_prob=pred_meta["signal_prob"],
        adverse_limit=cfg.adverse_limit,
        long_target_threshold=long_target_threshold,
        short_target_threshold=short_target_threshold,
        long_adverse_limit=long_adverse_limit,
        short_adverse_limit=short_adverse_limit,
        allow_overlap=cfg.allow_overlap_backtest,
        reverse_exit_prob=cfg.reverse_exit_prob,
        max_hold_minutes=cfg.max_hold_minutes,
        weak_period_cells=weak_period_cells,
        entry_px_ask=_spread_entry_ask,
        entry_px_bid=_spread_entry_bid,
        curr_ask=_spread_curr_ask,
        curr_bid=_spread_curr_bid,
        fut_ask=_spread_fut_ask,
        fut_bid=_spread_fut_bid,
    )
    print("── Directional PnL check ───────────────────────────────────────")
    print(f"   trades={pnl['trades']}  total=${pnl['total_pnl']:.2f}  avg_trade=${pnl['avg_trade']:.2f}")
    print(f"   avg_trades_per_day={float(pnl.get('avg_trades_per_day', 0.0)):.2f}")
    all_stats = pnl.get("all", {})
    print(
        "   drawdown: "
        f"trade=${float(all_stats.get('trade_max_drawdown', pnl.get('max_drawdown', 0.0))):.2f} "
        f"daily=${float(all_stats.get('daily_max_drawdown', 0.0)):.2f}"
    )
    if pnl["n_days"] >= 5:
        print(f"   n_days={pnl['n_days']}  avg_day=${pnl['avg_day']:.2f}  "
              f"positive_days={pnl['positive_days_pct']:.1f}%%")
    else:
        print(f"   n_days={pnl['n_days']}  (avg_day N/A – fewer than 5 trading days in test)")
    long_stats = pnl.get("long", {})
    short_stats = pnl.get("short", {})
    print(
        "   long: "
        f"trades={int(long_stats.get('trades', 0))} "
        f"win_rate={float(long_stats.get('win_rate') or 0.0):.1f}% "
        f"pnl=${float(long_stats.get('total_pnl', 0.0)):.2f}"
    )
    print(
        "   short: "
        f"trades={int(short_stats.get('trades', 0))} "
        f"win_rate={float(short_stats.get('win_rate') or 0.0):.1f}% "
        f"pnl=${float(short_stats.get('total_pnl', 0.0)):.2f}"
    )
    streaks = pnl.get("streaks", {})
    print(
        "   streaks: "
        f"max_win={int(streaks.get('max_win_streak', 0))} "
        f"max_loss={int(streaks.get('max_loss_streak', 0))} "
        f"current_win={int(streaks.get('current_win_streak', 0))} "
        f"current_loss={int(streaks.get('current_loss_streak', 0))}"
    )
    print(
        "   exits: "
        + ", ".join(f"{k}={v}" for k, v in dict(all_stats.get("exit_reason_counts", {})).items())
    )
    rev = all_stats.get("reverse_signal_stats", {})
    print(
        "   reverse_signal: "
        f"wins={int(rev.get('wins', 0))} "
        f"losses={int(rev.get('losses', 0))} "
        f"avg_pnl=${float(rev.get('avg_pnl') or 0.0):.2f} "
        f"win_pct={float(rev.get('win_rate_pct') or 0.0):.1f}% "
        f"loss_pct={float(rev.get('loss_rate_pct') or 0.0):.1f}%"
    )
    th = all_stats.get("target_hit_stats", {})
    print(
        "   target_hit: "
        f"trades={int(th.get('trades', 0))} "
        f"avg_pnl=${float(th.get('avg_pnl') or 0.0):.2f} "
    )
    to = all_stats.get("timeout_stats", {})
    print(
        "   timeout: "
        f"wins={int(to.get('wins', 0))} "
        f"losses={int(to.get('losses', 0))} "
        f"avg_pnl=${float(to.get('avg_pnl') or 0.0):.2f} "
        f"win_pct={float(to.get('win_rate_pct') or 0.0):.1f}% "
        f"loss_pct={float(to.get('loss_rate_pct') or 0.0):.1f}%"
    )

    if cfg.trades_out:
        trades_path = Path(cfg.trades_out)
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(trades_path, index=False)
        print(f"Saved trades  : {trades_path}")

    # ── Save ──────────────────────────────────────────────────────────────────
    t = _log("Saving model and report...")
    model_path = Path(cfg.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "stage1":        m1,
            "stage2":        m2,
            "config":        _public_config_dict(cfg),
            "execution_semantics": _execution_semantics(),
            "evaluation_mode": cfg.eval_mode,
            "threshold":     threshold,
            "horizon":       horizon,
            "long_target_threshold": long_target_threshold,
            "short_target_threshold": short_target_threshold,
            "long_adverse_limit": long_adverse_limit,
            "short_adverse_limit": short_adverse_limit,
            "stage1_min_prob": stage1_prob,
            "stage2_min_prob": stage2_prob,
            "stage2_min_prob_up": stage2_prob_up,
            "stage2_min_prob_down": stage2_prob_down,
            "stage1_min_prob_1m": cfg.stage1_min_prob_1m,
            "stage1_min_prob_15m": cfg.stage1_min_prob_15m,
            "stage2_min_prob_1m": cfg.stage2_min_prob_1m,
            "stage2_min_prob_15m": cfg.stage2_min_prob_15m,
            "window_15m":    cfg.window_15m,
            "min_15m_drop":  cfg.min_15m_drop,
            "min_15m_rise":  cfg.min_15m_rise,
            "two_branch":    two_branch,
            "two_branch_stage": cfg.two_branch_stage,
            "n_1m_feats":    n_1m_feats,
            "use_state_features": cfg.use_state_features,
            "use_stage1_day_ohl_utc2": cfg.use_stage1_day_ohl_utc2,
            "allow_overlap_backtest": cfg.allow_overlap_backtest,
            "reverse_exit_prob": cfg.reverse_exit_prob,
            "use_15m_wick_features": cfg.use_15m_wick_features,
            "wick_feature_min_range": cfg.wick_feature_min_range,
            "wick_feature_min_pct": cfg.wick_feature_min_pct,
            "wick_feature_min_volume": cfg.wick_feature_min_volume,
            "image_channel_names": image_channel_names,
            "state_feature_names": state_feature_names,
            "stage1_extra_feature_names": stage1_extra_feature_names,
            "feature_shape": [int(X_tensor.shape[1]), int(X_tensor.shape[2])],
            "label_map":     {0: "down", 1: "flat", 2: "up", 3: "risky"},
            "blocked_utc":   BLOCKED_UTC_WINDOWS,
            "trading_day_cutoff": {
                "timezone": "America/New_York",
                "hour": TRADING_DAY_CUTOFF_HOUR_NY,
            },
            "walkforward_config": {
                "init_train_months": cfg.wf_init_train_months,
                "retrain_days": cfg.wf_retrain_days,
                "max_train_days": cfg.wf_max_train_days,
                "min_train_samples": cfg.wf_min_train_samples,
                "disable_sweep": cfg.wf_disable_sweep,
                "sweep_flat_ratios": cfg.wf_sweep_flat_ratios,
                "sweep_stage1_probs": cfg.wf_sweep_stage1_probs,
                "sweep_stage2_probs": cfg.wf_sweep_stage2_probs,
                "sweep_stage2_long_probs": cfg.wf_sweep_stage2_long_probs,
                "sweep_stage2_short_probs": cfg.wf_sweep_stage2_short_probs,
                "sweep_val_ratio": cfg.wf_sweep_val_ratio,
                "sweep_min_val_samples": cfg.wf_sweep_min_val_samples,
                "anchor_mode": cfg.wf_anchor_mode,
            } if cfg.eval_mode == "walk_forward" else None,
            "walkforward_cycles": walkforward_cycles_report,
        },
        model_path,
    )

    report = {
        "config":               _public_config_dict(cfg),
        "execution_semantics":  _execution_semantics(),
        "threshold_used":       threshold,
        "horizon_used":         horizon,
        "long_target_threshold_used": long_target_threshold,
        "short_target_threshold_used": short_target_threshold,
        "long_adverse_limit_used": long_adverse_limit,
        "short_adverse_limit_used": short_adverse_limit,
        "min_window_range_used": min_window_range,
        "window_15m":           cfg.window_15m,
        "min_15m_drop":         cfg.min_15m_drop,
        "min_15m_rise":         cfg.min_15m_rise,
        "two_branch":           two_branch,
        "two_branch_stage":     cfg.two_branch_stage,
        "n_1m_feats":           n_1m_feats,
        "use_state_features":   cfg.use_state_features,
        "use_15m_wick_features": cfg.use_15m_wick_features,
        "wick_feature_min_range": cfg.wick_feature_min_range,
        "wick_feature_min_pct": cfg.wick_feature_min_pct,
        "wick_feature_min_volume": cfg.wick_feature_min_volume,
        "image_channel_names": image_channel_names,
        "state_feature_names": state_feature_names,
        "stage1_extra_feature_names": stage1_extra_feature_names,
        "feature_shape":        [int(X_tensor.shape[1]), int(X_tensor.shape[2])],
        "evaluation_mode":      cfg.eval_mode,
        "split_mode":           sample_split_mode,
        "n_kept":               kept,
        "n_skipped":            skipped,
        "train_samples":        split,
        "test_samples":         int(n_total - split),
        "stage1": {
            "accuracy":              float(s1_acc),
            "balanced_accuracy":     float(s1_bacc),
            "confusion_matrix":      s1_cm.tolist(),
            "confusion_labels":      ["flat", "trend"],
            "classification_report": s1_report,
        },
        "stage2": {
            "accuracy":              float(s2_acc),
            "balanced_accuracy":     float(s2_bacc),
            "confusion_matrix":      s2_cm.tolist(),
            "confusion_labels":      ["down", "up"],
            "classification_report": s2_report,
        },
        "full_confusion_matrix": full_cm.tolist(),
        "full_confusion_labels": ["down", "flat", "up", "risky"],
        "directional_pnl":      pnl,
        "trading_day_cutoff": {
            "timezone": "America/New_York",
            "hour": TRADING_DAY_CUTOFF_HOUR_NY,
        },
        "walkforward_config": {
            "init_train_months": cfg.wf_init_train_months,
            "retrain_days": cfg.wf_retrain_days,
            "max_train_days": cfg.wf_max_train_days,
            "min_train_samples": cfg.wf_min_train_samples,
            "disable_sweep": cfg.wf_disable_sweep,
            "sweep_flat_ratios": cfg.wf_sweep_flat_ratios,
            "sweep_stage1_probs": cfg.wf_sweep_stage1_probs,
            "sweep_stage2_probs": cfg.wf_sweep_stage2_probs,
            "sweep_stage2_long_probs": cfg.wf_sweep_stage2_long_probs,
            "sweep_stage2_short_probs": cfg.wf_sweep_stage2_short_probs,
            "sweep_val_ratio": cfg.wf_sweep_val_ratio,
            "sweep_min_val_samples": cfg.wf_sweep_min_val_samples,
            "anchor_mode": cfg.wf_anchor_mode,
        } if cfg.eval_mode == "walk_forward" else None,
        "walkforward_cycles": walkforward_cycles_report,
        "model_reuse": model_meta or None,
        "cache_info": cache_info or None,
    }
    report_path = Path(cfg.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved model  : {model_path}")
    print(f"Saved report : {report_path}")
    _log(f"All done — total elapsed: {_elapsed(t_main)}", t0=t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

