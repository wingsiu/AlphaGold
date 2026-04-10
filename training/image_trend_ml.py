#!/usr/bin/env python3
"""Image-like trend prediction from candle windows.

Pipeline:
1) Load gold minute bars from MySQL.
2) Resample to 1-minute candles by default.
3) Apply time-based no-trade filters (HKT open, London session, NY open).
4) Filter to keep only windows with price range > min_window_range.
5) Convert rolling windows into 9-channel image-like tensors.
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
import itertools
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import sys

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


def _is_blocked(ts: pd.Timestamp) -> bool:
    """Return True if UTC timestamp falls inside a no-trade window."""
    t = ts.hour * 60 + ts.minute
    for sh, sm, eh, em in BLOCKED_UTC_WINDOWS:
        if sh * 60 + sm <= t < eh * 60 + em:
            return True
    return False


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    table: str
    start_date: str
    end_date: str
    timeframe: str
    window: int
    min_window_range: float
    horizon: int
    trend_threshold: float
    adverse_limit: float
    test_size: float
    max_samples: Optional[int]
    optimize: bool
    optimize_prob: bool
    max_flat_ratio: float
    classifier: str
    stage1_min_prob: float
    stage2_min_prob: float
    random_state: int
    model_out: str
    report_out: str


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


# ── Image construction ────────────────────────────────────────────────────────

def _window_to_image(w: pd.DataFrame) -> np.ndarray:
    """Encode a candle window as a 9-channel tensor [channels, width].

    Channels:
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

    return np.stack(
        [open_rel, high_rel, low_rel, close_rel, body_rel, range_rel,
         vol_z, vol_rel, vol_diff_norm],
        axis=0,
    ).astype(np.float32)


# ── Label ─────────────────────────────────────────────────────────────────────

def _label(curr_close: float, future_high: float, future_low: float,
           threshold: float, adverse_limit: float) -> int:
    """
    up   (2): (future_high  - curr_close) / curr_close > threshold
              AND (curr_close - future_low)             < adverse_limit
    down (0): (curr_close   - future_low) / curr_close > threshold
              AND (future_high - curr_close)             < adverse_limit
    flat (1): otherwise
    """
    up_move      = (future_high - curr_close) / curr_close
    up_adverse   =  curr_close  - future_low
    down_move    = (curr_close  - future_low) / curr_close
    down_adverse =  future_high - curr_close

    up_ok   = up_move   > threshold and up_adverse   < adverse_limit
    down_ok = down_move > threshold and down_adverse < adverse_limit

    if up_ok and down_ok:
        return 2 if up_move >= down_move else 0
    if up_ok:
        return 2
    if down_ok:
        return 0
    return 1


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    df: pd.DataFrame,
    window: int,
    min_window_range: float,
    horizon: int,
    threshold: float,
    adverse_limit: float,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray, int, int]:
    tensors:   list[np.ndarray]   = []
    labels:    list[int]          = []
    ts:        list[pd.Timestamp] = []
    curr_list: list[float]        = []
    fut_list:  list[float]        = []
    skipped = 0
    total_iters = len(df) - horizon - (window - 1)
    t_ds = time.time()
    report_every = max(1, total_iters // 20)   # ~5% increments

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
        if _is_blocked(signal_ts):
            skipped += 1
            continue

        # ── range filter ─────────────────────────────────────────────────────
        w = df.iloc[i - window + 1 : i + 1]
        if float(w["high"].max() - w["low"].min()) <= min_window_range:
            skipped += 1
            continue

        curr_close   = float(df["close"].iloc[i])
        fut_w        = df.iloc[i + 1 : i + horizon + 1]
        future_high  = float(fut_w["high"].max())
        future_low   = float(fut_w["low"].min())
        future_close = float(df["close"].iloc[i + horizon])

        tensors.append(_window_to_image(w))
        labels.append(_label(curr_close, future_high, future_low, threshold, adverse_limit))
        ts.append(signal_ts)
        curr_list.append(curr_close)
        fut_list.append(future_close)

    if not tensors:
        raise ValueError("Dataset is empty – loosen filters or expand date range.")

    print(f"  [{_now()}] Dataset build done — kept={len(tensors):,}  skipped={skipped:,}  took={_elapsed(t_ds)}", flush=True)

    X      = np.stack(tensors, axis=0)
    y      = np.asarray(labels,    dtype=np.int64)
    t      = pd.DatetimeIndex(ts)
    curr_a = np.asarray(curr_list, dtype=np.float64)
    fut_a  = np.asarray(fut_list,  dtype=np.float64)
    return X, y, t, curr_a, fut_a, int(len(y)), skipped


def flatten_tensors(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


# ── Two-stage model ───────────────────────────────────────────────────────────

def _make_model(classifier: str = "gradient_boosting", random_state: int = 42):
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


def train_two_stage(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    max_flat_ratio: float = 4.0,
    classifier: str = "gradient_boosting",
) -> tuple:
    """
    Stage 1: flat(0) vs trend(1).
      - Flat class is subsampled to max_flat_ratio × n_trend before fitting
        to prevent the model being overwhelmed by the majority class.
      - Sample weights are passed to balance remaining class imbalance.
    Stage 2: down(0) vs up(1) on trend-only samples.
    """
    rng = np.random.default_rng(random_state)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    trend_mask = y != 1
    n_trend    = int(trend_mask.sum())
    flat_idx   = np.where(~trend_mask)[0]
    trend_idx  = np.where(trend_mask)[0]

    if n_trend < 10:
        raise ValueError("Too few trend samples to train stage 1.")

    # Subsample flat class
    max_flat   = min(len(flat_idx), int(max_flat_ratio * n_trend))
    keep_flat  = rng.choice(flat_idx, size=max_flat, replace=False)
    s1_idx     = np.sort(np.concatenate([keep_flat, trend_idx]))
    X_s1       = X[s1_idx]
    y_s1       = (y[s1_idx] != 1).astype(np.int64)

    sw_s1 = compute_sample_weight("balanced", y_s1)
    m1    = _make_model(classifier, random_state)
    if isinstance(m1, Pipeline):
        m1.fit(X_s1, y_s1)                     # LR already uses class_weight
    else:
        m1.fit(X_s1, y_s1, sample_weight=sw_s1)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    if n_trend < 10:
        raise ValueError("Too few trend samples to train stage 2.")
    X_s2  = X[trend_mask]
    y_s2  = (y[trend_mask] == 2).astype(np.int64)   # 1=up, 0=down
    sw_s2 = compute_sample_weight("balanced", y_s2)
    m2    = _make_model(classifier, random_state)
    if isinstance(m2, Pipeline):
        m2.fit(X_s2, y_s2)
    else:
        m2.fit(X_s2, y_s2, sample_weight=sw_s2)

    return m1, m2


def predict_two_stage(
    X: np.ndarray,
    m1,
    m2,
    stage1_min_prob: float = 0.55,
    stage2_min_prob: float = 0.55,
) -> np.ndarray:
    """Returns 3-class labels: 0=down, 1=flat, 2=up.

    - Stage 1 must pass P(trend) >= stage1_min_prob.
    - Stage 2 must pass confidence max(P(up), P(down)) >= stage2_min_prob.
    - Otherwise keep class=flat.
    """
    result = np.ones(len(X), dtype=np.int64)

    p1 = m1.predict_proba(X)
    c1 = list(getattr(m1, "classes_", [0, 1]))
    idx_trend = c1.index(1) if 1 in c1 else -1
    if idx_trend < 0:
        return result
    trend_prob = p1[:, idx_trend]
    trend_mask = trend_prob >= stage1_min_prob

    if trend_mask.any():
        x2 = X[trend_mask]
        p2 = m2.predict_proba(x2)
        c2 = list(getattr(m2, "classes_", [0, 1]))
        idx_up = c2.index(1) if 1 in c2 else -1
        if idx_up >= 0:
            up_prob = p2[:, idx_up]
            conf = np.maximum(up_prob, 1.0 - up_prob)
            sure = conf >= stage2_min_prob
            idx_global = np.where(trend_mask)[0]
            idx_sure = idx_global[sure]
            is_up = up_prob[sure] >= 0.5
            result[idx_sure] = np.where(is_up, 2, 0)

    return result


# ── Optimiser  (sweeps min_window_range only; threshold & horizon are fixed) ──

_OPT_RANGES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]


def optimize_min_range(
    bars: pd.DataFrame,
    window: int,
    threshold: float,
    horizon: int,
    adverse_limit: float,
    max_flat_ratio: float,
    classifier: str,
    random_state: int,
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
            X, y, ts, curr_a, fut_a, kept, skipped = build_dataset(
                bars, window, rng_val, horizon, threshold, adverse_limit,
            )
        except ValueError:
            print("skip (empty)", flush=True)
            continue

        n_trend    = int((y != 1).sum())
        trend_pct  = n_trend / max(kept, 1) * 100
        if kept < 100 or trend_pct < 2.0:
            print(f"skip  kept={kept}  trend={trend_pct:.1f}%%", flush=True)
            continue

        sp = int(kept * 0.7)
        if sp < 60 or (kept - sp) < 30:
            print("skip (split too small)", flush=True)
            continue

        Xf = flatten_tensors(X)
        try:
            m1, m2 = train_two_stage(
                Xf[:sp], y[:sp], random_state, max_flat_ratio, classifier,
            )
        except ValueError as e:
            print(f"skip ({e})", flush=True)
            continue

        pred    = predict_two_stage(Xf[sp:], m1, m2)
        sig_mask = pred != 1
        if sig_mask.sum() == 0:
            print("skip (no signals)", flush=True)
            continue

        sides = np.where(pred[sig_mask] == 2, 1.0, -1.0)
        pnl   = float(np.sum((fut_a[sp:][sig_mask] - curr_a[sp:][sig_mask]) * sides))
        n_sig = int(sig_mask.sum())
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
    curr_val: np.ndarray,
    fut_val: np.ndarray,
    m1,
    m2,
) -> tuple[float, float]:
    """Tune stage probability gates on validation data using PnL-driven score."""
    best_score = -1e18
    best_pair = (_OPT_STAGE1_PROBS[0], _OPT_STAGE2_PROBS[0])
    rows: list[tuple[float, float, float, int, float]] = []

    combos = list(itertools.product(_OPT_STAGE1_PROBS, _OPT_STAGE2_PROBS))
    t_opt = time.time()
    for idx, (p1, p2) in enumerate(combos, 1):
        pred = predict_two_stage(X_val, m1, m2, stage1_min_prob=p1, stage2_min_prob=p2)
        sig_mask = pred != 1
        n_sig = int(sig_mask.sum())
        if n_sig == 0:
            rows.append((p1, p2, -1e18, 0, 0.0))
            continue

        sides = np.where(pred[sig_mask] == 2, 1.0, -1.0)
        pnl = float(np.sum((fut_val[sig_mask] - curr_val[sig_mask]) * sides))
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


# ── PnL check ─────────────────────────────────────────────────────────────────

def directional_pnl_report(
    ts:   pd.DatetimeIndex,
    pred: np.ndarray,
    curr: np.ndarray,
    fut:  np.ndarray,
) -> dict:
    rows = [
        {"ts": ts[i], "pnl": (fut[i] - curr[i]) * (1.0 if pred[i] == 2 else -1.0)}
        for i in range(len(pred)) if pred[i] != 1
    ]
    if not rows:
        return {"trades": 0, "total_pnl": 0.0, "avg_trade": 0.0,
                "n_days": 0, "avg_day": None, "positive_days_pct": None}

    pdf   = pd.DataFrame(rows)
    pdf["day"] = pd.to_datetime(pdf["ts"], utc=True).dt.floor("D")
    daily = pdf.groupby("day")["pnl"].sum()
    n     = len(daily)
    return {
        "trades":            int(len(pdf)),
        "total_pnl":         float(pdf["pnl"].sum()),
        "avg_trade":         float(pdf["pnl"].mean()),
        "n_days":            n,
        "avg_day":           float(daily.mean())              if n >= 5 else None,
        "positive_days_pct": float((daily > 0).mean() * 100) if n >= 5 else None,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Image-like candle trend ML model.")
    p.add_argument("--table",            default="gold_prices")
    p.add_argument("--start-date",       default="2025-05-20")
    p.add_argument("--end-date",         default="2026-04-10")
    p.add_argument("--timeframe",        default="1min")
    p.add_argument("--window",           type=int,   default=150)
    p.add_argument("--min-window-range", type=float, default=40.0,
                   help="Keep windows where high-low > this value (default: 40)")
    p.add_argument("--horizon",          type=int,   default=25,
                   help="Prediction horizon in bars (default: 25)")
    p.add_argument("--trend-threshold",  type=float, default=0.004,
                   help="Move threshold for up/down labels (0.004 = 0.4%%)")
    p.add_argument("--adverse-limit",    type=float, default=15.0,
                   help="Absolute adverse move cap in price units (default: 15)")
    p.add_argument("--test-size",        type=float, default=0.30)
    p.add_argument("--max-samples",      type=int,   default=None)
    p.add_argument("--optimize",         action="store_true",
                   help="Grid-search min-window-range on train split (threshold and horizon fixed)")
    p.add_argument("--optimize-prob",    action="store_true",
                   help="Grid-search stage1/stage2 probability gates on a train validation split")
    p.add_argument("--max-flat-ratio",   type=float, default=4.0,
                   help="Max ratio flat:trend samples in stage-1 training (default: 4.0)")
    p.add_argument("--classifier",       default="gradient_boosting",
                   choices=["gradient_boosting", "logistic"],
                   help="Model type for both stages (default: gradient_boosting)")
    p.add_argument("--stage1-min-prob",  type=float, default=0.55,
                   help="Stage-1 minimum P(trend) to enable stage-2 (default: 0.55)")
    p.add_argument("--stage2-min-prob",  type=float, default=0.55,
                   help="Stage-2 minimum directional confidence max(P(up),P(down)) (default: 0.55)")
    p.add_argument("--random-state",     type=int,   default=42)
    p.add_argument("--model-out",        default="training/image_trend_model.joblib")
    p.add_argument("--report-out",       default="training/image_trend_report.json")
    args = p.parse_args()
    return Config(
        table=args.table, start_date=args.start_date, end_date=args.end_date,
        timeframe=args.timeframe, window=args.window,
        min_window_range=args.min_window_range, horizon=args.horizon,
        trend_threshold=args.trend_threshold, adverse_limit=args.adverse_limit,
        test_size=args.test_size, max_samples=args.max_samples,
        optimize=args.optimize, optimize_prob=args.optimize_prob,
        max_flat_ratio=args.max_flat_ratio,
        classifier=args.classifier,
        stage1_min_prob=args.stage1_min_prob,
        stage2_min_prob=args.stage2_min_prob,
        random_state=args.random_state,
        model_out=args.model_out, report_out=args.report_out,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = parse_args()
    t_main = _log("Starting image-trend pipeline")

    t = _log("Loading data from MySQL...")
    raw  = DataLoader().load_data(cfg.table, start_date=cfg.start_date, end_date=cfg.end_date)
    bars = _prepare_ohlcv(raw, cfg.timeframe)
    _log(f"Bars after resample: {len(bars):,}", t0=t)

    bar_split  = int(len(bars) * (1.0 - cfg.test_size))
    train_bars = bars.iloc[:bar_split].copy()

    threshold = cfg.trend_threshold
    horizon   = cfg.horizon
    min_window_range = cfg.min_window_range

    if cfg.optimize:
        t = _log("Optimising min-window-range on train split (threshold/horizon fixed)...")
        min_window_range = optimize_min_range(
            train_bars,
            cfg.window,
            threshold,
            horizon,
            cfg.adverse_limit,
            cfg.max_flat_ratio,
            cfg.classifier,
            cfg.random_state,
        )
        _log(f"→ Best min_window_range={min_window_range}", t0=t)

    t = _log(
        f"Building image-like dataset  (window={cfg.window}, horizon={horizon}, "
        f"threshold={threshold}, min_range={min_window_range})..."
    )
    X_tensor, y, ts, curr_a, fut_a, kept, skipped = build_dataset(
        bars, cfg.window, min_window_range, horizon, threshold, cfg.adverse_limit,
    )
    _log(f"Dataset ready: kept={kept:,}  skipped={skipped:,}", t0=t)

    if cfg.max_samples and kept > cfg.max_samples:
        X_tensor = X_tensor[-cfg.max_samples:]
        y        = y[-cfg.max_samples:]
        ts       = ts[-cfg.max_samples:]
        curr_a   = curr_a[-cfg.max_samples:]
        fut_a    = fut_a[-cfg.max_samples:]

    X_flat  = flatten_tensors(X_tensor)
    n_total = len(y)
    split   = int(n_total * (1.0 - cfg.test_size))

    if split < 100 or (n_total - split) < 50:
        raise ValueError("Not enough samples. Increase date range.")

    X_tr, X_te = X_flat[:split],  X_flat[split:]
    y_tr, y_te = y[:split],       y[split:]
    ts_te      = ts[split:]
    curr_te    = curr_a[split:]
    fut_te     = fut_a[split:]

    _log(f"Split: train={split:,}  test={n_total-split:,}")
    print("Class balance train (down/flat/up):", np.bincount(y_tr, minlength=3).tolist())
    print("Class balance test  (down/flat/up):", np.bincount(y_te, minlength=3).tolist())

    stage1_prob = cfg.stage1_min_prob
    stage2_prob = cfg.stage2_min_prob

    if cfg.optimize_prob:
        t = _log("Optimising stage probability thresholds on train-validation split...")
        split_fit = int(len(X_tr) * 0.8)
        if split_fit < 100 or (len(X_tr) - split_fit) < 50:
            _log("Not enough train samples for prob optimization; using provided thresholds", t0=t)
        else:
            X_fit, y_fit = X_tr[:split_fit], y_tr[:split_fit]
            X_val = X_tr[split_fit:]
            curr_val = curr_te[:0]
            fut_val = fut_te[:0]
            # Use aligned close arrays from the training side for validation PnL score.
            curr_tr = curr_a[:split]
            fut_tr = fut_a[:split]
            curr_val = curr_tr[split_fit:]
            fut_val = fut_tr[split_fit:]

            m1_tmp, m2_tmp = train_two_stage(
                X_fit,
                y_fit,
                cfg.random_state,
                cfg.max_flat_ratio,
                cfg.classifier,
            )
            stage1_prob, stage2_prob = optimize_prob_thresholds(
                X_val,
                curr_val,
                fut_val,
                m1_tmp,
                m2_tmp,
            )
            _log(f"→ Best probability gates: stage1={stage1_prob:.2f}, stage2={stage2_prob:.2f}", t0=t)

    t = _log("Training two-stage model...")
    m1, m2 = train_two_stage(
        X_tr,
        y_tr,
        cfg.random_state,
        cfg.max_flat_ratio,
        cfg.classifier,
    )
    _log("Training done", t0=t)
    y_pred = predict_two_stage(
        X_te,
        m1,
        m2,
        stage1_min_prob=stage1_prob,
        stage2_min_prob=stage2_prob,
    )

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    y_te_s1   = (y_te   != 1).astype(np.int64)
    y_pred_s1 = (y_pred != 1).astype(np.int64)
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
    trend_mask = y_te != 1
    if trend_mask.sum() > 0:
        pred_tr   = y_pred[trend_mask]
        true_tr   = y_te[trend_mask]
        dir_mask  = pred_tr != 1
        if dir_mask.sum() > 0:
            yp2 = (pred_tr[dir_mask] == 2).astype(np.int64)
            yt2 = (true_tr[dir_mask] == 2).astype(np.int64)
            s2_acc  = accuracy_score(yt2, yp2)
            s2_bacc = balanced_accuracy_score(yt2, yp2)
            s2_cm   = confusion_matrix(yt2, yp2, labels=[0, 1])
            s2_report = classification_report(
                yt2, yp2, target_names=["down", "up"],
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

    # ── Full 3-class ──────────────────────────────────────────────────────────
    full_cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
    print("── Full 3-class (down / flat / up) ─────────────────────────────")
    print("   Confusion matrix  (rows=true, cols=pred)  [down | flat | up]")
    print(f"             pred_down  pred_flat  pred_up")
    print(f"   true_down    {full_cm[0,0]:>5}      {full_cm[0,1]:>5}    {full_cm[0,2]:>5}")
    print(f"   true_flat    {full_cm[1,0]:>5}      {full_cm[1,1]:>5}    {full_cm[1,2]:>5}")
    print(f"   true_up      {full_cm[2,0]:>5}      {full_cm[2,1]:>5}    {full_cm[2,2]:>5}")
    print(classification_report(
        y_te,
        y_pred,
        labels=[0, 1, 2],
        target_names=["down", "flat", "up"],
        zero_division=0,
    ))

    # ── PnL check ─────────────────────────────────────────────────────────────
    pnl = directional_pnl_report(ts_te, y_pred, curr_te, fut_te)
    print("── Directional PnL check ───────────────────────────────────────")
    print(f"   trades={pnl['trades']}  total=${pnl['total_pnl']:.2f}  avg_trade=${pnl['avg_trade']:.2f}")
    if pnl["n_days"] >= 5:
        print(f"   n_days={pnl['n_days']}  avg_day=${pnl['avg_day']:.2f}  "
              f"positive_days={pnl['positive_days_pct']:.1f}%%")
    else:
        print(f"   n_days={pnl['n_days']}  (avg_day N/A – fewer than 5 trading days in test)")

    # ── Save ──────────────────────────────────────────────────────────────────
    t = _log("Saving model and report...")
    model_path = Path(cfg.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "stage1":        m1,
            "stage2":        m2,
            "config":        asdict(cfg),
            "threshold":     threshold,
            "horizon":       horizon,
            "stage1_min_prob": stage1_prob,
            "stage2_min_prob": stage2_prob,
            "feature_shape": [int(X_tensor.shape[1]), int(X_tensor.shape[2])],
            "label_map":     {0: "down", 1: "flat", 2: "up"},
            "blocked_utc":   BLOCKED_UTC_WINDOWS,
        },
        model_path,
    )

    report = {
        "config":               asdict(cfg),
        "threshold_used":       threshold,
        "horizon_used":         horizon,
        "min_window_range_used": min_window_range,
        "stage1_min_prob":      stage1_prob,
        "stage2_min_prob":      stage2_prob,
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
        "full_confusion_labels": ["down", "flat", "up"],
        "directional_pnl":      pnl,
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

