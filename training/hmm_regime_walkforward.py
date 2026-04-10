#!/usr/bin/env python3
"""Walk-forward HMM regime prediction without lookahead bias.

This script predicts the *next bar* regime by fitting an HMM only on data
available up to the current bar. No future rows are used during each fit.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Allow running as a script from training/ while importing project modules.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import DataLoader
try:
    from training.training import prepare_gold_data
except ModuleNotFoundError:
    # Support direct script execution where `training` is not a package.
    from training import prepare_gold_data


@dataclass
class HMMConfig:
    n_states: int = 3
    train_window: int = 4000
    min_train_rows: int = 700
    lookback_vol: int = 20
    retrain_step: int = 1
    regime_threshold_pct: float = 0.03
    random_state: int = 7
    progress_every: int = 1000
    hmm_n_iter: int = 80
    horizon_bars: int = 15


def _load_prices(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv:
        csv_path = Path(args.input_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        raw = pd.read_csv(csv_path)
        if args.price_column not in raw.columns:
            raise ValueError(f"Missing price column '{args.price_column}' in {csv_path}")

        out = pd.DataFrame({"close": pd.to_numeric(raw[args.price_column], errors="coerce")})

        if args.timestamp_column and args.timestamp_column in raw.columns:
            ts = pd.to_datetime(raw[args.timestamp_column], utc=True, errors="coerce")
            out.index = ts
            out = out[~out.index.isna()]
        else:
            out.index = pd.RangeIndex(start=0, stop=len(out), step=1)

        out = out.dropna(subset=["close"]).copy()
        return out

    if not args.start_date or not args.end_date:
        raise ValueError("--start-date and --end-date are required when loading from database")

    raw = DataLoader().load_data(args.table, start_date=args.start_date, end_date=args.end_date)
    prepared = prepare_gold_data(raw)
    return prepared[["close"]].dropna().copy()


def _make_observations(prices: pd.DataFrame, lookback_vol: int, horizon_bars: int) -> pd.DataFrame:
    out = prices.copy()
    out["ret_1"] = np.log(out["close"] / out["close"].shift(1))
    out["ret_5"] = np.log(out["close"] / out["close"].shift(5))
    out["vol_20"] = out["ret_1"].rolling(lookback_vol, min_periods=lookback_vol).std()
    out["target_ret_h"] = out["close"].shift(-horizon_bars) / out["close"] - 1.0
    out = out.dropna().copy()
    return out


def _fit_hmm(
    train_obs: np.ndarray,
    n_states: int,
    random_state: int,
    n_iter: int,
) -> tuple[StandardScaler, Any]:
    try:
        hmm_module = importlib.import_module("hmmlearn.hmm")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install hmmlearn (pip install -r requirements.txt)") from exc

    hmm_cls = getattr(hmm_module, "GaussianHMM", None)
    if hmm_cls is None:
        raise RuntimeError("hmmlearn is installed but GaussianHMM is unavailable")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_obs)

    model = hmm_cls(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
        min_covar=1e-6,
    )
    model.fit(scaled)
    return scaler, model


def _state_to_regime_map(states: np.ndarray, ret_series: np.ndarray, n_states: int) -> dict[int, str]:
    state_means: list[tuple[int, float]] = []
    for s in range(n_states):
        mask = states == s
        mean_ret = float(np.nanmean(ret_series[mask])) if np.any(mask) else -np.inf
        state_means.append((s, mean_ret))

    state_means.sort(key=lambda x: x[1])

    mapping: dict[int, str] = {}
    if n_states == 1:
        mapping[state_means[0][0]] = "neutral"
        return mapping
    if n_states == 2:
        mapping[state_means[0][0]] = "bearish"
        mapping[state_means[1][0]] = "bullish"
        return mapping

    low_state = state_means[0][0]
    high_state = state_means[-1][0]
    mapping[low_state] = "bearish"
    mapping[high_state] = "bullish"
    for state_id, _ in state_means[1:-1]:
        mapping[state_id] = "neutral"
    return mapping


def _classify_actual_regime(ret_pct: float, threshold_pct: float) -> str:
    if ret_pct >= threshold_pct:
        return "bullish"
    if ret_pct <= -threshold_pct:
        return "bearish"
    return "neutral"


def _gaussian_diag_emission_probs(model: Any, x_scaled: np.ndarray) -> np.ndarray:
    """Compute p(x|state) for a scaled observation under a diag-covariance GaussianHMM."""
    means = np.asarray(model.means_, dtype="float64")
    covars = np.asarray(model.covars_, dtype="float64")
    if covars.ndim == 3:
        covars = np.diagonal(covars, axis1=1, axis2=2)
    covars = np.maximum(covars, 1e-9)

    diff = x_scaled[None, :] - means
    log_det = np.log(2.0 * np.pi * covars).sum(axis=1)
    quad = ((diff * diff) / covars).sum(axis=1)
    log_pdf = -0.5 * (log_det + quad)

    log_pdf -= np.max(log_pdf)
    probs = np.exp(log_pdf)
    total = probs.sum()
    if total <= 0.0:
        return np.full_like(probs, 1.0 / len(probs), dtype="float64")
    return probs / total


def _normalize_prob_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float64").reshape(-1)
    total = float(v.sum())
    if total <= 0.0:
        return np.full_like(v, 1.0 / len(v), dtype="float64")
    return v / total


def run_walkforward(prices: pd.DataFrame, cfg: HMMConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    if cfg.horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")

    obs = _make_observations(prices, lookback_vol=cfg.lookback_vol, horizon_bars=cfg.horizon_bars)
    if len(obs) < cfg.min_train_rows + 2:
        raise ValueError(
            "Not enough rows after feature prep. Increase date range or lower --min-train-rows."
        )

    x_cols = ["ret_1", "ret_5", "vol_20"]
    values = obs[x_cols].to_numpy(dtype="float64")
    target_rets = obs["target_ret_h"].to_numpy(dtype="float64")

    start_idx = cfg.min_train_rows
    last_fit_anchor = -1
    scaler: StandardScaler | None = None
    model: Any | None = None
    mapping: dict[int, str] = {}
    alpha_t: np.ndarray | None = None

    rows: list[dict[str, object]] = []
    total_steps = max(0, len(obs) - (start_idx + cfg.horizon_bars))
    est_refits = (total_steps + max(cfg.retrain_step, 1) - 1) // max(cfg.retrain_step, 1)
    if est_refits > 5000:
        print(
            f"Warning: estimated refits={est_refits} is very high. "
            "Consider larger --retrain-step (e.g. 200-1000) or lower --train-window.",
            flush=True,
        )

    trans_h = None
    last_trans_h_anchor = -1

    for anchor in range(start_idx, len(obs) - cfg.horizon_bars):
        # Fit only using data up to `anchor` (inclusive), then predict `anchor + 1`.
        should_refit = (model is None) or ((anchor - last_fit_anchor) >= cfg.retrain_step)
        train_start = max(0, anchor - cfg.train_window + 1)

        if should_refit:
            train_x = values[train_start : anchor + 1]
            train_ret_1 = values[train_start : anchor + 1, 0]

            scaler, model = _fit_hmm(
                train_x,
                n_states=cfg.n_states,
                random_state=cfg.random_state,
                n_iter=cfg.hmm_n_iter,
            )
            train_scaled = scaler.transform(train_x)
            train_states = model.predict(train_scaled)
            mapping = _state_to_regime_map(train_states, train_ret_1, cfg.n_states)
            alpha_t = _normalize_prob_vec(model.predict_proba(train_scaled)[-1])
            last_fit_anchor = anchor
            trans_h = np.linalg.matrix_power(model.transmat_, cfg.horizon_bars)
            last_trans_h_anchor = anchor

        assert scaler is not None
        assert model is not None
        assert alpha_t is not None

        if trans_h is None or last_trans_h_anchor != last_fit_anchor:
            trans_h = np.linalg.matrix_power(model.transmat_, cfg.horizon_bars)
            last_trans_h_anchor = last_fit_anchor

        future_state_probs = _normalize_prob_vec(alpha_t @ trans_h)
        pred_state = int(np.argmax(future_state_probs))
        pred_regime = mapping.get(pred_state, "neutral")

        bearish_prob = float(sum(future_state_probs[s] for s, r in mapping.items() if r == "bearish"))
        bullish_prob = float(sum(future_state_probs[s] for s, r in mapping.items() if r == "bullish"))
        neutral_prob = max(0.0, 1.0 - bearish_prob - bullish_prob)

        target_idx = anchor + cfg.horizon_bars
        actual_ret_pct = float(target_rets[target_idx] * 100.0)
        actual_regime = _classify_actual_regime(actual_ret_pct, cfg.regime_threshold_pct)

        rows.append(
            {
                "idx": int(target_idx),
                "timestamp": str(obs.index[target_idx]),
                "predicted_state": pred_state,
                "predicted_regime": pred_regime,
                "horizon_bars": int(cfg.horizon_bars),
                "bullish_prob": bullish_prob,
                "neutral_prob": neutral_prob,
                "bearish_prob": bearish_prob,
                "actual_ret_h_pct": actual_ret_pct,
                "actual_regime": actual_regime,
                "is_correct": int(pred_regime == actual_regime),
            }
        )

        obs_next_idx = anchor + 1
        x_next_scaled = scaler.transform(values[obs_next_idx : obs_next_idx + 1])[0]
        emission_next = _gaussian_diag_emission_probs(model, x_next_scaled)
        one_step_probs = _normalize_prob_vec(alpha_t @ model.transmat_)
        alpha_t = _normalize_prob_vec(one_step_probs * emission_next)

        if cfg.progress_every > 0:
            done = (anchor - start_idx) + 1
            if done % cfg.progress_every == 0 or done == total_steps:
                pct_done = (done / total_steps) * 100.0 if total_steps else 100.0
                print(
                    f"  Progress: {done}/{total_steps} ({pct_done:.1f}%) | "
                    f"rows={len(rows)} | last_ts={obs.index[target_idx]}",
                    flush=True,
                )

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise ValueError("No predictions generated. Check training window and date range.")

    summary = {
        "rows": float(len(pred_df)),
        "accuracy": float(pred_df["is_correct"].mean()),
        "pred_bull_share": float(pred_df["predicted_regime"].eq("bullish").astype(float).mean()),
        "pred_bear_share": float(pred_df["predicted_regime"].eq("bearish").astype(float).mean()),
        "actual_bull_share": float(pred_df["actual_regime"].eq("bullish").astype(float).mean()),
        "actual_bear_share": float(pred_df["actual_regime"].eq("bearish").astype(float).mean()),
    }
    return pred_df, summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward HMM regime prediction (next-bar, bias-safe).")
    p.add_argument("--input-csv", default=None, help="Optional price CSV path")
    p.add_argument("--timestamp-column", default="timestamp", help="CSV timestamp column")
    p.add_argument("--price-column", default="close", help="CSV close price column")

    p.add_argument("--table", default="gold_prices")
    p.add_argument("--start-date", default=None, help="YYYY-MM-DD (DB mode)")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD (DB mode)")

    p.add_argument("--n-states", type=int, default=3)
    p.add_argument("--train-window", type=int, default=4000)
    p.add_argument("--min-train-rows", type=int, default=700)
    p.add_argument("--lookback-vol", type=int, default=20)
    p.add_argument("--retrain-step", type=int, default=250)
    p.add_argument("--regime-threshold-pct", type=float, default=0.03, help="Actual regime threshold in percent")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--hmm-n-iter", type=int, default=80, help="EM iterations per HMM fit")
    p.add_argument("--horizon-bars", type=int, default=15, help="Predict regime this many bars ahead")
    p.add_argument("--progress-every", type=int, default=1000, help="Print progress every N predictions (0 disables)")

    p.add_argument("--out", required=True, help="Prediction CSV output path")
    return p


def main() -> int:
    args = build_parser().parse_args()

    cfg = HMMConfig(
        n_states=args.n_states,
        train_window=args.train_window,
        min_train_rows=args.min_train_rows,
        lookback_vol=args.lookback_vol,
        retrain_step=args.retrain_step,
        regime_threshold_pct=args.regime_threshold_pct,
        random_state=args.seed,
        progress_every=args.progress_every,
        hmm_n_iter=args.hmm_n_iter,
        horizon_bars=args.horizon_bars,
    )

    prices = _load_prices(args)
    pred_df, summary = run_walkforward(prices, cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)

    print("Walk-forward HMM complete (no lookahead in fitting):")
    print(f"  Output: {out_path}")
    print(f"  Rows: {int(summary['rows'])}")
    print(f"  Accuracy: {summary['accuracy']:.4f}")
    print(f"  Pred bullish share: {summary['pred_bull_share']:.4f}")
    print(f"  Pred bearish share: {summary['pred_bear_share']:.4f}")
    print(f"  Actual bullish share: {summary['actual_bull_share']:.4f}")
    print(f"  Actual bearish share: {summary['actual_bear_share']:.4f}")
    print(f"  Horizon bars: {cfg.horizon_bars}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

