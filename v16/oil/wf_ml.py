"""v16 oil walk-forward ML — 14-day grid + multi-model (xgb/et/lgb/hgb)."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from v16.config.oil_config import OIL_ML_CONFIG, OIL_MODEL_DIR
from v16.oil.sim_15m import sim_15m_long, sim_15m_short

STRUCT_COLS = (
    "struct_trend",
    "struct_dist_pts",
    "struct_leg_pts",
    "struct_pullback_pct",
    "struct_leg_age_15m",
    "struct_aligned",
)


def wf_test_windows(
    test_start: pd.Timestamp,
    ts_max: pd.Timestamp,
    *,
    retrain_freq: str = "14D",
    retrain_days: int = 14,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Rolling OOS windows — 14D grid or monthly (no gold hybrid_config dependency)."""
    freq = str(retrain_freq).upper()
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(ts_max)
    if start.tz is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tz is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if freq in ("14D", "V15", "2W", "BIWEEKLY"):
        step = pd.Timedelta(days=int(retrain_days))
        cursor = start
        while cursor <= end:
            w_end = cursor + step
            windows.append((cursor, w_end))
            cursor = w_end
        return windows

    for period in pd.period_range(start, end, freq="M"):
        m_start = period.start_time.tz_localize("UTC")
        m_end = (period + 1).start_time.tz_localize("UTC")
        windows.append((m_start, m_end))
    return windows


def model_path(leg: str, tag: str) -> Path:
    p = OIL_MODEL_DIR / leg / f"{tag}.joblib"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@lru_cache(maxsize=256)
def _load_model_cached(path_str: str):
    return joblib.load(path_str)


def _fit_xgb(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    w = np.where(y == 1)[0]
    l = np.where(y == 0)[0]
    spw = len(l) / max(1, len(w))
    m = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        scale_pos_weight=spw,
        random_state=42,
        verbosity=0,
    )
    m.fit(X, y)
    return m


def _fit_lgb(X: np.ndarray, y: np.ndarray):
    w = np.where(y == 1)[0]
    l = np.where(y == 0)[0]
    spw = len(l) / max(1, len(w))
    m = lgb.LGBMClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        class_weight=None,
        scale_pos_weight=spw,
        random_state=42,
        verbosity=-1,
    )
    m.fit(X, y)
    return m


def _fit_model(name: str, X: np.ndarray, y: np.ndarray):
    name = name.lower()
    if name == "xgb":
        return _fit_xgb(X, y)
    if name == "lgb":
        if not _HAS_LGB:
            raise ValueError("lightgbm not installed")
        return _fit_lgb(X, y)
    if name == "et":
        return ExtraTreesClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ).fit(X, y.astype(int))
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.06,
            max_iter=200,
            min_samples_leaf=20,
            random_state=42,
        ).fit(X, y.astype(int))
    raise ValueError(f"Unknown model: {name}")


def _predict_proba(model, name: str, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _feature_matrix(
    d15: pd.DataFrame,
    sigs: list[dict],
    sig_indices: list[int],
    feat_names: list[str],
    struct_frame: pd.DataFrame | None,
) -> np.ndarray:
    cols = list(feat_names)
    if struct_frame is not None:
        cols = cols + [c for c in STRUCT_COLS if c in struct_frame.columns]
    rows = []
    for si in sig_indices:
        idx = sigs[si]["idx"]
        row = d15.iloc[idx]
        vals = [float(row.get(f, 0)) for f in feat_names]
        if struct_frame is not None:
            ts = d15.index[idx]
            if ts in struct_frame.index:
                sr = struct_frame.loc[ts]
            else:
                sr = struct_frame.reindex([ts]).iloc[0]
            for c in STRUCT_COLS:
                if c in struct_frame.columns:
                    vals.append(float(sr.get(c, 0)))
        rows.append(vals)
    return np.array(rows, dtype=float)


def walk_forward_oil_leg(
    d15: pd.DataFrame,
    sigs: list[dict],
    tp: float,
    sl: float,
    feat_names: list[str],
    leg: str,
    *,
    model_name: str = "xgb",
    ml_th: float = 0.55,
    stype: str | None = None,
    struct_frame: pd.DataFrame | None = None,
    save_models: bool = True,
    retrain_freq: str | None = None,
    side: str = "long",
) -> tuple[list[float], list[dict], np.ndarray]:
    """14D (or monthly) walk-forward ML on a 15m oil leg."""
    st = stype or leg
    if side == "short":
        p, tr, m = sim_15m_short(d15, sigs, tp, sl, st)
    else:
        p, tr, m = sim_15m_long(d15, sigs, tp, sl, st)
    if len(p) < 30:
        return [], [], np.array([])

    n_m = len(m)
    p = p[:n_m]
    tr = tr[:n_m]
    y = np.array([1.0 if p[i] > 0 else 0.0 for i in range(n_m)])
    tdates = pd.DatetimeIndex([d15.index[sigs[si]["idx"]] for si in m])
    X_all = _feature_matrix(d15, sigs, m, feat_names, struct_frame)

    cfg = OIL_ML_CONFIG
    rf = str(retrain_freq or cfg["retrain_freq"])
    rd = int(cfg["retrain_days"])
    tdays = int(cfg["train_days"])
    min_rows = int(cfg["min_train_rows"])
    wf_start = pd.Timestamp(cfg["wf_start"], tz="UTC")
    test_start = max(tdates.min() + pd.Timedelta(days=tdays), wf_start)

    pr = np.zeros(len(p))
    trained_any = False

    for w_start, w_end in wf_test_windows(
        test_start, tdates.max(), retrain_freq=rf, retrain_days=rd
    ):
        train_mask = tdates < w_start
        test_mask = (tdates >= w_start) & (tdates < w_end)
        if train_mask.sum() < min_rows or test_mask.sum() == 0:
            continue

        X_tr, y_tr = X_all[train_mask], y[train_mask]
        w_idx = np.where(y_tr == 1)[0]
        l_idx = np.where(y_tr == 0)[0]
        nm = min(len(w_idx), len(l_idx))
        if nm < 5:
            continue
        rng = np.random.RandomState(42 + int(w_start.strftime("%Y%m%d")))
        bal = np.concatenate([rng.choice(w_idx, nm, 0), rng.choice(l_idx, nm, 0)])
        Xb, yb = X_tr[bal], y_tr[bal]

        model = _fit_model(model_name, Xb, yb)
        trained_any = True
        tag = w_start.strftime("%Y-%m-%d")
        if save_models:
            joblib.dump(model, model_path(leg, tag))

        probs = _predict_proba(model, model_name, X_all[test_mask])
        for j, idx in enumerate(np.where(test_mask)[0]):
            pr[idx] = probs[j]

    if not trained_any and len(p) > 0:
        # Unscored trades stay at prob 0 — excluded by ML threshold (no pass-all fallback)
        pass

    return list(p), tr, pr


def filter_trades_by_ml(
    pnls: list[float],
    trades: list[dict],
    probas: np.ndarray,
    ml_th: float,
    leg: str,
) -> list[dict]:
    out = []
    for i in range(len(trades)):
        if probas[i] >= ml_th:
            out.append({**trades[i], "_leg": leg, "_prob": float(probas[i])})
    return out


def model_tag_for_bar(bar_ts: pd.Timestamp) -> str | None:
    """14D OOS window tag (model filename date) for a signal timestamp."""
    cfg = OIL_ML_CONFIG
    wf_start = pd.Timestamp(cfg["wf_start"], tz="UTC")
    ts = pd.Timestamp(bar_ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if ts < wf_start:
        return None
    step = pd.Timedelta(days=int(cfg["retrain_days"]))
    cursor = wf_start
    while cursor + step <= ts:
        cursor = cursor + step
    return cursor.strftime("%Y-%m-%d")


def _row_features(
    row,
    feat_names: list[str],
    struct_row: pd.Series | None = None,
) -> np.ndarray:
    vals = [float(row.get(f, 0) if hasattr(row, "get") else row[f]) for f in feat_names]
    if struct_row is not None:
        for c in STRUCT_COLS:
            if c in struct_row.index:
                vals.append(float(struct_row.get(c, 0)))
    return np.array([vals], dtype=float)


def score_oil_leg(
    leg: str,
    bar_ts: pd.Timestamp,
    row,
    feat_names: list[str],
    model_name: str,
    struct_row: pd.Series | None = None,
) -> float | None:
    """Point-in-time ML score using saved 14D walk-forward model."""
    tag = model_tag_for_bar(bar_ts)
    if tag is None:
        return None
    path = model_path(leg, tag)
    if not path.exists():
        return None
    model = _load_model_cached(str(path))
    X = _row_features(row, feat_names, struct_row)
    return float(_predict_proba(model, model_name, X)[0])
